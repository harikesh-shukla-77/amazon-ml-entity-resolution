#!/usr/bin/env python3
"""Amazon ML Challenge — final one-shot test pipeline (V3, memory-safe).

Uses the exact V10 feature definition and pairwise training recipe, but builds
training features on disk with NumPy memmaps so the process does not need to
hold millions of pair rows in RAM at once.

Stages:
  1. Rebuild S2/S3 V10 pairwise LogisticRegression models from V10 labeled data.
  2. Generate test candidates from the blocking families validated on train.
  3. Compute the same V6/V10 fuzzy features and transparent base score.
  4. Rank candidates by base score, create V10 model utility, and rank again.
  5. Write top1/top5/top10/top20 Parquets and CSV submissions.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/harikeshshukla/mla')
PROC = ROOT/'processed_dataset'
TRAIN = PROC/'train'
TEST = PROC/'test'
V10 = ROOT/'candidate_output_v10'
OUT = ROOT/'candidate_output_final_v6'
TMP = OUT/'duckdb_tmp'
PAIR_TMP = OUT/'pair_training_tmp'
RAW_TMP = OUT/'raw_blocks'
BASE_TMP = OUT/'base_parts'
UTIL_TMP = OUT/'utility_parts'
for p in (OUT,TMP,PAIR_TMP,RAW_TMP,BASE_TMP,UTIL_TMP):
    p.mkdir(parents=True, exist_ok=True)

S1_TEST=TEST/'test_source1.parquet'
S2_TEST=TEST/'test_source2.parquet'
S3_TEST=TEST/'test_source3.parquet'
LABELED_S2=V10/'train_labeled_base_s1_s2.parquet'
LABELED_S3=V10/'train_labeled_base_s1_s3.parquet'

MEMORY_LIMIT=os.environ.get('FINAL_V6_MEMORY','4GB')
THREADS=int(os.environ.get('FINAL_V6_THREADS','2'))
BATCH=int(os.environ.get('FINAL_V6_BATCH','100000'))
TOPKS=(1,5,10,20)
FEATURES=[
    'name_similarity','address_similarity','name_exact','address_exact','country_exact',
    'name_length_ratio','address_length_ratio','evidence_rows','evidence_file_count',
    'exact_key_count','base_score','log_base_rank','name_x_address','name_minus_address',
    'similarity_mean','similarity_min','exact_field_count',
]


def header(s):
    print('\n'+'='*100,flush=True); print(s,flush=True); print('='*100,flush=True)

def sql_path(p):
    return str(p).replace("'","''")

def qp(p):
    return "'"+sql_path(p)+"'"

def rows(con,p):
    return int(con.execute(f'SELECT COUNT(*) FROM read_parquet({qp(p)})').fetchone()[0])

def cols(con,p):
    return [r[0] for r in con.execute(f'DESCRIBE SELECT * FROM read_parquet({qp(p)})').fetchall()]

def conn():
    c=duckdb.connect()
    c.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    c.execute(f'PRAGMA threads={THREADS}')
    c.execute('PRAGMA preserve_insertion_order=false')
    c.execute(f"PRAGMA temp_directory='{sql_path(TMP)}'")
    return c

def require(p):
    if not p.exists(): raise FileNotFoundError(p)

def len_ratio(a,b):
    la=len(a); lb=len(b)
    if la==0 and lb==0: return 1.0
    if la==0 or lb==0: return 0.0
    return min(la,lb)/max(la,lb)

def processed_views(con,s1,target):
    for name,path in [('left_base',s1),('right_base',target)]:
        con.execute(f"""
        CREATE OR REPLACE TEMP VIEW {name} AS
        SELECT
          entity_id,
          business_name,
          business_address,
          country,
          coalesce(name_norm,'') AS name_norm,
          coalesce(address_norm,'') AS address_norm,
          coalesce(country_norm,'') AS country_norm,
          coalesce(name_key,'') AS name_key,
          coalesce(name_country_key,'') AS name_country_key,
          coalesce(address_country_key,'') AS address_country_key,
          coalesce(name_address_key,'') AS name_address_key,
          coalesce(full_key,'') AS full_key,
          coalesce(list_extract(regexp_extract_all(coalesce(address_norm,''),'[0-9]{{5,6}}'),-1),'') AS postal,
          regexp_replace(coalesce(address_norm,''),'[^0-9]','','g') AS addr_num
        FROM read_parquet({qp(path)})
        """)

def pair_sql(labeled):
    p=qp(labeled)
    return f"""
    WITH positives AS (
      SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY source1_entity_id ORDER BY base_rank, candidate_entity_id) AS p_rank
        FROM read_parquet({p})
        WHERE label=1
      ) WHERE p_rank<=1
    ),
    negatives AS (
      SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY source1_entity_id ORDER BY base_rank, candidate_entity_id) AS n_rank
        FROM read_parquet({p})
        WHERE label=0 AND base_rank<=100
      ) WHERE n_rank<=2
    ),
    joined AS (
      SELECT
        p.name_similarity pn,n.name_similarity nn,p.address_similarity pa,n.address_similarity na,
        p.name_exact pe,n.name_exact ne,p.address_exact pae,n.address_exact nae,
        p.country_exact pc,n.country_exact nc,p.name_length_ratio pl,n.name_length_ratio nl,
        p.address_length_ratio pal,n.address_length_ratio nal,p.evidence_rows per,n.evidence_rows ner,
        p.evidence_file_count pef,n.evidence_file_count nef,p.exact_key_count pek,n.exact_key_count nek,
        p.base_score ps,n.base_score ns,p.base_rank pr,n.base_rank nr
      FROM positives p JOIN negatives n USING(source1_entity_id)
    ),
    f AS (
      SELECT
        pn-nn AS name_similarity,
        pa-na AS address_similarity,
        pe-ne AS name_exact,
        pae-nae AS address_exact,
        pc-nc AS country_exact,
        pl-nl AS name_length_ratio,
        pal-nal AS address_length_ratio,
        per-ner AS evidence_rows,
        pef-nef AS evidence_file_count,
        pek-nek AS exact_key_count,
        ps-ns AS base_score,
        LN(1+pr)-LN(1+nr) AS log_base_rank,
        pn*pa-nn*na AS name_x_address,
        (pn-pa)-(nn-na) AS name_minus_address,
        ((pn+pa)/2)-((nn+na)/2) AS similarity_mean,
        LEAST(pn,pa)-LEAST(nn,na) AS similarity_min,
        (pe+pae+pc)-(ne+nae+nc) AS exact_field_count,
        1 AS pair_label
      FROM joined
    )
    SELECT * FROM f
    UNION ALL
    SELECT -name_similarity,-address_similarity,-name_exact,-address_exact,-country_exact,
           -name_length_ratio,-address_length_ratio,-evidence_rows,-evidence_file_count,
           -exact_key_count,-base_score,-log_base_rank,-name_x_address,-name_minus_address,
           -similarity_mean,-similarity_min,-exact_field_count,0 AS pair_label
    FROM f
    """

def build_pair_parquet(con,labeled,tag):
    out=PAIR_TMP/f'{tag}_pairs.parquet'
    if out.exists():
        return out,rows(con,out)
    header(f'BUILDING V10 PAIR TRAINING ON DISK: {tag}')
    con.execute(f"COPY ({pair_sql(labeled)}) TO {qp(out)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    n=rows(con,out)
    print(f'{tag} pair rows on disk: {n:,}',flush=True)
    return out,n

def fit_exact_v10(con,labeled,tag):
    pair_path,n=build_pair_parquet(con,labeled,tag)
    mm_x_path=PAIR_TMP/f'{tag}_X.dat'; mm_y_path=PAIR_TMP/f'{tag}_y.dat'
    X=np.memmap(mm_x_path,mode='w+',dtype='float32',shape=(n,len(FEATURES)))
    y=np.memmap(mm_y_path,mode='w+',dtype='int8',shape=(n,))

    header(f'PASS 1/2 STANDARDIZER: {tag}')
    scaler=StandardScaler()
    reader=con.execute(f'SELECT {", ".join(FEATURES)}, pair_label FROM read_parquet({qp(pair_path)})').to_arrow_reader(batch_size=BATCH)
    pos=0
    for b in reader:
        d=b.to_pandas()
        x=d[FEATURES].apply(pd.to_numeric,errors='coerce').fillna(0).astype('float32').to_numpy()
        yy=d.pair_label.astype('int8').to_numpy()
        scaler.partial_fit(x)
        k=len(d); X[pos:pos+k]=x; y[pos:pos+k]=yy; pos+=k
        if pos and pos%500000 < k: print(f'{tag} loaded: {pos:,}',flush=True)
    X.flush(); y.flush()

    header(f'PASS 2/2 LOGISTIC REGRESSION: {tag}')
    # Transform in-place to avoid a second full in-memory copy.
    mean=scaler.mean_.astype('float32'); scale=scaler.scale_.astype('float32'); scale[scale==0]=1.0
    for start in range(0,n,BATCH):
        end=min(n,start+BATCH)
        X[start:end]=(X[start:end]-mean)/scale
    X.flush()

    model=LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs',random_state=42)
    t=time.time(); model.fit(X,y); print(f'{tag} model fit: {time.time()-t:.1f}s',flush=True)

    meta={
      'features':FEATURES,
      'coef_scaled':model.coef_[0].astype(float).tolist(),
      'intercept':float(model.intercept_[0]),
      'scaler_mean':mean.astype(float).tolist(),
      'scaler_scale':scale.astype(float).tolist(),
    }
    X._mmap.close(); y._mmap.close()
    del X,y
    try: pair_path.unlink()
    except OSError: pass
    try: mm_x_path.unlink(); mm_y_path.unlink()
    except OSError: pass
    return meta

def make_test_candidates(con,target_name,target_path,out_path):
    if out_path.exists(): return rows(con,out_path)
    processed_views(con,S1_TEST,target_path)
    raw=RAW_TMP/target_name.lower()
    raw.mkdir(parents=True,exist_ok=True)
    blocks=[]
    for key in ('full_key','name_address_key','name_country_key','address_country_key','name_key'):
        blocks.append((f'v1_{key}',f"""
          SELECT l.entity_id source1_entity_id,r.entity_id candidate_entity_id,
                 'V1_{key}' evidence_group,1 is_exact
          FROM left_base l JOIN right_base r
          ON l.country_norm=r.country_norm AND l.{key}=r.{key}
          WHERE l.country_norm<>'' AND l.{key}<>''
        """))
    blocks.append(('v2_postal',"""
      SELECT l.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 'V2_postal' AS evidence_group, 0 AS is_exact
      FROM left_base l JOIN right_base r
      ON l.country_norm=r.country_norm AND l.postal=r.postal
      WHERE l.country_norm<>'' AND l.postal<>''
    """))
    if target_name=='S3':
        blocks.append(('v4_address_first4_last4_len',"""
          SELECT l.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 'V4_addr_shape' AS evidence_group, 0 AS is_exact
          FROM left_base l JOIN right_base r
          ON l.country_norm=r.country_norm
          AND substr(l.address_norm,1,4)=substr(r.address_norm,1,4)
          AND right(l.address_norm,4)=right(r.address_norm,4)
          AND length(l.address_norm)=length(r.address_norm)
          WHERE l.country_norm<>'' AND l.address_norm<>'' AND r.address_norm<>''
        """))
        # Safe V4 numeric-address block: reproduce the train-side idea but
        # cap each (country, numeric-address) bucket at 80 rows on BOTH sides.
        # The unrestricted equality version expands to ~4.18B S3 test pairs
        # and is unsafe; the train V4 block used the same small-block principle.
        blocks.append(('v4_num_address_safe',"""
          WITH lkeys AS (
            SELECT country_norm, addr_num
            FROM left_base
            WHERE country_norm<>'' AND addr_num<>''
            GROUP BY country_norm, addr_num
            HAVING COUNT(*)<=80
          ),
          rkeys AS (
            SELECT country_norm, addr_num
            FROM right_base
            WHERE country_norm<>'' AND addr_num<>''
            GROUP BY country_norm, addr_num
            HAVING COUNT(*)<=80
          )
          SELECT l.entity_id AS source1_entity_id,
                 r.entity_id AS candidate_entity_id,
                 'V4_num_address_safe' AS evidence_group,
                 0 AS is_exact
          FROM left_base l
          JOIN lkeys lk
            ON lk.country_norm=l.country_norm AND lk.addr_num=l.addr_num
          JOIN rkeys rk
            ON rk.country_norm=l.country_norm AND rk.addr_num=l.addr_num
          JOIN right_base r
            ON r.country_norm=rk.country_norm AND r.addr_num=rk.addr_num
        """))
    blocks.append(('v4_postal_name',"""
      SELECT l.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 'V4_postal_name' AS evidence_group, 0 AS is_exact
      FROM left_base l JOIN right_base r
      ON l.country_norm=r.country_norm AND l.postal=r.postal AND l.name_norm=r.name_norm
      WHERE l.country_norm<>'' AND l.postal<>'' AND l.name_norm<>''
    """))
    blocks.append(('v5_name_first3_postal',"""
      SELECT l.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 'V5_name_first3_postal' AS evidence_group, 0 AS is_exact
      FROM left_base l JOIN right_base r
      ON l.country_norm=r.country_norm
      AND substr(l.name_norm,1,3)=substr(r.name_norm,1,3)
      AND l.postal=r.postal
      WHERE l.country_norm<>'' AND l.postal<>'' AND substr(l.name_norm,1,3)<>''
    """))
    blocks.append(('v5_postal_addr_tail5',"""
      SELECT l.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 'V5_postal_addr_tail5' AS evidence_group, 0 AS is_exact
      FROM left_base l JOIN right_base r
      ON l.country_norm=r.country_norm AND l.postal=r.postal
      AND right(l.address_norm,5)=right(r.address_norm,5)
      WHERE l.country_norm<>'' AND l.postal<>'' AND l.address_norm<>'' AND r.address_norm<>''
    """))
    blocks.append(('v9_num_addr_name_prefix3',"""
      SELECT l.entity_id AS source1_entity_id, r.entity_id AS candidate_entity_id, 'V9.1' AS evidence_group, 0 AS is_exact
      FROM left_base l JOIN right_base r
      ON l.country_norm=r.country_norm AND l.addr_num=r.addr_num
      AND substr(l.name_norm,1,3)=substr(r.name_norm,1,3)
      WHERE l.country_norm<>'' AND l.addr_num<>'' AND substr(l.name_norm,1,3)<>''
    """))
    paths=[]
    for name,sql in blocks:
        p=raw/f'{name}.parquet'; paths.append(p)
        if p.exists():
            print(f'  reuse {p.name} ({rows(con,p):,})',flush=True); continue
        print(f'  generate {name}',flush=True)
        con.execute(f'COPY ({sql}) TO {qp(p)} (FORMAT PARQUET, COMPRESSION ZSTD)')
        print(f'    rows: {rows(con,p):,}',flush=True)
    # Safety check: every selected raw block must expose the same four columns.
    bad=[]
    for rp in paths:
        rc=cols(con,rp)
        expected={'source1_entity_id','candidate_entity_id','evidence_group','is_exact'}
        if not expected.issubset(set(rc)):
            bad.append((str(rp),rc))
    if bad:
        raise RuntimeError('Raw-block schema mismatch after generation: '+repr(bad))

    # Explicit file list: never use a wildcard glob here, because an old
    # failed intermediate (especially the 4.18B-pair S3 numeric block) must
    # not accidentally be pulled into consolidation.
    read_list = '[' + ','.join(qp(p) for p in paths) + ']'
    con.execute(f"""
      COPY (
        SELECT source1_entity_id,candidate_entity_id,
               COUNT(*)::INTEGER evidence_rows,
               COUNT(DISTINCT evidence_group)::INTEGER evidence_file_count,
               LEAST(5,SUM(is_exact))::INTEGER exact_key_count
        FROM read_parquet({read_list}, union_by_name=true)
        GROUP BY source1_entity_id,candidate_entity_id
      ) TO {qp(out_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    return rows(con,out_path)

def base_parts_and_rank(con,target_name,target_path,candidate_path):
    processed_views(con,S1_TEST,target_path)
    bdir=BASE_TMP/target_name.lower(); bdir.mkdir(parents=True,exist_ok=True)
    parts=sorted(bdir.glob('*.parquet'))
    if not parts:
        reader=con.execute(f"""
        SELECT c.source1_entity_id,c.candidate_entity_id,c.evidence_rows,c.evidence_file_count,c.exact_key_count,
               l.name_norm l_name,r.name_norm r_name,l.address_norm l_addr,r.address_norm r_addr,
               l.country_norm l_country,r.country_norm r_country
        FROM read_parquet({qp(candidate_path)}) c
        JOIN left_base l ON l.entity_id=c.source1_entity_id
        JOIN right_base r ON r.entity_id=c.candidate_entity_id
        """).to_arrow_reader(batch_size=BATCH)
        total=0; idx=0
        for ab in reader:
            d=ab.to_pandas()
            ln=d.l_name.fillna('').astype(str).tolist(); rn=d.r_name.fillna('').astype(str).tolist()
            la=d.l_addr.fillna('').astype(str).tolist(); ra=d.r_addr.fillna('').astype(str).tolist()
            lc=d.l_country.fillna('').astype(str).tolist(); rc=d.r_country.fillna('').astype(str).tolist()
            ns=np.empty(len(d),dtype='float32'); ads=np.empty(len(d),dtype='float32'); ne=np.empty(len(d),dtype='float32'); ae=np.empty(len(d),dtype='float32'); ce=np.empty(len(d),dtype='float32'); nl=np.empty(len(d),dtype='float32'); al=np.empty(len(d),dtype='float32'); base=np.empty(len(d),dtype='float32')
            for i in range(len(d)):
                ns[i]=fuzz.ratio(ln[i],rn[i])/100.0 if ln[i] and rn[i] else 0.0
                ads[i]=fuzz.ratio(la[i],ra[i])/100.0 if la[i] and ra[i] else 0.0
                ne[i]=1.0 if ln[i] and rn[i] and ln[i]==rn[i] else 0.0
                ae[i]=1.0 if la[i] and ra[i] and la[i]==ra[i] else 0.0
                ce[i]=1.0 if lc[i] and rc[i] and lc[i]==rc[i] else 0.0
                nl[i]=len_ratio(ln[i],rn[i]); al[i]=len_ratio(la[i],ra[i])
            ev=pd.to_numeric(d.evidence_file_count,errors='coerce').fillna(0).clip(0,4).to_numpy(dtype='float32')
            ek=pd.to_numeric(d.exact_key_count,errors='coerce').fillna(0).clip(0,5).to_numpy(dtype='float32')
            base=30*ne+25*ae+8*ce+22*ns+12*ads+2*nl+al+0.75*ev+0.25*ek
            out=pd.DataFrame({
                'source1_entity_id':d.source1_entity_id.astype(str), 'candidate_entity_id':d.candidate_entity_id.astype(str),
                'evidence_rows':pd.to_numeric(d.evidence_rows,errors='coerce').fillna(0).astype('float32'),
                'evidence_file_count':ev,'exact_key_count':ek,
                'name_similarity':ns,'address_similarity':ads,'name_exact':ne,'address_exact':ae,'country_exact':ce,
                'name_length_ratio':nl,'address_length_ratio':al,'base_score':base,
            })
            out.to_parquet(bdir/f'part_{idx:05d}.parquet',index=False,compression='zstd'); idx+=1; total+=len(out)
            if total and total%1000000 < len(out): print(f'  {target_name} base scored: {total:,}',flush=True)
    ranked=OUT/f'test_base_ranked_s1_{target_name.lower()}.parquet'
    if not ranked.exists():
        con.execute(f"""
        COPY (
          SELECT *,ROW_NUMBER() OVER(PARTITION BY source1_entity_id ORDER BY base_score DESC,candidate_entity_id) base_rank
          FROM read_parquet({qp(bdir/'*.parquet')})
        ) TO {qp(ranked)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    return ranked

def model_score(con,target_name,ranked,meta):
    out=OUT/f'test_scored_s1_{target_name.lower()}.parquet'
    if out.exists(): return rows(con,out)
    scaler_mean=np.array(meta['scaler_mean'],dtype='float32'); scaler_scale=np.array(meta['scaler_scale'],dtype='float32'); scaler_scale[scaler_scale==0]=1.0
    coef=np.array(meta['coef_scaled'],dtype='float32'); intercept=np.float32(meta['intercept'])
    udir=UTIL_TMP/target_name.lower(); udir.mkdir(parents=True,exist_ok=True)
    reader=con.execute(f'SELECT * FROM read_parquet({qp(ranked)})').to_arrow_reader(batch_size=BATCH)
    total=0; idx=0
    for ab in reader:
        d=ab.to_pandas()
        x=pd.DataFrame(index=d.index)
        for c in ['name_similarity','address_similarity','name_exact','address_exact','country_exact','name_length_ratio','address_length_ratio','evidence_rows','evidence_file_count','exact_key_count','base_score']:
            x[c]=pd.to_numeric(d[c],errors='coerce').fillna(0).astype('float32')
        rank=pd.to_numeric(d.base_rank,errors='coerce').fillna(1000000).astype('float64')
        x['log_base_rank']=np.log1p(rank).astype('float32')
        x['name_x_address']=(x.name_similarity*x.address_similarity).astype('float32')
        x['name_minus_address']=(x.name_similarity-x.address_similarity).astype('float32')
        x['similarity_mean']=((x.name_similarity+x.address_similarity)/2).astype('float32')
        x['similarity_min']=np.minimum(x.name_similarity,x.address_similarity).astype('float32')
        x['exact_field_count']=(x.name_exact+x.address_exact+x.country_exact).astype('float32')
        X=((x[FEATURES].to_numpy(dtype='float32')-scaler_mean)/scaler_scale)
        utility=(X@coef)+intercept
        pd.DataFrame({'source1_entity_id':d.source1_entity_id.astype(str),'candidate_entity_id':d.candidate_entity_id.astype(str),'v10_utility':utility.astype('float32')}).to_parquet(udir/f'part_{idx:05d}.parquet',index=False,compression='zstd')
        idx+=1; total+=len(d)
        if total and total%1000000 < len(d): print(f'  {target_name} model scored: {total:,}',flush=True)
    con.execute(f"""
      COPY (
        SELECT source1_entity_id,candidate_entity_id,v10_utility,
               ROW_NUMBER() OVER(PARTITION BY source1_entity_id ORDER BY v10_utility DESC,candidate_entity_id) v10_rank
        FROM read_parquet({qp(udir/'*.parquet')})
      ) TO {qp(out)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    return rows(con,out)

def topk(con,target_name,scored,k):
    out=OUT/f'test_ranked_top{k}_s1_{target_name.lower()}.parquet'
    if out.exists(): return
    con.execute(f"""
      COPY (
        SELECT source1_entity_id,candidate_entity_id,v10_utility,v10_rank AS rank
        FROM read_parquet({qp(scored)})
        WHERE v10_rank<={k}
      ) TO {qp(out)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

def submission(con,k,s2,s3):
    out=OUT/f'submission_top{k}.csv'
    if out.exists(): return
    con.execute(f"""
      COPY (
        WITH a AS (SELECT source1_entity_id,string_agg(candidate_entity_id,',' ORDER BY rank,candidate_entity_id) ids FROM read_parquet({qp(s2)}) GROUP BY source1_entity_id),
             b AS (SELECT source1_entity_id,string_agg(candidate_entity_id,',' ORDER BY rank,candidate_entity_id) ids FROM read_parquet({qp(s3)}) GROUP BY source1_entity_id)
        SELECT t.entity_id source1_entity_id,concat_ws(',',NULLIF(a.ids,''),NULLIF(b.ids,'')) matched_entity_ids
        FROM read_parquet({qp(S1_TEST)}) t LEFT JOIN a ON a.source1_entity_id=t.entity_id LEFT JOIN b ON b.source1_entity_id=t.entity_id
        ORDER BY t.entity_id
      ) TO {qp(out)} (HEADER,DELIMITER ',')
    """)

def main():
    t0=time.time(); header('AMAZON ML CHALLENGE — FINAL ONE-SHOT TEST PIPELINE V6')
    print(f'Project: {ROOT}\nOutput : {OUT}\nMemory : {MEMORY_LIMIT}\nThreads: {THREADS}\nBatch  : {BATCH}',flush=True)

    # Cleanup only the known-bad 4.18B-pair S3 intermediate produced by V4.
    # It is not used by this safe pipeline and only wastes disk space.
    bad_intermediate = ROOT/'candidate_output_final_v4/raw_blocks/s3/v4_num_address.parquet'
    if bad_intermediate.exists():
        try:
            bad_intermediate.unlink()
            print(f'Removed obsolete V4 S3 numeric intermediate: {bad_intermediate}',flush=True)
        except OSError as exc:
            print(f'WARNING: could not remove obsolete intermediate: {exc}',flush=True)
    for p in (S1_TEST,S2_TEST,S3_TEST,LABELED_S2,LABELED_S3): require(p)
    con=conn()
    try:
        header('REBUILDING EXACT V10 PAIRWISE MODELS — DISK BACKED')
        meta_s2=fit_exact_v10(con,LABELED_S2,'S2')
        meta_s3=fit_exact_v10(con,LABELED_S3,'S3')
        (OUT/'final_model_coefficients.json').write_text(json.dumps({'features':FEATURES,'S2':meta_s2,'S3':meta_s3},indent=2),encoding='utf-8')

        header('TEST CANDIDATE GENERATION')
        c2=OUT/'test_candidates_s1_s2.parquet'; c3=OUT/'test_candidates_s1_s3.parquet'
        print(f'S2 candidates: {make_test_candidates(con,"S2",S2_TEST,c2):,}',flush=True)
        print(f'S3 candidates: {make_test_candidates(con,"S3",S3_TEST,c3):,}',flush=True)

        header('TEST BASE SCORING + GLOBAL BASE RANK')
        r2=base_parts_and_rank(con,'S2',S2_TEST,c2); r3=base_parts_and_rank(con,'S3',S3_TEST,c3)

        header('TEST V10 MODEL SCORING')
        s2=OUT/'test_scored_s1_s2.parquet'; s3=OUT/'test_scored_s1_s3.parquet'
        print(f'S2 scored rows: {model_score(con,"S2",r2,meta_s2):,}',flush=True)
        print(f'S3 scored rows: {model_score(con,"S3",r3,meta_s3):,}',flush=True)

        header('TOP-K + SUBMISSIONS')
        for k in TOPKS:
            topk(con,'S2',s2,k); topk(con,'S3',s3,k)
            submission(con,k,OUT/f'test_ranked_top{k}_s1_s2.parquet',OUT/f'test_ranked_top{k}_s1_s3.parquet')
            print(f'Created submission_top{k}.csv',flush=True)

        manifest={'output':str(OUT),'test_rows':{'S1':rows(con,S1_TEST),'S2':rows(con,S2_TEST),'S3':rows(con,S3_TEST)},'candidate_rows':{'S2':rows(con,c2),'S3':rows(con,c3)},'scored_rows':{'S2':rows(con,s2),'S3':rows(con,s3)},'submissions':[str(OUT/f'submission_top{k}.csv') for k in TOPKS],'elapsed_seconds':round(time.time()-t0,2)}
        (OUT/'final_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        header('✅ FINAL ONE-SHOT PIPELINE COMPLETED')
        print(json.dumps(manifest,indent=2),flush=True)
    finally:
        con.close()

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('Interrupted.',file=sys.stderr); raise
