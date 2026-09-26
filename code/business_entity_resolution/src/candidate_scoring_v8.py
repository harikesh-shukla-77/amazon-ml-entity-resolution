
from pathlib import Path
import json
import time
import duckdb
import numpy as np
import pandas as pd

PROJECT = Path('/Users/harikeshshukla/mla')
TRAIN = PROJECT / 'processed_dataset/train'
GT = TRAIN / 'ground_truth_pairs.parquet'
V6 = PROJECT / 'candidate_output_v6'
OUT = PROJECT / 'candidate_output_v8'
TMP = OUT / 'tmp'

S2 = V6 / 'train_scored_s1_s2.parquet'
S3 = V6 / 'train_scored_s1_s3.parquet'

MEMORY = '4GB'
THREADS = 2
BATCH = 100_000
VALIDATION_MOD = 5
NEG_RANK = 100
NEG_PER_S1 = 2
POS_PER_S1 = 1
TOPK = 20

PAIR_FEATURES = [
    'name_similarity','address_similarity','name_exact','address_exact',
    'country_exact','name_length_ratio','address_length_ratio',
    'evidence_rows','evidence_file_count','exact_key_count','v6_score',
    'log_v6_rank','name_x_address','name_minus_address',
    'similarity_mean','similarity_min','exact_field_count'
]

def header(x):
    print('\n' + '='*100)
    print(x)
    print('='*100)

def qp(p):
    return "'" + str(p).replace("'", "''") + "'"

def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect(':memory:')
    td = TMP / 'duckdb_tmp'
    td.mkdir(parents=True, exist_ok=True)
    t = str(td).replace(chr(92), '/').replace(chr(39), chr(39)*2)
    c.execute(f"PRAGMA memory_limit='{MEMORY}'")
    c.execute(f'PRAGMA threads={THREADS}')
    c.execute(f"PRAGMA temp_directory='{t}'")
    return c

def make_features(df):
    x = pd.DataFrame(index=df.index)
    cols = [
        'name_similarity','address_similarity','name_exact','address_exact',
        'country_exact','name_length_ratio','address_length_ratio',
        'evidence_rows','evidence_file_count','exact_key_count','v6_score'
    ]
    for col in cols:
        x[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
    r = pd.to_numeric(df['v6_rank'], errors='coerce').fillna(1e6).astype('float64')
    x['log_v6_rank'] = np.log1p(r).astype('float32')
    x['name_x_address'] = (x['name_similarity'] * x['address_similarity']).astype('float32')
    x['name_minus_address'] = (x['name_similarity'] - x['address_similarity']).astype('float32')
    x['similarity_mean'] = ((x['name_similarity'] + x['address_similarity']) / 2).astype('float32')
    x['similarity_min'] = np.minimum(x['name_similarity'], x['address_similarity']).astype('float32')
    x['exact_field_count'] = (x['name_exact'] + x['address_exact'] + x['country_exact']).astype('float32')
    return x[PAIR_FEATURES]

def load_gt(c):
    c.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_s2 AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_parquet({qp(GT)})
        WHERE COALESCE(label,1)=1 AND starts_with(matched_entity_id,'S2-')
    """)
    c.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_s3 AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_parquet({qp(GT)})
        WHERE COALESCE(label,1)=1 AND starts_with(matched_entity_id,'S3-')
    """)

def labeled(c, path, gt, name):
    t = f'labeled_{name.lower()}'
    c.execute(f"""
        CREATE OR REPLACE TEMP TABLE {t} AS
        SELECT
            s.source1_entity_id, s.candidate_entity_id,
            CAST(s.rank AS BIGINT) v6_rank, CAST(s.score AS DOUBLE) v6_score,
            CAST(s.name_similarity AS DOUBLE) name_similarity,
            CAST(s.address_similarity AS DOUBLE) address_similarity,
            CAST(s.name_exact AS INTEGER) name_exact,
            CAST(s.address_exact AS INTEGER) address_exact,
            CAST(s.country_exact AS INTEGER) country_exact,
            CAST(s.name_length_ratio AS DOUBLE) name_length_ratio,
            CAST(s.address_length_ratio AS DOUBLE) address_length_ratio,
            CAST(s.evidence_rows AS DOUBLE) evidence_rows,
            CAST(s.evidence_file_count AS DOUBLE) evidence_file_count,
            CAST(s.exact_key_count AS DOUBLE) exact_key_count,
            CASE WHEN g.matched_entity_id IS NULL THEN 0 ELSE 1 END AS match_label,
            CASE WHEN MOD(ABS(HASH(s.source1_entity_id)),{VALIDATION_MOD})=0 THEN 1 ELSE 0 END is_validation
        FROM read_parquet({qp(path)}) s
        LEFT JOIN {gt} g
          ON s.source1_entity_id=g.source1_entity_id
         AND s.candidate_entity_id=g.matched_entity_id
    """)
    return t

def pair_sql(t, flag):
    cond = '' if flag is None else f'AND is_validation={flag}'
    return f"""
    WITH
    p AS (
        SELECT * FROM (
            SELECT t.*, ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id ORDER BY v6_rank, candidate_entity_id
            ) n
            FROM {t} t
            WHERE match_label=1 {cond}
        ) WHERE n<={POS_PER_S1}
    ),
    n AS (
        SELECT * FROM (
            SELECT t.*, ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id ORDER BY v6_rank, candidate_entity_id
            ) k
            FROM {t} t
            WHERE match_label=0 AND v6_rank<={NEG_RANK} {cond}
        ) WHERE k<={NEG_PER_S1}
    ),
    z AS (
        SELECT
            p.name_similarity pn,n.name_similarity nn,
            p.address_similarity pa,n.address_similarity na,
            p.name_exact pe,n.name_exact ne,
            p.address_exact pae,n.address_exact nae,
            p.country_exact pc,n.country_exact nc,
            p.name_length_ratio pl,n.name_length_ratio nl,
            p.address_length_ratio pal,n.address_length_ratio nal,
            p.evidence_rows per,n.evidence_rows ner,
            p.evidence_file_count pef,n.evidence_file_count nef,
            p.exact_key_count pek,n.exact_key_count nek,
            p.v6_score ps,n.v6_score ns,
            p.v6_rank pr,n.v6_rank nr
        FROM p JOIN n USING(source1_entity_id)
    ),
    f AS (
        SELECT
            pn-nn name_similarity, pa-na address_similarity,
            pe-ne name_exact, pae-nae address_exact, pc-nc country_exact,
            pl-nl name_length_ratio, pal-nal address_length_ratio,
            per-ner evidence_rows, pef-nef evidence_file_count,
            pek-nek exact_key_count, ps-ns v6_score,
            LN(1+pr)-LN(1+nr) log_v6_rank,
            pn*pa-nn*na name_x_address,
            (pn-pa)-(nn-na) name_minus_address,
            ((pn+pa)/2)-((nn+na)/2) similarity_mean,
            LEAST(pn,pa)-LEAST(nn,na) similarity_min,
            (pe+pae+pc)-(ne+nae+nc) exact_field_count,
            1 pair_label
        FROM z
    )
    SELECT * FROM f
    UNION ALL
    SELECT
        -name_similarity,-address_similarity,-name_exact,-address_exact,-country_exact,
        -name_length_ratio,-address_length_ratio,-evidence_rows,-evidence_file_count,
        -exact_key_count,-v6_score,-log_v6_rank,-name_x_address,-name_minus_address,
        -similarity_mean,-similarity_min,-exact_field_count,0 pair_label
    FROM f
    """

def collect(c, q, label):
    reader = c.execute(q).to_arrow_reader(batch_size=BATCH)
    parts=[]; rows=0
    for b in reader:
        d=b.to_pandas()
        if len(d):
            parts.append(d); rows += len(d)
    if not parts:
        raise RuntimeError(f'No pair data for {label}')
    d=pd.concat(parts, ignore_index=True)
    X=d[PAIR_FEATURES].apply(pd.to_numeric,errors='coerce').fillna(0).astype('float32').to_numpy()
    y=d.pair_label.astype('int8').to_numpy()
    print(f'{label}: {len(d):,} pair rows; positive={int((y==1).sum()):,}; negative={int((y==0).sum()):,}')
    return X,y

def fit(X,y,label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    XS=scaler.fit_transform(X).astype('float32')
    m=LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs',random_state=42)
    s=time.time(); m.fit(XS,y); print(f'{label} fit: {time.time()-s:.1f}s')
    return scaler,m

def rank(c,t,scaler,model,name,validation,outfile):
    where='WHERE is_validation=1' if validation else ''
    q=f"SELECT source1_entity_id,candidate_entity_id,v6_rank,v6_score,name_similarity,address_similarity,name_exact,address_exact,country_exact,name_length_ratio,address_length_ratio,evidence_rows,evidence_file_count,exact_key_count,match_label FROM {t} {where}"
    parts_dir=TMP / f"{name.lower()}_{'val' if validation else 'full'}"
    parts_dir.mkdir(parents=True,exist_ok=True)
    for p in parts_dir.glob('*.parquet'): p.unlink()
    reader=c.execute(q).to_arrow_reader(batch_size=BATCH); paths=[]; total=0; i=0
    for b in reader:
        d=b.to_pandas()
        if d.empty: continue
        X=make_features(d).to_numpy(dtype='float32',copy=False)
        XS=scaler.transform(X).astype('float32')
        d['v8_utility']=model.decision_function(XS)
        p=parts_dir/f'part_{i:05d}.parquet'; d.to_parquet(p,index=False,compression='zstd')
        paths.append(p); total+=len(d); i+=1
        if total and total%1_000_000<len(d): print(f'Scored: {total:,}')
    glob=str(parts_dir/'*.parquet').replace(chr(92),'/').replace(chr(39),chr(39)*2)
    outfile.unlink(missing_ok=True)
    c.execute(f"""
        COPY (
            SELECT *,
              ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id
                ORDER BY v8_utility DESC,candidate_entity_id
              ) rank
            FROM read_parquet('{glob}')
        ) TO {qp(outfile)} (FORMAT PARQUET,COMPRESSION ZSTD)
    """)
    for p in paths: p.unlink(missing_ok=True)
    try: parts_dir.rmdir()
    except OSError: pass
    return total

def recall(c,path,gt,name,val):
    filt=f'AND MOD(ABS(HASH(g.source1_entity_id)),{VALIDATION_MOD})=0' if val else ''
    total=c.execute(f'SELECT COUNT(*) FROM {gt}'+(' WHERE '+f'MOD(ABS(HASH(source1_entity_id)),{VALIDATION_MOD})=0' if val else '')).fetchone()[0]
    out={}
    for k in [1,3,5,10,20]:
        n=c.execute(f"""
          SELECT COUNT(*) FROM (
            SELECT DISTINCT g.source1_entity_id,g.matched_entity_id
            FROM {gt} g
            JOIN read_parquet({qp(path)}) r
              ON g.source1_entity_id=r.source1_entity_id
             AND g.matched_entity_id=r.candidate_entity_id
            WHERE r.rank<={k} {filt}
          )
        """).fetchone()[0]
        pct=100*n/total if total else 0
        out[f'recall_at_{k}']=pct; out[f'covered_at_{k}']=n
        print(f'{name} Recall@{k}: {pct:.2f}% ({n:,}/{total:,})')
    return out

def process(c,name,scored,gt):
    t=labeled(c,scored,gt,name)
    X,y=collect(c,pair_sql(t,0),name+' HOLDOUT TRAIN')
    scaler,model=fit(X,y,name+' HOLDOUT')
    val=TMP/f'v8_val_{name.lower()}.parquet'
    rank(c,t,scaler,model,name,True,val)
    vm=recall(c,val,gt,name,True)

    X2,y2=collect(c,pair_sql(t,None),name+' FULL TRAIN')
    scaler2,model2=fit(X2,y2,name+' FULL')
    full=OUT/f'train_ranked_v8_s1_{name.lower()}.parquet'
    rank(c,t,scaler2,model2,name,False,full)
    top=OUT/f'train_ranked_v8_top20_s1_{name.lower()}.parquet'
    c.execute(f"COPY (SELECT * FROM read_parquet({qp(full)}) WHERE rank<={TOPK}) TO {qp(top)} (FORMAT PARQUET,COMPRESSION ZSTD)")
    fm=recall(c,full,gt,name,False)
    return {'target':name,'validation':vm,'full_train':fm,'full_output':str(full),'top20_output':str(top)}

def main():
    header('AMAZON ML CHALLENGE - V8 PAIRWISE RANKING')
    print(f'Project: {PROJECT}')
    print(f'V6: {V6}')
    print(f'Output: {OUT}')
    for p in [GT,S2,S3]: require_file(p)
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
    c=setup(); load_gt(c)
    start=time.time()
    try:
        r2=process(c,'S2',S2,'gt_s2')
        r3=process(c,'S3',S3,'gt_s3')
    finally:
        c.close()
    summary={'version':'V8','method':'pairwise_logistic_ranking','results':{'s2':r2,'s3':r3},'elapsed_seconds':round(time.time()-start,2)}
    sp=OUT/'v8_summary.json'; sp.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    header('V8 FINAL SUMMARY')
    for k in ['s2','s3']:
        v=summary['results'][k]
        print(f"S1 -> {k.upper()} Holdout@20={v['validation']['recall_at_20']:.2f}% | Full@20={v['full_train']['recall_at_20']:.2f}%")
    print(f'Summary: {sp}')
    header('V8 COMPLETED')

def require_file(path):
    if not path.exists(): raise FileNotFoundError(str(path))

if __name__=='__main__':
    main()
