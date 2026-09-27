#!/usr/bin/env python3
"""
Amazon ML Challenge 2026
AUGMENTED V10 FRESH-DEV BLOCK IMPACT EXPERIMENT V1

Purpose
-------
Measure whether the transferable blocking families found on OLD/FRESH
diagnostics improve the actual V10 precision-heavy F0.5 policy on the
same fresh development split.

Frozen production submission is never touched.

Experiment:
  baseline candidate pool = existing V6 + V9.1 labeled pool
  augmented pool          = baseline + selected transferable blocks

S2 selected blocks:
  - postal_addr_f3_tail2_name_f1
  - name_f2_addr_tail3
  - name_f1_addr_f4_tail2

S3 selected blocks:
  - name_f3_addr_f2
  - postal_addr_f2_name_f2
  - name_f4_l2_len5
  - name_f3_l2_postal

Safety
------
- Does NOT read Final Holdout V2 GT.
- Does NOT read test data.
- Does NOT modify final candidate/output files.
- Does NOT modify final ZIP or GitHub release.
- Uses the already-created fresh_dev_s1.parquet only.
- Fresh S1 IDs are excluded from model training exactly as in the
  existing fresh-development confirmation script.
- New candidates are materialized only under this isolated audit folder.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"
GT = TRAIN / "ground_truth_pairs.parquet"
S1 = TRAIN / "train_source1.parquet"
S2 = TRAIN / "train_source2.parquet"
S3 = TRAIN / "train_source3.parquet"

BASELINE_LABELED = {
    "S2": PROJECT / "candidate_output_v10" / "train_labeled_base_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10" / "train_labeled_base_s1_s3.parquet",
}

FRESH_DIR = PROJECT / "validation_leakage_audit" / "f05_fresh_dev_joint_confirmation_v1"
FRESH_S1 = FRESH_DIR / "fresh_dev_s1.parquet"
EXISTING_FRESH_RANKED = {
    "S2": FRESH_DIR / "fresh_ranked_s2.parquet",
    "S3": FRESH_DIR / "fresh_ranked_s3.parquet",
}

OUT = PROJECT / "validation_leakage_audit" / "block_augmented_v10_fresh_v1"
TMP = OUT / "tmp"
BLOCK_PARTS = OUT / "block_parts"
NEW_FEATURE_PARTS = OUT / "new_feature_parts"

REPORT = OUT / "block_augmented_v10_fresh_v1_report.json"

MEMORY = os.environ.get("BLOCK_AUGMENT_MEMORY", "8GB")
THREADS = int(os.environ.get("BLOCK_AUGMENT_THREADS", "2"))
BATCH = int(os.environ.get("BLOCK_AUGMENT_BATCH", "100000"))
MAX_BUCKET_SIZE = int(os.environ.get("BLOCK_AUGMENT_MAX_BUCKET_SIZE", "100"))

CURRENT_S2 = 20.945884704589844
CURRENT_S3 = 20.945884704589844

CANDIDATE_S2 = 34.710811614990234
CANDIDATE_S3 = 22.125261306762695

FEATURES = [
    "name_similarity",
    "address_similarity",
    "name_exact",
    "address_exact",
    "country_exact",
    "name_length_ratio",
    "address_length_ratio",
    "evidence_rows",
    "evidence_file_count",
    "exact_key_count",
    "base_score",
    "log_base_rank",
    "name_x_address",
    "name_minus_address",
    "similarity_mean",
    "similarity_min",
    "exact_field_count",
]

POS_PER_S1 = 1
NEG_RANK = 100
NEG_PER_S1 = 2

SELECTED_BLOCKS = {
    "S2": [
        "postal_addr_f3_tail2_name_f1",
        "name_f2_addr_tail3",
        "name_f1_addr_f4_tail2",
    ],
    "S3": [
        "name_f3_addr_f2",
        "postal_addr_f2_name_f2",
        "name_f4_l2_len5",
        "name_f3_l2_postal",
    ],
}

BLOCKS = {
    "postal_addr_f3_tail2_name_f1": (
        "postal || '|' || substr(address_compact,1,3) "
        "|| '|' || right(address_compact,2) "
        "|| '|' || substr(name_compact,1,1)"
    ),
    "name_f2_addr_tail3": (
        "substr(name_compact,1,2) || '|' || right(address_compact,3)"
    ),
    "name_f1_addr_f4_tail2": (
        "substr(name_compact,1,1) || '|' || "
        "substr(address_compact,1,4) || '|' || "
        "right(address_compact,2)"
    ),
    "name_f3_addr_f2": (
        "substr(name_compact,1,3) || '|' || "
        "substr(address_compact,1,2)"
    ),
    "postal_addr_f2_name_f2": (
        "postal || '|' || substr(address_compact,1,2) "
        "|| '|' || substr(name_compact,1,2)"
    ),
    "name_f4_l2_len5": (
        "substr(name_compact,1,4) || '|' || right(name_compact,2) "
        "|| '|' || CAST(floor(length(name_compact)/5) AS BIGINT)"
    ),
    "name_f3_l2_postal": (
        "postal || '|' || substr(name_compact,1,3) "
        "|| '|' || right(name_compact,2)"
    ),
}


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def header(title: str) -> None:
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def count_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({qp(path)})"
        ).fetchone()[0]
    )


def connect() -> duckdb.DuckDBPyConnection:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={qp(TMP / 'duckdb_tmp')}")
    con.execute("SET max_temp_directory_size='50GB'")
    return con


def cleanup_own_outputs() -> None:
    """
    Delete only this experiment's own generated material.
    Never touch candidate_output_final_v10_threshold or student_resource/output.
    """
    if OUT.exists():
        for child in OUT.iterdir():
            if child.name == REPORT.name:
                try:
                    child.unlink()
                except OSError:
                    pass
            else:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    BLOCK_PARTS.mkdir(parents=True, exist_ok=True)
    NEW_FEATURE_PARTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# MODEL TRAINING — copied from the fresh-dev confirmation recipe
# ---------------------------------------------------------------------
def make_training_sql(path: Path) -> str:
    fresh = qp(FRESH_S1)
    p = qp(path)

    return f"""
    WITH eligible AS (
        SELECT *
        FROM read_parquet({p}) t
        WHERE NOT EXISTS (
            SELECT 1
            FROM read_parquet({fresh}) f
            WHERE CAST(f.source1_entity_id AS VARCHAR)
                = CAST(t.source1_entity_id AS VARCHAR)
        )
    ),
    positives AS (
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY source1_entity_id
                       ORDER BY base_rank, candidate_entity_id
                   ) AS rn
            FROM eligible
            WHERE label = 1
        )
        WHERE rn <= {POS_PER_S1}
    ),
    negatives AS (
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY source1_entity_id
                       ORDER BY base_rank, candidate_entity_id
                   ) AS rn
            FROM eligible
            WHERE label = 0
              AND base_rank <= {NEG_RANK}
        )
        WHERE rn <= {NEG_PER_S1}
    ),
    joined AS (
        SELECT
            p.name_similarity pn, n.name_similarity nn,
            p.address_similarity pa, n.address_similarity na,
            p.name_exact pe, n.name_exact ne,
            p.address_exact pae, n.address_exact nae,
            p.country_exact pc, n.country_exact nc,
            p.name_length_ratio pl, n.name_length_ratio nl,
            p.address_length_ratio pal, n.address_length_ratio nal,
            p.evidence_rows per, n.evidence_rows ner,
            p.evidence_file_count pef, n.evidence_file_count nef,
            p.exact_key_count pek, n.exact_key_count nek,
            p.base_score ps, n.base_score ns,
            p.base_rank pr, n.base_rank nr
        FROM positives p
        JOIN negatives n USING (source1_entity_id)
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
    SELECT
        -name_similarity,
        -address_similarity,
        -name_exact,
        -address_exact,
        -country_exact,
        -name_length_ratio,
        -address_length_ratio,
        -evidence_rows,
        -evidence_file_count,
        -exact_key_count,
        -base_score,
        -log_base_rank,
        -name_x_address,
        -name_minus_address,
        -similarity_mean,
        -similarity_min,
        -exact_field_count,
        0 AS pair_label
    FROM f
    """


def fit_model(con: duckdb.DuckDBPyConnection, target: str):
    path = BASELINE_LABELED[target]
    require(path)

    header(f"FIT FRESH-DEVELOPMENT MODEL — {target}")

    reader = con.execute(
        make_training_sql(path)
    ).to_arrow_reader(batch_size=BATCH)

    frames = []
    total = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue
        frames.append(d)
        total += len(d)
        if total and total % 1_000_000 < len(d):
            print(f"{target} training pairs loaded: {total:,}", flush=True)

    if not frames:
        raise RuntimeError(f"No training rows for {target}")

    d = pd.concat(frames, ignore_index=True)

    X = (
        d[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .to_numpy()
    )
    y = d["pair_label"].astype("int8").to_numpy()

    print(
        f"{target}: pair rows={len(d):,} | "
        f"positive={int((y == 1).sum()):,} | "
        f"negative={int((y == 0).sum()):,}",
        flush=True,
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype("float32")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    t0 = time.time()
    model.fit(Xs, y)
    print(f"{target} fit: {time.time() - t0:.1f}s", flush=True)

    return scaler, model


# ---------------------------------------------------------------------
# SOURCE PREPROCESSING — exact block-audit normalization
# ---------------------------------------------------------------------
def build_source_views(con: duckdb.DuckDBPyConnection) -> None:
    header("BUILD SOURCE VIEWS")

    for target, path in (("S2", S2), ("S3", S3)):
        require(path)

    require(S1)
    require(FRESH_S1)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fresh_s1_ids AS
        SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(FRESH_S1)})
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_src AS
        SELECT
            CAST(s.entity_id AS VARCHAR) AS entity_id,
            COALESCE(s.country_norm,'') AS country_norm,
            regexp_replace(
                COALESCE(s.name_norm,''),
                '[^a-z0-9]','','g'
            ) AS name_compact,
            regexp_replace(
                COALESCE(s.address_norm,''),
                '[^a-z0-9]','','g'
            ) AS address_compact,
            regexp_extract(
                COALESCE(s.address_norm,''),
                '[0-9]{{5,6}}',
                0
            ) AS postal,
            regexp_replace(
                regexp_replace(
                    COALESCE(s.address_norm,''),
                    '[^0-9]','','g'
                ),
                '^0+$','','g'
            ) AS addr_num
        FROM read_parquet({qp(S1)}) s
        INNER JOIN fresh_s1_ids f
            ON CAST(s.entity_id AS VARCHAR) = f.s1
    """)

    for target, path in (("S2", S2), ("S3", S3)):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {target.lower()}_src AS
            SELECT
                CAST(entity_id AS VARCHAR) AS entity_id,
                COALESCE(country_norm,'') AS country_norm,
                regexp_replace(
                    COALESCE(name_norm,''),
                    '[^a-z0-9]','','g'
                ) AS name_compact,
                regexp_replace(
                    COALESCE(address_norm,''),
                    '[^a-z0-9]','','g'
                ) AS address_compact,
                regexp_extract(
                    COALESCE(address_norm,''),
                    '[0-9]{{5,6}}',
                    0
                ) AS postal,
                regexp_replace(
                    regexp_replace(
                        COALESCE(address_norm,''),
                        '[^0-9]','','g'
                    ),
                    '^0+$','','g'
                ) AS addr_num
            FROM read_parquet({qp(path)})
        """)

    n = int(con.execute("SELECT COUNT(*) FROM fresh_s1_ids").fetchone()[0])
    print(f"Fresh S1 rows: {n:,}")


# ---------------------------------------------------------------------
# BASELINE PAIRS FOR DEDUPLICATION
# ---------------------------------------------------------------------
def materialize_baseline_pairs(
    con: duckdb.DuckDBPyConnection,
    target: str,
) -> Path:
    out = OUT / f"baseline_pairs_fresh_{target.lower()}.parquet"
    source = BASELINE_LABELED[target]

    con.execute(f"""
        COPY (
            SELECT DISTINCT
                CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
            FROM read_parquet({qp(source)})
            INNER JOIN fresh_s1_ids
              ON CAST(source1_entity_id AS VARCHAR) = fresh_s1_ids.s1
        )
        TO {qp(out)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print(f"{target} baseline fresh pairs: {count_rows(con, out):,}")
    return out


# ---------------------------------------------------------------------
# BLOCK CANDIDATE GENERATION
# ---------------------------------------------------------------------
def generate_new_block(
    con: duckdb.DuckDBPyConnection,
    target: str,
    block_name: str,
    baseline_pairs: Path,
) -> Path:
    expr = BLOCKS[block_name]
    target_table = target.lower() + "_src"

    left_view = f"left_{block_name}"
    right_view = f"right_{block_name}"

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW {left_view} AS
        SELECT
            entity_id,
            country_norm || '|' || ({expr}) AS block_key
        FROM s1_src
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW {right_view} AS
        SELECT
            entity_id,
            country_norm || '|' || ({expr}) AS block_key
        FROM {target_table}
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE lcnt AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {left_view}
        WHERE block_key <> ''
          AND NOT ends_with(block_key, '|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rcnt AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {right_view}
        WHERE block_key <> ''
          AND NOT ends_with(block_key, '|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    estimated = int(con.execute("""
        SELECT COALESCE(SUM(l.n*r.n),0)
        FROM lcnt l
        JOIN rcnt r USING(block_key)
    """).fetchone()[0])

    out = BLOCK_PARTS / target.lower() / f"{block_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    header(f"GENERATE {target} / {block_name}")
    print(f"Estimated pairs after bucket cap: {estimated:,}")

    if estimated == 0:
        raise RuntimeError(f"{target}/{block_name}: no shared valid keys")

    con.execute(f"""
        COPY (
            SELECT
                l.entity_id AS source1_entity_id,
                r.entity_id AS candidate_entity_id,
                '{block_name}' AS block_name
            FROM {left_view} l
            INNER JOIN {right_view} r
              ON l.block_key = r.block_key
            INNER JOIN lcnt lc
              ON lc.block_key = l.block_key
            INNER JOIN rcnt rc
              ON rc.block_key = r.block_key
            LEFT JOIN read_parquet({qp(baseline_pairs)}) b
              ON CAST(b.source1_entity_id AS VARCHAR) = CAST(l.entity_id AS VARCHAR)
             AND CAST(b.candidate_entity_id AS VARCHAR) = CAST(r.entity_id AS VARCHAR)
            WHERE b.source1_entity_id IS NULL
        )
        TO {qp(out)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    actual = count_rows(con, out)
    print(f"New candidates after baseline subtraction: {actual:,}")
    return out


def consolidate_new_candidates(
    con: duckdb.DuckDBPyConnection,
    target: str,
    paths: List[Path],
) -> Path:
    if not paths:
        raise RuntimeError(f"No block files for {target}")

    target_dir = BLOCK_PARTS / target.lower()
    explicit = "[" + ",".join(qp(p) for p in paths) + "]"

    out = OUT / f"new_candidates_fresh_{target.lower()}.parquet"
    out.unlink(missing_ok=True)

    con.execute(f"""
        COPY (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
                COUNT(*)::INTEGER AS evidence_rows,
                COUNT(DISTINCT block_name)::INTEGER AS evidence_file_count,
                0::INTEGER AS exact_key_count
            FROM read_parquet({explicit})
            GROUP BY source1_entity_id, candidate_entity_id
        )
        TO {qp(out)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows = count_rows(con, out)
    print(f"{target} deduplicated new candidates: {rows:,}")
    return out


# ---------------------------------------------------------------------
# NEW FEATURE SCORING
# ---------------------------------------------------------------------
def score_new_candidates(
    con: duckdb.DuckDBPyConnection,
    target: str,
    new_candidates: Path,
) -> Path:
    target_table = target.lower() + "_src"
    feature_base = OUT / f"new_feature_source_{target.lower()}.parquet"
    feature_base.unlink(missing_ok=True)

    con.execute(f"""
        COPY (
            SELECT
                c.source1_entity_id,
                c.candidate_entity_id,
                c.evidence_rows,
                c.evidence_file_count,
                c.exact_key_count,

                l.name_norm AS left_name,
                r.name_norm AS right_name,
                l.address_norm AS left_address,
                r.address_norm AS right_address,
                l.country_norm AS left_country,
                r.country_norm AS right_country,

                CASE
                    WHEN l.name_norm <> ''
                     AND l.name_norm = r.name_norm
                    THEN 1 ELSE 0
                END AS name_exact,

                CASE
                    WHEN l.address_norm <> ''
                     AND r.address_norm <> ''
                     AND l.address_norm = r.address_norm
                    THEN 1 ELSE 0
                END AS address_exact,

                CASE
                    WHEN l.country_norm <> ''
                     AND l.country_norm = r.country_norm
                    THEN 1 ELSE 0
                END AS country_exact,

                CASE
                    WHEN l.name_norm = '' OR r.name_norm = ''
                    THEN 0.0
                    ELSE
                        LEAST(length(l.name_norm), length(r.name_norm))::DOUBLE
                        /
                        GREATEST(length(l.name_norm), length(r.name_norm), 1)
                END AS name_length_ratio,

                CASE
                    WHEN l.address_norm = '' OR r.address_norm = ''
                    THEN 0.0
                    ELSE
                        LEAST(length(l.address_norm), length(r.address_norm))::DOUBLE
                        /
                        GREATEST(length(l.address_norm), length(r.address_norm), 1)
                END AS address_length_ratio

            FROM read_parquet({qp(new_candidates)}) c
            INNER JOIN read_parquet({qp(S1)}) l
              ON CAST(c.source1_entity_id AS VARCHAR)
               = CAST(l.entity_id AS VARCHAR)
            INNER JOIN read_parquet({qp(S2 if target == 'S2' else S3)}) r
              ON CAST(c.candidate_entity_id AS VARCHAR)
               = CAST(r.entity_id AS VARCHAR)
        )
        TO {qp(feature_base)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    part_dir = NEW_FEATURE_PARTS / target.lower()
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    reader = con.execute(f"""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            evidence_rows,
            evidence_file_count,
            exact_key_count,
            left_name,
            right_name,
            left_address,
            right_address,
            name_exact,
            address_exact,
            country_exact,
            name_length_ratio,
            address_length_ratio
        FROM read_parquet({qp(feature_base)})
    """).to_arrow_reader(batch_size=BATCH)

    total = 0
    idx = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue

        left_names = d["left_name"].fillna("").astype(str).tolist()
        right_names = d["right_name"].fillna("").astype(str).tolist()
        left_addr = d["left_address"].fillna("").astype(str).tolist()
        right_addr = d["right_address"].fillna("").astype(str).tolist()

        d["name_similarity"] = np.asarray(
            [
                ratio(a, b) / 100.0 if a and b else 0.0
                for a, b in zip(left_names, right_names)
            ],
            dtype="float32",
        )
        d["address_similarity"] = np.asarray(
            [
                ratio(a, b) / 100.0 if a and b else 0.0
                for a, b in zip(left_addr, right_addr)
            ],
            dtype="float32",
        )

        d["base_score"] = (
            30.0 * d["name_exact"].astype(float)
            + 25.0 * d["address_exact"].astype(float)
            + 8.0 * d["country_exact"].astype(float)
            + 22.0 * d["name_similarity"].astype(float)
            + 12.0 * d["address_similarity"].astype(float)
            + 2.0 * d["name_length_ratio"].astype(float)
            + 1.0 * d["address_length_ratio"].astype(float)
            + 0.75 * d["evidence_file_count"].astype(float).clip(upper=4.0)
            + 0.25 * d["exact_key_count"].astype(float).clip(upper=5.0)
        ).astype("float32")

        out = part_dir / f"part_{idx:05d}.parquet"
        d[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "evidence_rows",
                "evidence_file_count",
                "exact_key_count",
                "name_similarity",
                "address_similarity",
                "name_exact",
                "address_exact",
                "country_exact",
                "name_length_ratio",
                "address_length_ratio",
                "base_score",
            ]
        ].to_parquet(out, index=False, compression="zstd")

        total += len(d)
        idx += 1
        if total and total % 1_000_000 < len(d):
            print(f"{target} new pairs fuzzy-scored: {total:,}", flush=True)

    feature_base.unlink(missing_ok=True)

    if total == 0:
        raise RuntimeError(f"No scored new candidates for {target}")

    scored_new = OUT / f"new_features_fresh_{target.lower()}.parquet"
    scored_new.unlink(missing_ok=True)

    con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet({qp(part_dir / '*.parquet')})
        )
        TO {qp(scored_new)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print(f"{target} new candidate features: {count_rows(con, scored_new):,}")
    return scored_new


# ---------------------------------------------------------------------
# BASELINE + AUGMENTED COMBINED RANKING
# ---------------------------------------------------------------------
def make_baseline_fresh_features(
    con: duckdb.DuckDBPyConnection,
    target: str,
) -> Path:
    out = OUT / f"baseline_features_fresh_{target.lower()}.parquet"
    out.unlink(missing_ok=True)

    source = BASELINE_LABELED[target]
    con.execute(f"""
        COPY (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
                CAST(evidence_rows AS DOUBLE) AS evidence_rows,
                CAST(evidence_file_count AS DOUBLE) AS evidence_file_count,
                CAST(exact_key_count AS DOUBLE) AS exact_key_count,
                CAST(name_similarity AS DOUBLE) AS name_similarity,
                CAST(address_similarity AS DOUBLE) AS address_similarity,
                CAST(name_exact AS DOUBLE) AS name_exact,
                CAST(address_exact AS DOUBLE) AS address_exact,
                CAST(country_exact AS DOUBLE) AS country_exact,
                CAST(name_length_ratio AS DOUBLE) AS name_length_ratio,
                CAST(address_length_ratio AS DOUBLE) AS address_length_ratio,
                CAST(base_score AS DOUBLE) AS base_score
            FROM read_parquet({qp(source)})
            INNER JOIN fresh_s1_ids f
              ON CAST(source1_entity_id AS VARCHAR) = f.s1
        )
        TO {qp(out)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    return out


def build_ranked_base(
    con: duckdb.DuckDBPyConnection,
    target: str,
    baseline_features: Path,
    new_features: Path | None,
) -> Path:
    raw = OUT / f"combined_base_raw_{target.lower()}.parquet"
    ranked = OUT / f"combined_base_ranked_{target.lower()}.parquet"
    raw.unlink(missing_ok=True)
    ranked.unlink(missing_ok=True)

    if new_features is None:
        union_sql = f"""
            SELECT * FROM read_parquet({qp(baseline_features)})
        """
    else:
        union_sql = f"""
            SELECT * FROM read_parquet({qp(baseline_features)})
            UNION ALL
            SELECT * FROM read_parquet({qp(new_features)})
        """

    con.execute(f"""
        COPY (
            {union_sql}
        )
        TO {qp(raw)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    con.execute(f"""
        COPY (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY base_score DESC, candidate_entity_id
                ) AS base_rank
            FROM read_parquet({qp(raw)})
        )
        TO {qp(ranked)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    raw.unlink(missing_ok=True)
    print(f"{target} combined base rows: {count_rows(con, ranked):,}")
    return ranked


def score_ranked_with_model(
    con: duckdb.DuckDBPyConnection,
    target: str,
    ranked_base: Path,
    scaler: StandardScaler,
    model: LogisticRegression,
    tag: str,
) -> Path:
    util_dir = OUT / f"utility_parts_{tag}_{target.lower()}"
    if util_dir.exists():
        shutil.rmtree(util_dir)
    util_dir.mkdir(parents=True, exist_ok=True)

    reader = con.execute(
        f"SELECT * FROM read_parquet({qp(ranked_base)})"
    ).to_arrow_reader(batch_size=BATCH)

    total = 0
    idx = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue

        x = pd.DataFrame(index=d.index)
        numeric = [
            "name_similarity",
            "address_similarity",
            "name_exact",
            "address_exact",
            "country_exact",
            "name_length_ratio",
            "address_length_ratio",
            "evidence_rows",
            "evidence_file_count",
            "exact_key_count",
            "base_score",
        ]

        for c in numeric:
            x[c] = (
                pd.to_numeric(d[c], errors="coerce")
                .fillna(0)
                .astype("float32")
            )

        rank = (
            pd.to_numeric(d["base_rank"], errors="coerce")
            .fillna(1_000_000)
            .astype("float64")
        )

        x["log_base_rank"] = np.log1p(rank).astype("float32")
        x["name_x_address"] = (
            x["name_similarity"] * x["address_similarity"]
        ).astype("float32")
        x["name_minus_address"] = (
            x["name_similarity"] - x["address_similarity"]
        ).astype("float32")
        x["similarity_mean"] = (
            (x["name_similarity"] + x["address_similarity"]) / 2
        ).astype("float32")
        x["similarity_min"] = np.minimum(
            x["name_similarity"],
            x["address_similarity"],
        ).astype("float32")
        x["exact_field_count"] = (
            x["name_exact"] + x["address_exact"] + x["country_exact"]
        ).astype("float32")

        X = x[FEATURES].to_numpy(dtype="float32")
        Xs = scaler.transform(X).astype("float32")
        utility = model.decision_function(Xs)

        out = util_dir / f"part_{idx:05d}.parquet"
        pd.DataFrame(
            {
                "source1_entity_id": d["source1_entity_id"].astype(str),
                "candidate_entity_id": d["candidate_entity_id"].astype(str),
                "utility": utility.astype("float32"),
            }
        ).to_parquet(out, index=False, compression="zstd")

        total += len(d)
        idx += 1
        if total and total % 1_000_000 < len(d):
            print(f"{target} {tag} utility rows: {total:,}", flush=True)

    ranked = OUT / f"ranked_{tag}_{target.lower()}.parquet"
    ranked.unlink(missing_ok=True)

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                utility,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY utility DESC, candidate_entity_id
                ) AS rank
            FROM read_parquet({qp(util_dir / '*.parquet')})
        )
        TO {qp(ranked)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    shutil.rmtree(util_dir, ignore_errors=True)
    print(f"{target} {tag} ranked rows: {count_rows(con, ranked):,}")
    return ranked


# ---------------------------------------------------------------------
# REPRODUCTION CHECK AGAINST EXISTING FRESH RANKED FILE
# ---------------------------------------------------------------------
def compare_ranked_reproduction(
    con: duckdb.DuckDBPyConnection,
    target: str,
    recomputed: Path,
) -> dict:
    existing = EXISTING_FRESH_RANKED[target]
    require(existing)

    result = con.execute(f"""
        WITH a AS (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                CAST(candidate_entity_id AS VARCHAR) AS cand,
                CAST(utility AS DOUBLE) AS utility
            FROM read_parquet({qp(existing)})
            WHERE rank = 1
        ),
        b AS (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                CAST(candidate_entity_id AS VARCHAR) AS cand,
                CAST(utility AS DOUBLE) AS utility
            FROM read_parquet({qp(recomputed)})
            WHERE rank = 1
        )
        SELECT
            COUNT(*) AS rows_compared,
            SUM(CASE WHEN a.cand = b.cand THEN 1 ELSE 0 END) AS exact_top1_id,
            MAX(ABS(a.utility - b.utility)) AS max_abs_utility_diff,
            AVG(ABS(a.utility - b.utility)) AS mean_abs_utility_diff
        FROM a
        INNER JOIN b USING(s1)
    """).fetchone()

    rows_compared = int(result[0] or 0)
    exact_top1_id = int(result[1] or 0)
    max_diff = float(result[2] or 0.0)
    mean_diff = float(result[3] or 0.0)

    exact_pct = (
        100.0 * exact_top1_id / rows_compared
        if rows_compared else 0.0
    )

    out = {
        "rows_compared": rows_compared,
        "exact_top1_id": exact_top1_id,
        "exact_top1_pct": exact_pct,
        "max_abs_utility_diff": max_diff,
        "mean_abs_utility_diff": mean_diff,
        "pass": exact_top1_id == rows_compared and max_diff < 1e-5,
    }

    print(
        f"{target} reproduction: "
        f"top1 exact={exact_pct:.6f}% | "
        f"max utility diff={max_diff:.12g}"
    )

    return out


# ---------------------------------------------------------------------
# F0.5 EVALUATION
# ---------------------------------------------------------------------
def evaluate_policy(
    con: duckdb.DuckDBPyConnection,
    s2_ranked: Path,
    s3_ranked: Path,
    t2: float,
    t3: float,
) -> dict:
    sql = f"""
    WITH truth AS (
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            COUNT(*)::INTEGER AS truth_count
        FROM read_parquet({qp(GT)}) g
        INNER JOIN fresh_s1_ids f
          ON CAST(g.source1_entity_id AS VARCHAR) = f.s1
        WHERE COALESCE(g.label,1)=1
        GROUP BY 1
    ),
    ids AS (
        SELECT s1 FROM fresh_s1_ids
    ),
    s2 AS (
        SELECT
            CAST(r.source1_entity_id AS VARCHAR) AS s1,
            CAST(r.candidate_entity_id AS VARCHAR) AS cand,
            CASE
                WHEN r.rank=1 AND r.utility >= {t2}
                THEN 1 ELSE 0
            END AS pred,
            CASE
                WHEN r.rank=1
                 AND r.utility >= {t2}
                 AND EXISTS (
                     SELECT 1
                     FROM read_parquet({qp(GT)}) g
                     WHERE CAST(g.source1_entity_id AS VARCHAR)
                         = CAST(r.source1_entity_id AS VARCHAR)
                       AND CAST(g.matched_entity_id AS VARCHAR)
                         = CAST(r.candidate_entity_id AS VARCHAR)
                       AND COALESCE(g.label,1)=1
                 )
                THEN 1 ELSE 0
            END AS tp
        FROM read_parquet({qp(s2_ranked)}) r
        WHERE r.rank=1
    ),
    s3 AS (
        SELECT
            CAST(r.source1_entity_id AS VARCHAR) AS s1,
            CAST(r.candidate_entity_id AS VARCHAR) AS cand,
            CASE
                WHEN r.rank=1 AND r.utility >= {t3}
                THEN 1 ELSE 0
            END AS pred,
            CASE
                WHEN r.rank=1
                 AND r.utility >= {t3}
                 AND EXISTS (
                     SELECT 1
                     FROM read_parquet({qp(GT)}) g
                     WHERE CAST(g.source1_entity_id AS VARCHAR)
                         = CAST(r.source1_entity_id AS VARCHAR)
                       AND CAST(g.matched_entity_id AS VARCHAR)
                         = CAST(r.candidate_entity_id AS VARCHAR)
                       AND COALESCE(g.label,1)=1
                 )
                THEN 1 ELSE 0
            END AS tp
        FROM read_parquet({qp(s3_ranked)}) r
        WHERE r.rank=1
    ),
    per_s1 AS (
        SELECT
            ids.s1,
            COALESCE(t.truth_count,0) AS tc,
            COALESCE(s2.pred,0) + COALESCE(s3.pred,0) AS p,
            COALESCE(s2.tp,0) + COALESCE(s3.tp,0) AS tp
        FROM ids
        LEFT JOIN truth t ON t.s1=ids.s1
        LEFT JOIN s2 ON s2.s1=ids.s1
        LEFT JOIN s3 ON s3.s1=ids.s1
    ),
    scores AS (
        SELECT
            *,
            CASE
                WHEN tc=0 AND p=0 THEN 1.0
                WHEN tc=0 OR p=0 THEN 0.0
                ELSE 1.25 * tp / (0.25 * tc + p)
            END AS f05
        FROM per_s1
    )
    SELECT
        AVG(f05) AS macro_f05,
        SUM(tp)::DOUBLE / NULLIF(SUM(p),0) AS precision,
        SUM(tp)::DOUBLE / NULLIF(SUM(tc),0) AS recall,
        SUM(p) AS predicted,
        SUM(tp) AS tp,
        SUM(CASE WHEN tc=0 AND p=0 THEN 1 ELSE 0 END) AS correctly_empty,
        SUM(CASE WHEN tc=0 THEN p ELSE 0 END) AS fp_singletons,
        COUNT(*) AS s1_count
    FROM scores
    """

    row = con.execute(sql).fetchone()
    return {
        "threshold_s2": t2,
        "threshold_s3": t3,
        "macro_f05": float(row[0] or 0.0),
        "precision": float(row[1] or 0.0),
        "recall": float(row[2] or 0.0),
        "predicted": int(row[3] or 0),
        "tp": int(row[4] or 0),
        "correctly_empty": int(row[5] or 0),
        "fp_singletons": int(row[6] or 0),
        "s1_count": int(row[7] or 0),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    started = time.time()

    cleanup_own_outputs()

    header("AMAZON ML CHALLENGE — AUGMENTED V10 FRESH-DEV BLOCK IMPACT V1")
    print(f"Memory           : {MEMORY}")
    print(f"Threads          : {THREADS}")
    print(f"Batch            : {BATCH}")
    print(f"Max bucket size  : {MAX_BUCKET_SIZE}")
    print(f"Fresh S1         : {FRESH_S1}")
    print("Final Holdout GT : NOT READ")
    print("Test data        : NOT READ")
    print("Final outputs    : NOT MODIFIED")
    print()

    for p in [
        GT,
        S1,
        S2,
        S3,
        FRESH_S1,
        BASELINE_LABELED["S2"],
        BASELINE_LABELED["S3"],
        EXISTING_FRESH_RANKED["S2"],
        EXISTING_FRESH_RANKED["S3"],
    ]:
        require(p)

    con = connect()

    try:
        # =============================================================
        # 1. FIT EXACT FRESH-DEV MODELS
        # =============================================================
        s2_scaler, s2_model = fit_model(con, "S2")
        s3_scaler, s3_model = fit_model(con, "S3")

        # =============================================================
        # 2. SOURCE PREPROCESSING
        # =============================================================
        build_source_views(con)

        # =============================================================
        # 3. BASELINE RE-SCORE / REPRODUCTION CHECK
        # =============================================================
        baseline_reproduction = {}
        baseline_features = {}
        baseline_ranked = {}

        for target, scaler, model in (
            ("S2", s2_scaler, s2_model),
            ("S3", s3_scaler, s3_model),
        ):
            baseline_features[target] = make_baseline_fresh_features(con, target)

            baseline_ranked_base = build_ranked_base(
                con,
                target,
                baseline_features[target],
                None,
            )

            baseline_ranked[target] = score_ranked_with_model(
                con,
                target,
                baseline_ranked_base,
                scaler,
                model,
                "baseline_recomputed",
            )

            baseline_reproduction[target] = compare_ranked_reproduction(
                con,
                target,
                baseline_ranked[target],
            )

            if not baseline_reproduction[target]["pass"]:
                raise RuntimeError(
                    f"{target} baseline reproduction mismatch. "
                    "Stopping before block augmentation."
                )

        # =============================================================
        # 4. EVALUATE ORIGINAL FRESH RANKS + RECOMPUTED BASELINE
        # =============================================================
        original_current = evaluate_policy(
            con,
            EXISTING_FRESH_RANKED["S2"],
            EXISTING_FRESH_RANKED["S3"],
            CURRENT_S2,
            CURRENT_S3,
        )

        original_candidate = evaluate_policy(
            con,
            EXISTING_FRESH_RANKED["S2"],
            EXISTING_FRESH_RANKED["S3"],
            CANDIDATE_S2,
            CANDIDATE_S3,
        )

        reproduced_current = evaluate_policy(
            con,
            baseline_ranked["S2"],
            baseline_ranked["S3"],
            CURRENT_S2,
            CURRENT_S3,
        )

        reproduced_candidate = evaluate_policy(
            con,
            baseline_ranked["S2"],
            baseline_ranked["S3"],
            CANDIDATE_S2,
            CANDIDATE_S3,
        )

        # =============================================================
        # 5. GENERATE NEW BLOCK CANDIDATES
        # =============================================================
        block_results = {"S2": [], "S3": []}
        new_features = {}

        for target in ("S2", "S3"):
            baseline_pairs = materialize_baseline_pairs(con, target)
            paths = []

            for block_name in SELECTED_BLOCKS[target]:
                p = generate_new_block(
                    con,
                    target,
                    block_name,
                    baseline_pairs,
                )
                paths.append(p)

                n = count_rows(con, p)
                block_results[target].append({
                    "block": block_name,
                    "rows_after_baseline_subtraction": n,
                    "estimated_pairs_after_bucket_cap": None,
                })

            new_candidates = consolidate_new_candidates(
                con,
                target,
                paths,
            )

            block_results[target].append({
                "block": "__UNION__",
                "rows_after_dedup": count_rows(con, new_candidates),
            })

            new_features[target] = score_new_candidates(
                con,
                target,
                new_candidates,
            )

        # =============================================================
        # 6. BUILD AUGMENTED POOL + RESCORE ALL FRESH CANDIDATES
        # =============================================================
        augmented_ranked = {}

        for target, scaler, model in (
            ("S2", s2_scaler, s2_model),
            ("S3", s3_scaler, s3_model),
        ):
            augmented_base = build_ranked_base(
                con,
                target,
                baseline_features[target],
                new_features[target],
            )

            augmented_ranked[target] = score_ranked_with_model(
                con,
                target,
                augmented_base,
                scaler,
                model,
                "augmented",
            )

        # =============================================================
        # 7. POLICY EVALUATION
        # =============================================================
        augmented_current = evaluate_policy(
            con,
            augmented_ranked["S2"],
            augmented_ranked["S3"],
            CURRENT_S2,
            CURRENT_S3,
        )

        augmented_candidate = evaluate_policy(
            con,
            augmented_ranked["S2"],
            augmented_ranked["S3"],
            CANDIDATE_S2,
            CANDIDATE_S3,
        )

        def delta(a: dict, b: dict) -> dict:
            return {
                "macro_f05_delta": b["macro_f05"] - a["macro_f05"],
                "precision_delta": b["precision"] - a["precision"],
                "recall_delta": b["recall"] - a["recall"],
                "predicted_delta": b["predicted"] - a["predicted"],
                "tp_delta": b["tp"] - a["tp"],
                "correctly_empty_delta": b["correctly_empty"] - a["correctly_empty"],
                "fp_singletons_delta": b["fp_singletons"] - a["fp_singletons"],
            }

        report = {
            "version": "BLOCK_AUGMENTED_V10_FRESH_V1",
            "selected_blocks": SELECTED_BLOCKS,
            "parameters": {
                "memory": MEMORY,
                "threads": THREADS,
                "batch": BATCH,
                "max_bucket_size": MAX_BUCKET_SIZE,
                "current_threshold_s2": CURRENT_S2,
                "current_threshold_s3": CURRENT_S3,
                "candidate_threshold_s2": CANDIDATE_S2,
                "candidate_threshold_s3": CANDIDATE_S3,
            },
            "safety": {
                "final_holdout_gt_read": False,
                "test_data_read": False,
                "final_outputs_modified": False,
                "final_zip_modified": False,
                "github_release_modified": False,
            },
            "baseline_reproduction": baseline_reproduction,
            "policy_metrics": {
                "original_existing_ranked_current": original_current,
                "original_existing_ranked_candidate": original_candidate,
                "recomputed_baseline_current": reproduced_current,
                "recomputed_baseline_candidate": reproduced_candidate,
                "augmented_current": augmented_current,
                "augmented_candidate": augmented_candidate,
                "delta_augmented_vs_baseline_current": delta(
                    reproduced_current, augmented_current
                ),
                "delta_augmented_vs_baseline_candidate": delta(
                    reproduced_candidate, augmented_candidate
                ),
            },
            "block_generation": block_results,
            "paths": {
                "output_dir": str(OUT),
                "report": str(REPORT),
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        # =============================================================
        # 8. HUMAN-READABLE SUMMARY
        # =============================================================
        header("RESULT SUMMARY")

        print("\nCURRENT THRESHOLDS")
        print(
            f"Baseline  F0.5={reproduced_current['macro_f05']:.12f}  "
            f"P={reproduced_current['precision']:.12f}  "
            f"R={reproduced_current['recall']:.12f}"
        )
        print(
            f"Augmented F0.5={augmented_current['macro_f05']:.12f}  "
            f"P={augmented_current['precision']:.12f}  "
            f"R={augmented_current['recall']:.12f}"
        )
        print(
            f"Δ         F0.5={augmented_current['macro_f05'] - reproduced_current['macro_f05']:+.12f}  "
            f"P={augmented_current['precision'] - reproduced_current['precision']:+.12f}  "
            f"R={augmented_current['recall'] - reproduced_current['recall']:+.12f}"
        )

        print("\nOLD-VALIDATION SELECTED THRESHOLDS")
        print(
            f"Baseline  F0.5={reproduced_candidate['macro_f05']:.12f}  "
            f"P={reproduced_candidate['precision']:.12f}  "
            f"R={reproduced_candidate['recall']:.12f}"
        )
        print(
            f"Augmented F0.5={augmented_candidate['macro_f05']:.12f}  "
            f"P={augmented_candidate['precision']:.12f}  "
            f"R={augmented_candidate['recall']:.12f}"
        )
        print(
            f"Δ         F0.5={augmented_candidate['macro_f05'] - reproduced_candidate['macro_f05']:+.12f}  "
            f"P={augmented_candidate['precision'] - reproduced_candidate['precision']:+.12f}  "
            f"R={augmented_candidate['recall'] - reproduced_candidate['recall']:+.12f}"
        )

        print("\nBLOCK COUNTS")
        for target in ("S2", "S3"):
            union_row = [x for x in block_results[target] if x["block"] == "__UNION__"][0]
            print(
                f"{target}: selected new unique candidates = "
                f"{union_row['rows_after_dedup']:,}"
            )

        print(f"\nReport: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
