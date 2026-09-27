#!/usr/bin/env python3
"""
Amazon ML Challenge 2026
AUGMENTED V10 FRESH-DEV BLOCK IMPACT V3

Purpose
-------
Run the baseline and the transferable-block augmentation inside the SAME
fresh-development process and with the SAME freshly fitted V10 models.

This version deliberately does NOT attempt to reproduce an older persisted
fresh_ranked_*.parquet file. Previous V1/V2 experiments showed that persisted
ranking artifacts are not an exact reproducibility oracle even when the
training recipe appears identical. Instead, baseline and augmented results
share the exact same:
  - fresh S1 split
  - fresh-dev exclusion from training
  - pairwise training data
  - StandardScaler
  - LogisticRegression models
  - scoring implementation

Baseline:
  existing V6+V9.1 labeled candidate pool

Augmented:
  baseline + selected transferable blocks

S2 blocks:
  postal_addr_f3_tail2_name_f1
  name_f2_addr_tail3
  name_f1_addr_f4_tail2

S3 blocks:
  name_f3_addr_f2
  postal_addr_f2_name_f2
  name_f4_l2_len5
  name_f3_l2_postal

Safety
------
- Final Holdout V2 GT is never read.
- Test data is never read.
- Final output TSVs are never read or modified.
- Final ZIP is never read or modified.
- Only this experiment directory is written.
- The official fresh-dev script is imported for its exact fresh split,
  model-training, and baseline scoring functions; its output directory
  globals are redirected into this experiment directory.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio


# =====================================================================
# PATHS
# =====================================================================
PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"

FRESH_SCRIPT_PATH = PROJECT / "fresh_dev_joint_threshold_confirmation_v1.py"
GT = TRAIN / "ground_truth_pairs.parquet"
S1 = TRAIN / "train_source1.parquet"
S2 = TRAIN / "train_source2.parquet"
S3 = TRAIN / "train_source3.parquet"

BASELINE_LABELED = {
    "S2": PROJECT / "candidate_output_v10" / "train_labeled_base_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10" / "train_labeled_base_s1_s3.parquet",
}

AUDIT = PROJECT / "validation_leakage_audit"
PREVIOUS_HOLDOUT_S1 = AUDIT / "holdout" / "prospective_holdout_s1.parquet"
FINAL_HOLDOUT_S1 = AUDIT / "final_holdout_v2" / "final_holdout_s1.parquet"

OUT = AUDIT / "block_augmented_v10_fresh_v3"
TMP = OUT / "tmp"
BLOCK_PARTS = OUT / "block_parts"
NEW_FEATURE_PARTS = OUT / "new_feature_parts"
FRESH_OUT = OUT / "fresh_dev_core"

REPORT = OUT / "block_augmented_v10_fresh_v3_report.json"

MEMORY = os.environ.get("BLOCK_AUGMENT_V3_MEMORY", "8GB")
THREADS = int(os.environ.get("BLOCK_AUGMENT_V3_THREADS", "2"))
BATCH = int(os.environ.get("BLOCK_AUGMENT_V3_BATCH", "100000"))
MAX_BUCKET_SIZE = int(os.environ.get("BLOCK_AUGMENT_V3_MAX_BUCKET_SIZE", "100"))

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

# Exact block expressions from block_transferability_audit_v2.py.
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
        "substr(name_compact,1,1) || '|' || substr(address_compact,1,4) "
        "|| '|' || right(address_compact,2)"
    ),
    "name_f3_addr_f2": (
        "substr(name_compact,1,3) || '|' || substr(address_compact,1,2)"
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
]


# =====================================================================
# GENERAL HELPERS
# =====================================================================
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
    (TMP / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={qp(TMP / 'duckdb_tmp')}")
    con.execute("SET max_temp_directory_size='50GB'")
    return con


def reset_experiment_dir() -> None:
    if OUT.exists():
        for child in OUT.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    BLOCK_PARTS.mkdir(parents=True, exist_ok=True)
    NEW_FEATURE_PARTS.mkdir(parents=True, exist_ok=True)
    FRESH_OUT.mkdir(parents=True, exist_ok=True)


def load_original_fresh_module():
    """
    Import the exact existing fresh-dev implementation, but redirect its
    output globals into this isolated experiment directory.
    """
    require(FRESH_SCRIPT_PATH)

    spec = importlib.util.spec_from_file_location(
        "fresh_dev_original_v3",
        str(FRESH_SCRIPT_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load fresh-dev script module.")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Redirect all output locations used by the imported functions.
    mod.OUT = FRESH_OUT
    mod.TMP = FRESH_OUT / "tmp"
    mod.PARTS = FRESH_OUT / "score_parts"
    mod.REPORT = FRESH_OUT / "unused_report.json"

    return mod


# =====================================================================
# EXACT V10 MODEL / FRESH SPLIT FROM THE EXISTING SCRIPT
# =====================================================================
def prepare_fresh_core(con: duckdb.DuckDBPyConnection):
    mod = load_original_fresh_module()

    header("1. BUILD SAME FRESH-DEVELOPMENT SPLIT + FIT SAME V10 MODELS")

    # The imported build_fresh_split uses the exact original:
    #   old validation exclusion
    #   previous holdout exclusion
    #   final holdout S1 boundary exclusion
    #   FRESH_MOD=11
    #   FRESH_SALT="FRESH_DEV_JOINT_CONFIRMATION_V1"
    fresh_n = mod.build_fresh_split(con)

    print(f"Fresh development S1: {fresh_n:,}")

    # Fit exactly the same models as the original fresh-dev script.
    s2_scaler, s2_model = mod.fit_one(con, "S2")
    s3_scaler, s3_model = mod.fit_one(con, "S3")

    return mod, fresh_n, {
        "S2": (s2_scaler, s2_model),
        "S3": (s3_scaler, s3_model),
    }


# =====================================================================
# SOURCE VIEWS — EXACT BLOCK AUDIT PREPROCESSING
# =====================================================================
def build_source_views(con: duckdb.DuckDBPyConnection) -> None:
    header("2. BUILD EXACT BLOCK-AUDIT SOURCE VIEWS")

    require(S1)
    require(S2)
    require(S3)
    require(FRESH_OUT / "fresh_dev_s1.parquet")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fresh_s1_ids AS
        SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(FRESH_OUT / "fresh_dev_s1.parquet")})
    """)

    # Exact normalization logic from block_transferability_audit_v2.py:
    #   compact name/address: remove every non [a-z0-9]
    #   postal: first [0-9]{5,6} run
    #   addr_num: all digits, then remove all-zero values
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
          ON CAST(s.entity_id AS VARCHAR)=f.s1
    """)

    for target, path in (("S2", S2), ("S3", S3)):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {target.lower()}_src AS
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
            FROM read_parquet({qp(path)}) s
        """)

    print(
        "Source views ready: "
        f"S1 fresh={int(con.execute('SELECT COUNT(*) FROM s1_src').fetchone()[0]):,}"
    )


# =====================================================================
# BASELINE CANDIDATE/FEATURE POOL
# =====================================================================
def make_baseline_fresh_features(
    con: duckdb.DuckDBPyConnection,
    target: str,
) -> Path:
    """
    Copy the frozen V6+V9.1 candidate feature rows for ONLY the fresh S1 IDs.
    Preserve the original base_rank exactly as stored in the frozen labeled
    candidate pool.
    """
    out = OUT / f"baseline_features_fresh_{target.lower()}.parquet"
    source = BASELINE_LABELED[target]
    require(source)

    out.unlink(missing_ok=True)

    con.execute(f"""
        COPY (
            SELECT
                CAST(l.source1_entity_id AS VARCHAR) AS source1_entity_id,
                CAST(l.candidate_entity_id AS VARCHAR) AS candidate_entity_id,
                CAST(l.evidence_rows AS DOUBLE) AS evidence_rows,
                CAST(l.evidence_file_count AS DOUBLE) AS evidence_file_count,
                CAST(l.exact_key_count AS DOUBLE) AS exact_key_count,
                CAST(l.name_similarity AS DOUBLE) AS name_similarity,
                CAST(l.address_similarity AS DOUBLE) AS address_similarity,
                CAST(l.name_exact AS DOUBLE) AS name_exact,
                CAST(l.address_exact AS DOUBLE) AS address_exact,
                CAST(l.country_exact AS DOUBLE) AS country_exact,
                CAST(l.name_length_ratio AS DOUBLE) AS name_length_ratio,
                CAST(l.address_length_ratio AS DOUBLE) AS address_length_ratio,
                CAST(l.base_score AS DOUBLE) AS base_score,
                CAST(l.base_rank AS BIGINT) AS base_rank
            FROM read_parquet({qp(source)}) l
            INNER JOIN fresh_s1_ids f
              ON CAST(l.source1_entity_id AS VARCHAR)=f.s1
        )
        TO {qp(out)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print(
        f"{target} baseline feature rows: {count_rows(con, out):,}",
        flush=True,
    )
    return out


def score_ranked_base(
    con: duckdb.DuckDBPyConnection,
    target: str,
    ranked_base: Path,
    scaler,
    model,
    tag: str,
) -> Path:
    """
    Use the EXACT feature-to-utility scoring code from the original fresh
    script, but on a supplied ranked candidate pool.
    """
    out_dir = OUT / f"utility_parts_{tag}_{target.lower()}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

        p = out_dir / f"part_{idx:05d}.parquet"
        pd.DataFrame(
            {
                "source1_entity_id": d["source1_entity_id"].astype(str),
                "candidate_entity_id": d["candidate_entity_id"].astype(str),
                "utility": utility.astype("float32"),
            }
        ).to_parquet(p, index=False, compression="zstd")

        total += len(d)
        idx += 1

        if total and total % 1_000_000 < len(d):
            print(
                f"{target} {tag} utility rows: {total:,}",
                flush=True,
            )

    if total == 0:
        raise RuntimeError(f"No rows scored for {target}/{tag}")

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
            FROM read_parquet({qp(out_dir / '*.parquet')})
        )
        TO {qp(ranked)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"{target} {tag} ranked rows: {count_rows(con, ranked):,}")
    return ranked


# =====================================================================
# NEW BLOCK CANDIDATES
# =====================================================================
def generate_block(
    con: duckdb.DuckDBPyConnection,
    target: str,
    block_name: str,
    baseline_pairs: Path,
) -> tuple[Path, int, int]:
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
          AND NOT ends_with(block_key,'|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rcnt AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {right_view}
        WHERE block_key <> ''
          AND NOT ends_with(block_key,'|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    estimated = int(
        con.execute("""
            SELECT COALESCE(SUM(l.n*r.n),0)
            FROM lcnt l
            JOIN rcnt r USING(block_key)
        """).fetchone()[0]
    )

    out = BLOCK_PARTS / target.lower() / f"{block_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    header(f"3. GENERATE NEW BLOCK — {target} / {block_name}")
    print(f"Estimated pairs after bucket cap: {estimated:,}")

    if estimated == 0:
        raise RuntimeError(
            f"{target}/{block_name} has no shared valid block keys."
        )

    con.execute(f"""
        COPY (
            SELECT
                CAST(l.entity_id AS VARCHAR) AS source1_entity_id,
                CAST(r.entity_id AS VARCHAR) AS candidate_entity_id,
                '{block_name}' AS block_name
            FROM {left_view} l
            INNER JOIN {right_view} r
              ON l.block_key=r.block_key
            INNER JOIN lcnt lc
              ON lc.block_key=l.block_key
            INNER JOIN rcnt rc
              ON rc.block_key=r.block_key
            LEFT JOIN read_parquet({qp(baseline_pairs)}) b
              ON CAST(b.source1_entity_id AS VARCHAR)=CAST(l.entity_id AS VARCHAR)
             AND CAST(b.candidate_entity_id AS VARCHAR)=CAST(r.entity_id AS VARCHAR)
            WHERE b.source1_entity_id IS NULL
        )
        TO {qp(out)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    actual = count_rows(con, out)
    print(f"New candidates after baseline subtraction: {actual:,}")

    return out, estimated, actual


def consolidate_blocks(
    con: duckdb.DuckDBPyConnection,
    target: str,
    paths: List[Path],
) -> Path:
    if not paths:
        raise RuntimeError(f"No block outputs for {target}")

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

    n = count_rows(con, out)
    print(f"{target} unique new candidates across selected blocks: {n:,}")
    return out


# =====================================================================
# SCORE NEW CANDIDATES — V10 FEATURES
# =====================================================================
def score_new_candidates(
    con: duckdb.DuckDBPyConnection,
    target: str,
    new_candidates: Path,
) -> Path:
    target_path = S2 if target == "S2" else S3
    target_table = target.lower() + "_src"

    raw_features = OUT / f"raw_new_features_{target.lower()}.parquet"
    raw_features.unlink(missing_ok=True)

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
                    WHEN COALESCE(l.name_norm,'') <> ''
                     AND COALESCE(l.name_norm,'') = COALESCE(r.name_norm,'')
                    THEN 1 ELSE 0
                END AS name_exact,

                CASE
                    WHEN COALESCE(l.address_norm,'') <> ''
                     AND COALESCE(r.address_norm,'') <> ''
                     AND l.address_norm = r.address_norm
                    THEN 1 ELSE 0
                END AS address_exact,

                CASE
                    WHEN COALESCE(l.country_norm,'') <> ''
                     AND l.country_norm = r.country_norm
                    THEN 1 ELSE 0
                END AS country_exact,

                CASE
                    WHEN COALESCE(l.name_norm,'') = ''
                      OR COALESCE(r.name_norm,'') = ''
                    THEN 0.0
                    ELSE
                      LEAST(length(l.name_norm),length(r.name_norm))::DOUBLE
                      /
                      GREATEST(length(l.name_norm),length(r.name_norm),1)
                END AS name_length_ratio,

                CASE
                    WHEN COALESCE(l.address_norm,'') = ''
                      OR COALESCE(r.address_norm,'') = ''
                    THEN 0.0
                    ELSE
                      LEAST(length(l.address_norm),length(r.address_norm))::DOUBLE
                      /
                      GREATEST(length(l.address_norm),length(r.address_norm),1)
                END AS address_length_ratio

            FROM read_parquet({qp(new_candidates)}) c
            INNER JOIN read_parquet({qp(S1)}) l
              ON CAST(c.source1_entity_id AS VARCHAR)=CAST(l.entity_id AS VARCHAR)
            INNER JOIN read_parquet({qp(target_path)}) r
              ON CAST(c.candidate_entity_id AS VARCHAR)=CAST(r.entity_id AS VARCHAR)
        )
        TO {qp(raw_features)}
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
        FROM read_parquet({qp(raw_features)})
    """).to_arrow_reader(batch_size=BATCH)

    total = 0
    idx = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue

        ln = d["left_name"].fillna("").astype(str).tolist()
        rn = d["right_name"].fillna("").astype(str).tolist()
        la = d["left_address"].fillna("").astype(str).tolist()
        ra = d["right_address"].fillna("").astype(str).tolist()

        d["name_similarity"] = np.asarray(
            [
                ratio(a, b) / 100.0 if a and b else 0.0
                for a, b in zip(ln, rn)
            ],
            dtype="float32",
        )

        d["address_similarity"] = np.asarray(
            [
                ratio(a, b) / 100.0 if a and b else 0.0
                for a, b in zip(la, ra)
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
            print(
                f"{target} new feature rows scored: {total:,}",
                flush=True,
            )

    if total == 0:
        raise RuntimeError(f"No new feature rows for {target}")

    raw_features.unlink(missing_ok=True)

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

    shutil.rmtree(part_dir, ignore_errors=True)

    print(
        f"{target} new scored-feature rows: {count_rows(con, scored_new):,}"
    )
    return scored_new


# =====================================================================
# COMBINED BASE RANK
# =====================================================================
def build_augmented_ranked_base(
    con: duckdb.DuckDBPyConnection,
    target: str,
    baseline_features: Path,
    new_features: Path,
) -> Path:
    combined = OUT / f"combined_augmented_base_{target.lower()}.parquet"
    ranked = OUT / f"augmented_base_ranked_{target.lower()}.parquet"
    combined.unlink(missing_ok=True)
    ranked.unlink(missing_ok=True)

    # Baseline rows already have their frozen feature values. New block rows
    # have freshly computed equivalent V10 features.
    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                evidence_rows,
                evidence_file_count,
                exact_key_count,
                name_similarity,
                address_similarity,
                name_exact,
                address_exact,
                country_exact,
                name_length_ratio,
                address_length_ratio,
                base_score
            FROM read_parquet({qp(baseline_features)})
            UNION ALL
            SELECT
                source1_entity_id,
                candidate_entity_id,
                CAST(evidence_rows AS DOUBLE),
                CAST(evidence_file_count AS DOUBLE),
                CAST(exact_key_count AS DOUBLE),
                CAST(name_similarity AS DOUBLE),
                CAST(address_similarity AS DOUBLE),
                CAST(name_exact AS DOUBLE),
                CAST(address_exact AS DOUBLE),
                CAST(country_exact AS DOUBLE),
                CAST(name_length_ratio AS DOUBLE),
                CAST(address_length_ratio AS DOUBLE),
                CAST(base_score AS DOUBLE)
            FROM read_parquet({qp(new_features)})
        )
        TO {qp(combined)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Sanity check: duplicate pair IDs would corrupt the candidate pool.
    dup = int(
        con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT source1_entity_id, candidate_entity_id
                FROM read_parquet({qp(combined)})
                GROUP BY source1_entity_id, candidate_entity_id
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
    )
    if dup:
        raise RuntimeError(
            f"{target} augmented combined pool has {dup:,} duplicate pair IDs."
        )

    con.execute(f"""
        COPY (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY base_score DESC, candidate_entity_id
                ) AS base_rank
            FROM read_parquet({qp(combined)})
        )
        TO {qp(ranked)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    combined.unlink(missing_ok=True)

    print(
        f"{target} augmented base rows: {count_rows(con, ranked):,}"
    )
    return ranked


# =====================================================================
# CORRECT FRESH GT LOADER
# =====================================================================
def load_truth(con: duckdb.DuckDBPyConnection) -> Dict[str, int]:
    """
    Corrected version of the original script's load_truth() bug:
    gt_fresh has column `matched`, not `matched_entity_id`.
    """
    fresh = FRESH_OUT / "fresh_dev_s1.parquet"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_fresh AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT)}) g
        INNER JOIN read_parquet({qp(fresh)}) f
          ON CAST(g.source1_entity_id AS VARCHAR)=CAST(f.source1_entity_id AS VARCHAR)
        WHERE COALESCE(g.label,1)=1
    """)

    return {
        str(s1): int(cnt)
        for s1, cnt in con.execute(f"""
            SELECT
                f.source1_entity_id,
                COUNT(g.matched)
            FROM read_parquet({qp(fresh)}) f
            LEFT JOIN gt_fresh g
              ON CAST(f.source1_entity_id AS VARCHAR)=g.s1
            GROUP BY f.source1_entity_id
        """).fetchall()
    }


# =====================================================================
# POLICY EVALUATION
# =====================================================================
def evaluate_policy(
    con: duckdb.DuckDBPyConnection,
    s2_ranked: Path,
    s3_ranked: Path,
    t2: float,
    t3: float,
    truth: Dict[str, int],
) -> dict:
    ids = sorted(truth)

    # Load top1 rows from each ranked file into compact Python dictionaries.
    def top1(path: Path, prefix: str) -> Dict[str, tuple[float, int]]:
        rows = con.execute(f"""
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                CAST(utility AS DOUBLE) AS utility,
                CAST(candidate_entity_id AS VARCHAR) AS cand
            FROM read_parquet({qp(path)})
            WHERE rank=1
              AND CAST(candidate_entity_id AS VARCHAR) LIKE '{prefix}%'
        """).fetchall()

        gt_rows = []
        for s1, utility, cand in rows:
            gt_rows.append(
                (
                    str(s1),
                    float(utility),
                    1 if _is_truth_match(con, str(s1), str(cand)) else 0,
                )
            )

        return {
            s1: (utility, tp)
            for s1, utility, tp in gt_rows
        }

    s2 = top1(s2_ranked, "S2-")
    s3 = top1(s3_ranked, "S3-")

    score_sum = 0.0
    predicted = 0
    tp_total = 0
    correctly_empty = 0
    fp_singletons = 0
    truth_total = sum(truth.values())

    for s1 in ids:
        tc = truth[s1]
        p = 0
        tp = 0

        r2 = s2.get(s1)
        if r2 is not None and r2[0] >= t2:
            p += 1
            tp += r2[1]

        r3 = s3.get(s1)
        if r3 is not None and r3[0] >= t3:
            p += 1
            tp += r3[1]

        if tc == 0 and p == 0:
            f05 = 1.0
        elif tc == 0 or p == 0:
            f05 = 0.0
        else:
            f05 = 1.25 * tp / (0.25 * tc + p)

        score_sum += f05
        predicted += p
        tp_total += tp

        if tc == 0 and p == 0:
            correctly_empty += 1
        if tc == 0:
            fp_singletons += p

    return {
        "threshold_s2": t2,
        "threshold_s3": t3,
        "macro_f05": score_sum / len(ids),
        "precision": tp_total / predicted if predicted else 0.0,
        "recall": tp_total / truth_total if truth_total else 0.0,
        "predicted": predicted,
        "tp": tp_total,
        "correctly_empty": correctly_empty,
        "fp_singletons": fp_singletons,
        "s1_count": len(ids),
    }


def _is_truth_match(
    con: duckdb.DuckDBPyConnection,
    s1: str,
    cand: str,
) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM gt_fresh
        WHERE s1 = ? AND matched = ?
        LIMIT 1
        """,
        [s1, cand],
    ).fetchone()
    return row is not None


# =====================================================================
# BLOCK METRICS FROM MATERIALIZED OUTPUTS
# =====================================================================
def block_truth_stats(
    con: duckdb.DuckDBPyConnection,
    block_path: Path,
) -> dict:
    total = int(
        con.execute(f"""
            SELECT COUNT(*)
            FROM read_parquet({qp(block_path)})
        """).fetchone()[0]
    )

    new_gt = int(
        con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT
                    g.s1,
                    g.matched
                FROM gt_fresh g
                INNER JOIN read_parquet({qp(block_path)}) b
                  ON CAST(g.s1 AS VARCHAR)=CAST(b.source1_entity_id AS VARCHAR)
                 AND CAST(g.matched AS VARCHAR)=CAST(b.candidate_entity_id AS VARCHAR)
            )
        """).fetchone()[0]
    )

    return {
        "rows": total,
        "gt_matches": new_gt,
    }


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    started = time.time()
    reset_experiment_dir()

    header("AMAZON ML CHALLENGE — AUGMENTED V10 FRESH-DEV BLOCK IMPACT V3")
    print(f"Memory            : {MEMORY}")
    print(f"Threads           : {THREADS}")
    print(f"Batch             : {BATCH}")
    print(f"Max bucket size   : {MAX_BUCKET_SIZE}")
    print(f"Experiment OUT    : {OUT}")
    print("Final Holdout GT  : NOT READ")
    print("Test data         : NOT READ")
    print("Final outputs     : NOT MODIFIED")
    print("Final ZIP         : NOT MODIFIED")

    for p in [
        FRESH_SCRIPT_PATH,
        GT,
        S1,
        S2,
        S3,
        PREVIOUS_HOLDOUT_S1,
        FINAL_HOLDOUT_S1,
        BASELINE_LABELED["S2"],
        BASELINE_LABELED["S3"],
    ]:
        require(p)

    con = connect()

    try:
        # -------------------------------------------------------------
        # 1) SAME fresh split + SAME fresh models
        # -------------------------------------------------------------
        fresh_mod, fresh_n, models = prepare_fresh_core(con)

        # -------------------------------------------------------------
        # 2) SAME baseline scoring implementation, but isolated
        # -------------------------------------------------------------
        build_source_views(con)

        baseline_ranked = {}
        baseline_features = {}

        for target in ("S2", "S3"):
            baseline_features[target] = make_baseline_fresh_features(
                con, target
            )

            # Frozen V6+V9.1 pool already has base_rank. Preserve it.
            baseline_ranked[target] = score_ranked_base(
                con,
                target,
                baseline_features[target],
                models[target][0],
                models[target][1],
                "baseline",
            )

        # Corrected truth loader.
        truth = load_truth(con)

        # -------------------------------------------------------------
        # 3) Baseline policy metrics
        # -------------------------------------------------------------
        baseline_current = evaluate_policy(
            con,
            baseline_ranked["S2"],
            baseline_ranked["S3"],
            CURRENT_S2,
            CURRENT_S3,
            truth,
        )

        baseline_candidate = evaluate_policy(
            con,
            baseline_ranked["S2"],
            baseline_ranked["S3"],
            CANDIDATE_S2,
            CANDIDATE_S3,
            truth,
        )

        # -------------------------------------------------------------
        # 4) New transferable block candidates
        # -------------------------------------------------------------
        block_results = {"S2": [], "S3": []}
        new_features = {}

        for target in ("S2", "S3"):
            baseline_pairs = OUT / f"baseline_pairs_fresh_{target.lower()}.parquet"

            con.execute(f"""
                COPY (
                    SELECT DISTINCT
                        CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                        CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
                    FROM read_parquet({qp(BASELINE_LABELED[target])})
                    INNER JOIN fresh_s1_ids f
                      ON CAST(source1_entity_id AS VARCHAR)=f.s1
                )
                TO {qp(baseline_pairs)}
                (FORMAT PARQUET, COMPRESSION ZSTD)
            """)

            paths = []

            for block_name in SELECTED_BLOCKS[target]:
                p, estimated, actual = generate_block(
                    con,
                    target,
                    block_name,
                    baseline_pairs,
                )

                paths.append(p)

                stats = block_truth_stats(con, p)

                block_results[target].append({
                    "block": block_name,
                    "estimated_pairs_after_bucket_cap": estimated,
                    "new_candidate_rows": actual,
                    "new_gt_rows": stats["gt_matches"],
                    "new_gt_per_million_new_rows": (
                        1_000_000.0 * stats["gt_matches"] / actual
                        if actual else 0.0
                    ),
                })

            new_candidates = consolidate_blocks(
                con,
                target,
                paths,
            )

            union_stats = block_truth_stats(con, new_candidates)
            block_results[target].append({
                "block": "__UNION__",
                "new_candidate_rows": union_stats["rows"],
                "new_gt_rows": union_stats["gt_matches"],
            })

            new_features[target] = score_new_candidates(
                con,
                target,
                new_candidates,
            )

        # -------------------------------------------------------------
        # 5) Augmented scoring with SAME fitted models
        # -------------------------------------------------------------
        augmented_ranked = {}

        for target in ("S2", "S3"):
            augmented_base = build_augmented_ranked_base(
                con,
                target,
                baseline_features[target],
                new_features[target],
            )

            augmented_ranked[target] = score_ranked_base(
                con,
                target,
                augmented_base,
                models[target][0],
                models[target][1],
                "augmented",
            )

        # -------------------------------------------------------------
        # 6) Augmented policy metrics
        # -------------------------------------------------------------
        augmented_current = evaluate_policy(
            con,
            augmented_ranked["S2"],
            augmented_ranked["S3"],
            CURRENT_S2,
            CURRENT_S3,
            truth,
        )

        augmented_candidate = evaluate_policy(
            con,
            augmented_ranked["S2"],
            augmented_ranked["S3"],
            CANDIDATE_S2,
            CANDIDATE_S3,
            truth,
        )

        def metric_delta(base: dict, new: dict) -> dict:
            return {
                "macro_f05_delta": new["macro_f05"] - base["macro_f05"],
                "precision_delta": new["precision"] - base["precision"],
                "recall_delta": new["recall"] - base["recall"],
                "predicted_delta": new["predicted"] - base["predicted"],
                "tp_delta": new["tp"] - base["tp"],
                "correctly_empty_delta": (
                    new["correctly_empty"] - base["correctly_empty"]
                ),
                "fp_singletons_delta": (
                    new["fp_singletons"] - base["fp_singletons"]
                ),
            }

        # -------------------------------------------------------------
        # 7) Report
        # -------------------------------------------------------------
        report = {
            "version": "BLOCK_AUGMENTED_V10_FRESH_V3",
            "fresh_dev_s1": fresh_n,
            "fresh_dev_gt_pairs": sum(truth.values()),
            "selected_blocks": SELECTED_BLOCKS,
            "thresholds": {
                "current_s2": CURRENT_S2,
                "current_s3": CURRENT_S3,
                "candidate_s2": CANDIDATE_S2,
                "candidate_s3": CANDIDATE_S3,
            },
            "policy_metrics": {
                "baseline_current": baseline_current,
                "augmented_current": augmented_current,
                "delta_current": metric_delta(
                    baseline_current, augmented_current
                ),
                "baseline_candidate": baseline_candidate,
                "augmented_candidate": augmented_candidate,
                "delta_candidate": metric_delta(
                    baseline_candidate, augmented_candidate
                ),
            },
            "block_results": block_results,
            "safety": {
                "final_holdout_gt_read": False,
                "test_data_read": False,
                "final_outputs_modified": False,
                "final_zip_modified": False,
                "github_release_modified": False,
                "persisted_old_fresh_rankings_used_as_evaluation_oracle": False,
                "same_process_model_for_baseline_and_augmented": True,
            },
            "paths": {
                "experiment_dir": str(OUT),
                "fresh_core_dir": str(FRESH_OUT),
                "report": str(REPORT),
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        # -------------------------------------------------------------
        # 8) Human-readable summary
        # -------------------------------------------------------------
        header("FINAL EXPERIMENT SUMMARY")

        print("\nCURRENT THRESHOLDS")
        print(
            f"Baseline : F0.5={baseline_current['macro_f05']:.12f} "
            f"P={baseline_current['precision']:.12f} "
            f"R={baseline_current['recall']:.12f} "
            f"TP={baseline_current['tp']:,} "
            f"pred={baseline_current['predicted']:,} "
            f"singletonFP={baseline_current['fp_singletons']:,}"
        )
        print(
            f"Augmented: F0.5={augmented_current['macro_f05']:.12f} "
            f"P={augmented_current['precision']:.12f} "
            f"R={augmented_current['recall']:.12f} "
            f"TP={augmented_current['tp']:,} "
            f"pred={augmented_current['predicted']:,} "
            f"singletonFP={augmented_current['fp_singletons']:,}"
        )
        print(
            f"Delta    : F0.5={augmented_current['macro_f05'] - baseline_current['macro_f05']:+.12f} "
            f"P={augmented_current['precision'] - baseline_current['precision']:+.12f} "
            f"R={augmented_current['recall'] - baseline_current['recall']:+.12f}"
        )

        print("\nOLD-VALIDATION SELECTED THRESHOLDS")
        print(
            f"Baseline : F0.5={baseline_candidate['macro_f05']:.12f} "
            f"P={baseline_candidate['precision']:.12f} "
            f"R={baseline_candidate['recall']:.12f} "
            f"TP={baseline_candidate['tp']:,} "
            f"pred={baseline_candidate['predicted']:,} "
            f"singletonFP={baseline_candidate['fp_singletons']:,}"
        )
        print(
            f"Augmented: F0.5={augmented_candidate['macro_f05']:.12f} "
            f"P={augmented_candidate['precision']:.12f} "
            f"R={augmented_candidate['recall']:.12f} "
            f"TP={augmented_candidate['tp']:,} "
            f"pred={augmented_candidate['predicted']:,} "
            f"singletonFP={augmented_candidate['fp_singletons']:,}"
        )
        print(
            f"Delta    : F0.5={augmented_candidate['macro_f05'] - baseline_candidate['macro_f05']:+.12f} "
            f"P={augmented_candidate['precision'] - baseline_candidate['precision']:+.12f} "
            f"R={augmented_candidate['recall'] - baseline_candidate['recall']:+.12f}"
        )

        print("\nNEW CANDIDATE COUNTS")
        for target in ("S2", "S3"):
            union = [
                x for x in block_results[target]
                if x["block"] == "__UNION__"
            ][0]
            print(
                f"{target}: "
                f"{union['new_candidate_rows']:,} new unique pairs | "
                f"{union['new_gt_rows']:,} new GT pairs"
            )

            for r in block_results[target]:
                if r["block"] == "__UNION__":
                    continue
                print(
                    f"  {r['block']}: "
                    f"{r['new_candidate_rows']:,} pairs | "
                    f"{r['new_gt_rows']:,} GT | "
                    f"{r['new_gt_per_million_new_rows']:.2f}/M"
                )

        print(f"\nReport: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
