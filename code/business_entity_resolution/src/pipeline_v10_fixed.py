from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# AMAZON ML CHALLENGE - V10 FULL PIPELINE
# ============================================================
# Pipeline:
#   V1/V2/V4/V5 candidates
#          |
#          v
#   V6 scored/ranked baseline
#          |
#   + V9.1 new selective candidates
#          |
#          v
#   Score ONLY V9.1-new pairs with the same V6 feature recipe
#          |
#          v
#   Rebuild combined base_score/base_rank
#          |
#          v
#   V10 pairwise logistic ranking
#          |
#          v
#   Full S1->S2 and S1->S3 ranked candidate pools
#
# Important:
#   - V5, V8 and V9.1 outputs are NEVER modified.
#   - V9.1 candidates already present in V6 are excluded before
#     fuzzy scoring, so pairs are not double-counted.
#   - New V9.1 candidates receive V6-compatible transparent
#     similarity features and a synthetic base_score.
#   - The final model is retrained on the expanded candidate pool.
#   - Processing is batch/Parquet based for memory safety.
#
# Run from:
#   /Users/harikeshshukla/mla
#
# Environment:
#   source /Users/harikeshshukla/mla/.venv/bin/activate
#   python -u amazon_ml_pipeline_v10.py
#
# Default memory:
#   8GB DuckDB. Override with:
#     V10_MEMORY_LIMIT=10GB python -u amazon_ml_pipeline_v10.py
# ============================================================


PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"

S1_PATH = TRAIN / "train_source1.parquet"
S2_PATH = TRAIN / "train_source2.parquet"
S3_PATH = TRAIN / "train_source3.parquet"
GT_PATH = TRAIN / "ground_truth_pairs.parquet"

V6_DIR = PROJECT / "candidate_output_v6"
V9_1_DIR = PROJECT / "candidate_output_v9_1"

V6_S2 = V6_DIR / "train_scored_s1_s2.parquet"
V6_S3 = V6_DIR / "train_scored_s1_s3.parquet"

V9_1_S2_DIR = V9_1_DIR / "s1_s2_blocks" / "num_addr_name_prefix3_shards"
V9_1_S3_DIR = V9_1_DIR / "s1_s3_blocks" / "num_addr_name_prefix3_shards"

OUT = PROJECT / "candidate_output_v10"
TMP = OUT / "tmp"
NEW_PARTS = TMP / "new_scored_parts"

MEMORY_LIMIT = os.environ.get("V10_MEMORY_LIMIT", "8GB")
THREADS = int(os.environ.get("V10_THREADS", "4"))
BATCH = int(os.environ.get("V10_BATCH", "100_000"))

VALIDATION_MOD = 5
PAIR_POS_PER_S1 = 1
PAIR_NEG_RANK = 100
PAIR_NEG_PER_S1 = 2
TOPK = 20

# Same transparent V6 score recipe.
def base_score_formula(df: pd.DataFrame) -> pd.Series:
    return (
        30.0 * df["name_exact"].astype(float)
        + 25.0 * df["address_exact"].astype(float)
        + 8.0 * df["country_exact"].astype(float)
        + 22.0 * df["name_similarity"].astype(float)
        + 12.0 * df["address_similarity"].astype(float)
        + 2.0 * df["name_length_ratio"].astype(float)
        + 1.0 * df["address_length_ratio"].astype(float)
        + 0.75 * df["evidence_file_count"].astype(float).clip(upper=4.0)
        + 0.25 * df["exact_key_count"].astype(float).clip(upper=5.0)
    )


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


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required directory not found: {path}")


def setup_connection() -> duckdb.DuckDBPyConnection:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "duckdb_tmp").mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(
        f"PRAGMA temp_directory='{str(TMP / 'duckdb_tmp').replace(chr(92), '/').replace(chr(39), chr(39) * 2)}'"
    )
    return con


def load_ground_truth(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_s2 AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_parquet({qp(GT_PATH)})
        WHERE COALESCE(label, 1) = 1
          AND starts_with(matched_entity_id, 'S2-')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_s3 AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_parquet({qp(GT_PATH)})
        WHERE COALESCE(label, 1) = 1
          AND starts_with(matched_entity_id, 'S3-')
        """
    )


def v9_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.parquet"))


def make_existing_pair_filter_sql(
    v6_path: Path,
    s2_or_s3: str,
) -> str:
    # Existing V6 pair IDs are enough to exclude duplicates.
    return f"""
        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id
        FROM read_parquet({qp(v6_path)})
        WHERE starts_with(candidate_entity_id, '{s2_or_s3}-')
    """


def create_v9_new_table(
    con: duckdb.DuckDBPyConnection,
    *,
    target: str,
    v6_path: Path,
    v9_dir: Path,
) -> int:
    """
    Load V9.1 shard candidates and remove pairs already in V6.
    No GROUP BY on all 34M rows is used; each V9.1 pair exists in
    exactly one hash shard, so DISTINCT is only a safety de-dup.
    """
    files = v9_files(v9_dir)
    if not files:
        raise RuntimeError(f"No V9.1 Parquet shards found: {v9_dir}")

    shard_union = " UNION ALL ".join(
        f"""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            'num_addr_name_prefix3' AS block_name,
            1::INTEGER AS evidence_rows,
            1::INTEGER AS evidence_file_count,
            0::INTEGER AS exact_key_count
        FROM read_parquet({qp(p)})
        WHERE starts_with(candidate_entity_id, '{target}-')
        """
        for p in files
    )

    existing = make_existing_pair_filter_sql(v6_path, target)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v10_new_{target.lower()} AS
        SELECT
            n.source1_entity_id,
            n.candidate_entity_id,
            n.block_name,
            n.evidence_rows,
            n.evidence_file_count,
            n.exact_key_count
        FROM (
            {shard_union}
        ) n
        LEFT JOIN (
            {existing}
        ) e
          ON n.source1_entity_id = e.source1_entity_id
         AND n.candidate_entity_id = e.candidate_entity_id
        WHERE e.source1_entity_id IS NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY n.source1_entity_id, n.candidate_entity_id
            ORDER BY n.candidate_entity_id
        ) = 1
        """
    )

    return int(
        con.execute(
            f"SELECT COUNT(*) FROM v10_new_{target.lower()}"
        ).fetchone()[0]
    )


def create_new_feature_view(
    con: duckdb.DuckDBPyConnection,
    *,
    target: str,
    target_path: Path,
) -> None:
    """
    Join V9.1-only candidate pairs to source records and compute
    the same deterministic non-fuzzy features used by V6.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v10_s1 AS
        SELECT
            entity_id,
            COALESCE(name_norm, '') AS name_norm,
            COALESCE(address_norm, '') AS address_norm,
            COALESCE(country_norm, '') AS country_norm
        FROM read_parquet({qp(S1_PATH)})
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v10_target AS
        SELECT
            entity_id,
            COALESCE(name_norm, '') AS name_norm,
            COALESCE(address_norm, '') AS address_norm,
            COALESCE(country_norm, '') AS country_norm
        FROM read_parquet({qp(target_path)})
        """
    )

    new_table = f"v10_new_{target.lower()}"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v10_new_features_{target.lower()} AS
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.evidence_rows,
            c.evidence_file_count,
            c.exact_key_count,
            c.block_name,

            l.name_norm AS left_name,
            r.name_norm AS right_name,
            l.address_norm AS left_address,
            r.address_norm AS right_address,

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

        FROM {new_table} c
        INNER JOIN v10_s1 l
          ON c.source1_entity_id = l.entity_id
        INNER JOIN v10_target r
          ON c.candidate_entity_id = r.entity_id
        """
    )


def score_new_batches(
    con: duckdb.DuckDBPyConnection,
    *,
    target: str,
) -> tuple[Path, int]:
    """
    Score only V9.1-new candidates with RapidFuzz, in batches.
    Each output row is compatible with V6/V8 feature naming.
    """
    parts_dir = NEW_PARTS / target.lower()
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    query = f"""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            evidence_rows,
            evidence_file_count,
            exact_key_count,
            block_name,
            left_name,
            right_name,
            left_address,
            right_address,
            name_exact,
            address_exact,
            country_exact,
            name_length_ratio,
            address_length_ratio
        FROM v10_new_features_{target.lower()}
    """

    reader = con.execute(query).to_arrow_reader(batch_size=BATCH)

    total = 0
    part_index = 0

    for arrow_batch in reader:
        df = arrow_batch.to_pandas()
        if df.empty:
            continue

        names_left = df["left_name"].fillna("").astype(str).tolist()
        names_right = df["right_name"].fillna("").astype(str).tolist()
        addr_left = df["left_address"].fillna("").astype(str).tolist()
        addr_right = df["right_address"].fillna("").astype(str).tolist()

        df["name_similarity"] = np.asarray(
            [ratio(a, b) / 100.0 if a and b else 0.0
             for a, b in zip(names_left, names_right)],
            dtype="float32",
        )
        df["address_similarity"] = np.asarray(
            [ratio(a, b) / 100.0 if a and b else 0.0
             for a, b in zip(addr_left, addr_right)],
            dtype="float32",
        )

        df["base_score"] = base_score_formula(df).astype("float32")
        df["base_rank"] = np.int64(0)

        out_path = parts_dir / f"part_{part_index:05d}.parquet"

        cols = [
            "source1_entity_id",
            "candidate_entity_id",
            "base_score",
            "base_rank",
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
            "block_name",
        ]

        df[cols].to_parquet(
            out_path,
            index=False,
            compression="zstd",
        )

        total += len(df)
        part_index += 1

        if total and total % 1_000_000 < len(df):
            print(f"  {target} V9.1-new scored: {total:,}", flush=True)

    if total == 0:
        raise RuntimeError(f"No V9.1-new candidates remained for S1 -> {target}")

    return parts_dir, total


def build_combined_scored(
    con: duckdb.DuckDBPyConnection,
    *,
    target: str,
    v6_path: Path,
    new_parts_dir: Path,
) -> Path:
    """
    Build one expanded candidate Parquet.
    Existing V6 rows keep their V6 transparent score.
    V9.1 rows use the same score formula.
    """
    out_path = OUT / f"train_combined_base_s1_{target.lower()}.parquet"
    out_path.unlink(missing_ok=True)

    new_glob = qp(new_parts_dir / "*.parquet")

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                CAST(score AS DOUBLE) AS base_score,
                CAST(rank AS BIGINT) AS base_rank,
                CAST(name_similarity AS DOUBLE) AS name_similarity,
                CAST(address_similarity AS DOUBLE) AS address_similarity,
                CAST(name_exact AS INTEGER) AS name_exact,
                CAST(address_exact AS INTEGER) AS address_exact,
                CAST(country_exact AS INTEGER) AS country_exact,
                CAST(name_length_ratio AS DOUBLE) AS name_length_ratio,
                CAST(address_length_ratio AS DOUBLE) AS address_length_ratio,
                CAST(evidence_rows AS DOUBLE) AS evidence_rows,
                CAST(evidence_file_count AS DOUBLE) AS evidence_file_count,
                CAST(exact_key_count AS DOUBLE) AS exact_key_count,
                COALESCE(CAST(block_names AS VARCHAR), '') AS block_name
            FROM read_parquet({qp(v6_path)})

            UNION ALL

            SELECT
                source1_entity_id,
                candidate_entity_id,
                CAST(base_score AS DOUBLE) AS base_score,
                CAST(base_rank AS BIGINT) AS base_rank,
                CAST(name_similarity AS DOUBLE) AS name_similarity,
                CAST(address_similarity AS DOUBLE) AS address_similarity,
                CAST(name_exact AS INTEGER) AS name_exact,
                CAST(address_exact AS INTEGER) AS address_exact,
                CAST(country_exact AS INTEGER) AS country_exact,
                CAST(name_length_ratio AS DOUBLE) AS name_length_ratio,
                CAST(address_length_ratio AS DOUBLE) AS address_length_ratio,
                CAST(evidence_rows AS DOUBLE) AS evidence_rows,
                CAST(evidence_file_count AS DOUBLE) AS evidence_file_count,
                CAST(exact_key_count AS DOUBLE) AS exact_key_count,
                COALESCE(CAST(block_name AS VARCHAR), '') AS block_name
            FROM read_parquet({new_glob})
        )
        TO {qp(out_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    return out_path


def build_labeled_ranked_base(
    con: duckdb.DuckDBPyConnection,
    *,
    target: str,
    combined_path: Path,
) -> tuple[Path, str]:
    """
    Re-rank the expanded pool by transparent base_score.
    This creates the training/evaluation labels.
    """
    gt_table = f"gt_{target.lower()}"
    labeled_path = OUT / f"train_labeled_base_s1_{target.lower()}.parquet"
    labeled_path.unlink(missing_ok=True)

    validation_expr = (
        f"CASE WHEN MOD(ABS(HASH(c.source1_entity_id)), {VALIDATION_MOD}) = 0 "
        f"THEN 1 ELSE 0 END"
    )

    con.execute(
        f"""
        COPY (
            SELECT
                c.source1_entity_id,
                c.candidate_entity_id,
                c.base_score,
                ROW_NUMBER() OVER (
                    PARTITION BY c.source1_entity_id
                    ORDER BY c.base_score DESC, c.candidate_entity_id
                ) AS base_rank,
                c.name_similarity,
                c.address_similarity,
                c.name_exact,
                c.address_exact,
                c.country_exact,
                c.name_length_ratio,
                c.address_length_ratio,
                c.evidence_rows,
                c.evidence_file_count,
                c.exact_key_count,
                c.block_name,
                CASE WHEN g.matched_entity_id IS NULL THEN 0 ELSE 1 END AS label,
                {validation_expr} AS is_validation
            FROM read_parquet({qp(combined_path)}) c
            LEFT JOIN {gt_table} g
              ON c.source1_entity_id = g.source1_entity_id
             AND c.candidate_entity_id = g.matched_entity_id
        )
        TO {qp(labeled_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    row_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({qp(labeled_path)})"
        ).fetchone()[0]
    )

    return labeled_path, int(row_count)


def pair_training_sql(labeled_path: Path, validation_flag: int | None) -> str:
    cond = "" if validation_flag is None else f"AND is_validation = {validation_flag}"

    return f"""
        WITH
        positives AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY base_rank, candidate_entity_id
                    ) AS p_rank
                FROM read_parquet({qp(labeled_path)})
                WHERE label = 1
                  {cond}
            )
            WHERE p_rank <= {PAIR_POS_PER_S1}
        ),
        negatives AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY base_rank, candidate_entity_id
                    ) AS n_rank
                FROM read_parquet({qp(labeled_path)})
                WHERE label = 0
                  AND base_rank <= {PAIR_NEG_RANK}
                  {cond}
            )
            WHERE n_rank <= {PAIR_NEG_PER_S1}
        ),
        joined AS (
            SELECT
                p.name_similarity AS pn,
                n.name_similarity AS nn,
                p.address_similarity AS pa,
                n.address_similarity AS na,
                p.name_exact AS pe,
                n.name_exact AS ne,
                p.address_exact AS pae,
                n.address_exact AS nae,
                p.country_exact AS pc,
                n.country_exact AS nc,
                p.name_length_ratio AS pl,
                n.name_length_ratio AS nl,
                p.address_length_ratio AS pal,
                n.address_length_ratio AS nal,
                p.evidence_rows AS per,
                n.evidence_rows AS ner,
                p.evidence_file_count AS pef,
                n.evidence_file_count AS nef,
                p.exact_key_count AS pek,
                n.exact_key_count AS nek,
                p.base_score AS ps,
                n.base_score AS ns,
                p.base_rank AS pr,
                n.base_rank AS nr
            FROM positives p
            INNER JOIN negatives n
              ON p.source1_entity_id = n.source1_entity_id
        ),
        f AS (
            SELECT
                pn - nn AS name_similarity,
                pa - na AS address_similarity,
                pe - ne AS name_exact,
                pae - nae AS address_exact,
                pc - nc AS country_exact,
                pl - nl AS name_length_ratio,
                pal - nal AS address_length_ratio,
                per - ner AS evidence_rows,
                pef - nef AS evidence_file_count,
                pek - nek AS exact_key_count,
                ps - ns AS base_score,
                LN(1 + pr) - LN(1 + nr) AS log_base_rank,
                pn * pa - nn * na AS name_x_address,
                (pn - pa) - (nn - na) AS name_minus_address,
                ((pn + pa) / 2) - ((nn + na) / 2) AS similarity_mean,
                LEAST(pn, pa) - LEAST(nn, na) AS similarity_min,
                (pe + pae + pc) - (ne + nae + nc) AS exact_field_count,
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


def collect_training(
    con: duckdb.DuckDBPyConnection,
    *,
    labeled_path: Path,
    target: str,
    validation_flag: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    q = pair_training_sql(labeled_path, validation_flag)
    reader = con.execute(q).to_arrow_reader(batch_size=BATCH)

    frames: list[pd.DataFrame] = []
    total = 0

    for b in reader:
        d = b.to_pandas()
        if d.empty:
            continue
        frames.append(d)
        total += len(d)

    if not frames:
        raise RuntimeError(
            f"No training pairs produced for S1 -> {target}; "
            f"validation={validation_flag}"
        )

    df = pd.concat(frames, ignore_index=True)

    X = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
    y = df["pair_label"].astype("int8").to_numpy()

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    print(
        f"{target} pair training "
        f"{'HOLDOUT' if validation_flag == 1 else 'FULL'}: "
        f"{len(df):,} rows; "
        f"positive={positives:,}; negative={negatives:,}",
        flush=True,
    )

    return X, y


def fit_pairwise_model(X: np.ndarray, y: np.ndarray, target: str, mode: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

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
    print(
        f"{target} {mode} model fit: {time.time() - t0:.1f}s",
        flush=True,
    )

    return scaler, model


def score_and_rank(
    con: duckdb.DuckDBPyConnection,
    *,
    labeled_path: Path,
    target: str,
    scaler,
    model,
    validation_only: bool,
    output_path: Path,
) -> None:
    """
    Score numeric features in batches, then globally rank.
    No fuzzy strings are loaded here.
    """
    mode = "validation" if validation_only else "full"

    parts_dir = TMP / f"v10_model_parts_{target.lower()}_{mode}"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    where = "WHERE is_validation = 1" if validation_only else ""

    reader = con.execute(
        f"""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            base_rank,
            name_similarity,
            address_similarity,
            name_exact,
            address_exact,
            country_exact,
            name_length_ratio,
            address_length_ratio,
            evidence_rows,
            evidence_file_count,
            exact_key_count,
            base_score
        FROM read_parquet({qp(labeled_path)})
        {where}
        """
    ).to_arrow_reader(batch_size=BATCH)

    part_paths: list[Path] = []
    total = 0
    idx = 0

    for b in reader:
        d = b.to_pandas()
        if d.empty:
            continue

        x = pd.DataFrame(index=d.index)

        numeric_cols = [
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

        for col in numeric_cols:
            x[col] = pd.to_numeric(
                d[col], errors="coerce"
            ).fillna(0.0).astype("float32")

        rank = (
            pd.to_numeric(
                d["base_rank"], errors="coerce"
            )
            .fillna(1_000_000.0)
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
            (x["name_similarity"] + x["address_similarity"]) / 2.0
        ).astype("float32")
        x["similarity_min"] = np.minimum(
            x["name_similarity"],
            x["address_similarity"],
        ).astype("float32")
        x["exact_field_count"] = (
            x["name_exact"]
            + x["address_exact"]
            + x["country_exact"]
        ).astype("float32")

        X = x[FEATURES].to_numpy(dtype="float32", copy=False)
        Xs = scaler.transform(X).astype("float32")

        d["v10_utility"] = model.decision_function(Xs)

        out = parts_dir / f"part_{idx:05d}.parquet"
        d[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "v10_utility",
            ]
        ].to_parquet(
            out,
            index=False,
            compression="zstd",
        )

        idx += 1
        total += len(d)

        if total and total % 1_000_000 < len(d):
            print(
                f"{target} {mode} V10 scored: {total:,}",
                flush=True,
            )

    if not part_paths and idx == 0:
        raise RuntimeError(
            f"No rows scored for S1 -> {target}, validation={validation_only}"
        )

    part_paths = sorted(parts_dir.glob("*.parquet"))

    rank_glob = qp(parts_dir / "*.parquet")
    output_path.unlink(missing_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                v10_utility,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY v10_utility DESC, candidate_entity_id
                ) AS v10_rank
            FROM read_parquet({rank_glob})
        )
        TO {qp(output_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    shutil.rmtree(parts_dir, ignore_errors=True)


def evaluate_recall(
    con: duckdb.DuckDBPyConnection,
    *,
    ranked_path: Path,
    gt_table: str,
    target: str,
    validation_only: bool,
) -> dict:
    filt = (
        f"AND MOD(ABS(HASH(g.source1_entity_id)), {VALIDATION_MOD}) = 0"
        if validation_only
        else ""
    )

    total = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt_table} g
            WHERE 1=1
            {"AND MOD(ABS(HASH(g.source1_entity_id)), " + str(VALIDATION_MOD) + ") = 0" if validation_only else ""}
            """
        ).fetchone()[0]
    )

    result = {
        "ground_truth_pairs": total,
        "validation": validation_only,
    }

    for k in [1, 3, 5, 10, 20]:
        covered = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        g.source1_entity_id,
                        g.matched_entity_id
                    FROM {gt_table} g
                    INNER JOIN read_parquet({qp(ranked_path)}) r
                      ON g.source1_entity_id = r.source1_entity_id
                     AND g.matched_entity_id = r.candidate_entity_id
                    WHERE r.v10_rank <= {k}
                      {filt}
                )
                """
            ).fetchone()[0]
        )

        pct = 100.0 * covered / total if total else 0.0
        result[f"recall_at_{k}"] = pct
        result[f"covered_at_{k}"] = covered

        print(
            f"{target} {'HOLDOUT' if validation_only else 'FULL'} "
            f"Recall@{k}: {pct:.2f}% ({covered:,}/{total:,})",
            flush=True,
        )

    return result


def process_target(
    con: duckdb.DuckDBPyConnection,
    *,
    target: str,
    v6_path: Path,
    v9_dir: Path,
    target_path: Path,
) -> dict:
    header(f"V10 PROCESSING: S1 -> {target}")

    require_file(v6_path)
    require_dir(v9_dir)

    gt_table = f"gt_{target.lower()}"

    gt_total = int(
        con.execute(f"SELECT COUNT(*) FROM {gt_table}").fetchone()[0]
    )

    v6_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({qp(v6_path)})"
        ).fetchone()[0]
    )

    print(f"V6 candidate rows      : {v6_count:,}")
    print(f"Ground-truth pairs     : {gt_total:,}")

    before_covered = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT g.source1_entity_id, g.matched_entity_id
                FROM {gt_table} g
                INNER JOIN read_parquet({qp(v6_path)}) v
                  ON g.source1_entity_id = v.source1_entity_id
                 AND g.matched_entity_id = v.candidate_entity_id
            )
            """
        ).fetchone()[0]
    )

    print(
        f"V6 coverage            : "
        f"{100.0 * before_covered / gt_total:.2f}%",
        flush=True,
    )

    new_count = create_v9_new_table(
        con,
        target=target,
        v6_path=v6_path,
        v9_dir=v9_dir,
    )

    print(f"V9.1-new pairs         : {new_count:,}", flush=True)

    if new_count == 0:
        raise RuntimeError(f"No V9.1-new pairs for S1 -> {target}")

    create_new_feature_view(
        con,
        target=target,
        target_path=target_path,
    )

    new_parts_dir, new_scored_count = score_new_batches(
        con,
        target=target,
    )

    print(
        f"V9.1-new scored        : {new_scored_count:,}",
        flush=True,
    )

    combined_path = build_combined_scored(
        con,
        target=target,
        v6_path=v6_path,
        new_parts_dir=new_parts_dir,
    )

    combined_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({qp(combined_path)})"
        ).fetchone()[0]
    )

    print(
        f"Combined base pool     : {combined_count:,}",
        flush=True,
    )

    labeled_path, labeled_count = build_labeled_ranked_base(
        con,
        target=target,
        combined_path=combined_path,
    )

    print(
        f"Labeled base rows      : {labeled_count:,}",
        flush=True,
    )

    # Holdout model/evaluation.
    X_val, y_val = collect_training(
        con,
        labeled_path=labeled_path,
        target=target,
        validation_flag=1,
    )

    scaler_val, model_val = fit_pairwise_model(
        X_val, y_val, target, "HOLDOUT"
    )

    val_ranked = OUT / f"v10_validation_ranked_s1_{target.lower()}.parquet"

    score_and_rank(
        con,
        labeled_path=labeled_path,
        target=target,
        scaler=scaler_val,
        model=model_val,
        validation_only=True,
        output_path=val_ranked,
    )

    val_metrics = evaluate_recall(
        con,
        ranked_path=val_ranked,
        gt_table=gt_table,
        target=target,
        validation_only=True,
    )

    del X_val, y_val, scaler_val, model_val

    # Full model.
    X_full, y_full = collect_training(
        con,
        labeled_path=labeled_path,
        target=target,
        validation_flag=None,
    )

    scaler_full, model_full = fit_pairwise_model(
        X_full, y_full, target, "FULL"
    )

    ranked_full = OUT / f"train_ranked_v10_s1_{target.lower()}.parquet"

    score_and_rank(
        con,
        labeled_path=labeled_path,
        target=target,
        scaler=scaler_full,
        model=model_full,
        validation_only=False,
        output_path=ranked_full,
    )

    top20 = OUT / f"train_ranked_v10_top20_s1_{target.lower()}.parquet"
    top20.unlink(missing_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet({qp(ranked_full)})
            WHERE v10_rank <= {TOPK}
        )
        TO {qp(top20)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    full_metrics = evaluate_recall(
        con,
        ranked_path=ranked_full,
        gt_table=gt_table,
        target=target,
        validation_only=False,
    )

    return {
        "target": target,
        "ground_truth_pairs": gt_total,
        "v6_candidate_rows": v6_count,
        "v6_covered_truth": before_covered,
        "v6_coverage_pct": 100.0 * before_covered / gt_total,
        "v9_1_new_pairs": new_count,
        "v9_1_new_scored": new_scored_count,
        "combined_base_rows": combined_count,
        "labeled_rows": int(labeled_count),
        "validation": val_metrics,
        "full_train": full_metrics,
        "combined_base_output": str(combined_path),
        "labeled_base_output": str(labeled_path),
        "full_ranked_output": str(ranked_full),
        "top20_output": str(top20),
    }


def main() -> None:
    header("AMAZON ML CHALLENGE - V10 FULL EXPANDED PIPELINE")

    print(f"Project       : {PROJECT}")
    print(f"Output        : {OUT}")
    print(f"Memory limit  : {MEMORY_LIMIT}")
    print(f"Threads       : {THREADS}")
    print(f"Batch size    : {BATCH:,}")
    print(f"V6 S2         : {V6_S2}")
    print(f"V6 S3         : {V6_S3}")
    print(f"V9.1 S2       : {V9_1_S2_DIR}")
    print(f"V9.1 S3       : {V9_1_S3_DIR}")

    for p in [
        S1_PATH,
        S2_PATH,
        S3_PATH,
        GT_PATH,
        V6_S2,
        V6_S3,
    ]:
        require_file(p)

    require_dir(V9_1_S2_DIR)
    require_dir(V9_1_S3_DIR)

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    con = setup_connection()
    load_ground_truth(con)

    start = time.time()
    results = {}

    try:
        results["s2"] = process_target(
            con,
            target="S2",
            v6_path=V6_S2,
            v9_dir=V9_1_S2_DIR,
            target_path=S2_PATH,
        )

        # Release large S2 temp tables before S3.
        con.execute("DROP TABLE IF EXISTS v10_new_s2")
        con.execute("DROP TABLE IF EXISTS v10_new_features_s2")
        con.execute("DROP TABLE IF EXISTS v10_s1")
        con.execute("DROP TABLE IF EXISTS v10_target")

        results["s3"] = process_target(
            con,
            target="S3",
            v6_path=V6_S3,
            v9_dir=V9_1_S3_DIR,
            target_path=S3_PATH,
        )

    finally:
        con.close()

    summary = {
        "version": "V10",
        "method": "V6_transparent_score_plus_V9.1_candidates_then_pairwise_logistic",
        "memory_limit": MEMORY_LIMIT,
        "threads": THREADS,
        "batch_size": BATCH,
        "validation_mod": VALIDATION_MOD,
        "results": results,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    summary_path = OUT / "v10_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    header("V10 FINAL SUMMARY")

    for key in ("s2", "s3"):
        r = results[key]
        print(
            f"S1 -> {key.upper()} | "
            f"V6 coverage={r['v6_coverage_pct']:.2f}% | "
            f"V9.1-new={r['v9_1_new_pairs']:,} | "
            f"FULL Recall@20={r['full_train']['recall_at_20']:.2f}%",
            flush=True,
        )

    print(f"Summary: {summary_path}")
    header("✅ V10 COMPLETED")


if __name__ == "__main__":
    main()
