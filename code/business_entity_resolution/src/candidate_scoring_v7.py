from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# AMAZON ML CHALLENGE - V7 LEARNED RANKING
# ============================================================
#
# V6 already scores the complete V1+V2+V4+V5 candidate pools.
# V7 learns the ranking function from TRAIN ground truth instead
# of relying only on fixed hand-written weights.
#
# Validation protocol:
#   Source1 entities with
#       ABS(HASH(source1_entity_id)) % 5 = 0
#   are held out for validation.
#
# Training:
#   - positives = candidate pairs that are in ground truth
#   - hard negatives = high-ranked V6 non-matches
#   - deterministic sampling keeps RAM reasonable
#
# Model:
#   sklearn HistGradientBoostingClassifier
#
# Outputs:
#   candidate_output_v7/
#       train_ranked_v7_s1_s2.parquet
#       train_ranked_v7_s1_s3.parquet
#       train_ranked_v7_top20_s1_s2.parquet
#       train_ranked_v7_top20_s1_s3.parquet
#       v7_summary.json
#
# V1-V6 files are read-only.
# ============================================================


PROJECT = Path("/Users/harikeshshukla/mla")

TRAIN_DIR = PROJECT / "processed_dataset" / "train"
GT_PATH = TRAIN_DIR / "ground_truth_pairs.parquet"

V6_DIR = PROJECT / "candidate_output_v6"
OUT_DIR = PROJECT / "candidate_output_v7"
TMP_DIR = OUT_DIR / "tmp"

S2_V6 = V6_DIR / "train_scored_s1_s2.parquet"
S3_V6 = V6_DIR / "train_scored_s1_s3.parquet"

MEMORY_LIMIT = "4GB"
THREADS = 2
BATCH_SIZE = 100_000

TOP_K = 20
VALIDATION_MOD = 5

# Hard negatives used for model fitting.
NEGATIVE_RANK_LIMIT = 100
NEGATIVES_PER_S1 = 5

MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def qpath(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def require_sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is missing.\n"
            "Install it with:\n"
            "python -m pip install scikit-learn"
        ) from exc
    return HistGradientBoostingClassifier


def setup_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def setup_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")

    temp_dir = TMP_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_sql = str(temp_dir).replace(
        chr(92), "/"
    ).replace(
        chr(39), chr(39) * 2
    )

    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA temp_directory='{temp_sql}'")

    return con


def load_ground_truth(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_s2 AS
        SELECT
            source1_entity_id,
            matched_entity_id
        FROM read_parquet({qpath(GT_PATH)})
        WHERE
            COALESCE(label, 1) = 1
            AND starts_with(matched_entity_id, 'S2-')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_s3 AS
        SELECT
            source1_entity_id,
            matched_entity_id
        FROM read_parquet({qpath(GT_PATH)})
        WHERE
            COALESCE(label, 1) = 1
            AND starts_with(matched_entity_id, 'S3-')
        """
    )


def make_labeled_table(
    con: duckdb.DuckDBPyConnection,
    scored_path: Path,
    gt_table: str,
    target_name: str,
) -> str:
    table = f"v7_labeled_{target_name.lower()}"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT
            s.source1_entity_id,
            s.candidate_entity_id,
            CAST(s.rank AS BIGINT) AS v6_rank,
            CAST(s.score AS DOUBLE) AS v6_score,

            CAST(s.name_similarity AS DOUBLE) AS name_similarity,
            CAST(s.address_similarity AS DOUBLE) AS address_similarity,

            CAST(s.name_exact AS INTEGER) AS name_exact,
            CAST(s.address_exact AS INTEGER) AS address_exact,
            CAST(s.country_exact AS INTEGER) AS country_exact,

            CAST(s.name_length_ratio AS DOUBLE) AS name_length_ratio,
            CAST(s.address_length_ratio AS DOUBLE) AS address_length_ratio,

            CAST(s.evidence_rows AS DOUBLE) AS evidence_rows,
            CAST(s.evidence_file_count AS DOUBLE) AS evidence_file_count,
            CAST(s.exact_key_count AS DOUBLE) AS exact_key_count,

            CASE
                WHEN g.matched_entity_id IS NULL THEN 0
                ELSE 1
            END AS label,

            CASE
                WHEN MOD(
                    ABS(HASH(s.source1_entity_id)),
                    {VALIDATION_MOD}
                ) = 0 THEN 1
                ELSE 0
            END AS is_validation

        FROM read_parquet({qpath(scored_path)}) s

        LEFT JOIN {gt_table} g
            ON s.source1_entity_id = g.source1_entity_id
            AND s.candidate_entity_id = g.matched_entity_id
        """
    )

    return table


def collect_training_data(
    con: duckdb.DuckDBPyConnection,
    labeled_table: str,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Training split:
      all candidate-covered positives
      + deterministic high-ranked hard negatives.
    """
    header(f"V7 BUILD TRAINING DATA: S1 → {target_name}")

    query = f"""
    WITH ranked_negatives AS (
        SELECT
            t.*,
            ROW_NUMBER() OVER (
                PARTITION BY t.source1_entity_id
                ORDER BY t.v6_rank, t.candidate_entity_id
            ) AS neg_position
        FROM {labeled_table} t
        WHERE
            t.label = 0
            AND t.is_validation = 0
            AND t.v6_rank <= {NEGATIVE_RANK_LIMIT}
    ),

    selected_negatives AS (
        SELECT *
        FROM ranked_negatives
        WHERE
            neg_position <= {NEGATIVES_PER_S1}
            AND MOD(
                ABS(HASH(candidate_entity_id)),
                2
            ) = 0
    ),

    positives AS (
        SELECT *
        FROM {labeled_table}
        WHERE
            label = 1
            AND is_validation = 0
    )

    SELECT
        source1_entity_id,
        candidate_entity_id,
        v6_rank,
        v6_score,
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
        label

    FROM positives

    UNION ALL

    SELECT
        source1_entity_id,
        candidate_entity_id,
        v6_rank,
        v6_score,
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
        label

    FROM selected_negatives
    """

    reader = con.execute(query).to_arrow_reader(
        batch_size=BATCH_SIZE
    )

    frames: list[pd.DataFrame] = []
    total = 0

    for batch in reader:
        df = batch.to_pandas()

        if df.empty:
            continue

        frames.append(df)
        total += len(df)

        if total and total % 1_000_000 < len(df):
            print(f"Collected: {total:,}")

    if not frames:
        raise RuntimeError(f"No training data generated for {target_name}.")

    df = pd.concat(frames, ignore_index=True)

    x = make_features(df).astype("float32").to_numpy(copy=False)
    y = df["label"].astype("int8").to_numpy()

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    print(f"Training rows : {len(df):,}")
    print(f"Positive rows : {positives:,}")
    print(f"Negative rows : {negatives:,}")

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            f"Training split for {target_name} does not contain both classes."
        )

    return x, y


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric feature engineering.

    The model can learn non-linear interactions between:
      name similarity
      address similarity
      exactness
      country
      evidence
      V6 score/rank
    """
    x = pd.DataFrame(index=df.index)

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
        "v6_score",
    ]

    for col in numeric_cols:
        x[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0.0)

    rank = pd.to_numeric(
        df["v6_rank"],
        errors="coerce"
    ).fillna(1_000_000.0)

    x["log_v6_rank"] = np.log1p(
        rank.astype("float64")
    )

    x["name_minus_address"] = (
        x["name_similarity"] -
        x["address_similarity"]
    )

    x["name_x_address"] = (
        x["name_similarity"] *
        x["address_similarity"]
    )

    x["exact_field_count"] = (
        x["name_exact"] +
        x["address_exact"] +
        x["country_exact"]
    )

    x["similarity_mean"] = (
        x["name_similarity"] +
        x["address_similarity"]
    ) / 2.0

    x["similarity_min"] = np.minimum(
        x["name_similarity"],
        x["address_similarity"],
    )

    return x


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    target_name: str,
):
    HistGradientBoostingClassifier = require_sklearn()

    header(f"V7 MODEL FIT: S1 → {target_name}")

    model = HistGradientBoostingClassifier(
        **MODEL_PARAMS
    )

    # Balance the loss between positive and negative examples.
    pos_count = float(np.sum(y == 1))
    neg_count = float(np.sum(y == 0))

    pos_weight = (
        neg_count / pos_count
        if pos_count > 0
        else 1.0
    )

    sample_weight = np.where(
        y == 1,
        min(pos_weight, 20.0),
        1.0,
    ).astype("float32")

    print(f"Positive weight: {min(pos_weight, 20.0):.2f}")

    start = time.time()

    model.fit(
        x,
        y,
        sample_weight=sample_weight,
    )

    elapsed = time.time() - start
    print(f"Fit time: {elapsed:.1f}s")

    return model


def score_and_rank(
    con: duckdb.DuckDBPyConnection,
    labeled_table: str,
    model,
    target_name: str,
    output_path: Path,
    validation_only: bool,
) -> None:
    """
    Score a candidate table in batches, write temporary pieces,
    then globally rank with DuckDB.
    """
    mode = "validation" if validation_only else "full"

    header(
        f"V7 SCORING: S1 → {target_name} [{mode.upper()}]"
    )

    where_clause = (
        "WHERE is_validation = 1"
        if validation_only
        else ""
    )

    query = f"""
    SELECT
        source1_entity_id,
        candidate_entity_id,

        v6_rank,
        v6_score,

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

        label

    FROM {labeled_table}
    {where_clause}
    """

    parts_dir = (
        TMP_DIR
        / "score_parts"
        / target_name.lower()
        / mode
    )
    parts_dir.mkdir(parents=True, exist_ok=True)

    for p in parts_dir.glob("*.parquet"):
        p.unlink()

    reader = con.execute(query).to_arrow_reader(
        batch_size=BATCH_SIZE
    )

    part_paths: list[Path] = []
    total = 0
    index = 0

    for batch in reader:
        df = batch.to_pandas()

        if df.empty:
            continue

        features = make_features(df)

        probability = model.predict_proba(
            features.to_numpy(
                dtype="float32",
                copy=False,
            )
        )[:, 1]

        df["v7_probability"] = probability

        part = parts_dir / f"part_{index:05d}.parquet"

        df.to_parquet(
            part,
            index=False,
            compression="zstd",
        )

        part_paths.append(part)
        total += len(df)
        index += 1

        if total and total % 1_000_000 < len(df):
            print(f"Scored: {total:,}")

    if not part_paths:
        raise RuntimeError(
            f"No scoring rows generated for {target_name} [{mode}]."
        )

    glob = str(parts_dir / "*.parquet").replace(
        chr(92), "/"
    ).replace(
        chr(39), chr(39) * 2
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,

                v7_probability,

                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY
                        v7_probability DESC,
                        candidate_entity_id
                ) AS rank,

                v6_rank,
                v6_score,

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

                label

            FROM read_parquet('{glob}')
        )
        TO {qpath(output_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    print(f"Ranked rows: {total:,}")

    for p in part_paths:
        p.unlink(missing_ok=True)

    try:
        parts_dir.rmdir()
    except OSError:
        pass


def evaluate_recall(
    con: duckdb.DuckDBPyConnection,
    ranked_path: Path,
    gt_table: str,
    target_name: str,
    validation_only: bool,
) -> dict:
    mode = "validation" if validation_only else "full"

    header(
        f"V7 RECALL: S1 → {target_name} [{mode.upper()}]"
    )

    gt_filter = (
        f"""
        WHERE MOD(
            ABS(HASH(source1_entity_id)),
            {VALIDATION_MOD}
        ) = 0
        """
        if validation_only
        else ""
    )

    gt_total = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt_table}
            {gt_filter}
            """
        ).fetchone()[0]
    )

    result = {
        "target": target_name,
        "mode": mode,
        "ground_truth_pairs": gt_total,
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

                    INNER JOIN read_parquet({qpath(ranked_path)}) r
                        ON
                            g.source1_entity_id =
                                r.source1_entity_id
                            AND
                            g.matched_entity_id =
                                r.candidate_entity_id

                    WHERE
                        r.rank <= {k}

                        {
                            f"AND MOD(ABS(HASH(g.source1_entity_id)), {VALIDATION_MOD}) = 0"
                            if validation_only
                            else ""
                        }
                )
                """
            ).fetchone()[0]
        )

        pct = (
            100.0 * covered / gt_total
            if gt_total
            else 0.0
        )

        result[f"covered_at_{k}"] = covered
        result[f"recall_at_{k}"] = pct

        print(
            f"Recall@{k:<2}: "
            f"{pct:.2f}% "
            f"({covered:,}/{gt_total:,})"
        )

    return result


def top20_file(
    con: duckdb.DuckDBPyConnection,
    ranked_path: Path,
    target_name: str,
) -> Path:
    path = (
        OUT_DIR
        / f"train_ranked_v7_top20_s1_{target_name.lower()}.parquet"
    )

    path.unlink(missing_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet({qpath(ranked_path)})
            WHERE rank <= {TOP_K}
        )
        TO {qpath(path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    return path


def process_direction(
    con: duckdb.DuckDBPyConnection,
    *,
    target_name: str,
    scored_path: Path,
    gt_table: str,
) -> dict:
    labeled = make_labeled_table(
        con,
        scored_path,
        gt_table,
        target_name,
    )

    # --------------------------------------------------------
    # 1) TRAIN MODEL ON 80% OF SOURCE1 ENTITIES
    # --------------------------------------------------------
    x_train, y_train = collect_training_data(
        con,
        labeled,
        target_name,
    )

    model = fit_model(
        x_train,
        y_train,
        target_name,
    )

    # --------------------------------------------------------
    # 2) HOLDOUT VALIDATION
    # --------------------------------------------------------
    val_path = (
        TMP_DIR
        / f"validation_v7_s1_{target_name.lower()}.parquet"
    )

    score_and_rank(
        con,
        labeled,
        model,
        target_name,
        val_path,
        validation_only=True,
    )

    validation_metrics = evaluate_recall(
        con,
        val_path,
        gt_table,
        target_name,
        validation_only=True,
    )

    # --------------------------------------------------------
    # 3) REFIT ON BOTH TRAINING + VALIDATION EXAMPLES
    # --------------------------------------------------------
    header(
        f"V7 FULL-TRAIN REFIT: S1 → {target_name}"
    )

    # Build the same hard-negative strategy over all Source1 IDs.
    # This gives the final model access to all available TRAIN
    # labels without touching the test dataset.
    full_query = f"""
    WITH ranked_negatives AS (
        SELECT
            t.*,
            ROW_NUMBER() OVER (
                PARTITION BY t.source1_entity_id
                ORDER BY t.v6_rank, t.candidate_entity_id
            ) AS neg_position
        FROM {labeled} t
        WHERE
            t.label = 0
            AND t.v6_rank <= {NEGATIVE_RANK_LIMIT}
    ),

    selected_negatives AS (
        SELECT *
        FROM ranked_negatives
        WHERE
            neg_position <= {NEGATIVES_PER_S1}
            AND MOD(
                ABS(HASH(candidate_entity_id)),
                2
            ) = 0
    ),

    positives AS (
        SELECT *
        FROM {labeled}
        WHERE label = 1
    )

    SELECT
        source1_entity_id,
        candidate_entity_id,
        v6_rank,
        v6_score,
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
        label
    FROM positives

    UNION ALL

    SELECT
        source1_entity_id,
        candidate_entity_id,
        v6_rank,
        v6_score,
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
        label
    FROM selected_negatives
    """

    reader = con.execute(full_query).to_arrow_reader(
        batch_size=BATCH_SIZE
    )

    full_frames: list[pd.DataFrame] = []
    full_rows = 0

    for batch in reader:
        df = batch.to_pandas()

        if df.empty:
            continue

        full_frames.append(df)
        full_rows += len(df)

    full_df = pd.concat(
        full_frames,
        ignore_index=True,
    )

    x_full = make_features(full_df).astype(
        "float32"
    ).to_numpy(copy=False)

    y_full = full_df["label"].astype(
        "int8"
    ).to_numpy()

    full_model = fit_model(
        x_full,
        y_full,
        target_name + " FULL",
    )

    del full_frames
    del full_df
    del x_full
    del y_full

    # --------------------------------------------------------
    # 4) RANK COMPLETE TRAIN CANDIDATE POOL
    # --------------------------------------------------------
    full_path = (
        OUT_DIR
        / f"train_ranked_v7_s1_{target_name.lower()}.parquet"
    )

    score_and_rank(
        con,
        labeled,
        full_model,
        target_name,
        full_path,
        validation_only=False,
    )

    top_path = top20_file(
        con,
        full_path,
        target_name,
    )

    full_metrics = evaluate_recall(
        con,
        full_path,
        gt_table,
        target_name,
        validation_only=False,
    )

    return {
        "target": target_name,
        "training_rows_full": full_rows,
        "validation": validation_metrics,
        "full_train": full_metrics,
        "full_ranked_output": str(full_path),
        "top20_output": str(top_path),
    }


def main() -> None:
    header("AMAZON ML CHALLENGE - V7 LEARNED RANKING")

    print(f"Project           : {PROJECT}")
    print(f"V6 directory      : {V6_DIR}")
    print(f"Output             : {OUT_DIR}")
    print(f"Memory              : {MEMORY_LIMIT}")
    print(f"Threads             : {THREADS}")
    print(f"Batch size          : {BATCH_SIZE:,}")
    print(f"Hard-negative rank  : {NEGATIVE_RANK_LIMIT}")
    print(f"Negatives per S1    : {NEGATIVES_PER_S1}")
    print(f"Validation modulus  : {VALIDATION_MOD}")
    print(f"Top-K               : {TOP_K}")

    require_file(GT_PATH)
    require_file(S2_V6)
    require_file(S3_V6)

    setup_dirs()

    con = setup_duckdb()

    load_ground_truth(con)

    summary: dict = {
        "version": "V7",
        "model": "HistGradientBoostingClassifier",
        "model_params": MODEL_PARAMS,
        "validation_mod": VALIDATION_MOD,
        "negative_rank_limit": NEGATIVE_RANK_LIMIT,
        "negatives_per_s1": NEGATIVES_PER_S1,
        "top_k": TOP_K,
        "results": {},
    }

    start = time.time()

    try:
        summary["results"]["s2"] = process_direction(
            con,
            target_name="S2",
            scored_path=S2_V6,
            gt_table="gt_s2",
        )

        summary["results"]["s3"] = process_direction(
            con,
            target_name="S3",
            scored_path=S3_V6,
            gt_table="gt_s3",
        )

    finally:
        con.close()

    elapsed = time.time() - start

    summary["elapsed_seconds"] = round(
        elapsed,
        2,
    )

    summary_path = OUT_DIR / "v7_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    header("V7 FINAL SUMMARY")

    for key in ["s2", "s3"]:
        result = summary["results"][key]

        val = result["validation"]
        full = result["full_train"]

        print(
            f"S1 → {key.upper()}"
        )
        print(
            f"  Holdout Recall@1 : "
            f"{val['recall_at_1']:.2f}%"
        )
        print(
            f"  Holdout Recall@5 : "
            f"{val['recall_at_5']:.2f}%"
        )
        print(
            f"  Holdout Recall@20: "
            f"{val['recall_at_20']:.2f}%"
        )
        print(
            f"  Full Recall@1    : "
            f"{full['recall_at_1']:.2f}%"
        )
        print(
            f"  Full Recall@5    : "
            f"{full['recall_at_5']:.2f}%"
        )
        print(
            f"  Full Recall@20   : "
            f"{full['recall_at_20']:.2f}%"
        )

    print()
    print(f"Summary : {summary_path}")
    print(f"Elapsed : {elapsed:.1f}s")

    header("✅ V7 COMPLETED")


if __name__ == "__main__":
    main()
