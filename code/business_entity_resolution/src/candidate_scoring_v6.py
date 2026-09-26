from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


# ============================================================
# AMAZON ML CHALLENGE - V6 CANDIDATE SCORING / RANKING
# ============================================================
#
# V6 takes the candidate pools created by V1 + V2-SAFE + V4 + V5
# and ranks candidates using:
#
#   1. exact normalized name
#   2. exact normalized address
#   3. exact country
#   4. fuzzy name similarity
#   5. fuzzy address similarity
#   6. blocking/evidence count
#   7. small length-consistency signals
#
# It validates the ranking on TRAIN ground truth using:
#
#   Recall@1, @3, @5, @10, @20
#
# Output:
#   candidate_output_v6/
#       train_scored_s1_s2.parquet
#       train_scored_s1_s3.parquet
#       v6_summary.json
#
# This script DOES NOT modify V1/V2/V4/V5 outputs.
#
# Recommended environment:
#   cd /Users/harikeshshukla/mla
#   source /Users/harikeshshukla/mla/.venv/bin/activate
#
# Optional, strongly recommended for speed:
#   python -m pip install rapidfuzz
#
# Then:
#   python -u amazon_ml_candidate_scoring_v6.py
# ============================================================


PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN_DIR = PROJECT / "processed_dataset" / "train"

S1_PATH = TRAIN_DIR / "train_source1.parquet"
S2_PATH = TRAIN_DIR / "train_source2.parquet"
S3_PATH = TRAIN_DIR / "train_source3.parquet"
GT_PATH = TRAIN_DIR / "ground_truth_pairs.parquet"

V1_DIR = PROJECT / "candidate_output"
V2_DIR = PROJECT / "candidate_output_v2_safe"
V4_DIR = PROJECT / "candidate_output_v4"
V5_DIR = PROJECT / "candidate_output_v5"

OUT_DIR = PROJECT / "candidate_output_v6"
SUMMARY_PATH = OUT_DIR / "v6_summary.json"

MEMORY_LIMIT = "4GB"
TEMP_LIMIT = "8GB"
THREADS = 2

# Number of top-ranked candidates retained per S1.
TOP_K_SAVE = 20

# Fetch size for Python-side fuzzy scoring.
BATCH_SIZE = 100_000

# If a candidate has very weak fields, this keeps the score finite.
EPS = 1e-9


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def qpath(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


# ------------------------------------------------------------
# RapidFuzz is preferred. A stdlib fallback keeps the script
# functional even before installation, but it will be slower.
# ------------------------------------------------------------
try:
    from rapidfuzz.fuzz import ratio as rf_ratio

    HAVE_RAPIDFUZZ = True

    def fuzzy_ratio(a: str, b: str) -> float:
        return float(rf_ratio(a, b))

except Exception:
    from difflib import SequenceMatcher

    HAVE_RAPIDFUZZ = False

    def fuzzy_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return 100.0 * SequenceMatcher(None, a, b).ratio()


def candidate_files(root: Path) -> list[Path]:
    """
    Discover candidate parquet files dynamically.

    A candidate file must contain:
      source1_entity_id
      candidate_entity_id
    """
    if not root.exists():
        return []

    found: list[Path] = []

    for path in sorted(root.rglob("*.parquet")):
        try:
            c = duckdb.connect(database=":memory:")
            try:
                rows = c.execute(
                    f"DESCRIBE SELECT * FROM read_parquet({qpath(path)})"
                ).fetchall()
            finally:
                c.close()

            cols = {str(r[0]) for r in rows}

            if {
                "source1_entity_id",
                "candidate_entity_id",
            }.issubset(cols):
                found.append(path)

        except Exception:
            continue

    return found


def candidate_union_sql(
    paths: list[Path],
    target_prefix: str,
) -> str:
    """
    Union candidate files while filtering direction explicitly.

    Only candidate IDs with the requested target prefix are retained.
    This avoids accidentally mixing S2 files into S3 scoring and vice
    versa.
    """
    if not paths:
        raise RuntimeError("No candidate parquet files discovered.")

    pieces: list[str] = []

    for path in paths:
        source_file = str(path).replace("'", "''")

        # Introspect schema so V1, V2, V4 and V5 can coexist even
        # though their metadata columns differ.
        c = duckdb.connect(database=":memory:")
        try:
            schema_rows = c.execute(
                f"DESCRIBE SELECT * FROM read_parquet({qpath(path)})"
            ).fetchall()
        finally:
            c.close()

        cols = {str(r[0]) for r in schema_rows}

        block_expr = (
            "CAST(block_name AS VARCHAR)"
            if "block_name" in cols
            else "''"
        )

        match_keys_expr = (
            "CAST(match_keys AS VARCHAR)"
            if "match_keys" in cols
            else "''"
        )

        exact_count_expr = (
            "CAST(exact_key_count AS INTEGER)"
            if "exact_key_count" in cols
            else "0"
        )

        piece = f"""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            {block_expr} AS block_name,
            {match_keys_expr} AS match_keys,
            {exact_count_expr} AS exact_key_count,
            '{source_file}' AS source_file
        FROM read_parquet({qpath(path)})
        WHERE starts_with(candidate_entity_id, '{target_prefix}')
        """

        pieces.append(piece)

    return "\nUNION ALL\n".join(pieces)


def build_candidate_table(
    con: duckdb.DuckDBPyConnection,
    *,
    target_name: str,
    paths: list[Path],
) -> None:
    """
    Create a deduplicated candidate table with provenance counts.
    """
    target_prefix = target_name + "-"

    union_sql = candidate_union_sql(paths, target_prefix)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE candidates_raw AS
        SELECT
            source1_entity_id,
            candidate_entity_id,
            block_name,
            match_keys,
            exact_key_count,
            source_file
        FROM (
            {union_sql}
        )
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE candidates AS
        SELECT
            source1_entity_id,
            candidate_entity_id,

            COUNT(*) AS evidence_rows,

            COUNT(DISTINCT source_file) AS evidence_file_count,

            MAX(exact_key_count) AS exact_key_count,

            string_agg(
                DISTINCT
                CASE
                    WHEN block_name IS NULL THEN ''
                    ELSE block_name
                END,
                '|'
            ) AS block_names,

            string_agg(
                DISTINCT
                CASE
                    WHEN match_keys IS NULL THEN ''
                    ELSE match_keys
                END,
                '|'
            ) AS match_keys

        FROM candidates_raw
        GROUP BY
            source1_entity_id,
            candidate_entity_id
        """
    )


def create_feature_base(
    con: duckdb.DuckDBPyConnection,
    *,
    target_name: str,
    target_path: Path,
) -> None:
    """
    Join candidate IDs to the S1 and target records and create
    lightweight normalized features in DuckDB.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1 AS
        SELECT
            entity_id,
            coalesce(name_norm, '') AS name_norm,
            coalesce(address_norm, '') AS address_norm,
            coalesce(country_norm, '') AS country_norm
        FROM read_parquet({qpath(S1_PATH)})
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target AS
        SELECT
            entity_id,
            coalesce(name_norm, '') AS name_norm,
            coalesce(address_norm, '') AS address_norm,
            coalesce(country_norm, '') AS country_norm
        FROM read_parquet({qpath(target_path)})
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE feature_base AS
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            c.evidence_rows,
            c.evidence_file_count,
            c.exact_key_count,
            c.block_names,
            c.match_keys,

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
                WHEN l.name_norm = ''
                  OR r.name_norm = ''
                THEN 0.0
                ELSE
                    LEAST(
                        length(l.name_norm),
                        length(r.name_norm)
                    )::DOUBLE
                    /
                    GREATEST(
                        length(l.name_norm),
                        length(r.name_norm),
                        1
                    )
            END AS name_length_ratio,

            CASE
                WHEN l.address_norm = ''
                  OR r.address_norm = ''
                THEN 0.0
                ELSE
                    LEAST(
                        length(l.address_norm),
                        length(r.address_norm)
                    )::DOUBLE
                    /
                    GREATEST(
                        length(l.address_norm),
                        length(r.address_norm),
                        1
                    )
            END AS address_length_ratio

        FROM candidates c

        INNER JOIN s1 l
            ON c.source1_entity_id = l.entity_id

        INNER JOIN target r
            ON c.candidate_entity_id = r.entity_id
        """
    )


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Python-side fuzzy scoring.

    The score is intentionally transparent rather than a black box.
    It is designed as V6 validation/ranking, not as a claim that
    these weights are globally optimal.
    """
    if df.empty:
        return df

    names_left = df["left_name"].fillna("").astype(str).tolist()
    names_right = df["right_name"].fillna("").astype(str).tolist()

    addrs_left = df["left_address"].fillna("").astype(str).tolist()
    addrs_right = df["right_address"].fillna("").astype(str).tolist()

    name_sim = [
        fuzzy_ratio(a, b) / 100.0 if a and b else 0.0
        for a, b in zip(names_left, names_right)
    ]

    addr_sim = [
        fuzzy_ratio(a, b) / 100.0 if a and b else 0.0
        for a, b in zip(addrs_left, addrs_right)
    ]

    df["name_similarity"] = name_sim
    df["address_similarity"] = addr_sim

    # Transparent weighted score on 0-100 scale.
    #
    # Exact fields are strong evidence.
    # Fuzzy name/address are the main ranking features.
    # Evidence counts are deliberately small so they don't dominate
    # actual record similarity.
    df["score"] = (
        30.0 * df["name_exact"].astype(float)
        + 25.0 * df["address_exact"].astype(float)
        + 8.0 * df["country_exact"].astype(float)
        + 22.0 * df["name_similarity"].astype(float)
        + 12.0 * df["address_similarity"].astype(float)
        + 2.0 * df["name_length_ratio"].astype(float)
        + 1.0 * df["address_length_ratio"].astype(float)
        + 0.75 * (
            df["evidence_file_count"].astype(float).clip(upper=4.0)
        )
        + 0.25 * (
            df["exact_key_count"].astype(float).clip(upper=5.0)
        )
    )

    return df


def validate_ranking(
    con: duckdb.DuckDBPyConnection,
    scored_path: Path,
    gt_table: str,
) -> dict:
    """
    Evaluate recall@K over the complete ground-truth pair set.
    """
    result: dict[str, float | int] = {}

    gt_total = int(
        con.execute(f"SELECT COUNT(*) FROM {gt_table}").fetchone()[0]
    )

    result["ground_truth_pairs"] = gt_total

    for k in [1, 3, 5, 10, 20]:
        covered = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        g.source1_entity_id,
                        g.matched_entity_id
                    FROM {gt_table} g
                    INNER JOIN read_parquet({qpath(scored_path)}) s
                        ON g.source1_entity_id = s.source1_entity_id
                        AND g.matched_entity_id = s.candidate_entity_id
                    WHERE s.rank <= {k}
                    GROUP BY
                        g.source1_entity_id,
                        g.matched_entity_id
                )
                """
            ).fetchone()[0]
        )

        pct = 100.0 * covered / gt_total if gt_total else 0.0

        result[f"recall_at_{k}"] = pct
        result[f"covered_at_{k}"] = covered

    return result


def score_direction(
    con: duckdb.DuckDBPyConnection,
    *,
    target_name: str,
    target_path: Path,
    candidate_paths: list[Path],
    gt_table: str,
) -> dict:
    header(f"V6 SCORING: S1 → {target_name}")

    print(f"Candidate files: {len(candidate_paths)}")
    for p in candidate_paths:
        print(f"  {p}")

    build_candidate_table(
        con,
        target_name=target_name,
        paths=candidate_paths,
    )

    candidate_count = int(
        con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    )

    print(f"Unique candidate pairs: {candidate_count:,}")

    create_feature_base(
        con,
        target_name=target_name,
        target_path=target_path,
    )

    base_count = int(
        con.execute("SELECT COUNT(*) FROM feature_base").fetchone()[0]
    )

    print(f"Candidates after record join: {base_count:,}")

    # --------------------------------------------------------
    # Score in Arrow/Pandas batches to avoid loading all fuzzy
    # strings into RAM at once.
    # --------------------------------------------------------
    feature_query = """
        SELECT
            source1_entity_id,
            candidate_entity_id,

            evidence_rows,
            evidence_file_count,
            exact_key_count,
            block_names,
            match_keys,

            left_name,
            right_name,
            left_address,
            right_address,
            left_country,
            right_country,

            name_exact,
            address_exact,
            country_exact,

            name_length_ratio,
            address_length_ratio
        FROM feature_base
    """

    output_path = (
        OUT_DIR / f"train_scored_s1_{target_name.lower()}.parquet"
    )

    # Remove stale output so a rerun doesn't accidentally append/merge
    # with a previous version.
    output_path.unlink(missing_ok=True)

    # DuckDB Arrow record batch reader is memory-friendly.
    reader = con.execute(feature_query).fetch_record_batch(BATCH_SIZE)

    total_processed = 0
    all_parts: list[Path] = []

    part_index = 0

    for arrow_batch in reader:
        batch = arrow_batch.to_pandas()

        if batch.empty:
            continue

        batch = score_batch(batch)

        # Sort locally; global ranking is done after concatenation
        # through DuckDB, so batches do not need to fit entirely in RAM.
        part_path = (
            OUT_DIR
            / "parts"
            / target_name.lower()
            / f"part_{part_index:05d}.parquet"
        )
        part_path.parent.mkdir(parents=True, exist_ok=True)

        batch.to_parquet(
            part_path,
            index=False,
            compression="zstd",
        )

        all_parts.append(part_path)

        total_processed += len(batch)
        part_index += 1

        if total_processed % 1_000_000 < len(batch):
            print(f"Scored: {total_processed:,}")

    if not all_parts:
        raise RuntimeError(f"No scored rows produced for {target_name}.")

    print(f"Total scored rows: {total_processed:,}")
    print("Creating global rank per Source1...")

    parts_glob = str(
        OUT_DIR / "parts" / target_name.lower() / "part_*.parquet"
    ).replace("\\", "/").replace("'", "''")

    # Global ranking and top-K retention.
    #
    # Important: ties are broken by candidate_entity_id for
    # deterministic reruns.
    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,

                round(score, 6) AS score,

                row_number() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY
                        score DESC,
                        candidate_entity_id
                ) AS rank,

                round(name_similarity, 6) AS name_similarity,
                round(address_similarity, 6) AS address_similarity,

                name_exact,
                address_exact,
                country_exact,

                round(name_length_ratio, 6) AS name_length_ratio,
                round(address_length_ratio, 6) AS address_length_ratio,

                evidence_rows,
                evidence_file_count,
                exact_key_count,
                block_names,
                match_keys

            FROM read_parquet('{parts_glob}')
        )
        TO {qpath(output_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # Trim to top-K only, while preserving the complete ranking file
    # temporarily in the output parts. The final train file contains
    # TOP_K_SAVE per S1 at most.
    topk_path = (
        OUT_DIR / f"train_scored_top{TOP_K_SAVE}_s1_{target_name.lower()}.parquet"
    )
    topk_path.unlink(missing_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet({qpath(output_path)})
            WHERE rank <= {TOP_K_SAVE}
        )
        TO {qpath(topk_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # Validate from the top-K file only. Since recall@K <= TOP_K_SAVE,
    # this is exactly sufficient for K<=20.
    metrics = validate_ranking(
        con,
        topk_path,
        gt_table,
    )

    # Helpful candidate statistics.
    unique_s1 = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT source1_entity_id)
            FROM read_parquet({qpath(topk_path)})
            """
        ).fetchone()[0]
    )

    avg_candidates = (
        total_processed / unique_s1 if unique_s1 else 0.0
    )

    metrics.update(
        {
            "target": target_name,
            "input_candidate_pairs": candidate_count,
            "joined_candidate_pairs": base_count,
            "scored_rows": total_processed,
            "unique_s1_with_candidates": unique_s1,
            "average_input_candidates_per_s1": avg_candidates,
            "full_scored_output": str(output_path),
            "topk_output": str(topk_path),
            "top_k_saved": TOP_K_SAVE,
            "rapidfuzz_available": HAVE_RAPIDFUZZ,
        }
    )

    header(f"V6 RESULT: S1 → {target_name}")

    print(f"Input candidate pairs : {candidate_count:,}")
    print(f"Scored rows           : {total_processed:,}")
    print(f"Unique S1             : {unique_s1:,}")
    print(f"Average candidates/S1 : {avg_candidates:.2f}")

    for k in [1, 3, 5, 10, 20]:
        print(
            f"Recall@{k:<2}: "
            f"{metrics[f'recall_at_{k}']:.2f}% "
            f"({metrics[f'covered_at_{k}']:,}/"
            f"{metrics['ground_truth_pairs']:,})"
        )

    print(f"Full scored output : {output_path}")
    print(f"Top-{TOP_K_SAVE} output   : {topk_path}")

    return metrics


def cleanup_parts() -> None:
    """
    Keep the final scoring files and remove only temporary per-batch
    scoring files to save disk space.
    """
    parts_root = OUT_DIR / "parts"

    if not parts_root.exists():
        return

    for path in sorted(parts_root.rglob("*.parquet")):
        try:
            path.unlink()
        except Exception:
            pass

    # Remove empty directories bottom-up.
    for path in sorted(
        [p for p in parts_root.rglob("*") if p.is_dir()],
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass

    try:
        parts_root.rmdir()
    except OSError:
        pass


def main() -> None:
    header("AMAZON ML CHALLENGE - V6 CANDIDATE SCORING / RANKING")

    print(f"Project      : {PROJECT}")
    print(f"Train dir    : {TRAIN_DIR}")
    print(f"Output       : {OUT_DIR}")
    print(f"Memory       : {MEMORY_LIMIT}")
    print(f"Threads      : {THREADS}")
    print(f"Batch size   : {BATCH_SIZE:,}")
    print(f"Top-K saved  : {TOP_K_SAVE}")
    print(f"RapidFuzz    : {HAVE_RAPIDFUZZ}")

    if not HAVE_RAPIDFUZZ:
        print(
            "\nWARNING: rapidfuzz is not installed. "
            "The script will use Python's difflib fallback, "
            "which can be much slower."
        )

    for path in [S1_PATH, S2_PATH, S3_PATH, GT_PATH]:
        require(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v1 = candidate_files(V1_DIR)
    v2 = candidate_files(V2_DIR)
    v4 = candidate_files(V4_DIR)
    v5 = candidate_files(V5_DIR)

    if not v1:
        raise RuntimeError("No V1 candidate files found.")

    all_candidate_files = v1 + v2 + v4 + v5

    header("DISCOVERED CANDIDATE SOURCES")

    print(f"V1: {len(v1)} files")
    print(f"V2: {len(v2)} files")
    print(f"V4: {len(v4)} files")
    print(f"V5: {len(v5)} files")
    print(f"Total source files: {len(all_candidate_files)}")

    con = duckdb.connect(database=":memory:")

    temp_dir = OUT_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute(f"PRAGMA temp_directory='{str(temp_dir).replace(chr(92), '/').replace(chr(39), chr(39)*2)}'")

    # Ground truth is already exploded into one row per positive pair.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_all AS
        SELECT
            source1_entity_id,
            matched_entity_id,
            label
        FROM read_parquet({qpath(GT_PATH)})
        WHERE COALESCE(label, 1) = 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_s2 AS
        SELECT *
        FROM gt_all
        WHERE starts_with(matched_entity_id, 'S2-')
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_s3 AS
        SELECT *
        FROM gt_all
        WHERE starts_with(matched_entity_id, 'S3-')
        """
    )

    results: dict[str, dict] = {}

    start = time.time()

    try:
        results["s2"] = score_direction(
            con,
            target_name="S2",
            target_path=S2_PATH,
            candidate_paths=all_candidate_files,
            gt_table="gt_s2",
        )

        results["s3"] = score_direction(
            con,
            target_name="S3",
            target_path=S3_PATH,
            candidate_paths=all_candidate_files,
            gt_table="gt_s3",
        )

    finally:
        con.close()

    elapsed = time.time() - start

    summary = {
        "version": "V6",
        "project": str(PROJECT),
        "memory_limit": MEMORY_LIMIT,
        "threads": THREADS,
        "batch_size": BATCH_SIZE,
        "top_k_saved": TOP_K_SAVE,
        "rapidfuzz_available": HAVE_RAPIDFUZZ,
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Temporary per-batch scoring files are no longer needed.
    cleanup_parts()

    header("V6 FINAL SUMMARY")

    for key in ["s2", "s3"]:
        item = results[key]
        print(
            f"S1 → {item['target']}: "
            f"Recall@1={item['recall_at_1']:.2f}% | "
            f"Recall@5={item['recall_at_5']:.2f}% | "
            f"Recall@10={item['recall_at_10']:.2f}% | "
            f"Recall@20={item['recall_at_20']:.2f}%"
        )

    print()
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Elapsed: {elapsed:.1f}s")

    header("✅ V6 COMPLETED")


if __name__ == "__main__":
    main()
