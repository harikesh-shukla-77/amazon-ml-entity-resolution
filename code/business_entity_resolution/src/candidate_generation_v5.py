from __future__ import annotations

import json
import time
from pathlib import Path


# ============================================================
# AMAZON ML CHALLENGE - V5 SAFE BLOCKING
# ============================================================
# Purpose:
#   Add only the block families that were useful in the
#   previous blocking diagnostics:
#
#   S1 -> S2:
#       1) name_first3_postal
#       2) postal_addr_tail5
#       3) name_last3_postal
#
#   S1 -> S3:
#       1) name_first3_postal
#       2) name_last3_postal
#       3) postal_addr_tail5
#
# This script:
#   - PRESERVES V1 / V2-SAFE / V4 outputs
#   - reads Parquet with DuckDB
#   - generates each block sequentially
#   - measures NEW ground-truth coverage after every block
#   - writes block Parquets and one consolidated V5 Parquet
#   - writes a JSON manifest
#
# Run from:
#   /Users/harikeshshukla/mla
#
# Environment:
#   source /Users/harikeshshukla/mla/.venv/bin/activate
#   python -u amazon_ml_candidate_generation_v5.py
# ============================================================

import duckdb


PROJECT = Path("/Users/harikeshshukla/mla")
PROCESSED_TRAIN = PROJECT / "processed_dataset" / "train"

S1_PATH = PROCESSED_TRAIN / "train_source1.parquet"
S2_PATH = PROCESSED_TRAIN / "train_source2.parquet"
S3_PATH = PROCESSED_TRAIN / "train_source3.parquet"
GT_PATH = PROCESSED_TRAIN / "ground_truth_pairs.parquet"

V1_DIR = PROJECT / "candidate_output"
V2_SAFE_DIR = PROJECT / "candidate_output_v2_safe"
V4_DIR = PROJECT / "candidate_output_v4"

OUT_DIR = PROJECT / "candidate_output_v5"
S2_OUT = OUT_DIR / "s1_s2_blocks"
S3_OUT = OUT_DIR / "s1_s3_blocks"
MANIFEST_PATH = OUT_DIR / "v5_manifest.json"

# Conservative settings for the 4 GB DuckDB setup used earlier.
MEMORY_LIMIT = "4GB"
TEMP_LIMIT = "8GB"
THREADS = 2

# Per-side block cardinality cap.
# The previous diagnostic showed the useful blocks below this
# range. Keeping the cap explicit protects against accidental
# huge joins.
MAX_BLOCK_SIZE = 150

# Maximum estimated candidate-pair count for a block.
MAX_BLOCK_PAIRS = 7_500_000


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def sql_path(path: Path) -> str:
    """Return a SQL-safe single-quoted path."""
    return "'" + str(path).replace("'", "''") + "'"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def candidate_files(root: Path) -> list[Path]:
    """
    Recursively find Parquet files that contain the two candidate ID
    columns. This makes V5 tolerant of the exact V1/V2/V4 filenames.
    """
    if not root.exists():
        return []

    files: list[Path] = []

    for path in sorted(root.rglob("*.parquet")):
        try:
            # Use a tiny schema-only query.
            con = duckdb.connect(database=":memory:")
            try:
                desc = con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet({sql_path(path)})"
                ).fetchall()
            finally:
                con.close()

            cols = {str(row[0]) for row in desc}
            if {
                "source1_entity_id",
                "candidate_entity_id",
            }.issubset(cols):
                files.append(path)

        except Exception:
            # Ignore non-candidate Parquet files.
            continue

    return files


def union_candidates_sql(paths: list[Path]) -> str:
    if not paths:
        raise RuntimeError("No candidate Parquet files were found.")

    parts = [
        (
            "SELECT "
            "source1_entity_id, "
            "candidate_entity_id "
            f"FROM read_parquet({sql_path(path)})"
        )
        for path in paths
    ]

    return "\nUNION ALL\n".join(parts)


def create_block_sql(table_name: str, key_expression: str) -> str:
    """
    Build a table containing:
      entity_id, country_norm, block_key
    plus only valid keys.
    """
    return f"""
    CREATE OR REPLACE TEMP TABLE {table_name} AS
    SELECT
        entity_id,
        country_norm,
        {key_expression} AS block_key
    FROM source_data
    WHERE
        country_norm IS NOT NULL
        AND trim(country_norm) <> ''
        AND {key_expression} IS NOT NULL
        AND trim({key_expression}) <> ''
    """


def estimate_pairs(
    con: duckdb.DuckDBPyConnection,
    left_block_table: str,
    right_block_table: str,
) -> int:
    sql = f"""
    WITH
    left_counts AS (
        SELECT block_key, COUNT(*) AS n
        FROM {left_block_table}
        GROUP BY block_key
    ),
    right_counts AS (
        SELECT block_key, COUNT(*) AS n
        FROM {right_block_table}
        GROUP BY block_key
    )
    SELECT COALESCE(SUM(l.n * r.n), 0)::BIGINT
    FROM left_counts l
    JOIN right_counts r
        ON l.block_key = r.block_key
    WHERE
        l.n <= {MAX_BLOCK_SIZE}
        AND r.n <= {MAX_BLOCK_SIZE}
    """
    return int(con.execute(sql).fetchone()[0] or 0)


def write_block(
    con: duckdb.DuckDBPyConnection,
    left_block_table: str,
    right_block_table: str,
    block_name: str,
    output_path: Path,
) -> int:
    """
    Generate one block directly to Parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = f"""
    SELECT
        l.entity_id AS source1_entity_id,
        r.entity_id AS candidate_entity_id,
        '{block_name}' AS block_name
    FROM {left_block_table} l
    INNER JOIN {right_block_table} r
        ON l.block_key = r.block_key
    INNER JOIN (
        SELECT block_key
        FROM {left_block_table}
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BLOCK_SIZE}
    ) lc
        ON l.block_key = lc.block_key
    INNER JOIN (
        SELECT block_key
        FROM {right_block_table}
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BLOCK_SIZE}
    ) rc
        ON r.block_key = rc.block_key
    """

    con.execute(
        f"""
        COPY (
            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                block_name
            FROM ({query})
        )
        TO {sql_path(output_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_path(output_path)})"
        ).fetchone()[0]
    )


def gt_stats_for_block(
    con: duckdb.DuckDBPyConnection,
    gt_table: str,
    block_path: Path,
    covered_gt_table: str,
) -> tuple[int, int]:
    """
    Returns:
      block_truth_matches,
      new_truth_matches
    """
    block_sql = f"read_parquet({sql_path(block_path)})"

    block_truth = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                g.source1_entity_id,
                g.matched_entity_id
            FROM {gt_table} g
            INNER JOIN {block_sql} c
                ON g.source1_entity_id = c.source1_entity_id
                AND g.matched_entity_id = c.candidate_entity_id
        )
        """
    ).fetchone()[0]

    new_truth = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                g.source1_entity_id,
                g.matched_entity_id
            FROM {gt_table} g
            INNER JOIN {block_sql} c
                ON g.source1_entity_id = c.source1_entity_id
                AND g.matched_entity_id = c.candidate_entity_id
            LEFT JOIN {covered_gt_table} x
                ON g.source1_entity_id = x.source1_entity_id
                AND g.matched_entity_id = x.matched_entity_id
            WHERE x.source1_entity_id IS NULL
        )
        """
    ).fetchone()[0]

    return int(block_truth or 0), int(new_truth or 0)


def add_block_truth_to_covered(
    con: duckdb.DuckDBPyConnection,
    gt_table: str,
    block_path: Path,
    covered_gt_table: str,
) -> None:
    con.execute(
        f"""
        INSERT INTO {covered_gt_table}
        SELECT DISTINCT
            g.source1_entity_id,
            g.matched_entity_id
        FROM {gt_table} g
        INNER JOIN read_parquet({sql_path(block_path)}) c
            ON g.source1_entity_id = c.source1_entity_id
            AND g.matched_entity_id = c.candidate_entity_id
        LEFT JOIN {covered_gt_table} x
            ON g.source1_entity_id = x.source1_entity_id
            AND g.matched_entity_id = x.matched_entity_id
        WHERE x.source1_entity_id IS NULL
        """
    )


def existing_covered_gt(
    con: duckdb.DuckDBPyConnection,
    gt_table: str,
    candidate_paths: list[Path],
    output_table: str,
) -> int:
    """
    Materialize only GT pairs already covered by all existing
    V1/V2/V4 candidate outputs. This is much smaller than
    materializing all candidate pairs.
    """
    existing_sql = union_candidates_sql(candidate_paths)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {output_table} AS
        SELECT DISTINCT
            g.source1_entity_id,
            g.matched_entity_id
        FROM {gt_table} g
        INNER JOIN (
            {existing_sql}
        ) e
            ON g.source1_entity_id = e.source1_entity_id
            AND g.matched_entity_id = e.candidate_entity_id
        """
    )

    return int(
        con.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]
    )


def consolidate_v5(
    con: duckdb.DuckDBPyConnection,
    block_paths: list[Path],
    output_path: Path,
) -> int:
    """
    Combine all V5 block files and deduplicate candidate pairs.
    Keep the block provenance.
    """
    if not block_paths:
        raise RuntimeError("No V5 block files to consolidate.")

    union_sql = union_candidates_sql(
        [
            # The helper only checks the two ID columns. We also need
            # block_name, which all V5 files contain.
            path
            for path in block_paths
        ]
    )

    # Rebuild using only the three V5 columns.
    parts = [
        f"""
        SELECT
            source1_entity_id,
            candidate_entity_id,
            block_name
        FROM read_parquet({sql_path(path)})
        """
        for path in block_paths
    ]
    v5_union = "\nUNION ALL\n".join(parts)

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                string_agg(DISTINCT block_name, '|') AS block_names,
                COUNT(DISTINCT block_name) AS block_hit_count
            FROM ({v5_union})
            GROUP BY
                source1_entity_id,
                candidate_entity_id
        )
        TO {sql_path(output_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_path(output_path)})"
        ).fetchone()[0]
    )


def process_direction(
    con: duckdb.DuckDBPyConnection,
    *,
    target_name: str,
    target_path: Path,
    gt_table: str,
    existing_candidate_paths: list[Path],
    block_specs: list[tuple[str, str]],
    output_dir: Path,
) -> dict:
    header(f"V5 PROCESSING: S1 → {target_name}")

    if not existing_candidate_paths:
        raise RuntimeError(
            f"No existing candidate files found for S1 → {target_name}."
        )

    print(f"Existing candidate files: {len(existing_candidate_paths)}")
    for path in existing_candidate_paths:
        print(f"  - {path}")

    # Existing coverage baseline.
    covered_table = f"covered_gt_{target_name.lower()}"
    existing_covered = existing_covered_gt(
        con,
        gt_table,
        existing_candidate_paths,
        covered_table,
    )

    gt_total = int(
        con.execute(f"SELECT COUNT(*) FROM {gt_table}").fetchone()[0]
    )
    baseline_coverage = (
        100.0 * existing_covered / gt_total if gt_total else 0.0
    )

    print()
    print(f"Ground-truth pairs        : {gt_total:,}")
    print(f"Existing covered          : {existing_covered:,}")
    print(f"Existing coverage         : {baseline_coverage:.2f}%")

    # Load source tables once.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE source_data AS
        SELECT
            entity_id,
            country_norm,
            name_norm,
            address_norm
        FROM read_parquet({sql_path(target_path)})
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_data AS
        SELECT
            entity_id,
            country_norm,
            name_norm,
            address_norm
        FROM read_parquet({sql_path(S1_PATH)})
        """
    )

    # Common lightweight derived fields.
    #
    # postal_like:
    #   - first try a UK-style alpha-numeric postal code
    #   - otherwise use the LAST 4-6 digit run in the address
    #
    # The "last numeric run" fallback helps countries with numeric
    # postal codes such as India while avoiding a full fuzzy join.
    for table_name, source_table in [
        ("s1_blocks", "s1_data"),
        ("target_blocks", "source_data"),
    ]:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {table_name} AS
            SELECT
                entity_id,
                country_norm,
                regexp_replace(
                    coalesce(name_norm, ''),
                    '[[:space:]]+',
                    '',
                    'g'
                ) AS name_compact,
                regexp_replace(
                    coalesce(address_norm, ''),
                    '[[:space:]]+',
                    '',
                    'g'
                ) AS address_compact,
                coalesce(
                    nullif(
                        regexp_extract(
                            lower(coalesce(address_norm, '')),
                            '([a-z]{{1,2}}[0-9][a-z0-9]?[0-9][a-z]{{2}})',
                            1
                        ),
                        ''
                    ),
                    nullif(
                        reverse(
                            regexp_extract(
                                reverse(lower(coalesce(address_norm, ''))),
                                '([0-9]{{4,6}})',
                                1
                            )
                        ),
                        ''
                    )
                ) AS postal_like
            FROM {source_table}
            """
        )

    # Base left block keys are reused for all blocks.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_keys AS
        SELECT
            entity_id,
            country_norm,
            name_compact,
            address_compact,
            postal_like,
            CASE
                WHEN length(name_compact) >= 3
                    AND postal_like IS NOT NULL
                    AND trim(postal_like) <> ''
                THEN country_norm || '|' ||
                     substr(name_compact, 1, 3) || '|' ||
                     postal_like
            END AS name_first3_postal,
            CASE
                WHEN length(name_compact) >= 3
                    AND postal_like IS NOT NULL
                    AND trim(postal_like) <> ''
                THEN country_norm || '|' ||
                     right(name_compact, 3) || '|' ||
                     postal_like
            END AS name_last3_postal,
            CASE
                WHEN length(address_compact) >= 5
                    AND postal_like IS NOT NULL
                    AND trim(postal_like) <> ''
                THEN country_norm || '|' ||
                     postal_like || '|' ||
                     right(address_compact, 5)
            END AS postal_addr_tail5
        FROM s1_blocks
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target_keys AS
        SELECT
            entity_id,
            country_norm,
            name_compact,
            address_compact,
            postal_like,
            CASE
                WHEN length(name_compact) >= 3
                    AND postal_like IS NOT NULL
                    AND trim(postal_like) <> ''
                THEN country_norm || '|' ||
                     substr(name_compact, 1, 3) || '|' ||
                     postal_like
            END AS name_first3_postal,
            CASE
                WHEN length(name_compact) >= 3
                    AND postal_like IS NOT NULL
                    AND trim(postal_like) <> ''
                THEN country_norm || '|' ||
                     right(name_compact, 3) || '|' ||
                     postal_like
            END AS name_last3_postal,
            CASE
                WHEN length(address_compact) >= 5
                    AND postal_like IS NOT NULL
                    AND trim(postal_like) <> ''
                THEN country_norm || '|' ||
                     postal_like || '|' ||
                     right(address_compact, 5)
            END AS postal_addr_tail5
        FROM target_blocks
        """
    )

    block_paths: list[Path] = []
    block_results: list[dict] = []

    for block_name, key_column in block_specs:
        header(f"BLOCK: {target_name} / {block_name}")

        # Create small block tables for this key.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE left_block AS
            SELECT
                entity_id,
                {key_column} AS block_key
            FROM s1_keys
            WHERE {key_column} IS NOT NULL
              AND trim({key_column}) <> ''
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE right_block AS
            SELECT
                entity_id,
                {key_column} AS block_key
            FROM target_keys
            WHERE {key_column} IS NOT NULL
              AND trim({key_column}) <> ''
            """
        )

        estimated = estimate_pairs(con, "left_block", "right_block")
        print(f"Estimated candidate pairs: {estimated:,}")

        if estimated == 0:
            print("SKIP: no shared block keys.")
            block_results.append(
                {
                    "block_name": block_name,
                    "estimated_pairs": 0,
                    "status": "skip_empty",
                }
            )
            continue

        if estimated > MAX_BLOCK_PAIRS:
            print(
                f"SKIP: estimate exceeds MAX_BLOCK_PAIRS={MAX_BLOCK_PAIRS:,}."
            )
            block_results.append(
                {
                    "block_name": block_name,
                    "estimated_pairs": estimated,
                    "status": "skip_too_large",
                }
            )
            continue

        output_path = output_dir / f"{block_name}.parquet"

        start = time.time()
        generated = write_block(
            con,
            "left_block",
            "right_block",
            block_name,
            output_path,
        )
        elapsed = time.time() - start

        block_truth, new_truth = gt_stats_for_block(
            con,
            gt_table,
            output_path,
            covered_table,
        )

        add_block_truth_to_covered(
            con,
            gt_table,
            output_path,
            covered_table,
        )

        combined_covered = int(
            con.execute(f"SELECT COUNT(*) FROM {covered_table}").fetchone()[0]
        )
        combined_coverage = (
            100.0 * combined_covered / gt_total if gt_total else 0.0
        )

        print(f"Generated candidates : {generated:,}")
        print(f"Time                  : {elapsed:.2f}s")
        print(f"Truth pairs in block  : {block_truth:,}")
        print(f"NEW truth matches     : {new_truth:,}")
        print(f"Combined covered      : {combined_covered:,}")
        print(f"Combined coverage     : {combined_coverage:.2f}%")

        block_paths.append(output_path)
        block_results.append(
            {
                "block_name": block_name,
                "estimated_pairs": estimated,
                "generated_candidates": generated,
                "block_truth_matches": block_truth,
                "new_truth_matches": new_truth,
                "combined_covered": combined_covered,
                "combined_coverage_pct": combined_coverage,
                "elapsed_seconds": round(elapsed, 3),
                "status": "generated",
                "output": str(output_path),
            }
        )

    # Consolidated V5 candidate file for this direction.
    consolidated = OUT_DIR / f"train_candidates_v5_s1_{target_name.lower()} .parquet"
    consolidated = Path(str(consolidated).replace(" ", ""))

    if block_paths:
        consolidated_count = consolidate_v5(
            con,
            block_paths,
            consolidated,
        )
    else:
        consolidated_count = 0

    final_covered = int(
        con.execute(f"SELECT COUNT(*) FROM {covered_table}").fetchone()[0]
    )
    final_coverage = (
        100.0 * final_covered / gt_total if gt_total else 0.0
    )

    header(f"FINAL V5 RESULT: S1 → {target_name}")
    print(f"Ground truth                  : {gt_total:,}")
    print(f"Existing covered              : {existing_covered:,}")
    print(f"Existing coverage             : {baseline_coverage:.2f}%")
    print(f"V5 final covered              : {final_covered:,}")
    print(f"V5 final coverage             : {final_coverage:.2f}%")
    print(
        f"Improvement vs existing       : "
        f"{final_coverage - baseline_coverage:+.2f} points"
    )
    print(f"V5 block candidates           : {consolidated_count:,}")
    print(f"V5 consolidated output        : {consolidated}")

    return {
        "target": target_name,
        "ground_truth_pairs": gt_total,
        "existing_candidate_files": [str(p) for p in existing_candidate_paths],
        "existing_covered": existing_covered,
        "existing_coverage_pct": baseline_coverage,
        "final_covered": final_covered,
        "final_coverage_pct": final_coverage,
        "improvement_vs_existing_points": final_coverage - baseline_coverage,
        "v5_consolidated_candidates": consolidated_count,
        "v5_consolidated_output": str(consolidated),
        "blocks": block_results,
    }


def main() -> None:
    header("AMAZON ML CHALLENGE - V5 SAFE BLOCKING")

    print(f"Project        : {PROJECT}")
    print(f"Processed train: {PROCESSED_TRAIN}")
    print(f"Output         : {OUT_DIR}")
    print(f"Memory limit   : {MEMORY_LIMIT}")
    print(f"Threads        : {THREADS}")
    print(f"Temp limit     : {TEMP_LIMIT}")
    print(f"Max block size : {MAX_BLOCK_SIZE}")
    print(f"Max block pairs: {MAX_BLOCK_PAIRS:,}")

    for path in [S1_PATH, S2_PATH, S3_PATH, GT_PATH]:
        require_file(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S2_OUT.mkdir(parents=True, exist_ok=True)
    S3_OUT.mkdir(parents=True, exist_ok=True)

    # Existing candidate files are discovered dynamically so we do
    # not depend on uncertain filenames.
    v1_files = candidate_files(V1_DIR)
    v2_files = candidate_files(V2_SAFE_DIR)
    v4_files = candidate_files(V4_DIR)

    if not v1_files:
        raise RuntimeError(
            f"No V1 candidate parquet files found under {V1_DIR}"
        )

    header("DISCOVERED EXISTING CANDIDATE FILES")
    print(f"V1 files : {len(v1_files)}")
    for p in v1_files:
        print(f"  V1  {p}")

    print(f"V2 files : {len(v2_files)}")
    for p in v2_files:
        print(f"  V2  {p}")

    print(f"V4 files : {len(v4_files)}")
    for p in v4_files:
        print(f"  V4  {p}")

    # Ground truth.
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA temp_directory='/Users/harikeshshukla/mla/candidate_output_v5/duckdb_tmp'")

    # Temp tables are used only for GT and derived block keys.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_all AS
        SELECT
            source1_entity_id,
            matched_entity_id,
            label
        FROM read_parquet({sql_path(GT_PATH)})
        WHERE COALESCE(label, 1) = 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_s2 AS
        SELECT *
        FROM gt_all
        WHERE matched_entity_id LIKE 'S2-%'
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE gt_s3 AS
        SELECT *
        FROM gt_all
        WHERE matched_entity_id LIKE 'S3-%'
        """
    )

    # These exact block families were selected from the earlier
    # diagnostics because they were materially useful and still
    # small enough to test safely.
    s2_specs = [
        ("name_first3_postal", "name_first3_postal"),
        ("postal_addr_tail5", "postal_addr_tail5"),
        ("name_last3_postal", "name_last3_postal"),
    ]

    s3_specs = [
        ("name_first3_postal", "name_first3_postal"),
        ("name_last3_postal", "name_last3_postal"),
        ("postal_addr_tail5", "postal_addr_tail5"),
    ]

    results = {}

    try:
        results["s2"] = process_direction(
            con,
            target_name="S2",
            target_path=S2_PATH,
            gt_table="gt_s2",
            existing_candidate_paths=v1_files + v2_files + v4_files,
            block_specs=s2_specs,
            output_dir=S2_OUT,
        )

        results["s3"] = process_direction(
            con,
            target_name="S3",
            target_path=S3_PATH,
            gt_table="gt_s3",
            existing_candidate_paths=v1_files + v2_files + v4_files,
            block_specs=s3_specs,
            output_dir=S3_OUT,
        )

    finally:
        con.close()

    # Manifest.
    manifest = {
        "version": "V5",
        "project": str(PROJECT),
        "memory_limit": MEMORY_LIMIT,
        "threads": THREADS,
        "temp_limit": TEMP_LIMIT,
        "max_block_size": MAX_BLOCK_SIZE,
        "max_block_pairs": MAX_BLOCK_PAIRS,
        "results": results,
        "output_dir": str(OUT_DIR),
        "manifest": str(MANIFEST_PATH),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    header("V5 FINAL SUMMARY")

    print(
        f"S1 → S2  existing: "
        f"{results['s2']['existing_coverage_pct']:.2f}%   "
        f"V5: {results['s2']['final_coverage_pct']:.2f}%   "
        f"change: {results['s2']['improvement_vs_existing_points']:+.2f} points"
    )

    print(
        f"S1 → S3  existing: "
        f"{results['s3']['existing_coverage_pct']:.2f}%   "
        f"V5: {results['s3']['final_coverage_pct']:.2f}%   "
        f"change: {results['s3']['improvement_vs_existing_points']:+.2f} points"
    )

    print()
    print("S2 consolidated:")
    print(results["s2"]["v5_consolidated_output"])

    print("S3 consolidated:")
    print(results["s3"]["v5_consolidated_output"])

    print("Manifest:")
    print(MANIFEST_PATH)

    header("✅ V5 COMPLETED")


if __name__ == "__main__":
    main()
