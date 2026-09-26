from pathlib import Path
import json
import time

import duckdb


# ============================================================
# AMAZON ML CHALLENGE
# V2 SAFE CANDIDATE GENERATION
#
# Important:
#   V1 is NOT modified.
#
#   V2-SAFE:
#     - Direct Parquet queries
#     - No huge UNION ALL
#     - No 5M-row permanent pandas merge
#     - Sequential blocking
#     - Block-size estimation before JOIN
#     - Automatic block skipping
#     - V1 + V2 combined ground-truth coverage
#
# This version focuses on SAFE candidate generation.
# Fuzzy scoring comes after we have a manageable candidate pool.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_DIR = ROOT / "processed_dataset" / "train"

SOURCE1 = TRAIN_DIR / "train_source1.parquet"
SOURCE2 = TRAIN_DIR / "train_source2.parquet"
SOURCE3 = TRAIN_DIR / "train_source3.parquet"

GROUND_TRUTH = TRAIN_DIR / "ground_truth_pairs.parquet"

V1_DIR = ROOT / "candidate_output"

V1_S2 = V1_DIR / "train_candidates_s1_s2.parquet"
V1_S3 = V1_DIR / "train_candidates_s1_s3.parquet"

OUTPUT_DIR = ROOT / "candidate_output_v2_safe"

S2_OUT = OUTPUT_DIR / "s1_s2_blocks"
S3_OUT = OUTPUT_DIR / "s1_s3_blocks"

TEMP_DIR = ROOT / "duckdb_tmp_v2_safe"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
S2_OUT.mkdir(parents=True, exist_ok=True)
S3_OUT.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFETY LIMITS
# ============================================================

MEMORY_LIMIT = "4GB"
MAX_TEMP_DIRECTORY_SIZE = "8GB"
THREADS = 2

# Maximum number of rows allowed in a single block
MAX_BLOCK_SIZE = 250

# Do not materialize a block whose estimated pair count
# exceeds this value.
MAX_BLOCK_PAIRS = 8_000_000

# Maximum additional candidates for one S1→target direction.
MAX_TOTAL_ADDITIONAL = 24_000_000


# ============================================================
# BLOCK DEFINITIONS
# ============================================================

BLOCKS = [
    (
        "name_p5",
        "substr(name_compact, 1, 5)",
    ),
    (
        "name_s5",
        "right(name_compact, 5)",
    ),
    (
        "name_mid5",
        """
        CASE
            WHEN length(name_compact) >= 10
            THEN substr(
                name_compact,
                CAST(
                    floor(
                        (length(name_compact) - 5) / 2
                    ) + 1
                    AS BIGINT
                ),
                5
            )
            ELSE name_compact
        END
        """,
    ),
    (
        "address_p8",
        "substr(address_compact, 1, 8)",
    ),
    (
        "address_s8",
        "right(address_compact, 8)",
    ),
    (
        "postal",
        "regexp_extract(address_norm, '[0-9]{5,6}', 0)",
    ),
    (
        "name_p4_num",
        """
        CASE
            WHEN addr_num <> ''
            THEN
                substr(name_compact, 1, 4)
                || '|'
                || addr_num
            ELSE ''
        END
        """,
    ),
    (
        "name_s4_num",
        """
        CASE
            WHEN addr_num <> ''
            THEN
                right(name_compact, 4)
                || '|'
                || addr_num
            ELSE ''
        END
        """,
    ),
]


# ============================================================
# SQL HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def base_relation(path: Path) -> str:
    """
    Return a projected Parquet relation with only the fields
    required for blocking.

    Keeping the projection small reduces memory and I/O.
    """

    p = sql_path(path)

    return f"""
    (
        SELECT
            entity_id,
            country_norm,
            COALESCE(name_norm, '') AS name_norm,
            COALESCE(address_norm, '') AS address_norm,

            regexp_replace(
                COALESCE(name_norm, ''),
                '[^a-z0-9]',
                '',
                'g'
            ) AS name_compact,

            regexp_replace(
                COALESCE(address_norm, ''),
                '[^a-z0-9]',
                '',
                'g'
            ) AS address_compact,

            regexp_extract(
                COALESCE(address_norm, ''),
                '[0-9]{{1,6}}',
                0
            ) AS addr_num

        FROM read_parquet('{p}')
    )
    """


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# DUCKDB
# ============================================================

DB_FILE = OUTPUT_DIR / "candidate_generation_v2_safe.duckdb"

con = duckdb.connect(str(DB_FILE))

con.execute(
    f"SET memory_limit='{MEMORY_LIMIT}'"
)

con.execute(
    f"SET threads={THREADS}"
)

con.execute(
    f"SET temp_directory='{sql_path(TEMP_DIR)}'"
)

con.execute(
    f"SET max_temp_directory_size='{MAX_TEMP_DIRECTORY_SIZE}'"
)

con.execute(
    "SET preserve_insertion_order=false"
)

con.execute(
    "SET enable_progress_bar=false"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    SOURCE1,
    SOURCE2,
    SOURCE3,
    GROUND_TRUTH,
    V1_S2,
    V1_S3,
]

for f in required_files:
    if not f.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{f}"
        )


# ============================================================
# GROUND TRUTH COUNTS
# ============================================================

def truth_count(prefix: str) -> int:

    result = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{sql_path(GROUND_TRUTH)}'
        )
        WHERE matched_entity_id
              LIKE '{prefix}-%'
        """
    ).fetchone()[0]

    return int(result)


# ============================================================
# V1 COVERAGE TABLE
# ============================================================

def create_covered_table(
    prefix: str,
    v1_file: Path,
):

    table_name = (
        "covered_s2"
        if prefix == "S2"
        else "covered_s3"
    )

    print_header(
        f"INITIAL V1 COVERAGE: S1 → {prefix}"
    )

    con.execute(
        f"DROP TABLE IF EXISTS {table_name}"
    )

    con.execute(
        f"""
        CREATE TEMP TABLE {table_name} AS

        SELECT DISTINCT

            g.source1_entity_id,

            g.matched_entity_id

        FROM read_parquet(
            '{sql_path(GROUND_TRUTH)}'
        ) g

        INNER JOIN read_parquet(
            '{sql_path(v1_file)}'
        ) c

        ON
            g.source1_entity_id
            = c.source1_entity_id

        AND
            g.matched_entity_id
            = c.candidate_entity_id

        WHERE
            g.matched_entity_id
            LIKE '{prefix}-%'
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    total = truth_count(prefix)

    coverage = (
        count / total * 100
        if total
        else 0
    )

    print(
        f"Ground truth pairs : {total:,}"
    )

    print(
        f"V1 covered         : {count:,}"
    )

    print(
        f"V1 coverage        : {coverage:.2f}%"
    )

    return table_name, total, count


# ============================================================
# BLOCK ESTIMATION
# ============================================================

def estimate_block_pairs(
    target_file: Path,
    key_expression: str,
) -> int:

    left_relation = base_relation(SOURCE1)
    right_relation = base_relation(target_file)

    query = f"""
    WITH

    left_blocks AS (

        SELECT
            country_norm,
            {key_expression} AS block_key,
            COUNT(*) AS n

        FROM {left_relation}

        WHERE
            country_norm <> ''

        GROUP BY
            country_norm,
            block_key

        HAVING
            block_key <> ''

        AND
            COUNT(*) <= {MAX_BLOCK_SIZE}
    ),

    right_blocks AS (

        SELECT
            country_norm,
            {key_expression} AS block_key,
            COUNT(*) AS n

        FROM {right_relation}

        WHERE
            country_norm <> ''

        GROUP BY
            country_norm,
            block_key

        HAVING
            block_key <> ''

        AND
            COUNT(*) <= {MAX_BLOCK_SIZE}
    )

    SELECT
        COALESCE(
            SUM(
                l.n::BIGINT
                * r.n::BIGINT
            ),
            0
        )

    FROM left_blocks l

    INNER JOIN right_blocks r

        ON
            l.country_norm
            = r.country_norm

        AND
            l.block_key
            = r.block_key
    """

    result = con.execute(query).fetchone()[0]

    return int(result or 0)


# ============================================================
# BLOCK TRUTH AUDIT
#
# This measures how many true pairs share this block,
# before applying the candidate-pair generation.
# ============================================================

def audit_truth_block(
    target_file: Path,
    prefix: str,
    key_expression: str,
) -> int:

    left_relation = base_relation(SOURCE1)
    right_relation = base_relation(target_file)

    query = f"""
    SELECT COUNT(*)

    FROM read_parquet(
        '{sql_path(GROUND_TRUTH)}'
    ) g

    INNER JOIN {left_relation} l

        ON
            g.source1_entity_id
            = l.entity_id

    INNER JOIN {right_relation} r

        ON
            g.matched_entity_id
            = r.entity_id

    WHERE
        g.matched_entity_id
        LIKE '{prefix}-%'

    AND
        l.country_norm
        = r.country_norm

    AND
        l.country_norm <> ''

    AND
        {key_expression.replace("name_compact", "l.name_compact")
                        .replace("address_compact", "l.address_compact")
                        .replace("name_norm", "l.name_norm")
                        .replace("address_norm", "l.address_norm")
                        .replace("addr_num", "l.addr_num")}
        =
        {key_expression.replace("name_compact", "r.name_compact")
                        .replace("address_compact", "r.address_compact")
                        .replace("name_norm", "r.name_norm")
                        .replace("address_norm", "r.address_norm")
                        .replace("addr_num", "r.addr_num")}

    AND
        (
            {key_expression.replace("name_compact", "l.name_compact")
                           .replace("address_compact", "l.address_compact")
                           .replace("name_norm", "l.name_norm")
                           .replace("address_norm", "l.address_norm")
                           .replace("addr_num", "l.addr_num")}
            <> ''
        )
    """

    result = con.execute(query).fetchone()[0]

    return int(result or 0)


# ============================================================
# GENERATE ONE BLOCK
# ============================================================

def generate_block(
    target_file: Path,
    prefix: str,
    block_name: str,
    key_expression: str,
    output_file: Path,
) -> int:

    left_relation = base_relation(SOURCE1)
    right_relation = base_relation(target_file)

    query = f"""
    WITH

    left_raw AS (

        SELECT
            entity_id,
            country_norm,
            name_norm,
            address_norm,
            {key_expression} AS block_key

        FROM {left_relation}

        WHERE
            country_norm <> ''
    ),

    left_counts AS (

        SELECT
            country_norm,
            block_key,
            COUNT(*) AS block_n

        FROM left_raw

        WHERE
            block_key <> ''

        GROUP BY
            country_norm,
            block_key

        HAVING
            COUNT(*) <= {MAX_BLOCK_SIZE}
    ),

    right_raw AS (

        SELECT
            entity_id,
            country_norm,
            name_norm,
            address_norm,
            {key_expression} AS block_key

        FROM {right_relation}

        WHERE
            country_norm <> ''
    ),

    right_counts AS (

        SELECT
            country_norm,
            block_key,
            COUNT(*) AS block_n

        FROM right_raw

        WHERE
            block_key <> ''

        GROUP BY
            country_norm,
            block_key

        HAVING
            COUNT(*) <= {MAX_BLOCK_SIZE}
    )

    SELECT

        l.entity_id
            AS source1_entity_id,

        r.entity_id
            AS candidate_entity_id,

        '{block_name}'
            AS block_name

    FROM left_raw l

    INNER JOIN left_counts lc

        ON
            l.country_norm
            = lc.country_norm

        AND
            l.block_key
            = lc.block_key

    INNER JOIN right_raw r

        ON
            l.country_norm
            = r.country_norm

        AND
            l.block_key
            = r.block_key

    INNER JOIN right_counts rc

        ON
            r.country_norm
            = rc.country_norm

        AND
            r.block_key
            = rc.block_key
    """

    if output_file.exists():
        output_file.unlink()

    print(
        f"\nGenerating: {block_name}"
    )

    start = time.time()

    con.execute(
        f"""
        COPY (
            {query}
        )
        TO '{sql_path(output_file)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{sql_path(output_file)}'
        )
        """
    ).fetchone()[0]

    elapsed = time.time() - start

    print(
        f"Generated candidates : {count:,}"
    )

    print(
        f"Time                  : {elapsed:.2f}s"
    )

    return int(count)


# ============================================================
# UPDATE COMBINED GROUND-TRUTH COVERAGE
# ============================================================

def update_coverage(
    covered_table: str,
    candidate_file: Path,
    prefix: str,
):

    before = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {covered_table}
        """
    ).fetchone()[0]

    # Truth pairs found in this block
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE block_truth AS

        SELECT DISTINCT

            g.source1_entity_id,

            g.matched_entity_id

        FROM read_parquet(
            ?
        ) c

        INNER JOIN read_parquet(
            ?
        ) g

        ON
            g.source1_entity_id
            = c.source1_entity_id

        AND
            g.matched_entity_id
            = c.candidate_entity_id

        WHERE
            g.matched_entity_id
            LIKE ?
        """,
        [
            str(candidate_file),
            str(GROUND_TRUTH),
            f"{prefix}-%",
        ],
    )

    block_found = con.execute(
        """
        SELECT COUNT(*)
        FROM block_truth
        """
    ).fetchone()[0]

    # Add only previously unseen truth pairs
    con.execute(
        f"""
        INSERT INTO {covered_table}

        SELECT
            b.source1_entity_id,
            b.matched_entity_id

        FROM block_truth b

        LEFT JOIN {covered_table} c

        ON
            b.source1_entity_id
            = c.source1_entity_id

        AND
            b.matched_entity_id
            = c.matched_entity_id

        WHERE
            c.source1_entity_id IS NULL
        """
    )

    after = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {covered_table}
        """
    ).fetchone()[0]

    new_found = after - before

    return int(block_found), int(new_found)


# ============================================================
# PROCESS ONE DIRECTION
# ============================================================

def process_direction(
    target_file: Path,
    prefix: str,
    v1_file: Path,
    output_dir: Path,
):

    print_header(
        f"V2-SAFE: S1 → {prefix}"
    )

    covered_table, truth_total, v1_covered = (
        create_covered_table(
            prefix,
            v1_file,
        )
    )

    total_added = 0

    audit_rows = []

    for block_name, key_expression in BLOCKS:

        print_header(
            f"BLOCK AUDIT: {prefix} / {block_name}"
        )

        if total_added >= MAX_TOTAL_ADDITIONAL:

            print(
                "Maximum additional candidate budget "
                "reached. Remaining blocks skipped."
            )

            break

        try:

            estimated = estimate_block_pairs(
                target_file,
                key_expression,
            )

            print(
                f"Estimated candidate pairs "
                f"(after block-size cap): "
                f"{estimated:,}"
            )

            if estimated == 0:

                print(
                    "SKIP: zero estimated candidates."
                )

                continue

            if estimated > MAX_BLOCK_PAIRS:

                print(
                    "SKIP: block is too large."
                )

                audit_rows.append({
                    "block": block_name,
                    "estimated_pairs": estimated,
                    "truth_pairs": None,
                    "status": "SKIPPED_TOO_LARGE",
                })

                continue

            # ----------------------------------------------
            # Truth audit
            # ----------------------------------------------

            truth_pairs = audit_truth_block(
                target_file,
                prefix,
                key_expression,
            )

            truth_pct = (
                truth_pairs
                / truth_total
                * 100
                if truth_total
                else 0
            )

            print(
                f"Truth pairs sharing block: "
                f"{truth_pairs:,}"
            )

            print(
                f"Block truth coverage "
                f"(pre-cap audit): "
                f"{truth_pct:.2f}%"
            )

            remaining_budget = (
                MAX_TOTAL_ADDITIONAL
                - total_added
            )

            if estimated > remaining_budget:

                print(
                    f"SKIP: estimated {estimated:,} "
                    f"exceeds remaining budget "
                    f"{remaining_budget:,}."
                )

                audit_rows.append({
                    "block": block_name,
                    "estimated_pairs": estimated,
                    "truth_pairs": truth_pairs,
                    "status": "SKIPPED_BUDGET",
                })

                continue

            # ----------------------------------------------
            # Generate
            # ----------------------------------------------

            block_file = (
                output_dir
                / f"{prefix.lower()}_{block_name}.parquet"
            )

            count = generate_block(
                target_file,
                prefix,
                block_name,
                key_expression,
                block_file,
            )

            total_added += count

            # ----------------------------------------------
            # Combined V1 + V2 coverage
            # ----------------------------------------------

            block_found, new_found = update_coverage(
                covered_table,
                block_file,
                prefix,
            )

            combined = con.execute(
                f"""
                SELECT COUNT(*)
                FROM {covered_table}
                """
            ).fetchone()[0]

            combined_pct = (
                combined
                / truth_total
                * 100
                if truth_total
                else 0
            )

            print(
                f"\nBlock truth matches : "
                f"{block_found:,}"
            )

            print(
                f"New truth matches   : "
                f"{new_found:,}"
            )

            print(
                f"Combined covered    : "
                f"{combined:,}"
            )

            print(
                f"Combined coverage   : "
                f"{combined_pct:.2f}%"
            )

            print(
                f"Additional candidates: "
                f"{total_added:,}"
            )

            audit_rows.append({
                "block": block_name,
                "estimated_pairs": estimated,
                "generated_pairs": count,
                "truth_pairs": truth_pairs,
                "new_truth_pairs": new_found,
                "combined_coverage_pct": round(
                    combined_pct,
                    4,
                ),
                "status": "GENERATED",
            })

        except Exception as exc:

            print(
                f"\nERROR in block {block_name}:"
            )

            print(
                repr(exc)
            )

            print(
                "Block skipped; continuing safely."
            )

            audit_rows.append({
                "block": block_name,
                "estimated_pairs": None,
                "generated_pairs": None,
                "truth_pairs": None,
                "new_truth_pairs": None,
                "status": "ERROR_SKIPPED",
                "error": repr(exc),
            })

    # ========================================================
    # FINAL COVERAGE
    # ========================================================

    final_covered = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {covered_table}
        """
    ).fetchone()[0]

    final_pct = (
        final_covered
        / truth_total
        * 100
        if truth_total
        else 0
    )

    improvement_points = (
        final_pct
        - (
            v1_covered
            / truth_total
            * 100
        )
    )

    summary = {
        "direction": f"S1->{prefix}",
        "truth_pairs": truth_total,
        "v1_covered": v1_covered,
        "v1_coverage_pct": round(
            v1_covered / truth_total * 100,
            4,
        ),
        "v2_combined_covered": final_covered,
        "v2_combined_coverage_pct": round(
            final_pct,
            4,
        ),
        "coverage_improvement_points": round(
            improvement_points,
            4,
        ),
        "additional_candidates": total_added,
        "blocks": audit_rows,
    }

    print_header(
        f"FINAL V2-SAFE RESULT: S1 → {prefix}"
    )

    print(
        f"Ground truth        : {truth_total:,}"
    )

    print(
        f"V1 covered          : {v1_covered:,}"
    )

    print(
        f"V1 coverage         : "
        f"{v1_covered / truth_total * 100:.2f}%"
    )

    print(
        f"V2 combined covered : {final_covered:,}"
    )

    print(
        f"V2 combined         : "
        f"{final_pct:.2f}%"
    )

    print(
        f"Improvement         : "
        f"{improvement_points:+.2f} percentage points"
    )

    print(
        f"Additional candidates: "
        f"{total_added:,}"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

print_header(
    "AMAZON ML CHALLENGE - V2 SAFE"
)

print(
    f"Project       : {ROOT}"
)

print(
    f"Memory limit  : {MEMORY_LIMIT}"
)

print(
    f"Threads       : {THREADS}"
)

print(
    f"Temp limit    : {MAX_TEMP_DIRECTORY_SIZE}"
)

print(
    f"Max block size: {MAX_BLOCK_SIZE}"
)

print(
    f"Max block pair: {MAX_BLOCK_PAIRS:,}"
)

print(
    f"Max additions : {MAX_TOTAL_ADDITIONAL:,}"
)


overall_start = time.time()

result_s2 = process_direction(
    SOURCE2,
    "S2",
    V1_S2,
    S2_OUT,
)

result_s3 = process_direction(
    SOURCE3,
    "S3",
    V1_S3,
    S3_OUT,
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = {
    "configuration": {
        "memory_limit": MEMORY_LIMIT,
        "threads": THREADS,
        "max_temp_directory_size": MAX_TEMP_DIRECTORY_SIZE,
        "max_block_size": MAX_BLOCK_SIZE,
        "max_block_pairs": MAX_BLOCK_PAIRS,
        "max_total_additional": MAX_TOTAL_ADDITIONAL,
    },
    "results": {
        "S1_to_S2": result_s2,
        "S1_to_S3": result_s3,
    },
    "elapsed_seconds": round(
        time.time() - overall_start,
        2,
    ),
}

manifest_file = (
    OUTPUT_DIR
    / "v2_safe_manifest.json"
)

manifest_file.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "V2-SAFE FINAL SUMMARY"
)

print(
    f"\nS1 → S2"
)

print(
    f"V1 : "
    f"{result_s2['v1_coverage_pct']:.2f}%"
)

print(
    f"V2 : "
    f"{result_s2['v2_combined_coverage_pct']:.2f}%"
)

print(
    f"Change: "
    f"{result_s2['coverage_improvement_points']:+.2f} points"
)

print(
    f"\nS1 → S3"
)

print(
    f"V1 : "
    f"{result_s3['v1_coverage_pct']:.2f}%"
)

print(
    f"V2 : "
    f"{result_s3['v2_combined_coverage_pct']:.2f}%"
)

print(
    f"Change: "
    f"{result_s3['coverage_improvement_points']:+.2f} points"
)

print_header(
    "OUTPUT"
)

print(
    OUTPUT_DIR
)

print(
    f"\nS2 blocks:"
)

print(
    S2_OUT
)

print(
    f"\nS3 blocks:"
)

print(
    S3_OUT
)

print(
    f"\nManifest:"
)

print(
    manifest_file
)

print_header(
    "✅ V2-SAFE COMPLETED"
)

con.close()
