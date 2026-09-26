from pathlib import Path
import json
import time

import duckdb


# ============================================================
# AMAZON ML CHALLENGE
# V4 - TIGHT COMPOSITE BLOCKING
#
# V1 = exact normalized-key baseline
# V2 = postal block improvement
# V3 = not used because its blocks were too broad / buggy
#
# V4 goal:
#   Recover additional true matches with much tighter
#   composite signatures while keeping joins bounded.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN = ROOT / "processed_dataset" / "train"

S1_FILE = TRAIN / "train_source1.parquet"
S2_FILE = TRAIN / "train_source2.parquet"
S3_FILE = TRAIN / "train_source3.parquet"

GT_FILE = TRAIN / "ground_truth_pairs.parquet"

V1_DIR = ROOT / "candidate_output"

V1_S2 = V1_DIR / "train_candidates_s1_s2.parquet"
V1_S3 = V1_DIR / "train_candidates_s1_s3.parquet"

V2_DIR = ROOT / "candidate_output_v2_safe"

V2_S2_POSTAL = (
    V2_DIR
    / "s1_s2_blocks"
    / "s2_postal.parquet"
)

V2_S3_POSTAL = (
    V2_DIR
    / "s1_s3_blocks"
    / "s3_postal.parquet"
)

OUTPUT_DIR = ROOT / "candidate_output_v4"

S2_OUT = OUTPUT_DIR / "s1_s2"
S3_OUT = OUTPUT_DIR / "s1_s3"

TEMP_DIR = ROOT / "duckdb_tmp_v4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
S2_OUT.mkdir(parents=True, exist_ok=True)
S3_OUT.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFETY SETTINGS
# ============================================================

MEMORY_LIMIT = "4GB"
TEMP_LIMIT = "8GB"
THREADS = 2

# Individual block size cap
MAX_BLOCK_SIZE = 80

# Maximum generated pairs by one block
MAX_BLOCK_PAIRS = 4_000_000

# Maximum additional pairs per direction
MAX_TOTAL_ADDITIONAL = 16_000_000


# ============================================================
# BLOCK DEFINITIONS
#
# These are deliberately much tighter than V2/V3.
# ============================================================

BLOCKS = [

    (
        "name_first3_last3_len",
        """
        substr(name_compact, 1, 3)
        || '|'
        || right(name_compact, 3)
        || '|'
        || CAST(
            floor(length(name_compact) / 5)
            AS BIGINT
        )
        """
    ),

    (
        "name_first4_last2_len",
        """
        substr(name_compact, 1, 4)
        || '|'
        || right(name_compact, 2)
        || '|'
        || CAST(
            floor(length(name_compact) / 5)
            AS BIGINT
        )
        """
    ),

    (
        "name_first2_last4_len",
        """
        substr(name_compact, 1, 2)
        || '|'
        || right(name_compact, 4)
        || '|'
        || CAST(
            floor(length(name_compact) / 5)
            AS BIGINT
        )
        """
    ),

    (
        "name_prefix3_suffix2_len",
        """
        substr(name_compact, 1, 3)
        || '|'
        || right(name_compact, 2)
        || '|'
        || CAST(
            floor(length(name_compact) / 3)
            AS BIGINT
        )
        """
    ),

    (
        "address_first4_last4_len",
        """
        substr(address_compact, 1, 4)
        || '|'
        || right(address_compact, 4)
        || '|'
        || CAST(
            floor(length(address_compact) / 10)
            AS BIGINT
        )
        """
    ),

    (
        "name_num_signature",
        """
        CASE

            WHEN addr_num <> ''

            THEN

                substr(name_compact, 1, 3)
                || '|'
                || right(name_compact, 2)
                || '|'
                || addr_num

            ELSE ''

        END
        """
    ),

    (
        "num_address_signature",
        """
        CASE

            WHEN addr_num <> ''

            THEN

                addr_num
                || '|'
                || substr(address_compact, 1, 3)
                || '|'
                || right(address_compact, 3)

            ELSE ''

        END
        """
    ),

    (
        "postal_name_signature",
        """
        CASE

            WHEN postal <> ''

            THEN

                postal
                || '|'
                || substr(name_compact, 1, 3)
                || '|'
                || right(name_compact, 2)

            ELSE ''

        END
        """
    ),
]


# ============================================================
# SQL HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# DUCKDB
# ============================================================

DB_FILE = OUTPUT_DIR / "candidate_generation_v4.duckdb"

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
    f"SET max_temp_directory_size='{TEMP_LIMIT}'"
)

con.execute(
    "SET preserve_insertion_order=false"
)

con.execute(
    "SET enable_progress_bar=false"
)


# ============================================================
# SOURCE RELATION
# ============================================================

def source_relation(path: Path) -> str:

    p = sql_path(path)

    return f"""
    (
        SELECT

            entity_id,

            country_norm,

            COALESCE(
                name_norm,
                ''
            ) AS name_norm,

            COALESCE(
                address_norm,
                ''
            ) AS address_norm,

            regexp_replace(
                COALESCE(
                    name_norm,
                    ''
                ),
                '[^a-z0-9]',
                '',
                'g'
            ) AS name_compact,

            regexp_replace(
                COALESCE(
                    address_norm,
                    ''
                ),
                '[^a-z0-9]',
                '',
                'g'
            ) AS address_compact,

            regexp_extract(
                COALESCE(
                    address_norm,
                    ''
                ),
                '[0-9]{{1,6}}',
                0
            ) AS addr_num,

            regexp_extract(
                COALESCE(
                    address_norm,
                    ''
                ),
                '[0-9]{{5,6}}',
                0
            ) AS postal

        FROM read_parquet(
            '{p}'
        )
    )
    """


# ============================================================
# FILE CHECK
# ============================================================

required = [
    S1_FILE,
    S2_FILE,
    S3_FILE,
    GT_FILE,
    V1_S2,
    V1_S3,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# CREATE SOURCE 1 TABLE
# ============================================================

header(
    "CREATING SOURCE 1"
)

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE s1 AS

    SELECT *

    FROM {source_relation(S1_FILE)}
    """
)

s1_count = con.execute(
    """
    SELECT COUNT(*)
    FROM s1
    """
).fetchone()[0]

print(
    f"Source1 rows: {s1_count:,}"
)


# ============================================================
# CREATE GROUND TRUTH TABLE
# ============================================================

header(
    "LOADING GROUND TRUTH"
)

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE gt AS

    SELECT

        source1_entity_id,

        matched_entity_id

    FROM read_parquet(
        '{sql_path(GT_FILE)}'
    )
    """
)

gt_count = con.execute(
    """
    SELECT COUNT(*)
    FROM gt
    """
).fetchone()[0]

print(
    f"Ground-truth pairs: {gt_count:,}"
)


# ============================================================
# INITIAL COVERAGE TABLE
#
# Start with V1.
# ============================================================

def initialize_coverage(
    prefix: str,
    v1_file: Path,
):

    table = (
        "covered_s2"
        if prefix == "S2"
        else "covered_s3"
    )

    header(
        f"INITIAL V1 COVERAGE: S1 → {prefix}"
    )

    con.execute(
        f"DROP TABLE IF EXISTS {table}"
    )

    con.execute(
        f"""
        CREATE TEMP TABLE {table} AS

        SELECT DISTINCT

            g.source1_entity_id,

            g.matched_entity_id

        FROM gt g

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

    covered = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM gt
        WHERE matched_entity_id
              LIKE '{prefix}-%'
        """
    ).fetchone()[0]

    pct = (
        covered / total * 100
        if total
        else 0
    )

    print(
        f"GT pairs       : {total:,}"
    )

    print(
        f"V1 covered     : {covered:,}"
    )

    print(
        f"V1 coverage    : {pct:.2f}%"
    )

    return table, int(total), int(covered)


# ============================================================
# ADD PREVIOUS V2 POSTAL BLOCK
# ============================================================

def add_existing_candidate_file(
    covered_table: str,
    candidate_file: Path,
    prefix: str,
    label: str,
):

    if not candidate_file.exists():

        print(
            f"\n{label}: file not found, skipped."
        )

        return 0

    header(
        f"ADDING EXISTING {label}"
    )

    before = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {covered_table}
        """
    ).fetchone()[0]

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE prior_truth AS

        SELECT DISTINCT

            g.source1_entity_id,

            g.matched_entity_id

        FROM gt g

        INNER JOIN read_parquet(
            ?
        ) c

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
            f"{prefix}-%"
        ]
    )

    con.execute(
        f"""
        INSERT INTO {covered_table}

        SELECT

            p.source1_entity_id,

            p.matched_entity_id

        FROM prior_truth p

        LEFT JOIN {covered_table} c

        ON
            p.source1_entity_id
            = c.source1_entity_id

        AND
            p.matched_entity_id
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

    print(
        f"New truth matches from {label}: "
        f"{new_found:,}"
    )

    return int(new_found)


# ============================================================
# CREATE RIGHT TABLE
# ============================================================

def create_right_table(
    table_name: str,
    target_file: Path,
):

    header(
        f"CREATING {table_name.upper()}"
    )

    con.execute(
        f"DROP TABLE IF EXISTS {table_name}"
    )

    con.execute(
        f"""
        CREATE TEMP TABLE {table_name} AS

        SELECT *

        FROM {source_relation(target_file)}
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    print(
        f"{table_name} rows: {count:,}"
    )


# ============================================================
# BUILD BLOCK EXPRESSIONS
# ============================================================

def key_with_country(
    expression: str
) -> str:

    return f"""
        country_norm
        || '||'
        || (
            {expression}
        )
    """


# ============================================================
# ESTIMATE BLOCK SIZE
# ============================================================

def estimate_block(
    right_table: str,
    expression: str,
) -> int:

    key = key_with_country(expression)

    query = f"""
    WITH

    left_blocks AS (

        SELECT

            {key} AS block_key,

            COUNT(*) AS n

        FROM s1

        WHERE
            country_norm <> ''

        GROUP BY
            block_key

        HAVING
            block_key <> ''

        AND
            COUNT(*) <= {MAX_BLOCK_SIZE}
    ),

    right_blocks AS (

        SELECT

            {key} AS block_key,

            COUNT(*) AS n

        FROM {right_table}

        WHERE
            country_norm <> ''

        GROUP BY
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
            l.block_key
            = r.block_key
    """

    value = con.execute(
        query
    ).fetchone()[0]

    return int(value or 0)


# ============================================================
# GENERATE BLOCK
# ============================================================

def generate_block(
    right_table: str,
    expression: str,
    block_name: str,
    output_file: Path,
):

    key = key_with_country(expression)

    if output_file.exists():
        output_file.unlink()

    query = f"""
    WITH

    left_base AS (

        SELECT

            entity_id,

            {key} AS block_key

        FROM s1

        WHERE
            country_norm <> ''

    ),

    left_counts AS (

        SELECT

            block_key,

            COUNT(*) AS block_n

        FROM left_base

        WHERE
            block_key <> ''

        GROUP BY
            block_key

        HAVING
            COUNT(*) <= {MAX_BLOCK_SIZE}
    ),

    right_base AS (

        SELECT

            entity_id,

            {key} AS block_key

        FROM {right_table}

        WHERE
            country_norm <> ''

    ),

    right_counts AS (

        SELECT

            block_key,

            COUNT(*) AS block_n

        FROM right_base

        WHERE
            block_key <> ''

        GROUP BY
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

    FROM left_base l

    INNER JOIN left_counts lc

        ON
            l.block_key
            = lc.block_key

    INNER JOIN right_base r

        ON
            l.block_key
            = r.block_key

    INNER JOIN right_counts rc

        ON
            r.block_key
            = rc.block_key
    """

    start = time.time()

    print(
        f"\nGenerating block: {block_name}"
    )

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
        f"Generated candidates: {count:,}"
    )

    print(
        f"Time: {elapsed:.2f}s"
    )

    return int(count)


# ============================================================
# UPDATE COVERAGE FROM ONE BLOCK
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

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE block_truth AS

        SELECT DISTINCT

            g.source1_entity_id,

            g.matched_entity_id

        FROM gt g

        INNER JOIN read_parquet(
            ?
        ) c

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
            f"{prefix}-%"
        ]
    )

    block_truth = con.execute(
        """
        SELECT COUNT(*)
        FROM block_truth
        """
    ).fetchone()[0]

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

    new_truth = after - before

    return int(block_truth), int(new_truth)


# ============================================================
# PROCESS ONE DIRECTION
# ============================================================

def process_direction(
    prefix: str,
    right_table: str,
    right_file: Path,
    v1_file: Path,
    old_postal_file: Path,
    output_dir: Path,
):

    header(
        f"V4 PROCESSING: S1 → {prefix}"
    )

    covered_table, truth_total, v1_covered = (
        initialize_coverage(
            prefix,
            v1_file,
        )
    )

    # --------------------------------------------------------
    # Add V2 postal candidates
    # --------------------------------------------------------

    postal_new = add_existing_candidate_file(
        covered_table,
        old_postal_file,
        prefix,
        "V2 POSTAL",
    )

    best_after_postal = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {covered_table}
        """
    ).fetchone()[0]

    best_postal_pct = (
        best_after_postal
        / truth_total
        * 100
        if truth_total
        else 0
    )

    print(
        f"Coverage after V2 postal: "
        f"{best_postal_pct:.2f}%"
    )

    # --------------------------------------------------------
    # Create right table
    # --------------------------------------------------------

    create_right_table(
        right_table,
        right_file,
    )

    additional_total = 0

    block_results = []

    # --------------------------------------------------------
    # Process each tight block
    # --------------------------------------------------------

    for block_name, expression in BLOCKS:

        header(
            f"BLOCK: {prefix} / {block_name}"
        )

        if additional_total >= MAX_TOTAL_ADDITIONAL:

            print(
                "Additional candidate budget reached."
            )

            break

        try:

            estimated = estimate_block(
                right_table,
                expression,
            )

            print(
                f"Estimated pairs: "
                f"{estimated:,}"
            )

            if estimated == 0:

                print(
                    "SKIP: zero pairs."
                )

                block_results.append({
                    "block": block_name,
                    "estimated": 0,
                    "generated": 0,
                    "new_truth": 0,
                    "status": "ZERO"
                })

                continue

            if estimated > MAX_BLOCK_PAIRS:

                print(
                    "SKIP: block too large."
                )

                block_results.append({
                    "block": block_name,
                    "estimated": estimated,
                    "generated": 0,
                    "new_truth": 0,
                    "status": "TOO_LARGE"
                })

                continue

            if (
                additional_total
                + estimated
                > MAX_TOTAL_ADDITIONAL
            ):

                print(
                    "SKIP: total budget exceeded."
                )

                block_results.append({
                    "block": block_name,
                    "estimated": estimated,
                    "generated": 0,
                    "new_truth": 0,
                    "status": "BUDGET"
                })

                continue

            output_file = (
                output_dir
                / f"{prefix.lower()}_{block_name}.parquet"
            )

            generated = generate_block(
                right_table,
                expression,
                block_name,
                output_file,
            )

            additional_total += generated

            block_truth, new_truth = (
                update_coverage(
                    covered_table,
                    output_file,
                    prefix,
                )
            )

            current_covered = con.execute(
                f"""
                SELECT COUNT(*)
                FROM {covered_table}
                """
            ).fetchone()[0]

            current_pct = (
                current_covered
                / truth_total
                * 100
                if truth_total
                else 0
            )

            print(
                f"Truth pairs in block: "
                f"{block_truth:,}"
            )

            print(
                f"New truth pairs: "
                f"{new_truth:,}"
            )

            print(
                f"Combined covered: "
                f"{current_covered:,}"
            )

            print(
                f"Combined coverage: "
                f"{current_pct:.2f}%"
            )

            print(
                f"Additional candidates: "
                f"{additional_total:,}"
            )

            block_results.append({
                "block": block_name,
                "estimated": estimated,
                "generated": generated,
                "truth_in_block": block_truth,
                "new_truth": new_truth,
                "coverage": round(
                    current_pct,
                    4,
                ),
                "status": "GENERATED"
            })

        except Exception as exc:

            print(
                f"\nERROR in block {block_name}:"
            )

            print(
                repr(exc)
            )

            print(
                "Skipping this block."
            )

            block_results.append({
                "block": block_name,
                "estimated": None,
                "generated": 0,
                "new_truth": 0,
                "status": "ERROR",
                "error": repr(exc)
            })

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

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

    v1_pct = (
        v1_covered
        / truth_total
        * 100
        if truth_total
        else 0
    )

    improvement_over_v1 = (
        final_pct - v1_pct
    )

    improvement_over_v2_postal = (
        final_pct - best_postal_pct
    )

    header(
        f"FINAL V4 RESULT: S1 → {prefix}"
    )

    print(
        f"Ground truth             : "
        f"{truth_total:,}"
    )

    print(
        f"V1 covered               : "
        f"{v1_covered:,}"
    )

    print(
        f"V1 coverage              : "
        f"{v1_pct:.2f}%"
    )

    print(
        f"V2-postal covered        : "
        f"{best_after_postal:,}"
    )

    print(
        f"V2-postal coverage       : "
        f"{best_postal_pct:.2f}%"
    )

    print(
        f"V4 covered               : "
        f"{final_covered:,}"
    )

    print(
        f"V4 coverage              : "
        f"{final_pct:.2f}%"
    )

    print(
        f"Improvement vs V1        : "
        f"{improvement_over_v1:+.2f} points"
    )

    print(
        f"Improvement vs V2-postal : "
        f"{improvement_over_v2_postal:+.2f} points"
    )

    print(
        f"Additional V4 candidates: "
        f"{additional_total:,}"
    )

    return {
        "direction": f"S1->{prefix}",
        "truth_total": truth_total,
        "v1_covered": v1_covered,
        "v1_coverage": round(v1_pct, 4),
        "v2_postal_covered": best_after_postal,
        "v2_postal_coverage": round(
            best_postal_pct,
            4
        ),
        "v4_covered": final_covered,
        "v4_coverage": round(
            final_pct,
            4
        ),
        "improvement_vs_v1": round(
            improvement_over_v1,
            4
        ),
        "improvement_vs_v2_postal": round(
            improvement_over_v2_postal,
            4
        ),
        "additional_candidates": additional_total,
        "blocks": block_results,
        "v2_postal_new_truth": postal_new,
    }


# ============================================================
# RUN S2
# ============================================================

header(
    "AMAZON ML CHALLENGE - V4"
)

print(
    f"Memory limit       : {MEMORY_LIMIT}"
)

print(
    f"Threads             : {THREADS}"
)

print(
    f"Max block size      : {MAX_BLOCK_SIZE}"
)

print(
    f"Max block pairs     : {MAX_BLOCK_PAIRS:,}"
)

print(
    f"Max extra candidates: {MAX_TOTAL_ADDITIONAL:,}"
)


overall_start = time.time()


result_s2 = process_direction(
    prefix="S2",
    right_table="s2",
    right_file=S2_FILE,
    v1_file=V1_S2,
    old_postal_file=V2_S2_POSTAL,
    output_dir=S2_OUT,
)


# ============================================================
# RUN S3
# ============================================================

result_s3 = process_direction(
    prefix="S3",
    right_table="s3",
    right_file=S3_FILE,
    v1_file=V1_S3,
    old_postal_file=V2_S3_POSTAL,
    output_dir=S3_OUT,
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = {

    "configuration": {
        "memory_limit": MEMORY_LIMIT,
        "temp_limit": TEMP_LIMIT,
        "threads": THREADS,
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
    OUTPUT_DIR / "v4_manifest.json"
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

header(
    "V4 FINAL SUMMARY"
)

print(
    "\nS1 → S2"
)

print(
    f"V1          : "
    f"{result_s2['v1_coverage']:.2f}%"
)

print(
    f"V2-postal   : "
    f"{result_s2['v2_postal_coverage']:.2f}%"
)

print(
    f"V4          : "
    f"{result_s2['v4_coverage']:.2f}%"
)

print(
    f"V4 vs V1    : "
    f"{result_s2['improvement_vs_v1']:+.2f} points"
)

print(
    f"V4 vs V2    : "
    f"{result_s2['improvement_vs_v2_postal']:+.2f} points"
)


print(
    "\nS1 → S3"
)

print(
    f"V1          : "
    f"{result_s3['v1_coverage']:.2f}%"
)

print(
    f"V2-postal   : "
    f"{result_s3['v2_postal_coverage']:.2f}%"
)

print(
    f"V4          : "
    f"{result_s3['v4_coverage']:.2f}%"
)

print(
    f"V4 vs V1    : "
    f"{result_s3['improvement_vs_v1']:+.2f} points"
)

print(
    f"V4 vs V2    : "
    f"{result_s3['improvement_vs_v2_postal']:+.2f} points"
)


header(
    "OUTPUT"
)

print(
    OUTPUT_DIR
)

print(
    "\nS2 blocks:"
)

print(
    S2_OUT
)

print(
    "\nS3 blocks:"
)

print(
    S3_OUT
)

print(
    "\nManifest:"
)

print(
    manifest_file
)


header(
    "✅ V4 COMPLETED"
)

con.close()
