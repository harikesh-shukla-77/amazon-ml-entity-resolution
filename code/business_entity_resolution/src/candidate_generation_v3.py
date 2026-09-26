from pathlib import Path
import json
import time

import duckdb


# ============================================================
# AMAZON ML CHALLENGE
# V3 - TARGETED / SAFE BLOCKING
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

V2_DIR = ROOT / "candidate_output_v2_safe"

V3_DIR = ROOT / "candidate_output_v3"

V3_S2_DIR = V3_DIR / "s1_s2"
V3_S3_DIR = V3_DIR / "s1_s3"

TEMP_DIR = ROOT / "duckdb_tmp_v3"

V3_DIR.mkdir(parents=True, exist_ok=True)
V3_S2_DIR.mkdir(parents=True, exist_ok=True)
V3_S3_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFETY SETTINGS
# ============================================================

MEMORY_LIMIT = "4GB"
TEMP_LIMIT = "8GB"
THREADS = 2

# Smaller block cap than V2
MAX_BLOCK_SIZE = 150

# Maximum materialized pairs per block
MAX_BLOCK_PAIRS = 6_000_000

# Maximum additional pairs per direction
MAX_TOTAL_ADDITIONAL = 20_000_000


# ============================================================
# BLOCKS
# ============================================================

BLOCKS = [

    # --------------------------------------------------------
    # 1. Remove common legal/business suffixes
    # --------------------------------------------------------

    (
        "name_core_country",
        "name_core"
    ),

    # --------------------------------------------------------
    # 2. Core-name prefix + suffix
    # Helps small edits in the middle of names
    # --------------------------------------------------------

    (
        "name_core_ps4",
        """
        substr(name_core_compact, 1, 4)
        || '|'
        || right(name_core_compact, 4)
        """
    ),

    # --------------------------------------------------------
    # 3. Original name prefix + suffix
    # --------------------------------------------------------

    (
        "name_ps4",
        """
        substr(name_compact, 1, 4)
        || '|'
        || right(name_compact, 4)
        """
    ),

    # --------------------------------------------------------
    # 4. Name prefix + address number
    # --------------------------------------------------------

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
        """
    ),

    # --------------------------------------------------------
    # 5. Name suffix + address number
    # --------------------------------------------------------

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
        """
    ),

    # --------------------------------------------------------
    # 6. Canonical address prefix
    # --------------------------------------------------------

    (
        "address_core_p7",
        "substr(address_compact_canon, 1, 7)"
    ),

    # --------------------------------------------------------
    # 7. Canonical address suffix
    # --------------------------------------------------------

    (
        "address_core_s7",
        "right(address_compact_canon, 7)"
    ),

    # --------------------------------------------------------
    # 8. Address number + name prefix
    # --------------------------------------------------------

    (
        "num_name_p4",
        """
        CASE
            WHEN addr_num <> ''
            THEN
                addr_num
                || '|'
                || substr(name_compact, 1, 4)
            ELSE ''
        END
        """
    ),
]


# ============================================================
# HELPERS
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

DB_FILE = V3_DIR / "candidate_generation_v3.duckdb"

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
# FILE CHECK
# ============================================================

required = [
    SOURCE1,
    SOURCE2,
    SOURCE3,
    GROUND_TRUTH,
    V1_S2,
    V1_S3,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


# ============================================================
# BASE RELATION
# ============================================================

def relation(path: Path) -> str:

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

            regexp_replace(

                regexp_replace(

                    regexp_replace(

                        regexp_replace(

                            regexp_replace(

                                regexp_replace(

                                    regexp_replace(

                                        regexp_replace(

                                            regexp_replace(

                                                lower(
                                                    COALESCE(
                                                        name_norm,
                                                        ''
                                                    )
                                                ),

                                                ' (incorporated|inc|corporation|corp)$',
                                                ''

                                            ),

                                            ' (company|co)$',
                                            ''

                                        ),

                                        ' (limited|ltd)$',
                                        ''

                                    ),

                                    ' (llc|plc)$',
                                    ''

                                ),

                                ' (private|pvt)$',
                                ''

                            ),

                            ' (sarl|sas)$',
                            ''

                        ),

                        ' (gmbh|bv)$',
                        ''

                    ),

                    '[^a-z0-9 ]',
                    ' ',
                    'g'

                ),

                '\\s+',
                ' ',
                'g'

            ) AS name_core,

            regexp_replace(

                regexp_replace(

                    regexp_replace(

                        regexp_replace(

                            regexp_replace(

                                regexp_replace(

                                    regexp_replace(

                                        regexp_replace(

                                            regexp_replace(

                                                regexp_replace(

                                                    regexp_replace(

                                                        lower(
                                                            COALESCE(
                                                                address_norm,
                                                                ''
                                                            )
                                                        ),

                                                        ' street',
                                                        ' st',
                                                        'g'
                                                    ),

                                                    ' road',
                                                    ' rd',
                                                    'g'
                                                ),

                                                ' avenue',
                                                ' ave',
                                                'g'
                                            ),

                                            ' boulevard',
                                            ' blvd',
                                            'g'
                                        ),

                                        ' drive',
                                        ' dr',
                                        'g'
                                    ),

                                    ' lane',
                                    ' ln',
                                    'g'
                                ),

                                ' highway',
                                ' hwy',
                                'g'
                            ),

                            ' parkway',
                            ' pkwy',
                            'g'
                        ),

                        ' place',
                        ' pl',
                        'g'
                    ),

                    ' court',
                    ' ct',
                    'g'
                ),

                ' apartment',
                ' apt',
                'g'
            ) AS address_canon

        FROM read_parquet('{p}')
    )
    """


# ============================================================
# TEMP TABLE SOURCE 1
# ============================================================

header("CREATING SOURCE 1")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE s1 AS

    SELECT

        *,

        regexp_replace(
            name_core,
            '[^a-z0-9]',
            '',
            'g'
        ) AS name_core_compact,

        regexp_replace(
            address_canon,
            '[^a-z0-9]',
            '',
            'g'
        ) AS address_compact_canon

    FROM {relation(SOURCE1)}
    """
)

s1_count = con.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

print(
    f"Source1 rows: {s1_count:,}"
)


# ============================================================
# GROUND TRUTH
# ============================================================

header("GROUND TRUTH")

gt_total = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_parquet(
        '{sql_path(GROUND_TRUTH)}'
    )
    """
).fetchone()[0]

print(
    f"Total GT pairs: {gt_total:,}"
)


# ============================================================
# INITIAL V1 COVERAGE
# ============================================================

def initialize_coverage(
    prefix: str,
    v1_file: Path
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

    covered = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{sql_path(GROUND_TRUTH)}'
        )
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
        f"GT pairs : {total:,}"
    )

    print(
        f"V1 covered: {covered:,}"
    )

    print(
        f"V1 coverage: {pct:.2f}%"
    )

    return table, int(total), int(covered)


# ============================================================
# ESTIMATE BLOCK
# ============================================================

def estimate_block(
    target_file: Path,
    key_expr: str
):

    left_sql = """
        SELECT
            country_norm,
            {key} AS block_key,
            COUNT(*) AS n
        FROM s1
        WHERE country_norm <> ''
        GROUP BY country_norm, block_key
        HAVING block_key <> ''
           AND COUNT(*) <= {max_size}
    """.format(
        key=key_expr,
        max_size=MAX_BLOCK_SIZE
    )

    right_sql = """
        SELECT
            country_norm,
            {key} AS block_key,
            COUNT(*) AS n
        FROM {relation}
        WHERE country_norm <> ''
        GROUP BY country_norm, block_key
        HAVING block_key <> ''
           AND COUNT(*) <= {max_size}
    """.format(
        key=key_expr,
        relation=relation(target_file),
        max_size=MAX_BLOCK_SIZE
    )

    query = f"""
    WITH

    l AS (
        {left_sql}
    ),

    r AS (
        {right_sql}
    )

    SELECT
        COALESCE(
            SUM(
                l.n::BIGINT
                * r.n::BIGINT
            ),
            0
        )

    FROM l

    INNER JOIN r

        ON
            l.country_norm
            = r.country_norm

        AND
            l.block_key
            = r.block_key
    """

    value = con.execute(query).fetchone()[0]

    return int(value or 0)


# ============================================================
# GENERATE BLOCK
# ============================================================

def generate_block(
    target_file: Path,
    prefix: str,
    block_name: str,
    key_expr: str,
    output_file: Path
):

    right_rel = relation(target_file)

    # We enrich the right side with V3 keys.
    right_sql = f"""
    (
        SELECT

            *,

            regexp_replace(
                name_core,
                '[^a-z0-9]',
                '',
                'g'
            ) AS name_core_compact,

            regexp_replace(
                address_canon,
                '[^a-z0-9]',
                '',
                'g'
            ) AS address_compact_canon

        FROM {right_rel}
    )
    """

    query = f"""
    WITH

    l AS (

        SELECT

            entity_id,
            country_norm,
            name_core_compact,
            address_compact_canon,
            name_compact,
            addr_num,

            {key_expr}
                AS block_key

        FROM s1

        WHERE
            country_norm <> ''

    ),

    l_counts AS (

        SELECT

            country_norm,
            block_key,
            COUNT(*) AS block_n

        FROM l

        WHERE
            block_key <> ''

        GROUP BY

            country_norm,
            block_key

        HAVING
            COUNT(*) <= {MAX_BLOCK_SIZE}
    ),

    r AS (

        SELECT

            entity_id,
            country_norm,
            name_core_compact,
            address_compact_canon,
            name_compact,
            addr_num,

            {key_expr}
                AS block_key

        FROM {right_sql}

        WHERE
            country_norm <> ''

    ),

    r_counts AS (

        SELECT

            country_norm,
            block_key,
            COUNT(*) AS block_n

        FROM r

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

    FROM l

    INNER JOIN l_counts lc

        ON
            l.country_norm
            = lc.country_norm

        AND
            l.block_key
            = lc.block_key

    INNER JOIN r

        ON
            l.country_norm
            = r.country_norm

        AND
            l.block_key
            = r.block_key

    INNER JOIN r_counts rc

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
        f"\nGenerating {block_name}..."
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

    print(
        f"Generated: {count:,}"
    )

    print(
        f"Time: {time.time() - start:.2f}s"
    )

    return int(count)


# ============================================================
# UPDATE COVERAGE
# ============================================================

def update_coverage(
    covered_table: str,
    candidate_file: Path,
    prefix: str
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

        FROM read_parquet(?) c

        INNER JOIN read_parquet(?) g

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

    return (
        int(block_truth),
        int(after - before)
    )


# ============================================================
# PROCESS DIRECTION
# ============================================================

def process_direction(
    target_file: Path,
    prefix: str,
    v1_file: Path,
    output_dir: Path
):

    table, truth_total, v1_covered = (
        initialize_coverage(
            prefix,
            v1_file
        )
    )

    initial_pct = (
        v1_covered
        / truth_total
        * 100
        if truth_total
        else 0
    )

    additional_total = 0

    audit = []

    for block_name, key_expr in BLOCKS:

        header(
            f"V3 BLOCK: {prefix} / {block_name}"
        )

        if additional_total >= MAX_TOTAL_ADDITIONAL:

            print(
                "Maximum additional candidate budget reached."
            )

            break

        try:

            estimated = estimate_block(
                target_file,
                key_expr
            )

            print(
                f"Estimated pairs: "
                f"{estimated:,}"
            )

            if estimated == 0:

                print(
                    "SKIP: no pairs."
                )

                continue

            if estimated > MAX_BLOCK_PAIRS:

                print(
                    "SKIP: block too large."
                )

                audit.append({
                    "block": block_name,
                    "estimated": estimated,
                    "generated": 0,
                    "new_truth": 0,
                    "status": "SKIPPED_TOO_LARGE"
                })

                continue

            if (
                additional_total
                + estimated
                > MAX_TOTAL_ADDITIONAL
            ):

                print(
                    "SKIP: total candidate budget exceeded."
                )

                audit.append({
                    "block": block_name,
                    "estimated": estimated,
                    "generated": 0,
                    "new_truth": 0,
                    "status": "SKIPPED_BUDGET"
                })

                continue

            output_file = (
                output_dir
                / f"{prefix.lower()}_{block_name}.parquet"
            )

            generated = generate_block(
                target_file,
                prefix,
                block_name,
                key_expr,
                output_file
            )

            additional_total += generated

            truth_found, new_truth = (
                update_coverage(
                    table,
                    output_file,
                    prefix
                )
            )

            current_covered = con.execute(
                f"""
                SELECT COUNT(*)
                FROM {table}
                """
            ).fetchone()[0]

            current_pct = (
                current_covered
                / truth_total
                * 100
                if truth_total
                else 0
            )

            improvement = (
                current_pct
                - initial_pct
            )

            print(
                f"Truth pairs in block : "
                f"{truth_found:,}"
            )

            print(
                f"New truth pairs       : "
                f"{new_truth:,}"
            )

            print(
                f"Combined covered      : "
                f"{current_covered:,}"
            )

            print(
                f"Combined coverage     : "
                f"{current_pct:.2f}%"
            )

            print(
                f"V3 improvement        : "
                f"{improvement:+.2f} points"
            )

            audit.append({
                "block": block_name,
                "estimated": estimated,
                "generated": generated,
                "truth_in_block": truth_found,
                "new_truth": new_truth,
                "coverage_pct": round(
                    current_pct,
                    4
                ),
                "status": "GENERATED"
            })

        except Exception as exc:

            print(
                f"\nERROR in {block_name}:"
            )

            print(
                repr(exc)
            )

            print(
                "This block will be skipped."
            )

            audit.append({
                "block": block_name,
                "estimated": None,
                "generated": 0,
                "new_truth": 0,
                "status": "ERROR",
                "error": repr(exc)
            })

    final_covered = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    final_pct = (
        final_covered
        / truth_total
        * 100
        if truth_total
        else 0
    )

    improvement = (
        final_pct
        - initial_pct
    )

    print(
        "\n" + "=" * 100
    )

    print(
        f"FINAL V3 RESULT: S1 → {prefix}"
    )

    print(
        "=" * 100
    )

    print(
        f"Ground truth : {truth_total:,}"
    )

    print(
        f"V1 covered   : {v1_covered:,}"
    )

    print(
        f"V1 coverage  : {initial_pct:.2f}%"
    )

    print(
        f"V3 covered   : {final_covered:,}"
    )

    print(
        f"V3 coverage  : {final_pct:.2f}%"
    )

    print(
        f"Improvement  : {improvement:+.2f} points"
    )

    print(
        f"Extra candidates: {additional_total:,}"
    )

    return {
        "direction": f"S1->{prefix}",
        "truth_total": truth_total,
        "v1_covered": v1_covered,
        "v1_coverage": round(
            initial_pct,
            4
        ),
        "v3_covered": final_covered,
        "v3_coverage": round(
            final_pct,
            4
        ),
        "improvement_points": round(
            improvement,
            4
        ),
        "additional_candidates": additional_total,
        "blocks": audit
    }


# ============================================================
# RUN
# ============================================================

header(
    "AMAZON ML CHALLENGE - V3"
)

print(
    f"Memory limit       : {MEMORY_LIMIT}"
)

print(
    f"Threads            : {THREADS}"
)

print(
    f"Max block size     : {MAX_BLOCK_SIZE}"
)

print(
    f"Max block pairs    : {MAX_BLOCK_PAIRS:,}"
)

print(
    f"Max extra pairs    : {MAX_TOTAL_ADDITIONAL:,}"
)


overall_start = time.time()


result_s2 = process_direction(
    SOURCE2,
    "S2",
    V1_S2,
    V3_S2_DIR
)


result_s3 = process_direction(
    SOURCE3,
    "S3",
    V1_S3,
    V3_S3_DIR
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = {
    "config": {
        "memory_limit": MEMORY_LIMIT,
        "threads": THREADS,
        "max_block_size": MAX_BLOCK_SIZE,
        "max_block_pairs": MAX_BLOCK_PAIRS,
        "max_total_additional": MAX_TOTAL_ADDITIONAL
    },
    "results": {
        "S1_to_S2": result_s2,
        "S1_to_S3": result_s3
    },
    "elapsed_seconds": round(
        time.time() - overall_start,
        2
    )
}


manifest_file = (
    V3_DIR / "v3_manifest.json"
)

manifest_file.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False
    )
)


# ============================================================
# FINAL
# ============================================================

header(
    "V3 FINAL SUMMARY"
)

print(
    "\nS1 → S2"
)

print(
    f"V1: {result_s2['v1_coverage']:.2f}%"
)

print(
    f"V3: {result_s2['v3_coverage']:.2f}%"
)

print(
    f"Change: "
    f"{result_s2['improvement_points']:+.2f} points"
)


print(
    "\nS1 → S3"
)

print(
    f"V1: {result_s3['v1_coverage']:.2f}%"
)

print(
    f"V3: {result_s3['v3_coverage']:.2f}%"
)

print(
    f"Change: "
    f"{result_s3['improvement_points']:+.2f} points"
)


print(
    "\nOutput:"
)

print(
    V3_DIR
)

print(
    "\nManifest:"
)

print(
    manifest_file
)


header(
    "✅ V3 COMPLETED"
)

con.close()