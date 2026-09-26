from pathlib import Path
import time
import duckdb


# ============================================================
# AMAZON ML CHALLENGE - CANDIDATE GENERATION V2
#
# V1 baseline is preserved.
#
# V2:
#   - Reuses V1 candidates
#   - Adds memory-efficient approximate blocking
#   - Uses DuckDB directly on Parquet
#   - Uses Jaro-Winkler similarity for ranking
#   - Evaluates raw + Top-K recall against ground truth
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

PROCESSED = ROOT / "processed_dataset"
TRAIN_DIR = PROCESSED / "train"

BASELINE_DIR = ROOT / "candidate_output"

OUTPUT_DIR = ROOT / "candidate_output_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = ROOT / "duckdb_tmp_v2"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = OUTPUT_DIR / "candidate_generation_v2.duckdb"

MEMORY_LIMIT = "4GB"
THREADS = 4

# Number of final candidates retained per Source1 entity
TOP_K = 50


# ============================================================
# SOURCE FILES
# ============================================================

SOURCE1_FILE = (
    TRAIN_DIR / "train_source1.parquet"
)

SOURCE2_FILE = (
    TRAIN_DIR / "train_source2.parquet"
)

SOURCE3_FILE = (
    TRAIN_DIR / "train_source3.parquet"
)

GROUND_TRUTH_FILE = (
    TRAIN_DIR / "ground_truth_pairs.parquet"
)

BASELINE_S2_FILE = (
    BASELINE_DIR
    / "train_candidates_s1_s2.parquet"
)

BASELINE_S3_FILE = (
    BASELINE_DIR
    / "train_candidates_s1_s3.parquet"
)


# ============================================================
# APPROXIMATE BLOCKS
#
# The baseline candidates are always retained.
# These blocks add candidates that V1 missed.
# ============================================================

BLOCKS = [
    (
        "name_p6",
        "name_p6",
        2000,
    ),
    (
        "name_s6",
        "name_s6",
        2000,
    ),
    (
        "name_mid6",
        "name_mid6",
        1500,
    ),
    (
        "address_p8",
        "address_p8",
        2500,
    ),
    (
        "address_s8",
        "address_s8",
        2500,
    ),
    (
        "zip6",
        "zip6",
        1500,
    ),
    (
        "name_p4_num",
        "name_p4_num",
        1500,
    ),
    (
        "name_s4_num",
        "name_s4_num",
        1500,
    ),
]


# ============================================================
# UTILITIES
# ============================================================

def sql_path(path: Path) -> str:
    """
    Convert filesystem path to a SQL-safe string.
    """
    return path.as_posix().replace("'", "''")


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_time(start: float, label: str):
    elapsed = time.time() - start
    print(f"{label}: {elapsed:.2f} seconds")


# ============================================================
# CONNECT DUCKDB
# ============================================================

print_header("AMAZON ML - CANDIDATE GENERATION V2")

print(f"Project : {ROOT}")
print(f"Train   : {TRAIN_DIR}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Memory  : {MEMORY_LIMIT}")
print(f"Threads : {THREADS}")
print(f"Top-K   : {TOP_K}")


con = duckdb.connect(str(DB_FILE))

con.execute(
    f"PRAGMA memory_limit='{MEMORY_LIMIT}'"
)

con.execute(
    f"PRAGMA threads={THREADS}"
)

temp_sql_path = sql_path(TEMP_DIR)

con.execute(
    f"PRAGMA temp_directory='{temp_sql_path}'"
)

con.execute(
    "SET preserve_insertion_order=false"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    SOURCE1_FILE,
    SOURCE2_FILE,
    SOURCE3_FILE,
    GROUND_TRUTH_FILE,
    BASELINE_S2_FILE,
    BASELINE_S3_FILE,
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


# ============================================================
# LOAD SOURCE TABLE
# ============================================================

def create_source_table(
    table_name: str,
    parquet_file: Path,
):

    print_header(
        f"CREATING TABLE: {table_name}"
    )

    path = sql_path(parquet_file)

    sql = f"""
    CREATE OR REPLACE TEMP TABLE {table_name} AS

    WITH base AS (

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            name_norm,
            address_norm,
            country_norm,

            regexp_replace(
                coalesce(name_norm, ''),
                '[^a-z0-9]',
                '',
                'g'
            ) AS name_compact,

            regexp_replace(
                coalesce(address_norm, ''),
                '[^a-z0-9]',
                '',
                'g'
            ) AS address_compact,

            regexp_extract(
                coalesce(address_norm, ''),
                '[0-9]{{1,6}}',
                0
            ) AS addr_num

        FROM read_parquet('{path}')
    )

    SELECT
        *,

        substr(
            name_compact,
            1,
            6
        ) AS name_p6,

        right(
            name_compact,
            6
        ) AS name_s6,

        CASE

            WHEN length(name_compact) <= 6
                THEN name_compact

            ELSE substr(
                name_compact,
                CAST(
                    floor(
                        (
                            length(name_compact) - 6
                        ) / 2
                    ) + 1
                    AS BIGINT
                ),
                6
            )

        END AS name_mid6,

        substr(
            address_compact,
            1,
            8
        ) AS address_p8,

        right(
            address_compact,
            8
        ) AS address_s8,

        regexp_extract(
            coalesce(address_norm, ''),
            '[0-9]{{5,6}}',
            0
        ) AS zip6,

        CASE

            WHEN addr_num <> ''
                THEN
                    substr(
                        name_compact,
                        1,
                        4
                    )
                    || '|'
                    || addr_num

            ELSE ''

        END AS name_p4_num,

        CASE

            WHEN addr_num <> ''
                THEN
                    right(
                        name_compact,
                        4
                    )
                    || '|'
                    || addr_num

            ELSE ''

        END AS name_s4_num

    FROM base
    """

    con.execute(sql)

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
# SOURCE 1
# ============================================================

create_source_table(
    "s1",
    SOURCE1_FILE,
)


# ============================================================
# GROUND TRUTH
# ============================================================

print_header("LOADING GROUND TRUTH")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE ground_truth AS
    SELECT
        source1_entity_id,
        matched_entity_id
    FROM read_parquet(
        '{sql_path(GROUND_TRUTH_FILE)}'
    )
    """
)

gt_total = con.execute(
    """
    SELECT COUNT(*)
    FROM ground_truth
    """
).fetchone()[0]

print(
    f"Total ground-truth pairs: {gt_total:,}"
)


# ============================================================
# GET BASELINE COVERAGE
# ============================================================

def baseline_coverage(
    baseline_file: Path,
    prefix: str,
):

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE baseline_tmp AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM read_parquet(
            '{sql_path(baseline_file)}'
        )
        """
    )

    truth_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ground_truth
        WHERE matched_entity_id
              LIKE '{prefix}-%'
        """
    ).fetchone()[0]

    found = con.execute(
        f"""
        SELECT COUNT(*)

        FROM ground_truth g

        INNER JOIN baseline_tmp b

        ON
            g.source1_entity_id
            = b.source1_entity_id

        AND
            g.matched_entity_id
            = b.candidate_entity_id

        WHERE g.matched_entity_id
              LIKE '{prefix}-%'
        """
    ).fetchone()[0]

    coverage = (
        found / truth_count * 100
        if truth_count
        else 0
    )

    return truth_count, found, coverage


print_header("V1 BASELINE COVERAGE")

truth_s2, found_s2, base_cov_s2 = (
    baseline_coverage(
        BASELINE_S2_FILE,
        "S2",
    )
)

print(
    f"S1 → S2 truth pairs : {truth_s2:,}"
)

print(
    f"V1 covered          : {found_s2:,}"
)

print(
    f"V1 coverage         : {base_cov_s2:.2f}%"
)


truth_s3, found_s3, base_cov_s3 = (
    baseline_coverage(
        BASELINE_S3_FILE,
        "S3",
    )
)

print(
    f"\nS1 → S3 truth pairs : {truth_s3:,}"
)

print(
    f"V1 covered          : {found_s3:,}"
)

print(
    f"V1 coverage         : {base_cov_s3:.2f}%"
)


# ============================================================
# BLOCK SQL
# ============================================================

def make_block_query(
    right_table: str,
    block_name: str,
    block_column: str,
    max_block_size: int,
):

    sql = f"""

    WITH left_base AS (

        SELECT
            entity_id,
            country_norm,
            {block_column} AS block_key

        FROM s1

        WHERE
            {block_column} <> ''

    ),

    left_blocks AS (

        SELECT
            *,
            COUNT(*) OVER (
                PARTITION BY
                    country_norm,
                    block_key
            ) AS block_n

        FROM left_base

    ),

    right_base AS (

        SELECT
            entity_id,
            country_norm,
            {block_column} AS block_key

        FROM {right_table}

        WHERE
            {block_column} <> ''

    ),

    right_blocks AS (

        SELECT
            *,
            COUNT(*) OVER (
                PARTITION BY
                    country_norm,
                    block_key
            ) AS block_n

        FROM right_base

    )

    SELECT

        l.entity_id
            AS source1_entity_id,

        r.entity_id
            AS candidate_entity_id,

        1
            AS block_hit,

        '{block_name}'
            AS block_name

    FROM left_blocks l

    INNER JOIN right_blocks r

        ON
            l.country_norm
            = r.country_norm

        AND
            l.block_key
            = r.block_key

    WHERE
        l.block_n <= {max_block_size}

    AND
        r.block_n <= {max_block_size}
    """

    return sql


# ============================================================
# PROCESS ONE SOURCE PAIR
# ============================================================

def process_pair(
    right_table: str,
    right_file: Path,
    baseline_file: Path,
    truth_prefix: str,
    output_name: str,
):

    pair_start = time.time()

    print_header(
        f"V2 PROCESSING: S1 → {truth_prefix}"
    )

    # --------------------------------------------------------
    # CREATE RIGHT TABLE
    # --------------------------------------------------------

    create_source_table(
        right_table,
        right_file,
    )

    # --------------------------------------------------------
    # LOAD V1 BASELINE CANDIDATES
    # --------------------------------------------------------

    print_header(
        "LOADING V1 CANDIDATES"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v1_candidates AS

        SELECT
            source1_entity_id,
            candidate_entity_id,

            match_keys
                AS base_exact_keys,

            exact_key_count
                AS evidence_count,

            TRUE
                AS from_v1

        FROM read_parquet(
            '{sql_path(baseline_file)}'
        )
        """
    )

    v1_count = con.execute(
        """
        SELECT COUNT(*)
        FROM v1_candidates
        """
    ).fetchone()[0]

    print(
        f"V1 candidates: {v1_count:,}"
    )

    # --------------------------------------------------------
    # APPROXIMATE BLOCK CANDIDATES
    # --------------------------------------------------------

    print_header(
        "GENERATING APPROXIMATE BLOCK CANDIDATES"
    )

    block_queries = []

    for (
        block_name,
        block_column,
        max_block_size,
    ) in BLOCKS:

        print(
            f"Preparing block: "
            f"{block_name} "
            f"(max block = {max_block_size:,})"
        )

        block_queries.append(
            make_block_query(
                right_table,
                block_name,
                block_column,
                max_block_size,
            )
        )

    union_sql = "\nUNION ALL\n".join(
        f"({q})" for q in block_queries
    )

    approx_start = time.time()

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE approx_candidates_raw AS

        {union_sql}
        """
    )

    approx_raw_count = con.execute(
        """
        SELECT COUNT(*)
        FROM approx_candidates_raw
        """
    ).fetchone()[0]

    print(
        f"\nApproximate raw candidates: "
        f"{approx_raw_count:,}"
    )

    print_time(
        approx_start,
        "Approximate block generation",
    )

    # --------------------------------------------------------
    # COMBINE V1 + APPROXIMATE CANDIDATES
    # --------------------------------------------------------

    print_header(
        "COMBINING V1 + V2 CANDIDATES"
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE raw_candidates AS

        WITH combined AS (

            SELECT

                source1_entity_id,
                candidate_entity_id,

                base_exact_keys,

                evidence_count,

                from_v1

            FROM v1_candidates

            UNION ALL

            SELECT

                source1_entity_id,
                candidate_entity_id,

                ''
                    AS base_exact_keys,

                block_hit
                    AS evidence_count,

                FALSE
                    AS from_v1

            FROM approx_candidates_raw
        )

        SELECT

            source1_entity_id,

            candidate_entity_id,

            MAX(
                base_exact_keys
            ) AS base_exact_keys,

            SUM(
                evidence_count
            ) AS evidence_count,

            BOOL_OR(
                from_v1
            ) AS from_v1

        FROM combined

        GROUP BY

            source1_entity_id,
            candidate_entity_id
        """
    )

    raw_count = con.execute(
        """
        SELECT COUNT(*)
        FROM raw_candidates
        """
    ).fetchone()[0]

    raw_source1 = con.execute(
        """
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM raw_candidates
        """
    ).fetchone()[0]

    print(
        f"V2 raw candidates: "
        f"{raw_count:,}"
    )

    print(
        f"Source1 entities with candidates: "
        f"{raw_source1:,}"
    )

    # --------------------------------------------------------
    # RAW V2 COVERAGE
    # --------------------------------------------------------

    print_header(
        "RAW V2 GROUND-TRUTH COVERAGE"
    )

    raw_found = con.execute(
        f"""
        SELECT COUNT(*)

        FROM ground_truth g

        INNER JOIN raw_candidates c

        ON
            g.source1_entity_id
            = c.source1_entity_id

        AND
            g.matched_entity_id
            = c.candidate_entity_id

        WHERE
            g.matched_entity_id
            LIKE '{truth_prefix}-%'
        """
    ).fetchone()[0]

    raw_coverage = (
        raw_found
        / truth_s2 * 100
        if truth_prefix == "S2" and truth_s2
        else raw_found
        / truth_s3 * 100
        if truth_prefix == "S3" and truth_s3
        else 0
    )

    baseline_cov = (
        base_cov_s2
        if truth_prefix == "S2"
        else base_cov_s3
    )

    print(
        f"V1 coverage : "
        f"{baseline_cov:.2f}%"
    )

    print(
        f"V2 raw      : "
        f"{raw_coverage:.2f}%"
    )

    print(
        f"Additional truth pairs found: "
        f"{raw_found - (found_s2 if truth_prefix == 'S2' else found_s3):,}"
    )

    # --------------------------------------------------------
    # SCORE CANDIDATES
    # --------------------------------------------------------

    print_header(
        "V2 FUZZY SCORING"
    )

    score_start = time.time()

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE scored AS

        WITH similarities AS (

            SELECT

                c.source1_entity_id,

                c.candidate_entity_id,

                c.base_exact_keys,

                c.evidence_count,

                c.from_v1,

                s1.business_name
                    AS source1_business_name,

                r.business_name
                    AS candidate_business_name,

                s1.business_address
                    AS source1_business_address,

                r.business_address
                    AS candidate_business_address,

                s1.country
                    AS source1_country,

                r.country
                    AS candidate_country,

                s1.name_norm
                    AS source1_name_norm,

                r.name_norm
                    AS candidate_name_norm,

                s1.address_norm
                    AS source1_address_norm,

                r.address_norm
                    AS candidate_address_norm,

                jaro_winkler_similarity(

                    coalesce(
                        s1.name_norm,
                        ''
                    ),

                    coalesce(
                        r.name_norm,
                        ''
                    )

                ) AS name_similarity,

                CASE

                    WHEN
                        coalesce(
                            s1.address_norm,
                            ''
                        ) <> ''

                    AND
                        coalesce(
                            r.address_norm,
                            ''
                        ) <> ''

                    THEN

                        jaro_winkler_similarity(

                            s1.address_norm,

                            r.address_norm
                        )

                    ELSE 0

                END AS address_similarity,

                CASE

                    WHEN
                        coalesce(
                            s1.name_norm,
                            ''
                        ) <> ''

                    AND
                        coalesce(
                            r.name_norm,
                            ''
                        ) <> ''

                    THEN

                        jaccard(

                            s1.name_norm,

                            r.name_norm
                        )

                    ELSE 0

                END AS name_jaccard,

                CASE

                    WHEN
                        coalesce(
                            s1.address_norm,
                            ''
                        ) <> ''

                    AND
                        coalesce(
                            r.address_norm,
                            ''
                        ) <> ''

                    THEN

                        jaccard(

                            s1.address_norm,

                            r.address_norm
                        )

                    ELSE 0

                END AS address_jaccard

            FROM raw_candidates c

            INNER JOIN s1

                ON
                    c.source1_entity_id
                    = s1.entity_id

            INNER JOIN {right_table} r

                ON
                    c.candidate_entity_id
                    = r.entity_id
        )

        SELECT

            *,

            LEAST(

                1.0,

                CASE

                    WHEN
                        source1_address_norm <> ''

                    AND
                        candidate_address_norm <> ''

                    THEN

                        (
                            0.45
                            * name_similarity

                            +

                            0.40
                            * address_similarity

                            +

                            0.10
                            * name_jaccard

                            +

                            0.05
                            * address_jaccard
                        )

                    ELSE

                        (
                            0.75
                            * name_similarity

                            +

                            0.25
                            * name_jaccard
                        )

                END

                +

                CASE

                    WHEN
                        source1_name_norm
                        <> ''

                    AND
                        source1_name_norm
                        = candidate_name_norm

                    THEN 0.10

                    ELSE 0

                END

                +

                CASE

                    WHEN
                        source1_address_norm
                        <> ''

                    AND
                        source1_address_norm
                        = candidate_address_norm

                    THEN 0.15

                    ELSE 0

                END

                +

                (
                    0.01
                    * LEAST(
                        evidence_count,
                        5
                    )
                )

            ) AS match_score

        FROM similarities
        """
    )

    scored_count = con.execute(
        """
        SELECT COUNT(*)
        FROM scored
        """
    ).fetchone()[0]

    print(
        f"Scored candidates: "
        f"{scored_count:,}"
    )

    print_time(
        score_start,
        "Fuzzy scoring",
    )

    # --------------------------------------------------------
    # RANK PER SOURCE1
    # --------------------------------------------------------

    print_header(
        f"RANKING TOP {TOP_K}"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE ranked AS

        SELECT

            *,

            ROW_NUMBER() OVER (

                PARTITION BY
                    source1_entity_id

                ORDER BY

                    match_score DESC,

                    CASE

                        WHEN
                            source1_address_norm <> ''

                        AND
                            source1_address_norm
                            = candidate_address_norm

                        THEN 1

                        ELSE 0

                    END DESC,

                    CASE

                        WHEN
                            source1_name_norm <> ''

                        AND
                            source1_name_norm
                            = candidate_name_norm

                        THEN 1

                        ELSE 0

                    END DESC,

                    evidence_count DESC,

                    candidate_entity_id

            ) AS rank_in_source1

        FROM scored
        """
    )

    # --------------------------------------------------------
    # TOP-K COVERAGE
    # --------------------------------------------------------

    print_header(
        "TOP-K GROUND-TRUTH COVERAGE"
    )

    topk_found = con.execute(
        f"""
        SELECT COUNT(*)

        FROM ground_truth g

        INNER JOIN ranked r

        ON
            g.source1_entity_id
            = r.source1_entity_id

        AND
            g.matched_entity_id
            = r.candidate_entity_id

        WHERE
            g.matched_entity_id
            LIKE '{truth_prefix}-%'

        AND
            r.rank_in_source1 <= {TOP_K}
        """
    ).fetchone()[0]

    truth_total = (
        truth_s2
        if truth_prefix == "S2"
        else truth_s3
    )

    topk_coverage = (
        topk_found
        / truth_total
        * 100
        if truth_total
        else 0
    )

    print(
        f"V1 coverage     : "
        f"{baseline_cov:.2f}%"
    )

    print(
        f"V2 raw coverage : "
        f"{raw_coverage:.2f}%"
    )

    print(
        f"V2 Top-{TOP_K}    : "
        f"{topk_coverage:.2f}%"
    )

    print(
        f"\nTruth pairs covered by Top-{TOP_K}: "
        f"{topk_found:,} / {truth_total:,}"
    )

    # --------------------------------------------------------
    # SAVE TOP-K CANDIDATES
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / output_name
    )

    if output_file.exists():
        output_file.unlink()

    print_header(
        "SAVING TOP-K CANDIDATES"
    )

    con.execute(
        f"""
        COPY (

            SELECT

                source1_entity_id,

                candidate_entity_id,

                match_score,

                name_similarity,

                address_similarity,

                name_jaccard,

                address_jaccard,

                evidence_count,

                from_v1,

                base_exact_keys,

                source1_business_name,

                candidate_business_name,

                source1_business_address,

                candidate_business_address,

                source1_country,

                candidate_country,

                rank_in_source1

            FROM ranked

            WHERE
                rank_in_source1 <= {TOP_K}

        )

        TO '{sql_path(output_file)}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    saved_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{sql_path(output_file)}'
        )
        """
    ).fetchone()[0]

    print(
        f"Saved rows: {saved_count:,}"
    )

    print(
        f"Output: {output_file}"
    )

    # --------------------------------------------------------
    # SAMPLE RESULTS
    # --------------------------------------------------------

    print_header(
        "TOP CANDIDATE EXAMPLES"
    )

    sample = con.execute(
        f"""
        SELECT

            source1_entity_id,

            candidate_entity_id,

            ROUND(
                match_score,
                4
            ) AS score,

            ROUND(
                name_similarity,
                4
            ) AS name_sim,

            ROUND(
                address_similarity,
                4
            ) AS address_sim,

            evidence_count,

            from_v1,

            source1_business_name,

            candidate_business_name

        FROM ranked

        WHERE
            rank_in_source1 <= {TOP_K}

        ORDER BY

            match_score DESC

        LIMIT 20
        """
    ).fetchdf()

    print(sample.to_string(index=False))

    print_time(
        pair_start,
        f"Total S1 → {truth_prefix} processing",
    )

    # --------------------------------------------------------
    # CLEAN TEMP TABLES
    # --------------------------------------------------------

    con.execute(
        "DROP TABLE IF EXISTS v1_candidates"
    )

    con.execute(
        "DROP TABLE IF EXISTS approx_candidates_raw"
    )

    con.execute(
        "DROP TABLE IF EXISTS raw_candidates"
    )

    con.execute(
        "DROP TABLE IF EXISTS scored"
    )

    con.execute(
        "DROP TABLE IF EXISTS ranked"
    )

    con.execute(
        f"DROP TABLE IF EXISTS {right_table}"
    )

    return {
        "prefix": truth_prefix,
        "v1_coverage": baseline_cov,
        "raw_coverage": raw_coverage,
        "topk_coverage": topk_coverage,
        "v1_candidates": v1_count,
        "v2_raw_candidates": raw_count,
        "topk_candidates": saved_count,
    }


# ============================================================
# RUN S1 → S2
# ============================================================

result_s2 = process_pair(
    right_table="s2",
    right_file=SOURCE2_FILE,
    baseline_file=BASELINE_S2_FILE,
    truth_prefix="S2",
    output_name=(
        "train_candidates_v2_s1_s2_top50.parquet"
    ),
)


# ============================================================
# RUN S1 → S3
# ============================================================

result_s3 = process_pair(
    right_table="s3",
    right_file=SOURCE3_FILE,
    baseline_file=BASELINE_S3_FILE,
    truth_prefix="S3",
    output_name=(
        "train_candidates_v2_s1_s3_top50.parquet"
    ),
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "V1 vs V2 FINAL COMPARISON"
)

print(
    "\nS1 → S2"
)

print(
    f"V1 candidates : "
    f"{result_s2['v1_candidates']:,}"
)

print(
    f"V2 raw        : "
    f"{result_s2['v2_raw_candidates']:,}"
)

print(
    f"V1 coverage   : "
    f"{result_s2['v1_coverage']:.2f}%"
)

print(
    f"V2 raw        : "
    f"{result_s2['raw_coverage']:.2f}%"
)

print(
    f"V2 Top-{TOP_K}      : "
    f"{result_s2['topk_coverage']:.2f}%"
)


print(
    "\nS1 → S3"
)

print(
    f"V1 candidates : "
    f"{result_s3['v1_candidates']:,}"
)

print(
    f"V2 raw        : "
    f"{result_s3['v2_raw_candidates']:,}"
)

print(
    f"V1 coverage   : "
    f"{result_s3['v1_coverage']:.2f}%"
)

print(
    f"V2 raw        : "
    f"{result_s3['raw_coverage']:.2f}%"
)

print(
    f"V2 Top-{TOP_K}      : "
    f"{result_s3['topk_coverage']:.2f}%"
)


print_header(
    "OUTPUT FILES"
)

print(
    OUTPUT_DIR
    / "train_candidates_v2_s1_s2_top50.parquet"
)

print(
    OUTPUT_DIR
    / "train_candidates_v2_s1_s3_top50.parquet"
)


print_header(
    "✅ V2 CANDIDATE GENERATION COMPLETED"
)


con.close()
