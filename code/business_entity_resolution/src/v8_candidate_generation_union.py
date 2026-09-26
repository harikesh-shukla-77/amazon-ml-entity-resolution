from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import duckdb

# ============================================================
# AMAZON ML CHALLENGE - V8 CANDIDATE GENERATION
# FINAL 4-BLOCK DEVELOPMENT ARCHITECTURE
#
# Purpose:
#   Generate the selected deterministic blocking union discovered
#   by V1-V7, without reading GT/holdout during generation.
#
# S2 blocks:
#   1) num_addr_name_prefix2
#   2) addr_last2_postal
#   3) name_last_token_postal
#   4) name_first1_postal
#
# S3 blocks:
#   1) num_addr_name_prefix2
#   2) name_last_token_postal
#   3) addr_last2_postal
#   4) name_first1_postal
#
# Safety:
#   - NEVER reads prospective holdout
#   - NEVER reads ground truth
#   - NEVER modifies V6/V7/V8/V9/V10 outputs
#   - Writes only candidate_output_v8/
#   - Per-block hash sharding, V9.1-style
#   - Exact V8 union is deduplicated by source/candidate pair
#   - Existing V6 + V9.1 pairs are partitioned once and removed
#     from the V8-new pool, so V10 sees genuinely new pairs only.
#
# Modes:
#   V8_MODE=plan     -> estimates block cost + shard feasibility only
#   V8_MODE=generate -> generates blocks + exact union + new-vs-baseline
#
# Suggested environment on the MacBook Air M4:
#   V8_MEMORY=8GB V8_THREADS=4
# ============================================================

PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"

S1_PATH = TRAIN / "train_source1.parquet"
S2_PATH = TRAIN / "train_source2.parquet"
S3_PATH = TRAIN / "train_source3.parquet"

V6_DIR = PROJECT / "candidate_output_v6"
V9_1_DIR = PROJECT / "candidate_output_v9_1"

V8_DIR = PROJECT / "candidate_output_v8"
TMP_DIR = V8_DIR / "tmp"
STAGE_DIR = TMP_DIR / "block_stage"
BASELINE_STAGE_DIR = TMP_DIR / "baseline_pair_shards"
UNION_DIR = V8_DIR / "union_exact"
NEW_DIR = V8_DIR / "new_vs_v6_v9_1"
MANIFEST = V8_DIR / "v8_manifest.json"

V6_S2 = V6_DIR / "train_scored_s1_s2.parquet"
V6_S3 = V6_DIR / "train_scored_s1_s3.parquet"
V9_S2 = V9_1_DIR / "s1_s2_blocks" / "num_addr_name_prefix3_shards"
V9_S3 = V9_1_DIR / "s1_s3_blocks" / "num_addr_name_prefix3_shards"

MEMORY = os.environ.get("V8_MEMORY", "8GB")
THREADS = int(os.environ.get("V8_THREADS", "4"))
TARGET_SHARD_PAIRS = int(os.environ.get("V8_TARGET_SHARD_PAIRS", "1500000"))
MAX_SHARDS = int(os.environ.get("V8_MAX_SHARDS", "64"))
UNION_SHARDS = int(os.environ.get("V8_UNION_SHARDS", "64"))
MODE = os.environ.get("V8_MODE", "plan").strip().lower()

# Exact logical definitions copied from the V3 diagnostic universe.
BLOCK_DEFS = {
    "num_addr_name_prefix2": {
        "expr": "country_norm || '|' || addr_num || '|' || substr(name_compact,1,2)",
        "valid": "country_norm <> '' AND name_norm <> '' AND addr_num <> ''",
    },
    "addr_last2_postal": {
        "expr": "country_norm || '|' || postal_like || '|' || right(address_compact,2)",
        "valid": "country_norm <> '' AND postal_like <> '' AND address_norm <> ''",
    },
    "name_last_token_postal": {
        "expr": (
            "country_norm || '|' || postal_like || '|' "
            "|| regexp_extract(name_norm,'([^[:space:]]+)$',1)"
        ),
        "valid": "country_norm <> '' AND name_norm <> '' AND postal_like <> ''",
    },
    "name_first1_postal": {
        "expr": "country_norm || '|' || postal_like || '|' || substr(name_compact,1,1)",
        "valid": "country_norm <> '' AND name_norm <> '' AND postal_like <> ''",
    },
}

TARGET_BLOCKS = {
    "S2": [
        "num_addr_name_prefix2",
        "addr_last2_postal",
        "name_last_token_postal",
        "name_first1_postal",
    ],
    "S3": [
        "num_addr_name_prefix2",
        "name_last_token_postal",
        "addr_last2_postal",
        "name_first1_postal",
    ],
}


def header(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110, flush=True)


def sql_path(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required directory not found: {path}")


def candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.parquet") if "duckdb_tmp" not in p.parts)


def setup_connection() -> duckdb.DuckDBPyConnection:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    (TMP_DIR / "duckdb_tmp").mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory={sql_path(TMP_DIR / 'duckdb_tmp')}")
    return con


def build_source_views(con: duckdb.DuckDBPyConnection) -> None:
    header("1. BUILD NORMALIZED SOURCE VIEWS")

    def build_one(table: str, path: Path) -> None:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {table} AS
            SELECT
                entity_id,
                COALESCE(name_norm,'') AS name_norm,
                COALESCE(address_norm,'') AS address_norm,
                COALESCE(country_norm,'') AS country_norm,
                regexp_replace(
                    COALESCE(name_norm,''),
                    '[[:space:]]+','',
                    'g'
                ) AS name_compact,
                regexp_replace(
                    COALESCE(address_norm,''),
                    '[[:space:]]+','',
                    'g'
                ) AS address_compact,
                COALESCE(
                    NULLIF(
                        regexp_extract(
                            lower(COALESCE(address_norm,'')),
                            '([a-z]{{1,2}}[0-9][a-z0-9]?[0-9][a-z]{{2}})',
                            1
                        ),
                        ''
                    ),
                    NULLIF(
                        reverse(
                            regexp_extract(
                                reverse(lower(COALESCE(address_norm,''))),
                                '([0-9]{{4,6}})',
                                1
                            )
                        ),
                        ''
                    )
                ) AS postal_like,
                regexp_replace(
                    COALESCE(address_norm,''),
                    '[^0-9]+','',
                    'g'
                ) AS addr_num
            FROM read_parquet({sql_path(path)})
            """
        )

    build_one("s1", S1_PATH)
    build_one("s2", S2_PATH)
    build_one("s3", S3_PATH)
    print("Source views ready.", flush=True)


def make_keys(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    key_table: str,
    block_name: str,
    shards: int,
) -> None:
    d = BLOCK_DEFS[block_name]
    expr = d["expr"]
    valid = d["valid"]

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {key_table} AS
        SELECT
            entity_id,
            block_key,
            CAST(hash(block_key) % {shards} AS BIGINT) AS shard
        FROM (
            SELECT
                entity_id,
                {expr} AS block_key
            FROM {source_table}
            WHERE {valid}
        ) x
        WHERE block_key IS NOT NULL
          AND trim(block_key) <> ''
          AND block_key NOT LIKE '%||%'
        """
    )


def max_shard_pairs(con: duckdb.DuckDBPyConnection, key1: str, key2: str) -> int:
    return int(
        con.execute(
            f"""
            SELECT COALESCE(MAX(pair_count), 0)
            FROM (
                SELECT a.shard, COUNT(*) AS pair_count
                FROM {key1} a
                INNER JOIN {key2} b
                  ON a.block_key = b.block_key
                 AND a.shard = b.shard
                GROUP BY a.shard
            )
            """
        ).fetchone()[0]
    )


def choose_shards(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    target_table: str,
    block_name: str,
) -> tuple[int, int]:
    shards = 16
    while True:
        make_keys(con, source_table, "v8_s1_keys", block_name, shards)
        make_keys(con, target_table, "v8_t_keys", block_name, shards)
        mx = max_shard_pairs(con, "v8_s1_keys", "v8_t_keys")
        if mx <= TARGET_SHARD_PAIRS or shards >= MAX_SHARDS:
            return shards, mx
        shards *= 2


def estimate_block(
    con: duckdb.DuckDBPyConnection,
    target: str,
    block_name: str,
) -> dict:
    source_table = "s1"
    target_table = target.lower()
    t0 = time.time()

    shards, mx = choose_shards(con, source_table, target_table, block_name)
    est = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM v8_s1_keys a
            INNER JOIN v8_t_keys b
              ON a.block_key = b.block_key
             AND a.shard = b.shard
            """
        ).fetchone()[0]
    )

    return {
        "target": target,
        "block_name": block_name,
        "estimated_candidate_pairs": est,
        "shards": shards,
        "largest_shard_pairs": mx,
        "seconds": round(time.time() - t0, 3),
    }


def plan(con: duckdb.DuckDBPyConnection) -> dict:
    header("V8 PLAN — NO GT / NO HOLDOUT")
    results: dict[str, list[dict]] = {}

    for target in ("S2", "S3"):
        header(f"PLAN {target}")
        results[target] = []
        for idx, block in enumerate(TARGET_BLOCKS[target], 1):
            print(f"[{idx}/{len(TARGET_BLOCKS[target])}] {block}", flush=True)
            row = estimate_block(con, target, block)
            results[target].append(row)
            print(
                f"  candidates : {row['estimated_candidate_pairs']:,}\n"
                f"  shards     : {row['shards']}\n"
                f"  max shard  : {row['largest_shard_pairs']:,}\n"
                f"  seconds    : {row['seconds']:.3f}",
                flush=True,
            )

    manifest = {
        "version": "V8-plan",
        "mode": MODE,
        "memory": MEMORY,
        "threads": THREADS,
        "target_shard_pairs": TARGET_SHARD_PAIRS,
        "max_shards": MAX_SHARDS,
        "union_shards": UNION_SHARDS,
        "targets": TARGET_BLOCKS,
        "results": results,
        "ground_truth_read": False,
        "prospective_holdout_read": False,
    }
    V8_DIR.mkdir(parents=True, exist_ok=True)
    (V8_DIR / "v8_plan_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def stage_block(
    con: duckdb.DuckDBPyConnection,
    target: str,
    block_name: str,
    shards: int,
) -> dict:
    target_table = target.lower()
    block_root = STAGE_DIR / target.lower() / block_name
    if block_root.exists():
        shutil.rmtree(block_root)
    block_root.mkdir(parents=True, exist_ok=True)

    make_keys(con, "s1", "v8_s1_keys", block_name, shards)
    make_keys(con, target_table, "v8_t_keys", block_name, shards)

    total = 0
    t0 = time.time()

    for shard in range(shards):
        local_dir = block_root / f"local_shard={shard:02d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  {target} {block_name}: local shard {shard + 1:02d}/{shards:02d}",
            flush=True,
        )

        out_glob = sql_path(local_dir)
        con.execute(
            f"""
            COPY (
                SELECT
                    a.entity_id AS source1_entity_id,
                    b.entity_id AS candidate_entity_id,
                    '{block_name}' AS block_name,
                    CAST(hash(
                        CAST(a.entity_id AS VARCHAR) || '|' ||
                        CAST(b.entity_id AS VARCHAR)
                    ) % {UNION_SHARDS} AS BIGINT) AS pair_shard
                FROM v8_s1_keys a
                INNER JOIN v8_t_keys b
                  ON a.block_key = b.block_key
                 AND a.shard = b.shard
                WHERE a.shard = {shard}
            )
            TO {out_glob}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (pair_shard))
            """
        )

        # Count rows by scanning only files from this local shard.
        local_files = sorted(local_dir.rglob("*.parquet"))
        if local_files:
            file_sql = ",".join(sql_path(p) for p in local_files)
            n = int(con.execute(f"SELECT COUNT(*) FROM read_parquet([{file_sql}])").fetchone()[0])
        else:
            n = 0
        print(f"    candidates: {n:,}", flush=True)
        total += n

    return {
        "target": target,
        "block_name": block_name,
        "shards": shards,
        "candidate_rows": total,
        "seconds": round(time.time() - t0, 3),
        "stage_dir": str(block_root),
    }


def collect_pair_shard_files(root: Path, pair_shard: int) -> list[Path]:
    """Return only parquet files belonging to the exact pair_shard directory.

    IMPORTANT: do not use substring matching here. A token such as
    ``pair_shard=2`` would also match ``pair_shard=20`` ... ``pair_shard=29``
    and silently inflate the exact-union input.
    """
    if not root.exists():
        return []

    token = f"pair_shard={pair_shard}"
    out = [
        p
        for p in root.rglob("*.parquet")
        if token in p.parts
    ]
    return sorted(out)


def merge_exact_union(target: str) -> dict:
    header(f"V8 EXACT UNION — {target}")
    target_stage = STAGE_DIR / target.lower()
    union_target = UNION_DIR / target.lower()
    if union_target.exists():
        shutil.rmtree(union_target)
    union_target.mkdir(parents=True, exist_ok=True)

    total = 0
    nonempty = 0
    t0 = time.time()

    for shard in range(UNION_SHARDS):
        files = collect_pair_shard_files(target_stage, shard)

        # Each local block shard can contribute at most one parquet file to a
        # given pair_shard. There are 80 local block shards per target
        # (32 + 16 + 16 + 16), so >80 inputs is a hard correctness failure.
        expected_max_files = sum(
            len(list((target_stage / block).glob("local_shard=*")))
            for block in TARGET_BLOCKS[target]
        )
        if len(files) > expected_max_files:
            raise RuntimeError(
                f"{target} pair_shard={shard}: collected {len(files)} files; "
                f"maximum possible is {expected_max_files}. "
                f"This indicates incorrect pair_shard path matching."
            )

        out_path = union_target / f"union_shard_{shard:03d}_of_{UNION_SHARDS:03d}.parquet"
        if not files:
            # Keep a tiny empty parquet with the expected schema.
            con = duckdb.connect(":memory:")
            con.execute(
                f"""
                COPY (
                    SELECT
                        CAST(NULL AS VARCHAR) AS source1_entity_id,
                        CAST(NULL AS VARCHAR) AS candidate_entity_id,
                        CAST(NULL AS INTEGER) AS evidence_rows,
                        CAST(NULL AS INTEGER) AS evidence_file_count,
                        CAST(NULL AS INTEGER) AS exact_key_count,
                        CAST(NULL AS VARCHAR) AS block_name
                    WHERE FALSE
                ) TO {sql_path(out_path)}
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            con.close()
            continue

        con = duckdb.connect(":memory:")
        con.execute(f"PRAGMA memory_limit='{MEMORY}'")
        con.execute(f"PRAGMA threads={THREADS}")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute(f"PRAGMA temp_directory={sql_path(TMP_DIR / 'duckdb_tmp')}")

        paths_sql = ",".join(sql_path(p) for p in files)
        print(f"  union shard {shard + 1:03d}/{UNION_SHARDS:03d}: {len(files)} files", flush=True)
        con.execute(
            f"""
            COPY (
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    COUNT(*)::INTEGER AS evidence_rows,
                    COUNT(DISTINCT block_name)::INTEGER AS evidence_file_count,
                    0::INTEGER AS exact_key_count,
                    string_agg(DISTINCT block_name, '|' ORDER BY block_name) AS block_name
                FROM read_parquet([{paths_sql}])
                GROUP BY source1_entity_id, candidate_entity_id
            ) TO {sql_path(out_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        n = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_path(out_path)})").fetchone()[0])
        con.close()
        total += n
        nonempty += 1
        print(f"    exact unique rows: {n:,}", flush=True)

    return {
        "target": target,
        "union_rows": total,
        "nonempty_shards": nonempty,
        "union_shards": UNION_SHARDS,
        "seconds": round(time.time() - t0, 3),
        "output_dir": str(union_target),
    }


def partition_baseline(target: str) -> dict:
    header(f"V8 PARTITION BASELINE V6 + V9.1 — {target}")
    target_dir = BASELINE_STAGE_DIR / target.lower()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    v6_path = V6_S2 if target == "S2" else V6_S3
    v9_dir = V9_S2 if target == "S2" else V9_S3

    require_file(v6_path)
    require_dir(v9_dir)
    v9_files = candidate_files(v9_dir)
    if not v9_files:
        raise RuntimeError(f"No V9.1 parquet files found in {v9_dir}")

    # Read baseline once and partition by deterministic pair hash.
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory={sql_path(TMP_DIR / 'duckdb_tmp')}")

    v9_union = "\nUNION ALL\n".join(
        f"""
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM read_parquet({sql_path(p)})
        WHERE starts_with(candidate_entity_id, '{target}-')
        """
        for p in v9_files
    )

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                CAST(hash(
                    CAST(source1_entity_id AS VARCHAR) || '|' ||
                    CAST(candidate_entity_id AS VARCHAR)
                ) % {UNION_SHARDS} AS BIGINT) AS pair_shard
            FROM (
                SELECT
                    source1_entity_id,
                    candidate_entity_id
                FROM read_parquet({sql_path(v6_path)})
                WHERE starts_with(candidate_entity_id, '{target}-')

                UNION ALL

                {v9_union}
            ) x
        ) TO {sql_path(target_dir)}
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (pair_shard))
        """
    )
    partitioned_files = candidate_files(target_dir)
    if partitioned_files:
        paths_sql = ",".join(sql_path(p) for p in partitioned_files)
        n = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet([{paths_sql}])"
            ).fetchone()[0]
        )
    else:
        n = 0
    con.close()

    return {
        "target": target,
        "baseline_rows_partitioned": n,
        "union_shards": UNION_SHARDS,
        "output_dir": str(target_dir),
    }


def new_vs_baseline(target: str) -> dict:
    header(f"V8 NEW-PAIR FILTER — {target}")
    union_target = UNION_DIR / target.lower()
    baseline_target = BASELINE_STAGE_DIR / target.lower()
    new_target = NEW_DIR / target.lower()
    if new_target.exists():
        shutil.rmtree(new_target)
    new_target.mkdir(parents=True, exist_ok=True)

    total = 0
    t0 = time.time()

    for shard in range(UNION_SHARDS):
        union_path = union_target / f"union_shard_{shard:03d}_of_{UNION_SHARDS:03d}.parquet"
        base_files = collect_pair_shard_files(baseline_target, shard)
        out_path = new_target / f"new_shard_{shard:03d}_of_{UNION_SHARDS:03d}.parquet"

        con = duckdb.connect(":memory:")
        con.execute(f"PRAGMA memory_limit='{MEMORY}'")
        con.execute(f"PRAGMA threads={THREADS}")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute(f"PRAGMA temp_directory={sql_path(TMP_DIR / 'duckdb_tmp')}")

        if base_files:
            base_sql = ",".join(sql_path(p) for p in base_files)
            base_ref = f"read_parquet([{base_sql}])"
            anti = f"""
                LEFT JOIN (
                    SELECT DISTINCT source1_entity_id, candidate_entity_id
                    FROM {base_ref}
                ) b
                  ON u.source1_entity_id = b.source1_entity_id
                 AND u.candidate_entity_id = b.candidate_entity_id
                WHERE b.source1_entity_id IS NULL
            """
        else:
            anti = ""

        con.execute(
            f"""
            COPY (
                SELECT
                    u.source1_entity_id,
                    u.candidate_entity_id,
                    u.evidence_rows,
                    u.evidence_file_count,
                    u.exact_key_count,
                    u.block_name
                FROM read_parquet({sql_path(union_path)}) u
                {anti}
            ) TO {sql_path(out_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        n = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_path(out_path)})").fetchone()[0])
        con.close()
        total += n
        print(f"  new shard {shard + 1:03d}/{UNION_SHARDS:03d}: {n:,} rows", flush=True)

    return {
        "target": target,
        "new_rows_vs_v6_v9_1": total,
        "seconds": round(time.time() - t0, 3),
        "output_dir": str(new_target),
    }


def generate(con: duckdb.DuckDBPyConnection) -> dict:
    header("V8 GENERATION — NO GT / NO HOLDOUT")

    # Only our own V8 output may be replaced.
    if V8_DIR.exists():
        shutil.rmtree(V8_DIR)
    V8_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    (TMP_DIR / "duckdb_tmp").mkdir(parents=True, exist_ok=True)

    block_results: list[dict] = []

    for target in ("S2", "S3"):
        # Determine shard counts using the same logical key/shard recipe
        # used by the V9.1 generator, but without reading GT.
        for block in TARGET_BLOCKS[target]:
            row = estimate_block(con, target, block)
            print(
                f"{target:2s} {block:28s} "
                f"estimated={row['estimated_candidate_pairs']:,} "
                f"shards={row['shards']} "
                f"max_shard={row['largest_shard_pairs']:,}",
                flush=True,
            )
            gen = stage_block(con, target, block, row["shards"])
            gen["estimated_candidate_pairs"] = row["estimated_candidate_pairs"]
            gen["largest_shard_pairs"] = row["largest_shard_pairs"]
            block_results.append(gen)

    union_results = []
    baseline_results = []
    new_results = []

    for target in ("S2", "S3"):
        union_results.append(merge_exact_union(target))
        baseline_results.append(partition_baseline(target))
        new_results.append(new_vs_baseline(target))

    manifest = {
        "version": "V8",
        "mode": MODE,
        "memory": MEMORY,
        "threads": THREADS,
        "target_shard_pairs": TARGET_SHARD_PAIRS,
        "max_shards": MAX_SHARDS,
        "union_shards": UNION_SHARDS,
        "targets": TARGET_BLOCKS,
        "blocks": block_results,
        "exact_unions": union_results,
        "baseline_partition": baseline_results,
        "new_vs_baseline": new_results,
        "ground_truth_read": False,
        "prospective_holdout_read": False,
        "existing_outputs_modified": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    header("V8 COMPLETE")
    for row in union_results:
        target = row["target"]
        new_row = next(x for x in new_results if x["target"] == target)
        print(
            f"{target}: exact V8 union = {row['union_rows']:,} | "
            f"new vs V6+V9.1 = {new_row['new_rows_vs_v6_v9_1']:,}",
            flush=True,
        )
    print(f"Manifest: {MANIFEST}")
    return manifest


def main() -> None:
    header("AMAZON ML CHALLENGE — V8 FINAL 4-BLOCK CANDIDATE GENERATOR")
    print(f"Project      : {PROJECT}")
    print(f"Mode         : {MODE}")
    print(f"Memory       : {MEMORY}")
    print(f"Threads      : {THREADS}")
    print(f"Target shard : {TARGET_SHARD_PAIRS:,} pairs")
    print(f"Max shards   : {MAX_SHARDS}")
    print(f"Union shards : {UNION_SHARDS}")
    print("PROSPECTIVE HOLDOUT IS NOT READ.")
    print("GROUND TRUTH IS NOT READ.")
    print("V6/V7/V8/V9/V10 OUTPUTS ARE NOT MODIFIED.")

    for p in (S1_PATH, S2_PATH, S3_PATH, V6_S2, V6_S3):
        require_file(p)
    require_dir(V9_S2)
    require_dir(V9_S3)

    con = setup_connection()
    try:
        build_source_views(con)
        if MODE == "plan":
            result = plan(con)
            header("V8 PLAN COMPLETE")
            for target, rows in result["results"].items():
                print(target)
                for row in rows:
                    print(
                        f"  {row['block_name']:28s} "
                        f"{row['estimated_candidate_pairs']:>12,} candidates | "
                        f"{row['shards']:>2} shards | "
                        f"max {row['largest_shard_pairs']:,}",
                        flush=True,
                    )
        elif MODE == "generate":
            generate(con)
        else:
            raise ValueError("V8_MODE must be 'plan' or 'generate'")
    finally:
        con.close()


if __name__ == "__main__":
    main()
