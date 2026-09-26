from __future__ import annotations

import json
import time
from pathlib import Path
import duckdb

# ============================================================
# AMAZON ML CHALLENGE - V9.1 SELECTIVE NUMERIC-ADDRESS BLOCKING
# ============================================================
# Purpose:
#   Take the strongest selective block discovered by the V9 probe:
#       num_addr + name_prefix3
#   and generate ONLY this block for S1->S2 and S1->S3.
#
# Why this block:
#   V9 probe measured:
#     S2: ~8.0M candidate pairs
#     S3: ~9.1M candidate pairs
#   with very high truth density in both directions.
#
# Safety:
#   - 4 GB DuckDB memory limit
#   - 2 threads
#   - 64 hash shards maximum
#   - NEVER consolidates all shards into one giant GROUP BY
#   - Existing V1/V2/V4/V5 outputs are preserved
#   - Each shard is written independently
#   - Coverage is measured against existing V1/V2/V4/V5
#
# Output:
#   candidate_output_v9_1/
#     s1_s2_blocks/num_addr_name_prefix3_shards/*.parquet
#     s1_s3_blocks/num_addr_name_prefix3_shards/*.parquet
#     v9_1_manifest.json
# ============================================================

PROJECT = Path('/Users/harikeshshukla/mla')
TRAIN = PROJECT / 'processed_dataset' / 'train'
S1_PATH = TRAIN / 'train_source1.parquet'
S2_PATH = TRAIN / 'train_source2.parquet'
S3_PATH = TRAIN / 'train_source3.parquet'
GT_PATH = TRAIN / 'ground_truth_pairs.parquet'

V1_DIR = PROJECT / 'candidate_output'
V2_DIR = PROJECT / 'candidate_output_v2_safe'
V4_DIR = PROJECT / 'candidate_output_v4'
V5_DIR = PROJECT / 'candidate_output_v5'
OUT_DIR = PROJECT / 'candidate_output_v9_1'
S2_OUT = OUT_DIR / 's1_s2_blocks' / 'num_addr_name_prefix3_shards'
S3_OUT = OUT_DIR / 's1_s3_blocks' / 'num_addr_name_prefix3_shards'
MANIFEST = OUT_DIR / 'v9_1_manifest.json'

MEMORY_LIMIT = '4GB'
THREADS = 2
MAX_SHARDS = 64
TARGET_SHARD_PAIRS = 1_500_000
BLOCK_NAME = 'num_addr_name_prefix3'


def header(title: str) -> None:
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100, flush=True)


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob('*.parquet') if 'duckdb_tmp' not in p.parts)


def union_candidates(paths: list[Path], prefix: str) -> str:
    parts = []
    for p in sorted(set(paths)):
        parts.append(
            f"SELECT source1_entity_id, candidate_entity_id "
            f"FROM read_parquet({sql_path(p)}) "
            f"WHERE candidate_entity_id LIKE '{prefix}-%'"
        )
    if not parts:
        raise RuntimeError('No existing candidate files found.')
    return '\nUNION ALL\n'.join(parts)


def make_source(con: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT
            entity_id,
            country_norm,
            regexp_replace(coalesce(name_norm, ''), '[[:space:]]+', '', 'g') AS name_compact,
            regexp_replace(coalesce(address_norm, ''), '[[:space:]]+', '', 'g') AS address_compact
        FROM read_parquet({sql_path(path)})
        """
    )


def make_keys(con: duckdb.DuckDBPyConnection, source_table: str, key_table: str, shards: int) -> None:
    # Numeric address signature + first three compact name characters.
    # Empty numeric signature is rejected so the block does not explode on
    # addresses with no digits.
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
                country_norm || '|' ||
                regexp_replace(address_compact, '[^0-9]+', '', 'g') || '|' ||
                substr(name_compact, 1, 3) AS block_key
            FROM {source_table}
        ) x
        WHERE block_key IS NOT NULL
          AND trim(block_key) <> ''
          AND block_key NOT LIKE '%||%'
        """
    )


def existing_coverage(con: duckdb.DuckDBPyConnection, target: str, gt_table: str) -> int:
    paths = (
        candidate_files(V1_DIR)
        + candidate_files(V2_DIR)
        + candidate_files(V4_DIR)
        + [p for p in candidate_files(V5_DIR) if p.name.startswith('train_candidates_v5_s1_')]
    )
    paths = sorted(set(paths))
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE existing_cov AS
        SELECT DISTINCT g.source1_entity_id, g.matched_entity_id
        FROM {gt_table} g
        INNER JOIN ({union_candidates(paths, target)}) c
          ON g.source1_entity_id = c.source1_entity_id
         AND g.matched_entity_id = c.candidate_entity_id
        """
    )
    return int(con.execute('SELECT COUNT(*) FROM existing_cov').fetchone()[0])


def block_stats(
    con: duckdb.DuckDBPyConnection,
    gt_table: str,
    key1: str,
    key2: str,
) -> tuple[int, int, int]:
    estimated = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {key1} a
            INNER JOIN {key2} b
              ON a.block_key = b.block_key
             AND a.shard = b.shard
            """
        ).fetchone()[0]
    )

    truth = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt_table} g
            INNER JOIN {key1} a ON g.source1_entity_id = a.entity_id
            INNER JOIN {key2} b ON g.matched_entity_id = b.entity_id
            WHERE a.block_key = b.block_key
            """
        ).fetchone()[0]
    )

    new_truth = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt_table} g
            INNER JOIN {key1} a ON g.source1_entity_id = a.entity_id
            INNER JOIN {key2} b ON g.matched_entity_id = b.entity_id
            LEFT JOIN existing_cov e
              ON g.source1_entity_id = e.source1_entity_id
             AND g.matched_entity_id = e.matched_entity_id
            WHERE a.block_key = b.block_key
              AND e.source1_entity_id IS NULL
            """
        ).fetchone()[0]
    )
    return estimated, truth, new_truth


def choose_shards(
    con: duckdb.DuckDBPyConnection,
    key1: str,
    key2: str,
    start: int,
) -> int:
    shards = start
    while True:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE shard_sizes AS
            SELECT a.shard, COUNT(*) AS pair_count
            FROM {key1} a
            INNER JOIN {key2} b
              ON a.block_key = b.block_key
             AND a.shard = b.shard
            GROUP BY a.shard
            """
        )
        mx = int(con.execute('SELECT COALESCE(MAX(pair_count), 0) FROM shard_sizes').fetchone()[0])
        if mx <= TARGET_SHARD_PAIRS or shards >= MAX_SHARDS:
            return shards
        shards *= 2
        # caller rebuilds the keys with the new shard count
        return shards


def generate_shards(
    con: duckdb.DuckDBPyConnection,
    key1: str,
    key2: str,
    out_dir: Path,
    shard_count: int,
) -> tuple[list[Path], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = 0
    for shard in range(shard_count):
        path = out_dir / f'{BLOCK_NAME}_shard_{shard:02d}_of_{shard_count:02d}.parquet'
        print(f'  shard {shard + 1:02d}/{shard_count:02d} ...', flush=True)
        con.execute(
            f"""
            COPY (
                SELECT
                    a.entity_id AS source1_entity_id,
                    b.entity_id AS candidate_entity_id,
                    '{BLOCK_NAME}' AS block_name,
                    {shard} AS shard
                FROM {key1} a
                INNER JOIN {key2} b
                  ON a.block_key = b.block_key
                 AND a.shard = b.shard
                WHERE a.shard = {shard}
            ) TO {sql_path(path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        n = int(con.execute(f'SELECT COUNT(*) FROM read_parquet({sql_path(path)})').fetchone()[0])
        print(f'    candidates: {n:,}', flush=True)
        paths.append(path)
        total += n
    return paths, total


def final_coverage(
    con: duckdb.DuckDBPyConnection,
    target: str,
    gt_table: str,
    new_paths: list[Path],
) -> int:
    base = (
        candidate_files(V1_DIR)
        + candidate_files(V2_DIR)
        + candidate_files(V4_DIR)
        + [p for p in candidate_files(V5_DIR) if p.name.startswith('train_candidates_v5_s1_')]
    )
    all_paths = sorted(set(base + new_paths))
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE final_cov AS
        SELECT DISTINCT g.source1_entity_id, g.matched_entity_id
        FROM {gt_table} g
        INNER JOIN (
            {union_candidates(all_paths, target)}
        ) c
          ON g.source1_entity_id = c.source1_entity_id
         AND g.matched_entity_id = c.candidate_entity_id
        """
    )
    return int(con.execute('SELECT COUNT(*) FROM final_cov').fetchone()[0])


def process(con: duckdb.DuckDBPyConnection, target: str, target_path: Path, out_dir: Path) -> dict:
    header(f'V9.1 PROCESSING: S1 -> {target}')
    gt_table = f'gt_{target.lower()}'
    target_table = f'source_{target.lower()}'
    s1_keys = 's1_keys'
    tgt_keys = 'tgt_keys'

    gt_total = int(con.execute(f'SELECT COUNT(*) FROM {gt_table}').fetchone()[0])
    covered = existing_coverage(con, target, gt_table)
    base_pct = 100.0 * covered / gt_total
    print(f'Ground-truth pairs : {gt_total:,}')
    print(f'Existing covered   : {covered:,}')
    print(f'Existing coverage  : {base_pct:.2f}%')

    make_source(con, 'source_s1', S1_PATH)
    make_source(con, target_table, target_path)

    # First build with 16 shards. If a shard is still too large, rebuild at 32/64.
    shards = 16
    while True:
        make_keys(con, 'source_s1', s1_keys, shards)
        make_keys(con, target_table, tgt_keys, shards)
        mx = int(con.execute(
            f"""
            SELECT COALESCE(MAX(pair_count), 0) FROM (
                SELECT a.shard, COUNT(*) AS pair_count
                FROM {s1_keys} a
                INNER JOIN {tgt_keys} b
                  ON a.block_key = b.block_key
                 AND a.shard = b.shard
                GROUP BY a.shard
            )
            """
        ).fetchone()[0])
        if mx <= TARGET_SHARD_PAIRS or shards >= MAX_SHARDS:
            break
        shards *= 2

    est, truth, new_truth = block_stats(con, gt_table, s1_keys, tgt_keys)
    print(f'Block estimated candidates : {est:,}')
    print(f'Truth pairs in block       : {truth:,}')
    print(f'NEW truth pairs            : {new_truth:,}')
    print(f'Shards                     : {shards}')

    if new_truth == 0:
        print('ACTION: no NEW truth; generation skipped')
        return {
            'target': target,
            'status': 'skip_no_new_truth',
            'ground_truth_pairs': gt_total,
            'existing_covered': covered,
            'existing_coverage_pct': base_pct,
            'new_truth_pairs': 0,
            'generated_candidates': 0,
            'shards': shards,
        }

    start = time.time()
    paths, generated = generate_shards(con, s1_keys, tgt_keys, out_dir, shards)
    elapsed = time.time() - start
    final_cov = final_coverage(con, target, gt_table, paths)
    final_pct = 100.0 * final_cov / gt_total

    header(f'V9.1 FINAL RESULT: S1 -> {target}')
    print(f'Ground truth             : {gt_total:,}')
    print(f'Existing covered         : {covered:,}')
    print(f'Existing coverage        : {base_pct:.2f}%')
    print(f'NEW truth from V9.1      : {new_truth:,}')
    print(f'Generated candidates     : {generated:,}')
    print(f'Generation time          : {elapsed:.2f}s')
    print(f'Final covered            : {final_cov:,}')
    print(f'Final coverage           : {final_pct:.2f}%')
    print(f'Coverage improvement     : {final_pct - base_pct:+.2f} points')
    print(f'Output                   : {out_dir}')

    return {
        'target': target,
        'status': 'generated',
        'block_name': BLOCK_NAME,
        'ground_truth_pairs': gt_total,
        'existing_covered': covered,
        'existing_coverage_pct': base_pct,
        'block_estimated_candidates': est,
        'block_truth_pairs': truth,
        'new_truth_pairs': new_truth,
        'generated_candidates': generated,
        'shards': shards,
        'generation_seconds': round(elapsed, 3),
        'final_covered': final_cov,
        'final_coverage_pct': final_pct,
        'coverage_improvement_points': final_pct - base_pct,
        'output_dir': str(out_dir),
    }


def main() -> None:
    header('AMAZON ML CHALLENGE - V9.1 SELECTIVE NUMERIC-ADDRESS BLOCKING')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_file(S1_PATH)
    require_file(S2_PATH)
    require_file(S3_PATH)
    require_file(GT_PATH)

    con = duckdb.connect(':memory:')
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}'")
    con.execute(f'PRAGMA threads={THREADS}')
    con.execute('PRAGMA preserve_insertion_order=false')
    tmp = OUT_DIR / 'duckdb_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{str(tmp).replace(chr(92), '/').replace(chr(39), chr(39)+chr(39))}'")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gt_all AS
        SELECT source1_entity_id, matched_entity_id, label
        FROM read_parquet({sql_path(GT_PATH)})
        WHERE COALESCE(label, 1) = 1
        """
    )
    con.execute("CREATE OR REPLACE TEMP TABLE gt_s2 AS SELECT * FROM gt_all WHERE matched_entity_id LIKE 'S2-%'")
    con.execute("CREATE OR REPLACE TEMP TABLE gt_s3 AS SELECT * FROM gt_all WHERE matched_entity_id LIKE 'S3-%'")

    results = {}
    try:
        results['s2'] = process(con, 'S2', S2_PATH, S2_OUT)
        results['s3'] = process(con, 'S3', S3_PATH, S3_OUT)
    finally:
        con.close()

    manifest = {
        'version': 'V9.1',
        'block': BLOCK_NAME,
        'memory_limit': MEMORY_LIMIT,
        'threads': THREADS,
        'target_shard_pairs': TARGET_SHARD_PAIRS,
        'results': results,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    header('V9.1 COMPLETE')
    print(f'S2 final coverage : {results["s2"].get("final_coverage_pct", results["s2"]["existing_coverage_pct"]):.2f}%')
    print(f'S3 final coverage : {results["s3"].get("final_coverage_pct", results["s3"]["existing_coverage_pct"]):.2f}%')
    print(f'Manifest           : {MANIFEST}')


if __name__ == '__main__':
    main()
