#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb

PROJECT = Path('/Users/harikeshshukla/mla')
TRAIN = PROJECT / 'processed_dataset' / 'train'
GT_PATH = TRAIN / 'ground_truth_pairs.parquet'
V8_DIR = PROJECT / 'candidate_output_v8'
V8_UNION_DIR = V8_DIR / 'union_exact'
V8_BASELINE_DIR = V8_DIR / 'tmp' / 'baseline_pair_shards'
AUDIT_DIR = PROJECT / 'validation_leakage_audit' / 'v8_validation_coverage'
REPORT_PATH = AUDIT_DIR / 'v8_validation_coverage_report.json'
VALIDATION_MOD = 5
EXPECTED = {
    'S2': {'gt_validation_pairs': 739151, 'missed_by_v10_baseline_pairs': 261662, 'four_block_recoveries': 13848, 'v7_four_block_union_rows': 56294673, 'v8_union_rows': 56294673},
    'S3': {'gt_validation_pairs': 787926, 'missed_by_v10_baseline_pairs': 288857, 'four_block_recoveries': 20993, 'v7_four_block_union_rows': 54395314, 'v8_union_rows': 54395314},
}

def qpath(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"

def header(title: str) -> None:
    print('\n' + '=' * 116)
    print(title)
    print('=' * 116, flush=True)

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Required file missing: {path}')

def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f'Required directory missing: {path}')

def parquet_list_sql(paths: list[Path]) -> str:
    if not paths:
        raise RuntimeError('No parquet files found.')
    return '[' + ','.join(qpath(p) for p in paths) + ']'

def candidate_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*.parquet') if p.is_file())

def setup_connection(memory: str, threads: int, tmp_dir: Path) -> duckdb.DuckDBPyConnection:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(':memory:')
    con.execute(f"PRAGMA memory_limit='{memory}'")
    con.execute(f'PRAGMA threads={threads}')
    con.execute('PRAGMA preserve_insertion_order=false')
    con.execute(f'PRAGMA temp_directory={qpath(tmp_dir)}')
    return con

def make_gt_table(con: duckdb.DuckDBPyConnection, target: str) -> dict[str, int]:
    table = f'gt_{target.lower()}_validation'
    pred = f"""
        COALESCE(label, 1) = 1
        AND starts_with(CAST(matched_entity_id AS VARCHAR), '{target}-')
        AND MOD(ABS(HASH(CAST(source1_entity_id AS VARCHAR))), {VALIDATION_MOD}) = 0
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                        CAST(matched_entity_id AS VARCHAR) AS matched_entity_id
        FROM read_parquet({qpath(GT_PATH)})
        WHERE {pred}
    """)
    raw_count = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({qpath(GT_PATH)}) WHERE {pred}").fetchone()[0])
    pair_count = int(con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
    s1_count = int(con.execute(f'SELECT COUNT(DISTINCT source1_entity_id) FROM {table}').fetchone()[0])
    return {'raw_gt_rows': raw_count, 'distinct_gt_pairs': pair_count, 'gt_s1_entities': s1_count}

def count_union_duplicates(con: duckdb.DuckDBPyConnection, target: str) -> dict[str, int]:
    files = candidate_files(V8_UNION_DIR / target.lower())
    paths = parquet_list_sql(files)
    raw_rows = int(con.execute(f'SELECT COUNT(*) FROM read_parquet({paths})').fetchone()[0])
    distinct_pairs = int(con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT source1_entity_id, candidate_entity_id FROM read_parquet({paths}))").fetchone()[0])
    return {'raw_rows': raw_rows, 'distinct_pairs': distinct_pairs, 'duplicate_rows': raw_rows - distinct_pairs, 'file_count': len(files)}

def make_v8_table(con: duckdb.DuckDBPyConnection, target: str) -> tuple[str, list[Path]]:
    table = f'v8_{target.lower()}_union'
    files = candidate_files(V8_UNION_DIR / target.lower())
    paths = parquet_list_sql(files)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT DISTINCT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id,
            CAST(evidence_rows AS INTEGER) AS evidence_rows,
            CAST(evidence_file_count AS INTEGER) AS evidence_file_count,
            CAST(exact_key_count AS INTEGER) AS exact_key_count,
            CAST(block_name AS VARCHAR) AS block_name
        FROM read_parquet({paths})
    """)
    return table, files

def make_baseline_table(con: duckdb.DuckDBPyConnection, target: str) -> tuple[str, list[Path]]:
    table = f'baseline_{target.lower()}'
    files = candidate_files(V8_BASELINE_DIR / target.lower())
    paths = parquet_list_sql(files)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                        CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet({paths})
    """)
    return table, files

def audit_target(con: duckdb.DuckDBPyConnection, target: str) -> dict:
    header(f'V8 VALIDATION COVERAGE AUDIT — S1 → {target}')
    gt_info = make_gt_table(con, target)
    gt_table = f'gt_{target.lower()}_validation'
    v8_table, v8_files = make_v8_table(con, target)
    baseline_table, baseline_files = make_baseline_table(con, target)
    v8_rows = int(con.execute(f'SELECT COUNT(*) FROM {v8_table}').fetchone()[0])
    baseline_rows = int(con.execute(f'SELECT COUNT(*) FROM {baseline_table}').fetchone()[0])
    v8_duplicate_info = count_union_duplicates(con, target)
    baseline_covered = int(con.execute(f"""
        SELECT COUNT(*) FROM {gt_table} g
        INNER JOIN {baseline_table} b ON g.source1_entity_id=b.source1_entity_id AND g.matched_entity_id=b.candidate_entity_id
    """).fetchone()[0])
    v8_covered = int(con.execute(f"""
        SELECT COUNT(*) FROM {gt_table} g
        INNER JOIN {v8_table} v ON g.source1_entity_id=v.source1_entity_id AND g.matched_entity_id=v.candidate_entity_id
    """).fetchone()[0])
    new_recovered = int(con.execute(f"""
        SELECT COUNT(*) FROM {gt_table} g
        INNER JOIN {v8_table} v ON g.source1_entity_id=v.source1_entity_id AND g.matched_entity_id=v.candidate_entity_id
        LEFT JOIN {baseline_table} b ON g.source1_entity_id=b.source1_entity_id AND g.matched_entity_id=b.candidate_entity_id
        WHERE b.source1_entity_id IS NULL
    """).fetchone()[0])
    baseline_missed = int(con.execute(f"""
        SELECT COUNT(*) FROM {gt_table} g
        LEFT JOIN {baseline_table} b ON g.source1_entity_id=b.source1_entity_id AND g.matched_entity_id=b.candidate_entity_id
        WHERE b.source1_entity_id IS NULL
    """).fetchone()[0])
    v8_missed = int(con.execute(f"""
        SELECT COUNT(*) FROM {gt_table} g
        LEFT JOIN {v8_table} v ON g.source1_entity_id=v.source1_entity_id AND g.matched_entity_id=v.candidate_entity_id
        WHERE v.source1_entity_id IS NULL
    """).fetchone()[0])
    baseline_covered_s1 = int(con.execute(f"""
        SELECT COUNT(DISTINCT g.source1_entity_id) FROM {gt_table} g
        INNER JOIN {baseline_table} b ON g.source1_entity_id=b.source1_entity_id AND g.matched_entity_id=b.candidate_entity_id
    """).fetchone()[0])
    v8_covered_s1 = int(con.execute(f"""
        SELECT COUNT(DISTINCT g.source1_entity_id) FROM {gt_table} g
        INNER JOIN {v8_table} v ON g.source1_entity_id=v.source1_entity_id AND g.matched_entity_id=v.candidate_entity_id
    """).fetchone()[0])
    combo_rows = con.execute(f"""
        SELECT COALESCE(v.evidence_file_count,0) AS evidence_file_count, COUNT(*) AS gt_pairs
        FROM {gt_table} g
        INNER JOIN {v8_table} v ON g.source1_entity_id=v.source1_entity_id AND g.matched_entity_id=v.candidate_entity_id
        LEFT JOIN {baseline_table} b ON g.source1_entity_id=b.source1_entity_id AND g.matched_entity_id=b.candidate_entity_id
        WHERE b.source1_entity_id IS NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    pattern_rows = con.execute(f"""
        SELECT v.block_name, COUNT(*) AS gt_pairs
        FROM {gt_table} g
        INNER JOIN {v8_table} v ON g.source1_entity_id=v.source1_entity_id AND g.matched_entity_id=v.candidate_entity_id
        LEFT JOIN {baseline_table} b ON g.source1_entity_id=b.source1_entity_id AND g.matched_entity_id=b.candidate_entity_id
        WHERE b.source1_entity_id IS NULL
        GROUP BY 1 ORDER BY gt_pairs DESC, v.block_name LIMIT 30
    """).fetchall()
    gt_pairs = gt_info['distinct_gt_pairs']
    baseline_pct = 100.0 * baseline_covered / gt_pairs if gt_pairs else 0.0
    v8_pct = 100.0 * v8_covered / gt_pairs if gt_pairs else 0.0
    new_pct = 100.0 * new_recovered / baseline_missed if baseline_missed else 0.0
    expansion_pct = 100.0 * (v8_rows - baseline_rows) / baseline_rows if baseline_rows else 0.0
    rows_per_gt = (v8_rows - baseline_rows) / new_recovered if new_recovered else None
    result = {
        'target': target,
        'gt': gt_info,
        'v8_union': {**v8_duplicate_info},
        'baseline': {'distinct_pairs': baseline_rows, 'file_count': len(baseline_files)},
        'coverage': {
            'baseline_covered_gt_pairs': baseline_covered,
            'baseline_coverage_pct': baseline_pct,
            'v8_covered_gt_pairs': v8_covered,
            'v8_coverage_pct': v8_pct,
            'baseline_missed_gt_pairs': baseline_missed,
            'v8_missed_gt_pairs': v8_missed,
            'new_gt_pairs_recovered_by_v8_vs_baseline': new_recovered,
            'new_recovery_pct_of_baseline_misses': new_pct,
            'baseline_covered_s1': baseline_covered_s1,
            'v8_covered_s1': v8_covered_s1,
            's1_gain': v8_covered_s1 - baseline_covered_s1,
        },
        'economics': {
            'baseline_candidate_pairs': baseline_rows,
            'v8_candidate_pairs': v8_rows,
            'new_candidate_pairs': v8_rows - baseline_rows,
            'candidate_expansion_pct': expansion_pct,
            'new_candidate_rows_per_recovered_gt_pair': rows_per_gt,
        },
        'complementarity': {
            'new_recovered_by_evidence_file_count': [{'evidence_file_count': int(a), 'gt_pairs': int(b)} for a,b in combo_rows],
            'top_new_recovered_block_patterns': [{'block_pattern': str(a), 'gt_pairs': int(b)} for a,b in pattern_rows],
        },
        'expected_checks': {
            'expected_v7_four_block_union_rows': EXPECTED[target]['v7_four_block_union_rows'],
            'expected_v8_union_rows': EXPECTED[target]['v8_union_rows'],
            'expected_four_block_recoveries_from_v7_forensics': EXPECTED[target]['four_block_recoveries'],
            'expected_v7_missed_baseline_pairs': EXPECTED[target]['missed_by_v10_baseline_pairs'],
        },
    }
    print(f"GT validation pairs           : {gt_pairs:,}")
    print(f"Baseline candidate pairs      : {baseline_rows:,}")
    print(f"V8 exact-union candidate pairs: {v8_rows:,}")
    print(f"Candidate expansion           : {v8_rows-baseline_rows:,} ({expansion_pct:.2f}%)")
    print(f"Baseline GT coverage          : {baseline_covered:,} ({baseline_pct:.4f}%)")
    print(f"V8 GT coverage                : {v8_covered:,} ({v8_pct:.4f}%)")
    print(f"Baseline-missed GT             : {baseline_missed:,}")
    print(f"V8-missed GT                   : {v8_missed:,}")
    print(f"NEW GT recovered by V8         : {new_recovered:,}")
    print(f"New recovery of baseline misses: {new_pct:.4f}%")
    print(f"S1 entities baseline covered  : {baseline_covered_s1:,}")
    print(f"S1 entities V8 covered        : {v8_covered_s1:,}")
    print(f"S1 gain                        : {v8_covered_s1-baseline_covered_s1:,}")
    return result

def run_audit(memory: str, threads: int) -> dict:
    header('AMAZON ML CHALLENGE — V8 OLD-VALIDATION COVERAGE AUDIT')
    print(f'Project              : {PROJECT}')
    print(f'Memory               : {memory}')
    print(f'Threads              : {threads}')
    print('Prospective holdout  : NOT READ')
    print('V8 generator output  : NOT MODIFIED')
    print('V6/V7/V9/V10 outputs : NOT MODIFIED')
    print('GT use               : validation audit only, after V8 generation')
    require_file(GT_PATH)
    require_dir(V8_UNION_DIR / 's2')
    require_dir(V8_UNION_DIR / 's3')
    require_dir(V8_BASELINE_DIR / 's2')
    require_dir(V8_BASELINE_DIR / 's3')
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = AUDIT_DIR / 'duckdb_tmp'
    con = setup_connection(memory, threads, tmp_dir)
    t0 = time.time()
    try:
        results = {'S2': audit_target(con, 'S2'), 'S3': audit_target(con, 'S3')}
    finally:
        con.close()
    failures = []
    for target, row in results.items():
        exp = EXPECTED[target]
        if row['v8_union']['distinct_pairs'] != exp['v8_union_rows']:
            failures.append(f"{target}: V8 union rows {row['v8_union']['distinct_pairs']:,} != expected {exp['v8_union_rows']:,}")
        if row['v8_union']['duplicate_rows'] != 0:
            failures.append(f"{target}: V8 union has {row['v8_union']['duplicate_rows']:,} duplicate rows")
        if row['coverage']['baseline_missed_gt_pairs'] != exp['missed_by_v10_baseline_pairs']:
            failures.append(f"{target}: baseline-missed GT {row['coverage']['baseline_missed_gt_pairs']:,} != V7 forensic {exp['missed_by_v10_baseline_pairs']:,}")
        if row['coverage']['new_gt_pairs_recovered_by_v8_vs_baseline'] != exp['four_block_recoveries']:
            failures.append(f"{target}: V8 new recovered GT {row['coverage']['new_gt_pairs_recovered_by_v8_vs_baseline']:,} != V7 four-block recovery {exp['four_block_recoveries']:,}")
        if row['coverage']['v8_missed_gt_pairs'] != exp['missed_by_v10_baseline_pairs'] - exp['four_block_recoveries']:
            failures.append(f"{target}: remaining V8-missed GT does not equal baseline misses minus four-block recoveries")
    report = {
        'version': 'V8_VALIDATION_COVERAGE_AUDIT_V1',
        'validation_mod': VALIDATION_MOD,
        'prospective_holdout_read': False,
        'generator_outputs_modified': False,
        'elapsed_seconds': round(time.time()-t0, 3),
        'results': results,
        'strict_invariant_failures': failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    header('AUDIT COMPLETE')
    for target in ('S2','S3'):
        row=results[target]
        print(f"{target}: baseline={row['baseline']['distinct_pairs']:,} | V8={row['v8_union']['distinct_pairs']:,} | new_GT={row['coverage']['new_gt_pairs_recovered_by_v8_vs_baseline']:,} | V8_missed={row['coverage']['v8_missed_gt_pairs']:,}", flush=True)
    if failures:
        print('\nSTRICT CHECK: FAIL')
        for f in failures: print('  - ' + f)
        raise RuntimeError('V8 validation coverage audit failed strict invariants.')
    print('\nSTRICT CHECK: PASS')
    print(f'Report: {REPORT_PATH}')
    return report

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--memory', default=os.environ.get('V8_AUDIT_MEMORY','8GB'))
    ap.add_argument('--threads', type=int, default=int(os.environ.get('V8_AUDIT_THREADS','4')))
    args=ap.parse_args()
    run_audit(args.memory,args.threads)

if __name__ == '__main__':
    main()
