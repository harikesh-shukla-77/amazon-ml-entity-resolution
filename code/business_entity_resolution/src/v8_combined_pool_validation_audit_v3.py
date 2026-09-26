#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb

PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"
GT_PATH = TRAIN / "ground_truth_pairs.parquet"

V8_DIR = PROJECT / "candidate_output_v8"
V8_UNION_DIR = V8_DIR / "union_exact"
V8_NEW_DIR = V8_DIR / "new_vs_v6_v9_1"
V8_BASELINE_DIR = V8_DIR / "tmp" / "baseline_pair_shards"
AUDIT_DIR = PROJECT / "validation_leakage_audit" / "v8_validation_coverage"
REPORT_PATH = AUDIT_DIR / "v8_validation_coverage_report.json"

VALIDATION_MOD = 5

EXPECTED = {
    "S2": {
        "gt_validation_pairs": 739_151,
        "baseline_pairs": 24_382_935,
        "baseline_covered": 477_489,
        "baseline_missed": 261_662,
        "v8_union_rows": 56_294_673,
        "v8_new_rows": 45_750_584,
        "new_recovered_gt": 13_848,
    },
    "S3": {
        "gt_validation_pairs": 787_926,
        "baseline_pairs": 29_215_410,
        "baseline_covered": 499_069,
        "baseline_missed": 288_857,
        "v8_union_rows": 54_395_314,
        "v8_new_rows": 44_112_929,
        "new_recovered_gt": 20_993,
    },
}


def qpath(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def header(title: str) -> None:
    print("\n" + "=" * 116)
    print(title)
    print("=" * 116, flush=True)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory missing: {path}")


def parquet_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def paths_sql(paths: list[Path]) -> str:
    if not paths:
        raise RuntimeError("No parquet files found.")
    return "[" + ",".join(qpath(p) for p in paths) + "]"


def setup_connection(memory: str, threads: int, tmp_dir: Path) -> duckdb.DuckDBPyConnection:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{memory}'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory={qpath(tmp_dir)}")
    return con


def create_gt(con: duckdb.DuckDBPyConnection, target: str) -> str:
    table = f"gt_{target.lower()}_validation"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT DISTINCT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(matched_entity_id AS VARCHAR) AS matched_entity_id
        FROM read_parquet({qpath(GT_PATH)})
        WHERE COALESCE(label, 1) = 1
          AND starts_with(CAST(matched_entity_id AS VARCHAR), '{target}-')
          AND MOD(ABS(HASH(CAST(source1_entity_id AS VARCHAR))), {VALIDATION_MOD}) = 0
        """
    )
    return table


def create_pair_table(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    table: str,
    columns_sql: str,
) -> tuple[str, int]:
    files = parquet_files(root)
    paths = paths_sql(files)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT DISTINCT
            {columns_sql}
        FROM read_parquet({paths})
        """
    )
    n = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return table, n


def coverage_count(
    con: duckdb.DuckDBPyConnection,
    gt_table: str,
    cand_table: str,
) -> int:
    return int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt_table} g
            INNER JOIN {cand_table} c
              ON g.source1_entity_id = c.source1_entity_id
             AND g.matched_entity_id = c.candidate_entity_id
            """
        ).fetchone()[0]
    )


def missed_count(
    con: duckdb.DuckDBPyConnection,
    gt_table: str,
    cand_table: str,
) -> int:
    return int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt_table} g
            LEFT JOIN {cand_table} c
              ON g.source1_entity_id = c.source1_entity_id
             AND g.matched_entity_id = c.candidate_entity_id
            WHERE c.source1_entity_id IS NULL
            """
        ).fetchone()[0]
    )


def run_target(con: duckdb.DuckDBPyConnection, target: str) -> dict:
    header(f"V8 COMBINED-CANDIDATE VALIDATION AUDIT — S1 → {target}")

    gt = create_gt(con, target)
    gt_pairs = int(con.execute(f"SELECT COUNT(*) FROM {gt}").fetchone()[0])

    baseline_root = V8_BASELINE_DIR / target.lower()
    union_root = V8_UNION_DIR / target.lower()
    new_root = V8_NEW_DIR / target.lower()

    baseline, baseline_rows = create_pair_table(
        con,
        baseline_root,
        f"baseline_{target.lower()}",
        "CAST(source1_entity_id AS VARCHAR) AS source1_entity_id, "
        "CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id",
    )

    v8_union, v8_union_rows = create_pair_table(
        con,
        union_root,
        f"v8_union_{target.lower()}",
        "CAST(source1_entity_id AS VARCHAR) AS source1_entity_id, "
        "CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id",
    )

    v8_new, v8_new_rows = create_pair_table(
        con,
        new_root,
        f"v8_new_{target.lower()}",
        "CAST(source1_entity_id AS VARCHAR) AS source1_entity_id, "
        "CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id",
    )

    # Important architecture point:
    # V8 union is the FOUR-BLOCK pool.
    # v8_new is V8 union MINUS the existing baseline.
    # The actual expanded V10 candidate pool is:
    #       baseline UNION v8_new
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE combined_{target.lower()} AS
        SELECT source1_entity_id, candidate_entity_id FROM {baseline}
        UNION ALL
        SELECT source1_entity_id, candidate_entity_id FROM {v8_new}
        """
    )
    combined_raw = int(
        con.execute(f"SELECT COUNT(*) FROM combined_{target.lower()}").fetchone()[0]
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE combined_distinct_{target.lower()} AS
        SELECT DISTINCT source1_entity_id, candidate_entity_id
        FROM combined_{target.lower()}
        """
    )
    combined = f"combined_distinct_{target.lower()}"
    combined_rows = int(con.execute(f"SELECT COUNT(*) FROM {combined}").fetchone()[0])
    combined_dups = combined_raw - combined_rows

    # Hard structural invariant before any coverage arithmetic:
    # V8-new is generated by anti-joining V8 union against baseline, so any
    # overlap here means the new-pair filtering stage is incorrect.
    overlap = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {v8_new} n
            INNER JOIN {baseline} b
              ON n.source1_entity_id = b.source1_entity_id
             AND n.candidate_entity_id = b.candidate_entity_id
            """
        ).fetchone()[0]
    )
    if overlap != 0:
        raise RuntimeError(
            f"{target}: V8-new overlaps baseline by {overlap:,} pairs."
        )

    # Baseline coverage.
    baseline_cov = coverage_count(con, gt, baseline)
    baseline_missed = missed_count(con, gt, baseline)

    # V8 union alone is intentionally NOT treated as the final expanded pool.
    v8_union_cov = coverage_count(con, gt, v8_union)
    v8_union_missed = missed_count(con, gt, v8_union)

    # New V8-only recovery.
    new_cov = coverage_count(con, gt, v8_new)
    new_recovered = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM {gt} g
            INNER JOIN {v8_new} n
              ON g.source1_entity_id = n.source1_entity_id
             AND g.matched_entity_id = n.candidate_entity_id
            LEFT JOIN {baseline} b
              ON g.source1_entity_id = b.source1_entity_id
             AND g.matched_entity_id = b.candidate_entity_id
            WHERE b.source1_entity_id IS NULL
            """
        ).fetchone()[0]
    )

    # Final expanded pool coverage: baseline UNION V8-new.
    combined_cov = coverage_count(con, gt, combined)
    combined_missed = missed_count(con, gt, combined)

    expansion_rows = combined_rows - baseline_rows
    expansion_pct = 100.0 * expansion_rows / baseline_rows
    new_recovery_pct = 100.0 * new_recovered / baseline_missed
    combined_coverage_pct = 100.0 * combined_cov / gt_pairs
    candidate_rows_per_new_gt = expansion_rows / new_recovered

    # S1-level coverage.
    baseline_s1 = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT g.source1_entity_id)
            FROM {gt} g
            INNER JOIN {baseline} b
              ON g.source1_entity_id = b.source1_entity_id
             AND g.matched_entity_id = b.candidate_entity_id
            """
        ).fetchone()[0]
    )
    combined_s1 = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT g.source1_entity_id)
            FROM {gt} g
            INNER JOIN {combined} c
              ON g.source1_entity_id = c.source1_entity_id
             AND g.matched_entity_id = c.candidate_entity_id
            """
        ).fetchone()[0]
    )

    result = {
        "target": target,
        "gt_validation_pairs": gt_pairs,
        "baseline_candidate_pairs": baseline_rows,
        "v8_union_candidate_pairs": v8_union_rows,
        "v8_new_candidate_pairs": v8_new_rows,
        "final_combined_candidate_pairs": combined_rows,
        "final_combined_duplicate_rows": combined_dups,
        "baseline_coverage": {
            "covered_gt_pairs": baseline_cov,
            "coverage_pct": 100.0 * baseline_cov / gt_pairs,
            "missed_gt_pairs": baseline_missed,
        },
        "v8_union_alone": {
            "covered_gt_pairs": v8_union_cov,
            "coverage_pct": 100.0 * v8_union_cov / gt_pairs,
            "missed_gt_pairs": v8_union_missed,
        },
        "v8_new_vs_baseline": {
            "new_recovered_gt_pairs": new_recovered,
            "new_recovery_pct_of_baseline_misses": new_recovery_pct,
            "new_gt_total_in_v8_new": new_cov,
        },
        "final_combined_pool": {
            "added_candidate_pairs": expansion_rows,
            "candidate_expansion_pct": expansion_pct,
            "covered_gt_pairs": combined_cov,
            "coverage_pct": combined_coverage_pct,
            "missed_gt_pairs": combined_missed,
            "gt_pair_gain_vs_baseline": combined_cov - baseline_cov,
            "candidate_rows_per_new_recovered_gt": candidate_rows_per_new_gt,
            "covered_s1_entities": combined_s1,
            "covered_s1_gain": combined_s1 - baseline_s1,
        },
        "integrity": {
            "baseline_v8new_overlap_pairs": overlap,
        },
    }

    print(f"Validation GT pairs          : {gt_pairs:,}")
    print(f"Baseline candidate pairs     : {baseline_rows:,}")
    print(f"V8 exact-union pairs         : {v8_union_rows:,}")
    print(f"V8 NEW-vs-baseline pairs     : {v8_new_rows:,}")
    print(f"Final combined pairs         : {combined_rows:,}")
    print(f"Candidate expansion          : +{expansion_rows:,} ({expansion_pct:.2f}%)")
    print(f"Baseline GT coverage         : {baseline_cov:,} ({100.0*baseline_cov/gt_pairs:.4f}%)")
    print(f"V8 union-alone GT coverage   : {v8_union_cov:,} ({100.0*v8_union_cov/gt_pairs:.4f}%)")
    print(f"New GT recovered by V8       : {new_recovered:,}")
    print(f"Recovery of baseline misses  : {new_recovery_pct:.4f}%")
    print(f"FINAL COMBINED GT coverage   : {combined_cov:,} ({combined_coverage_pct:.4f}%)")
    print(f"FINAL combined GT missed     : {combined_missed:,}")
    print(f"GT gain vs baseline          : +{combined_cov-baseline_cov:,}")
    print(f"Rows per new recovered GT    : {candidate_rows_per_new_gt:,.2f}")
    print(f"Baseline/V8-new overlap      : {overlap:,}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", default=os.environ.get("V8_AUDIT_MEMORY", "8GB"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("V8_AUDIT_THREADS", "4")))
    args = parser.parse_args()

    header("AMAZON ML CHALLENGE — V8 FINAL COMBINED-POOL VALIDATION AUDIT")
    print(f"Project                  : {PROJECT}")
    print(f"Memory                   : {args.memory}")
    print(f"Threads                  : {args.threads}")
    print("Prospective holdout      : NOT READ")
    print("V8 generator output      : NOT MODIFIED")
    print("V6/V7/V9/V10 outputs     : NOT MODIFIED")
    print("Ground truth             : validation audit only")

    require_file(GT_PATH)
    require_dir(V8_UNION_DIR / "s2")
    require_dir(V8_UNION_DIR / "s3")
    require_dir(V8_NEW_DIR / "s2")
    require_dir(V8_NEW_DIR / "s3")
    require_dir(V8_BASELINE_DIR / "s2")
    require_dir(V8_BASELINE_DIR / "s3")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    con = setup_connection(args.memory, args.threads, AUDIT_DIR / "duckdb_tmp")
    t0 = time.time()

    try:
        results = {
            "S2": run_target(con, "S2"),
            "S3": run_target(con, "S3"),
        }
    finally:
        con.close()

    failures = []

    for target, row in results.items():
        exp = EXPECTED[target]

        if row["gt_validation_pairs"] != exp["gt_validation_pairs"]:
            failures.append(
                f"{target}: GT pairs {row['gt_validation_pairs']:,} != {exp['gt_validation_pairs']:,}"
            )

        if row["baseline_candidate_pairs"] != exp["baseline_pairs"]:
            failures.append(
                f"{target}: baseline pairs {row['baseline_candidate_pairs']:,} != {exp['baseline_pairs']:,}"
            )

        if row["v8_union_candidate_pairs"] != exp["v8_union_rows"]:
            failures.append(
                f"{target}: V8 union rows {row['v8_union_candidate_pairs']:,} != {exp['v8_union_rows']:,}"
            )

        if row["v8_new_candidate_pairs"] != exp["v8_new_rows"]:
            failures.append(
                f"{target}: V8-new rows {row['v8_new_candidate_pairs']:,} != {exp['v8_new_rows']:,}"
            )

        if row["baseline_coverage"]["covered_gt_pairs"] != exp["baseline_covered"]:
            failures.append(
                f"{target}: baseline covered GT {row['baseline_coverage']['covered_gt_pairs']:,} != {exp['baseline_covered']:,}"
            )

        if row["baseline_coverage"]["missed_gt_pairs"] != exp["baseline_missed"]:
            failures.append(
                f"{target}: baseline missed GT {row['baseline_coverage']['missed_gt_pairs']:,} != {exp['baseline_missed']:,}"
            )

        if row["v8_new_vs_baseline"]["new_recovered_gt_pairs"] != exp["new_recovered_gt"]:
            failures.append(
                f"{target}: V8-new recovered GT {row['v8_new_vs_baseline']['new_recovered_gt_pairs']:,} != {exp['new_recovered_gt']:,}"
            )

        # Core accounting identity:
        # final combined coverage = baseline coverage + genuinely new V8 recovery.
        expected_combined_cov = exp["baseline_covered"] + exp["new_recovered_gt"]
        if row["final_combined_pool"]["covered_gt_pairs"] != expected_combined_cov:
            failures.append(
                f"{target}: combined coverage {row['final_combined_pool']['covered_gt_pairs']:,} "
                f"!= expected {expected_combined_cov:,}"
            )

        if row["integrity"]["baseline_v8new_overlap_pairs"] != 0:
            failures.append(
                f"{target}: baseline overlaps V8-new by {row['integrity']['baseline_v8new_overlap_pairs']:,} pairs"
            )

        expected_combined_rows = exp["baseline_pairs"] + exp["v8_new_rows"]
        if row["final_combined_candidate_pairs"] != expected_combined_rows:
            failures.append(
                f"{target}: combined candidate rows {row['final_combined_candidate_pairs']:,} "
                f"!= expected {expected_combined_rows:,}"
            )

    report = {
        "version": "V8_COMBINED_POOL_VALIDATION_AUDIT_V2",
        "validation_mod": VALIDATION_MOD,
        "prospective_holdout_read": False,
        "generator_outputs_modified": False,
        "elapsed_seconds": round(time.time() - t0, 3),
        "results": results,
        "strict_invariant_failures": failures,
        "interpretation": (
            "V8 union is not itself the final candidate pool. "
            "The intended expanded pool is the existing V6+V9.1 baseline "
            "UNION V8-new pairs. Therefore validation coverage must be evaluated "
            "on that combined pool, not on V8 union alone."
        ),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header("AUDIT COMPLETE")
    for target in ("S2", "S3"):
        row = results[target]
        print(
            f"{target}: combined={row['final_combined_candidate_pairs']:,} | "
            f"coverage={row['final_combined_pool']['coverage_pct']:.4f}% | "
            f"new_GT={row['v8_new_vs_baseline']['new_recovered_gt_pairs']:,} | "
            f"remaining_GT_missed={row['final_combined_pool']['missed_gt_pairs']:,}",
            flush=True,
        )

    if failures:
        print("\nSTRICT CHECK: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise RuntimeError("V8 combined-pool validation audit failed strict invariants.")

    print("\nSTRICT CHECK: PASS")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
