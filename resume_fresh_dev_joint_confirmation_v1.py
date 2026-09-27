#!/usr/bin/env python3
"""
Resume-only continuation for fresh_dev_joint_threshold_confirmation_v1.py.

The original run successfully:
  - built the fresh split
  - trained fresh S2/S3 models
  - scored and ranked the fresh S2/S3 candidates

It then stopped on a typo in load_truth():
    COUNT(g.matched_entity_id)
while the temp table aliases that column as `matched`.

This continuation:
  - does NOT retrain
  - does NOT rescore
  - does NOT read Final Holdout GT
  - does NOT read test data
  - does NOT change final outputs
  - reuses fresh_ranked_s2.parquet and fresh_ranked_s3.parquet
  - resumes at truth loading + policy evaluation
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import duckdb
import importlib


PROJECT = Path("/Users/harikeshshukla/mla")
MODULE = "fresh_dev_joint_threshold_confirmation_v1"

# Import existing functions/constants, but do not call its main().
m = importlib.import_module(MODULE)

OUT = m.OUT
GT = m.GT
BATCH = m.BATCH
MEMORY = m.MEMORY
THREADS = m.THREADS

CURRENT_S2 = m.CURRENT_S2
CURRENT_S3 = m.CURRENT_S3
CANDIDATE_S2 = m.CANDIDATE_S2
CANDIDATE_S3 = m.CANDIDATE_S3

RANKED_S2 = OUT / "fresh_ranked_s2.parquet"
RANKED_S3 = OUT / "fresh_ranked_s3.parquet"
FRESH_S1 = OUT / "fresh_dev_s1.parquet"
REPORT = OUT / "f05_fresh_dev_joint_confirmation_v1_report.json"


def qp(path):
    return "'" + str(path).replace("'", "''") + "'"


def require(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required existing artifact: {path}")


def load_truth_fixed(con):
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_fresh AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT)}) g
        INNER JOIN read_parquet({qp(FRESH_S1)}) f
          ON CAST(g.source1_entity_id AS VARCHAR)
           = CAST(f.source1_entity_id AS VARCHAR)
        WHERE COALESCE(g.label, 1) = 1
    """)

    rows = con.execute(f"""
        SELECT
            CAST(f.source1_entity_id AS VARCHAR) AS source1_entity_id,
            COUNT(g.matched) AS truth_count
        FROM read_parquet({qp(FRESH_S1)}) f
        LEFT JOIN gt_fresh g
          ON CAST(f.source1_entity_id AS VARCHAR) = g.s1
        GROUP BY f.source1_entity_id
    """).fetchall()

    return {str(s1): int(cnt) for s1, cnt in rows}


def main():
    for p in (FRESH_S1, RANKED_S2, RANKED_S3, GT):
        require(p)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    started = time.time()

    try:
        print("=" * 112)
        print("AMAZON ML CHALLENGE — RESUME FRESH DEV JOINT CONFIRMATION")
        print("=" * 112)
        print("REUSING existing fresh split + ranked outputs")
        print("Retrain       : NO")
        print("Rescore       : NO")
        print("Final Holdout GT: NOT READ")
        print("Test data     : NOT READ")
        print("Final outputs : NOT MODIFIED")

        truth = load_truth_fixed(con)
        ids = sorted(truth)

        s2 = m.load_top1(con, RANKED_S2, "S2")
        s3 = m.load_top1(con, RANKED_S3, "S3")

        print(f"Fresh dev S1        : {len(ids):,}")
        print(f"Fresh dev GT pairs  : {sum(truth.values()):,}")
        print(f"S2 top1 rows        : {len(s2):,}")
        print(f"S3 top1 rows        : {len(s3):,}")

        current = m.evaluate_policy(
            ids, truth, s2, s3, CURRENT_S2, CURRENT_S3
        )

        old_candidate = m.evaluate_policy(
            ids, truth, s2, s3, CANDIDATE_S2, CANDIDATE_S3
        )

        fresh_joint_thresholds = m.exact_joint_sweep(
            ids, truth, s2, s3
        )

        fresh_best = m.evaluate_policy(
            ids,
            truth,
            s2,
            s3,
            fresh_joint_thresholds["threshold_s2"],
            fresh_joint_thresholds["threshold_s3"],
        )

        print("\nCURRENT FROZEN POLICY")
        for k, v in current.items():
            print(f"{k:18s}: {v}")

        print("\nOLD VALIDATION JOINT CANDIDATE")
        for k, v in old_candidate.items():
            print(f"{k:18s}: {v}")

        print("\nFRESH-DEV EXACT JOINT BEST")
        for k, v in fresh_best.items():
            print(f"{k:18s}: {v}")

        print(
            f"\nOLD-CANDIDATE delta vs CURRENT: "
            f"{old_candidate['macro_f05'] - current['macro_f05']:+.12f}"
        )
        print(
            f"FRESH-JOINT-BEST delta vs CURRENT: "
            f"{fresh_best['macro_f05'] - current['macro_f05']:+.12f}"
        )

        report = {
            "version": "F05_FRESH_DEV_JOINT_CONFIRMATION_V1_RESUMED",
            "fresh_dev_s1": len(ids),
            "fresh_dev_gt_pairs": sum(truth.values()),
            "current_frozen_policy": current,
            "old_validation_joint_candidate_policy": old_candidate,
            "fresh_dev_exact_joint_best": fresh_best,
            "old_candidate_delta_vs_current": (
                old_candidate["macro_f05"] - current["macro_f05"]
            ),
            "fresh_best_delta_vs_current": (
                fresh_best["macro_f05"] - current["macro_f05"]
            ),
            "final_holdout_gt_read": False,
            "test_data_read": False,
            "retrained": False,
            "rescored": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
            "resumed_after_load_truth_typo": True,
            "elapsed_seconds": round(time.time() - started, 3),
        }

        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n" + "=" * 112)
        print("RESUME COMPLETE")
        print("=" * 112)
        print(f"Report: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
