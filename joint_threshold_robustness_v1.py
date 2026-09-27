#!/usr/bin/env python3
"""
Amazon ML Challenge 2026
Joint-threshold robustness audit around the already-found candidate policy.

READ-ONLY:
- Uses OLD V7 validation only.
- Does NOT read Final Holdout V2.
- Does NOT read test data.
- Does NOT retrain.
- Does NOT generate candidates.
- Does NOT modify final outputs.

Purpose:
The exact joint sweep found:
    S2 = 34.710811614990234
    S3 = 22.125261306762695

This script checks whether that optimum is a sharp spike or part of a
stable local region by evaluating nearby observed threshold states.

For each direction, the "neighbor" is defined by rank among unique observed
top-1 utility threshold values. Offsets:
    -10, -5, -2, -1, 0, +1, +2, +5, +10

For each of the 9x9 combinations, macro F0.5 is evaluated exactly over the
same 441,094-entity validation universe.

The report also compares the current frozen policy:
    (20.945884704589844, 20.945884704589844)

and the joint candidate:
    (34.710811614990234, 22.125261306762695)

No final files are touched.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import duckdb


PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"

S1_PATH = TRAIN / "train_source1.parquet"
GT_PATH = TRAIN / "ground_truth_pairs.parquet"

RANKED = {
    "S2": PROJECT / "candidate_output_v10_v8_leakage_safe"
          / "baseline_true_validation_ranked_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10_v8_leakage_safe"
          / "baseline_true_validation_ranked_s1_s3.parquet",
}

OUT_DIR = PROJECT / "validation_leakage_audit" / "f05_joint_threshold_robustness_v1"
REPORT = OUT_DIR / "f05_joint_threshold_robustness_v1_report.json"

MEMORY = os.environ.get("F05_ROBUST_MEMORY", "8GB")
THREADS = int(os.environ.get("F05_ROBUST_THREADS", "2"))

VALIDATION_MOD = 5
EXPECTED_S1 = 441_094
EXPECTED_GT = 1_527_077

CURRENT_S2 = 20.945884704589844
CURRENT_S3 = 20.945884704589844

BEST_S2 = 34.710811614990234
BEST_S3 = 22.125261306762695

OFFSETS = [-10, -5, -2, -1, 0, 1, 2, 5, 10]


def header(s: str) -> None:
    print("\n" + "=" * 110)
    print(s)
    print("=" * 110, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def score_f05(truth: int, pred: int, tp: int) -> float:
    if truth == 0 and pred == 0:
        return 1.0
    if truth == 0 or pred == 0:
        return 0.0
    return 1.25 * tp / (0.25 * truth + pred)


def load_universe(con):
    require(S1_PATH)
    require(GT_PATH)
    require(RANKED["S2"])
    require(RANKED["S3"])

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_s1 AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(S1_PATH)})
        WHERE MOD(
            ABS(HASH(CAST(entity_id AS VARCHAR))),
            {VALIDATION_MOD}
        ) = 0
    """)

    n_s1 = int(con.execute(
        "SELECT COUNT(*) FROM validation_s1"
    ).fetchone()[0])

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_all AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1,
            CAST(matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)})
    """)

    gt_total = int(con.execute("""
        SELECT COUNT(*)
        FROM gt_all g
        JOIN validation_s1 v ON g.s1 = v.s1
    """).fetchone()[0])

    if n_s1 != EXPECTED_S1:
        raise RuntimeError(
            f"Validation S1 mismatch: expected {EXPECTED_S1:,}, got {n_s1:,}"
        )

    if gt_total != EXPECTED_GT:
        raise RuntimeError(
            f"Validation GT mismatch: expected {EXPECTED_GT:,}, got {gt_total:,}"
        )

    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_counts AS
        SELECT
            v.s1,
            COUNT(g.matched)::INTEGER AS truth_count
        FROM validation_s1 v
        LEFT JOIN gt_all g ON g.s1 = v.s1
        GROUP BY v.s1
    """)

    truth = {
        str(s1): int(cnt)
        for s1, cnt in con.execute("""
            SELECT s1, truth_count
            FROM truth_counts
        """).fetchall()
    }

    ids = list(truth)

    print(f"Validation S1 : {n_s1:,}")
    print(f"Validation GT : {gt_total:,}")

    return ids, truth


def load_top1(con, path: Path, target: str):
    gt_table = f"gt_{target.lower()}"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {gt_table} AS
        SELECT s1, matched
        FROM gt_all
        WHERE starts_with(matched, '{target}-')
    """)

    rows = con.execute(f"""
        WITH top1 AS (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                CAST(candidate_entity_id AS VARCHAR) AS candidate,
                CAST(v10_utility AS DOUBLE) AS utility
            FROM read_parquet({qp(path)})
            WHERE v10_rank = 1
        )
        SELECT
            t.s1,
            t.utility,
            CASE WHEN g.matched IS NULL THEN 0 ELSE 1 END AS tp
        FROM top1 t
        LEFT JOIN {gt_table} g
          ON t.s1 = g.s1
         AND t.candidate = g.matched
    """).fetchall()

    out = {}
    for s1, utility, tp in rows:
        out[str(s1)] = (float(utility), int(tp))

    print(f"{target} top1 rows : {len(out):,}")
    return out


def build_thresholds(values, best_value):
    unique = sorted(set(values), reverse=True)
    if not unique:
        raise RuntimeError("No utility values found.")

    # Closest exact observed state to the requested best value.
    center = min(
        range(len(unique)),
        key=lambda i: abs(unique[i] - best_value),
    )

    states = {}
    for offset in OFFSETS:
        idx = center + offset
        if 0 <= idx < len(unique):
            states[offset] = unique[idx]

    return unique, center, states


def evaluate_policy(ids, truth, s2, s3, t2, t3):
    total_score = 0.0
    predicted = 0
    tp_total = 0
    truth_total = sum(truth.values())
    correctly_empty = 0
    fp_singletons = 0

    for s1 in ids:
        tc = truth[s1]
        p = 0
        tp = 0

        r2 = s2.get(s1)
        if r2 is not None and r2[0] >= t2:
            p += 1
            tp += r2[1]

        r3 = s3.get(s1)
        if r3 is not None and r3[0] >= t3:
            p += 1
            tp += r3[1]

        total_score += score_f05(tc, p, tp)
        predicted += p
        tp_total += tp

        if tc == 0 and p == 0:
            correctly_empty += 1
        if tc == 0:
            fp_singletons += p

    macro = total_score / len(ids)
    precision = tp_total / predicted if predicted else 0.0
    recall = tp_total / truth_total if truth_total else 0.0

    return {
        "threshold_s2": t2,
        "threshold_s3": t3,
        "macro_f05": macro,
        "precision": precision,
        "recall": recall,
        "predicted": predicted,
        "tp": tp_total,
        "correctly_empty": correctly_empty,
        "fp_singletons": fp_singletons,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    header("AMAZON ML CHALLENGE — JOINT THRESHOLD ROBUSTNESS AUDIT V1")
    print(f"Memory        : {MEMORY}")
    print(f"Threads       : {THREADS}")
    print("Validation    : OLD V7 ONLY")
    print("Final holdout : NOT READ")
    print("Model         : NOT TRAINED")
    print("Candidate gen : NOT RUN")
    print("Final outputs : NOT MODIFIED")
    print(f"Candidate S2  : {BEST_S2:.15f}")
    print(f"Candidate S3  : {BEST_S3:.15f}")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    started = time.time()

    try:
        ids, truth = load_universe(con)

        s2 = load_top1(con, RANKED["S2"], "S2")
        s3 = load_top1(con, RANKED["S3"], "S3")

        s2_values, s2_center, s2_states = build_thresholds(
            [v[0] for v in s2.values()],
            BEST_S2,
        )
        s3_values, s3_center, s3_states = build_thresholds(
            [v[0] for v in s3.values()],
            BEST_S3,
        )

        header("1. THRESHOLD NEIGHBOR STATES")

        print("S2 states:")
        for off in OFFSETS:
            if off in s2_states:
                print(f"  {off:+3d}: {s2_states[off]:.15f}")

        print("S3 states:")
        for off in OFFSETS:
            if off in s3_states:
                print(f"  {off:+3d}: {s3_states[off]:.15f}")

        results = []

        header("2. EVALUATE LOCAL 2-D GRID")

        for o2 in OFFSETS:
            if o2 not in s2_states:
                continue

            for o3 in OFFSETS:
                if o3 not in s3_states:
                    continue

                r = evaluate_policy(
                    ids,
                    truth,
                    s2,
                    s3,
                    s2_states[o2],
                    s3_states[o3],
                )
                r["s2_offset"] = o2
                r["s3_offset"] = o3
                results.append(r)

        results.sort(
            key=lambda x: (
                x["macro_f05"],
                x["precision"],
                -x["predicted"],
            ),
            reverse=True,
        )

        current = evaluate_policy(
            ids, truth, s2, s3, CURRENT_S2, CURRENT_S3
        )

        candidate = evaluate_policy(
            ids, truth, s2, s3, BEST_S2, BEST_S3
        )

        print("\nCURRENT FROZEN POLICY")
        for k, v in current.items():
            print(f"{k:18s}: {v}")

        print("\nJOINT CANDIDATE")
        for k, v in candidate.items():
            print(f"{k:18s}: {v}")

        print("\nTOP 20 LOCAL GRID STATES")
        for i, r in enumerate(results[:20], 1):
            print(
                f"{i:2d}. "
                f"S2off={r['s2_offset']:+3d} "
                f"S3off={r['s3_offset']:+3d} | "
                f"S2={r['threshold_s2']:.9f} | "
                f"S3={r['threshold_s3']:.9f} | "
                f"F0.5={r['macro_f05']:.12f} | "
                f"P={r['precision']:.6f} | "
                f"R={r['recall']:.6f}"
            )

        best_local = results[0]

        candidate_rank = next(
            (
                i + 1
                for i, r in enumerate(results)
                if abs(r["threshold_s2"] - BEST_S2) < 1e-12
                and abs(r["threshold_s3"] - BEST_S3) < 1e-12
            ),
            None,
        )

        local_spread = best_local["macro_f05"] - results[-1]["macro_f05"]

        close_10 = [
            r for r in results
            if r["macro_f05"] >= best_local["macro_f05"] - 0.002
        ]

        report = {
            "version": "F05_JOINT_THRESHOLD_ROBUSTNESS_V1",
            "validation_mod": VALIDATION_MOD,
            "validation_s1": len(ids),
            "validation_gt_pairs": sum(truth.values()),
            "current_frozen_policy": current,
            "joint_candidate": candidate,
            "joint_candidate_local_grid_rank": candidate_rank,
            "best_local_state": best_local,
            "local_grid_size": len(results),
            "local_grid_macro_spread": local_spread,
            "states_within_0_002_of_best": len(close_10),
            "top20_local_states": results[:20],
            "offsets": OFFSETS,
            "final_holdout_read": False,
            "test_data_read": False,
            "model_retrained": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
            "elapsed_seconds": round(time.time() - started, 3),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("3. ROBUSTNESS SUMMARY")
        print(f"Candidate local-grid rank : {candidate_rank}")
        print(f"Best local F0.5           : {best_local['macro_f05']:.12f}")
        print(
            f"Candidate F0.5 delta      : "
            f"{candidate['macro_f05'] - current['macro_f05']:+.12f}"
        )
        print(
            f"States within 0.002 of best: "
            f"{len(close_10)} / {len(results)}"
        )
        print(f"Report: {REPORT}")
        print("\nSAFE: no final candidate/output files were changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
