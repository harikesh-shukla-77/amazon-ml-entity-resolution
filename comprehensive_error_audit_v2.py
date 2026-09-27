#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
COMPREHENSIVE ERROR / BOTTLENECK AUDIT V2

ONE-SHOT READ-ONLY DIAGNOSTIC.

This combines the highest-value diagnostics we can do WITHOUT changing the
current final system:

  A) Current confirmed policy metrics on OLD V7 validation and FRESH DEV
  B) Candidate-pool coverage and Recall@1/3/5/10/20
  C) Error triage:
       - accepted top1 true positives
       - threshold-rejected true top1
       - wrong top1 with true candidate <=20
       - wrong top1 with true candidate >20
       - wrong top1 with no true candidate in candidate pool
       - singleton false positives
  D) Utility / threshold discrimination around the accepted threshold
  E) Utility gap between wrong top1 and best true candidate
  F) Best-true rank distribution
  G) Accepted false-positive exact-field patterns
  H) Candidate-count distribution per S1
  I) S2/S3 prediction-combination diagnostics
  J) Fixed-threshold comparison:
       current = 20.945884704589844 / 20.945884704589844
       confirmed = 34.710811614990234 / 22.125261306762695

IMPORTANT DATA PROTOCOL
-----------------------
OLD validation:
  development/diagnostic set only.

FRESH DEV:
  blind confirmation set for the already-selected threshold policy.

NOT READ:
  Final Holdout V2 ground truth
  Test dataset
  Final submission outputs

NOT RUN:
  training
  retraining
  candidate generation

NOT MODIFIED:
  matching_results.tsv
  candidate_pairs.tsv
  final ZIP

No parameter/model is selected from FRESH DEV in this script. Fresh DEV is
diagnostic confirmation only.
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

OLD_RANKED = {
    "S2": PROJECT / "candidate_output_v10_v8_leakage_safe"
          / "baseline_true_validation_ranked_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10_v8_leakage_safe"
          / "baseline_true_validation_ranked_s1_s3.parquet",
}

FRESH_DIR = (
    PROJECT / "validation_leakage_audit"
    / "f05_fresh_dev_joint_confirmation_v1"
)

FRESH_S1 = FRESH_DIR / "fresh_dev_s1.parquet"
FRESH_RANKED = {
    "S2": FRESH_DIR / "fresh_ranked_s2.parquet",
    "S3": FRESH_DIR / "fresh_ranked_s3.parquet",
}

OUT = (
    PROJECT / "validation_leakage_audit"
    / "comprehensive_error_audit_v2"
)
REPORT = OUT / "comprehensive_error_audit_v2_report.json"

MEMORY = os.environ.get("COMPREHENSIVE_AUDIT_MEMORY", "8GB")
THREADS = int(os.environ.get("COMPREHENSIVE_AUDIT_THREADS", "2"))

OLD_VALIDATION_MOD = 5

CURRENT = {
    "S2": 20.945884704589844,
    "S3": 20.945884704589844,
}

CONFIRMED = {
    "S2": 34.710811614990234,
    "S3": 22.125261306762695,
}


def header(s: str) -> None:
    print("\n" + "=" * 116)
    print(s)
    print("=" * 116, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def cols(con, path: Path) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({qp(path)})"
        ).fetchall()
    }


def detect_rank_schema(con, path: Path) -> tuple[str, str]:
    c = cols(con, path)

    utility_col = next(
        (
            x for x in
            ["v10_utility", "utility", "v8_utility", "v7_probability"]
            if x in c
        ),
        None,
    )

    rank_col = next(
        (
            x for x in
            ["v10_rank", "rank", "final_holdout_rank"]
            if x in c
        ),
        None,
    )

    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - c

    if utility_col is None or rank_col is None or missing:
        raise RuntimeError(
            f"Schema problem in {path}. "
            f"utility={utility_col}, rank={rank_col}, "
            f"missing={sorted(missing)}, cols={sorted(c)}"
        )

    return utility_col, rank_col


def create_eval_universe(
    con: duckdb.DuckDBPyConnection,
    mode: str,
) -> tuple[int, int]:
    if mode == "OLD":
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE eval_ids AS
            SELECT DISTINCT
                CAST(entity_id AS VARCHAR) AS s1
            FROM read_parquet({qp(S1_PATH)})
            WHERE MOD(
                ABS(HASH(CAST(entity_id AS VARCHAR))),
                {OLD_VALIDATION_MOD}
            ) = 0
        """)
    elif mode == "FRESH":
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE eval_ids AS
            SELECT DISTINCT
                CAST(source1_entity_id AS VARCHAR) AS s1
            FROM read_parquet({qp(FRESH_S1)})
        """)
    else:
        raise ValueError(mode)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_pairs AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)}) g
        INNER JOIN eval_ids e
          ON CAST(g.source1_entity_id AS VARCHAR) = e.s1
        WHERE COALESCE(g.label, 1) = 1
    """)

    n = int(con.execute(
        "SELECT COUNT(*) FROM eval_ids"
    ).fetchone()[0])

    gt = int(con.execute(
        "SELECT COUNT(*) FROM gt_pairs"
    ).fetchone()[0])

    return n, gt


def analyze_target(
    con: duckdb.DuckDBPyConnection,
    target: str,
    ranked_path: Path,
    threshold: float,
) -> dict:
    utility_col, rank_col = detect_rank_schema(con, ranked_path)

    # The ranked files are complete candidate pools with ranking information.
    # We keep this as a TEMP VIEW to avoid copying tens of millions of rows.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW ranked_{target.lower()} AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1,
            CAST(candidate_entity_id AS VARCHAR) AS candidate,
            CAST({utility_col} AS DOUBLE) AS utility,
            CAST({rank_col} AS BIGINT) AS rnk
        FROM read_parquet({qp(ranked_path)})
    """)

    # Basic candidate-pool / recall quantities.
    gt_pairs = int(con.execute("""
        SELECT COUNT(*) FROM gt_pairs
    """).fetchone()[0])

    candidate_covered = int(con.execute(f"""
        SELECT COUNT(*)
        FROM gt_pairs g
        WHERE EXISTS (
            SELECT 1
            FROM ranked_{target.lower()} r
            WHERE r.s1 = g.s1
              AND r.candidate = g.matched
        )
    """).fetchone()[0])

    recall_pairs = {}
    for k in (1, 3, 5, 10, 20):
        covered = int(con.execute(f"""
            SELECT COUNT(*)
            FROM gt_pairs g
            INNER JOIN ranked_{target.lower()} r
              ON r.s1 = g.s1
             AND r.candidate = g.matched
            WHERE r.rnk <= {k}
        """).fetchone()[0])
        recall_pairs[str(k)] = {
            "covered": covered,
            "pct": (100.0 * covered / gt_pairs) if gt_pairs else 0.0,
        }

    # Per-S1 summary around top1 / true-match rank.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW summary_{target.lower()} AS
        WITH true_rank AS (
            SELECT
                g.s1,
                MIN(r.rnk) AS best_true_rank,
                MAX(r.utility) AS best_true_utility
            FROM gt_pairs g
            INNER JOIN ranked_{target.lower()} r
              ON r.s1 = g.s1
             AND r.candidate = g.matched
            GROUP BY g.s1
        ),
        top1 AS (
            SELECT
                s1,
                candidate AS top1_candidate,
                utility AS top1_utility
            FROM ranked_{target.lower()}
            WHERE rnk = 1
        ),
        truth_count AS (
            SELECT s1, COUNT(*) AS truth_count
            FROM gt_pairs
            GROUP BY s1
        ),
        all_s1 AS (
            SELECT s1 FROM eval_ids
        )
        SELECT
            a.s1,
            COALESCE(t.truth_count, 0)::INTEGER AS truth_count,
            o.top1_candidate,
            o.top1_utility,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM gt_pairs g
                    WHERE g.s1 = a.s1
                      AND g.matched = o.top1_candidate
                )
                THEN 1 ELSE 0
            END AS top1_tp,
            tr.best_true_rank,
            tr.best_true_utility,
            CASE
                WHEN tr.best_true_utility IS NOT NULL
                THEN o.top1_utility - tr.best_true_utility
                ELSE NULL
            END AS utility_gap
        FROM all_s1 a
        LEFT JOIN truth_count t ON t.s1 = a.s1
        LEFT JOIN top1 o ON o.s1 = a.s1
        LEFT JOIN true_rank tr ON tr.s1 = a.s1
    """)

    # Threshold acceptance / error buckets.
    accepted = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE top1_utility >= {threshold}
    """).fetchone()[0])

    accepted_tp = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE top1_utility >= {threshold}
          AND top1_tp = 1
    """).fetchone()[0])

    accepted_fp = accepted - accepted_tp

    rejected_top1_tp = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE top1_tp = 1
          AND (top1_utility < {threshold} OR top1_utility IS NULL)
    """).fetchone()[0])

    wrong_with_true_le20 = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND top1_tp = 0
          AND best_true_rank <= 20
    """).fetchone()[0])

    wrong_true_rank_gt20 = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND top1_tp = 0
          AND best_true_rank > 20
    """).fetchone()[0])

    wrong_candidate_missing = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND top1_tp = 0
          AND best_true_rank IS NULL
    """).fetchone()[0])

    singleton_fp = int(con.execute(f"""
        SELECT COUNT(*)
        FROM summary_{target.lower()}
        WHERE truth_count = 0
          AND top1_utility >= {threshold}
    """).fetchone()[0])

    # S1-level perfect-candidate availability.
    s1_with_any_candidate = int(con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT s1
            FROM ranked_{target.lower()}
        )
    """).fetchone()[0])

    s1_with_true_candidate = int(con.execute("""
        SELECT COUNT(DISTINCT g.s1)
        FROM gt_pairs g
        WHERE EXISTS (
            SELECT 1
            FROM ranked_{target} r
            WHERE r.s1 = g.s1
              AND r.candidate = g.matched
        )
    """.replace("{target}", target.lower())).fetchone()[0])

    # Candidate count distribution per S1.
    cand_stats = con.execute(f"""
        SELECT
            COUNT(*) AS s1s,
            AVG(n)::DOUBLE AS mean_candidates,
            APPROX_QUANTILE(n, 0.50) AS p50,
            APPROX_QUANTILE(n, 0.90) AS p90,
            APPROX_QUANTILE(n, 0.99) AS p99,
            MAX(n) AS max_candidates
        FROM (
            SELECT s1, COUNT(*) AS n
            FROM ranked_{target.lower()}
            GROUP BY s1
        )
    """).fetchone()

    # Utility-gap / true-rank diagnostics for wrong top1s where a true
    # candidate is actually in the pool.
    gap_stats = con.execute(f"""
        SELECT
            AVG(utility_gap)::DOUBLE AS mean_gap,
            APPROX_QUANTILE(utility_gap, 0.50) AS median_gap,
            APPROX_QUANTILE(utility_gap, 0.90) AS p90_gap,
            MIN(utility_gap) AS min_gap,
            MAX(utility_gap) AS max_gap,
            APPROX_QUANTILE(best_true_rank, 0.50) AS median_best_true_rank,
            APPROX_QUANTILE(best_true_rank, 0.90) AS p90_best_true_rank
        FROM summary_{target.lower()}
        WHERE truth_count > 0
          AND top1_tp = 0
          AND best_true_rank IS NOT NULL
    """).fetchone()

    # Utility bins centered on the production threshold.
    bins = con.execute(f"""
        SELECT
            CASE
                WHEN top1_utility IS NULL THEN 'NO_TOP1'
                WHEN top1_utility < {threshold} - 10 THEN 'BELOW_LT10'
                WHEN top1_utility < {threshold} - 5 THEN 'BELOW_10_TO_5'
                WHEN top1_utility < {threshold} THEN 'BELOW_5_TO_0'
                WHEN top1_utility < {threshold} + 5 THEN 'ABOVE_0_TO_5'
                WHEN top1_utility < {threshold} + 10 THEN 'ABOVE_5_TO_10'
                ELSE 'ABOVE_GT10'
            END AS bin,
            COUNT(*) AS n,
            SUM(CASE WHEN top1_tp = 1 THEN 1 ELSE 0 END) AS tp,
            SUM(
                CASE
                    WHEN truth_count = 0 AND top1_utility >= {threshold}
                    THEN 1 ELSE 0
                END
            ) AS singleton_fp
        FROM summary_{target.lower()}
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    # Accepted FP exactness is not always present, because some fresh ranked
    # files only contain IDs + utility. Return what can be diagnosed here.
    exactness_patterns = None
    rank_cols = cols(con, ranked_path)
    exact_cols = {
        "name_exact",
        "address_exact",
        "country_exact",
        "name_similarity",
        "address_similarity",
        "evidence_file_count",
    }

    if exact_cols.issubset(rank_cols):
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW features_{target.lower()} AS
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                CAST(candidate_entity_id AS VARCHAR) AS candidate,
                CAST({utility_col} AS DOUBLE) AS utility,
                CAST({rank_col} AS BIGINT) AS rnk,
                CAST(name_exact AS DOUBLE) AS name_exact,
                CAST(address_exact AS DOUBLE) AS address_exact,
                CAST(country_exact AS DOUBLE) AS country_exact,
                CAST(name_similarity AS DOUBLE) AS name_similarity,
                CAST(address_similarity AS DOUBLE) AS address_similarity,
                CAST(evidence_file_count AS DOUBLE) AS evidence_file_count
            FROM read_parquet({qp(ranked_path)})
            WHERE CAST({rank_col} AS BIGINT) = 1
        """)

        rows = con.execute(f"""
            SELECT
                name_exact,
                address_exact,
                country_exact,
                COUNT(*) AS n
            FROM features_{target.lower()} f
            INNER JOIN summary_{target.lower()} s
              ON f.s1 = s.s1
             AND f.candidate = s.top1_candidate
            WHERE s.top1_utility >= {threshold}
              AND s.top1_tp = 0
            GROUP BY 1,2,3
            ORDER BY n DESC
            LIMIT 20
        """).fetchall()

        exactness_patterns = [
            {
                "name_exact": int(r[0]),
                "address_exact": int(r[1]),
                "country_exact": int(r[2]),
                "count": int(r[3]),
            }
            for r in rows
        ]

    return {
        "target": target,
        "threshold": threshold,
        "s1_entities": int(con.execute(
            "SELECT COUNT(*) FROM eval_ids"
        ).fetchone()[0]),
        "ground_truth_pairs": gt_pairs,
        "candidate_covered_gt_pairs": candidate_covered,
        "candidate_missing_gt_pairs": gt_pairs - candidate_covered,
        "recall_at_k_pairs": recall_pairs,
        "accepted_top1_predictions": accepted,
        "accepted_top1_tp": accepted_tp,
        "accepted_top1_fp": accepted_fp,
        "accepted_precision": accepted_tp / accepted if accepted else 0.0,
        "threshold_rejected_true_top1": rejected_top1_tp,
        "wrong_top1_true_rank_le20": wrong_with_true_le20,
        "wrong_top1_true_rank_gt20": wrong_true_rank_gt20,
        "wrong_top1_candidate_missing": wrong_candidate_missing,
        "singleton_false_positives": singleton_fp,
        "s1_with_any_candidate": s1_with_any_candidate,
        "s1_with_true_candidate_in_pool": s1_with_true_candidate,
        "candidate_count_distribution": {
            "s1s_with_candidates": int(cand_stats[0] or 0),
            "mean": float(cand_stats[1] or 0.0),
            "p50": float(cand_stats[2] or 0.0),
            "p90": float(cand_stats[3] or 0.0),
            "p99": float(cand_stats[4] or 0.0),
            "max": int(cand_stats[5] or 0),
        },
        "wrong_top1_gap_stats": {
            "mean_utility_gap": (
                float(gap_stats[0])
                if gap_stats[0] is not None else None
            ),
            "median_utility_gap": (
                float(gap_stats[1])
                if gap_stats[1] is not None else None
            ),
            "p90_utility_gap": (
                float(gap_stats[2])
                if gap_stats[2] is not None else None
            ),
            "min_utility_gap": (
                float(gap_stats[3])
                if gap_stats[3] is not None else None
            ),
            "max_utility_gap": (
                float(gap_stats[4])
                if gap_stats[4] is not None else None
            ),
            "median_best_true_rank": (
                float(gap_stats[5])
                if gap_stats[5] is not None else None
            ),
            "p90_best_true_rank": (
                float(gap_stats[6])
                if gap_stats[6] is not None else None
            ),
        },
        "utility_bins": [
            {
                "bin": str(r[0]),
                "count": int(r[1]),
                "tp": int(r[2] or 0),
                "singleton_fp": int(r[3] or 0),
            }
            for r in bins
        ],
        "accepted_fp_exactness_patterns": exactness_patterns,
    }


def pair_policy_metrics(
    con: duckdb.DuckDBPyConnection,
    threshold_s2: float,
    threshold_s3: float,
) -> dict:
    """
    Evaluate both sources jointly at fixed thresholds on the current eval
    universe. Uses only top1 utilities and GT. No tuning occurs here.
    """
    con.execute("""
        CREATE OR REPLACE TEMP VIEW joint_top1 AS
        SELECT
            e.s1,
            tc.truth_count,
            s2.top1_utility AS s2_utility,
            s2.top1_tp AS s2_tp,
            s3.top1_utility AS s3_utility,
            s3.top1_tp AS s3_tp
        FROM eval_ids e
        LEFT JOIN (
            SELECT s1, top1_utility, top1_tp
            FROM summary_s2
        ) s2 ON s2.s1 = e.s1
        LEFT JOIN (
            SELECT s1, top1_utility, top1_tp
            FROM summary_s3
        ) s3 ON s3.s1 = e.s1
        LEFT JOIN (
            SELECT s1, truth_count
            FROM (
                SELECT s1, MAX(truth_count) AS truth_count
                FROM summary_s2
                GROUP BY s1
            )
        ) tc ON tc.s1 = e.s1
    """)

    row = con.execute(f"""
        SELECT
            SUM(
                CASE
                    WHEN truth_count = 0 AND pred = 0 THEN 1.0
                    WHEN truth_count = 0 THEN 0.0
                    WHEN pred = 0 THEN 0.0
                    ELSE
                        1.25 * tp
                        / (0.25 * truth_count + pred)
                END
            ) / COUNT(*) AS macro_f05,
            SUM(tp) AS tp,
            SUM(pred) AS predicted,
            SUM(CASE WHEN truth_count = 0 AND pred = 0 THEN 1 ELSE 0 END)
                AS correctly_empty,
            SUM(CASE WHEN truth_count = 0 THEN pred ELSE 0 END)
                AS fp_singletons
        FROM (
            SELECT
                truth_count,
                (CASE WHEN s2_utility >= {threshold_s2} THEN 1 ELSE 0 END)
                + (CASE WHEN s3_utility >= {threshold_s3} THEN 1 ELSE 0 END)
                    AS pred,
                (CASE
                    WHEN s2_utility >= {threshold_s2} THEN s2_tp ELSE 0 END)
                + (CASE
                    WHEN s3_utility >= {threshold_s3} THEN s3_tp ELSE 0 END)
                    AS tp
            FROM joint_top1
        )
    """).fetchone()

    tp = int(row[1] or 0)
    pred = int(row[2] or 0)
    truth_total = int(con.execute(
        "SELECT COUNT(*) FROM gt_pairs"
    ).fetchone()[0])

    return {
        "threshold_s2": threshold_s2,
        "threshold_s3": threshold_s3,
        "macro_f05": float(row[0] or 0.0),
        "precision": tp / pred if pred else 0.0,
        "recall": tp / truth_total if truth_total else 0.0,
        "predicted": pred,
        "tp": tp,
        "correctly_empty": int(row[3] or 0),
        "fp_singletons": int(row[4] or 0),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    required = [
        S1_PATH,
        GT_PATH,
        OLD_RANKED["S2"],
        OLD_RANKED["S3"],
        FRESH_S1,
        FRESH_RANKED["S2"],
        FRESH_RANKED["S3"],
    ]

    for p in required:
        require(p)

    header("AMAZON ML CHALLENGE — COMPREHENSIVE ERROR AUDIT V2")
    print(f"Memory        : {MEMORY}")
    print(f"Threads       : {THREADS}")
    print("Final Holdout GT : NOT READ")
    print("Test data        : NOT READ")
    print("Training         : NOT RUN")
    print("Candidate gen    : NOT RUN")
    print("Final outputs    : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    tmp = OUT / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory={qp(tmp)}")

    started = time.time()

    report = {
        "version": "COMPREHENSIVE_ERROR_AUDIT_V2",
        "current_policy": CURRENT,
        "confirmed_policy": CONFIRMED,
        "datasets": {},
        "joint_policy_comparison": {},
        "safety": {
            "final_holdout_gt_read": False,
            "test_data_read": False,
            "training_run": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
        },
    }

    try:
        # OLD
        header("1. OLD V7 VALIDATION")
        n_old, gt_old = create_eval_universe(con, "OLD")
        print(f"OLD validation S1: {n_old:,}")
        print(f"OLD validation GT: {gt_old:,}")

        old_s2 = analyze_target(
            con,
            "S2",
            OLD_RANKED["S2"],
            CONFIRMED["S2"],
        )
        old_s3 = analyze_target(
            con,
            "S3",
            OLD_RANKED["S3"],
            CONFIRMED["S3"],
        )

        old_joint_current = pair_policy_metrics(
            con,
            CURRENT["S2"],
            CURRENT["S3"],
        )
        old_joint_confirmed = pair_policy_metrics(
            con,
            CONFIRMED["S2"],
            CONFIRMED["S3"],
        )

        report["datasets"]["OLD"] = {
            "s1_entities": n_old,
            "gt_pairs": gt_old,
            "S2": old_s2,
            "S3": old_s3,
            "joint_current_policy": old_joint_current,
            "joint_confirmed_policy": old_joint_confirmed,
        }

        # FRESH
        header("2. FRESH DEVELOPMENT CONFIRMATION")
        n_fresh, gt_fresh = create_eval_universe(con, "FRESH")
        print(f"FRESH dev S1: {n_fresh:,}")
        print(f"FRESH dev GT: {gt_fresh:,}")

        fresh_s2 = analyze_target(
            con,
            "S2",
            FRESH_RANKED["S2"],
            CONFIRMED["S2"],
        )
        fresh_s3 = analyze_target(
            con,
            "S3",
            FRESH_RANKED["S3"],
            CONFIRMED["S3"],
        )

        fresh_joint_current = pair_policy_metrics(
            con,
            CURRENT["S2"],
            CURRENT["S3"],
        )
        fresh_joint_confirmed = pair_policy_metrics(
            con,
            CONFIRMED["S2"],
            CONFIRMED["S3"],
        )

        report["datasets"]["FRESH"] = {
            "s1_entities": n_fresh,
            "gt_pairs": gt_fresh,
            "S2": fresh_s2,
            "S3": fresh_s3,
            "joint_current_policy": fresh_joint_current,
            "joint_confirmed_policy": fresh_joint_confirmed,
        }

        # Cross-split generalization summary.
        report["generalization"] = {
            "old_confirmed_macro_f05": old_joint_confirmed["macro_f05"],
            "fresh_confirmed_macro_f05": fresh_joint_confirmed["macro_f05"],
            "old_current_macro_f05": old_joint_current["macro_f05"],
            "fresh_current_macro_f05": fresh_joint_current["macro_f05"],
            "fresh_confirmed_gain_vs_current": (
                fresh_joint_confirmed["macro_f05"]
                - fresh_joint_current["macro_f05"]
            ),
            "confirmed_policy_precision_fresh": (
                fresh_joint_confirmed["precision"]
            ),
            "confirmed_policy_recall_fresh": (
                fresh_joint_confirmed["recall"]
            ),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("3. KEY TRIAGE")
        for mode in ("OLD", "FRESH"):
            for target in ("S2", "S3"):
                x = report["datasets"][mode][target]
                print(
                    f"{mode} {target}: "
                    f"GT={x['ground_truth_pairs']:,} | "
                    f"candidate-missing={x['candidate_missing_gt_pairs']:,} | "
                    f"R@20={x['recall_at_k_pairs']['20']['pct']:.4f}% | "
                    f"accepted={x['accepted_top1_predictions']:,} | "
                    f"TP={x['accepted_top1_tp']:,} | "
                    f"FP={x['accepted_top1_fp']:,} | "
                    f"reject-TP={x['threshold_rejected_true_top1']:,} | "
                    f"wrong-rank<=20={x['wrong_top1_true_rank_le20']:,} | "
                    f"wrong-rank>20={x['wrong_top1_true_rank_gt20']:,} | "
                    f"candidate-missing-S1={x['wrong_top1_candidate_missing']:,}"
                )

        print("\nJOINT POLICY")
        for mode in ("OLD", "FRESH"):
            cur = report["datasets"][mode]["joint_current_policy"]
            conf = report["datasets"][mode]["joint_confirmed_policy"]
            print(
                f"{mode}: current F0.5={cur['macro_f05']:.12f} | "
                f"confirmed F0.5={conf['macro_f05']:.12f} | "
                f"delta={conf['macro_f05']-cur['macro_f05']:+.12f}"
            )

        print(f"\nReport: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")

    finally:
        con.close()

    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
