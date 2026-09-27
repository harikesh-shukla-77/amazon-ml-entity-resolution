#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
DEEP ERROR ANALYSIS V1 — THRESHOLD POLICY + RANKING + CANDIDATE RECALL

Read-only diagnostics only.

It analyzes:
  1) OLD V7 validation (used for development)
  2) FRESH development split (blind confirmation)

It does NOT:
  - train
  - retrain
  - generate candidates
  - read Final Holdout V2 GT
  - read test data
  - modify any final output

Purpose:
Identify the highest-value next improvement:
  candidate generation vs ranking vs threshold / false-positive control.

For each target source, it reports:
  - GT pairs
  - candidate-covered GT pairs
  - candidate-missing GT pairs
  - GT found at rank 1/3/5/10/20
  - accepted predictions under the confirmed thresholds
  - accepted TP / FP
  - threshold-rejected top1 TPs
  - wrong top1 cases where the true match is inside the candidate pool
  - wrong top1 cases where the true match is outside top20

It also reports score-gap diagnostics when a true candidate is present:
  top1 utility
  best true-match utility
  top1 utility - best-true utility
  best true rank

This is designed to tell us where another 0.03+ Macro F0.5 could realistically come from.
"""

from __future__ import annotations

import json
import math
import os
import statistics
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
    / "deep_error_analysis_v1"
)
REPORT = OUT / "deep_error_analysis_v1_report.json"

MEMORY = os.environ.get("ERROR_AUDIT_MEMORY", "8GB")
THREADS = int(os.environ.get("ERROR_AUDIT_THREADS", "2"))

OLD_VALIDATION_MOD = 5

THRESHOLD = {
    "S2": 34.710811614990234,
    "S3": 22.125261306762695,
}


def header(s: str) -> None:
    print("\n" + "=" * 112)
    print(s)
    print("=" * 112, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")


def columns(con, path: Path) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({qp(path)})"
        ).fetchall()
    }


def detect(path: Path, con) -> tuple[str, str]:
    cols = columns(con, path)

    utility = next(
        (c for c in ["v10_utility", "utility", "v8_utility", "v7_probability"]
         if c in cols),
        None,
    )
    rank = next(
        (c for c in ["v10_rank", "rank", "final_holdout_rank"]
         if c in cols),
        None,
    )

    if utility is None or rank is None:
        raise RuntimeError(
            f"Could not detect utility/rank columns in {path}."
            f" Available={sorted(cols)}"
        )

    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - cols
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    return utility, rank


def build_eval_ids(con, mode: str):
    if mode == "OLD":
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE eval_ids AS
            SELECT DISTINCT CAST(entity_id AS VARCHAR) AS s1
            FROM read_parquet({qp(S1_PATH)})
            WHERE MOD(
                ABS(HASH(CAST(entity_id AS VARCHAR))),
                {OLD_VALIDATION_MOD}
            ) = 0
        """)
    elif mode == "FRESH":
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE eval_ids AS
            SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS s1
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

    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_counts AS
        SELECT
            e.s1,
            COUNT(g.matched)::INTEGER AS truth_count
        FROM eval_ids e
        LEFT JOIN gt_pairs g
          ON e.s1 = g.s1
        GROUP BY e.s1
    """)

    n = int(con.execute(
        "SELECT COUNT(*) FROM eval_ids"
    ).fetchone()[0])

    gt = int(con.execute(
        "SELECT COUNT(*) FROM gt_pairs"
    ).fetchone()[0])

    return n, gt


def analyze_target(con, path: Path, target: str, mode: str):
    utility, rank = detect(path, con)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ranked_local AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1,
            CAST(candidate_entity_id AS VARCHAR) AS candidate,
            CAST({utility} AS DOUBLE) AS utility,
            CAST({rank} AS BIGINT) AS rnk,
            {("name_exact" if "name_exact" in columns(con, path) else "NULL")} AS name_exact,
            {("address_exact" if "address_exact" in columns(con, path) else "NULL")} AS address_exact,
            {("country_exact" if "country_exact" in columns(con, path) else "NULL")} AS country_exact,
            {("evidence_file_count" if "evidence_file_count" in columns(con, path) else "NULL")} AS evidence_file_count,
            {("evidence_rows" if "evidence_rows" in columns(con, path) else "NULL")} AS evidence_rows,
            {("name_similarity" if "name_similarity" in columns(con, path) else "NULL")} AS name_similarity,
            {("address_similarity" if "address_similarity" in columns(con, path) else "NULL")} AS address_similarity
        FROM read_parquet({qp(path)})
    """)

    # One row per S1 with top1/top20 summaries and best rank of any true match.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s1_summary AS
        SELECT
            e.s1,
            tc.truth_count,

            MAX(CASE WHEN r.rnk = 1 THEN r.candidate END) AS top1_candidate,
            MAX(CASE WHEN r.rnk = 1 THEN r.utility END) AS top1_utility,
            MAX(CASE WHEN r.rnk = 1 THEN r.name_similarity END) AS top1_name_similarity,
            MAX(CASE WHEN r.rnk = 1 THEN r.address_similarity END) AS top1_address_similarity,
            MAX(CASE WHEN r.rnk = 1 THEN r.name_exact END) AS top1_name_exact,
            MAX(CASE WHEN r.rnk = 1 THEN r.address_exact END) AS top1_address_exact,
            MAX(CASE WHEN r.rnk = 1 THEN r.country_exact END) AS top1_country_exact,
            MAX(CASE WHEN r.rnk = 1 THEN r.evidence_file_count END) AS top1_evidence_file_count,

            MIN(
                CASE
                    WHEN g.matched IS NOT NULL THEN r.rnk
                    ELSE NULL
                END
            ) AS best_true_rank,

            MAX(
                CASE
                    WHEN g.matched IS NOT NULL THEN r.utility
                    ELSE NULL
                END
            ) AS best_true_utility,

            MAX(
                CASE
                    WHEN r.rnk = 1
                     AND g.matched = r.candidate
                    THEN 1 ELSE 0
                END
            ) AS top1_tp,

            MAX(
                CASE
                    WHEN r.rnk <= 3 AND g.matched = r.candidate
                    THEN 1 ELSE 0
                END
            ) AS top3_tp,

            MAX(
                CASE
                    WHEN r.rnk <= 5 AND g.matched = r.candidate
                    THEN 1 ELSE 0
                END
            ) AS top5_tp,

            MAX(
                CASE
                    WHEN r.rnk <= 10 AND g.matched = r.candidate
                    THEN 1 ELSE 0
                END
            ) AS top10_tp,

            MAX(
                CASE
                    WHEN r.rnk <= 20 AND g.matched = r.candidate
                    THEN 1 ELSE 0
                END
            ) AS top20_tp

        FROM eval_ids e
        LEFT JOIN truth_counts tc
          ON e.s1 = tc.s1
        LEFT JOIN ranked_local r
          ON e.s1 = r.s1
        LEFT JOIN gt_pairs g
          ON r.s1 = g.s1
         AND r.candidate = g.matched
        GROUP BY e.s1, tc.truth_count
    """)

    threshold = THRESHOLD[target]

    # Candidate-level aggregates.
    candidate_covered = int(con.execute(f"""
        SELECT COUNT(*)
        FROM gt_pairs g
        WHERE EXISTS (
            SELECT 1
            FROM ranked_local r
            WHERE r.s1 = g.s1
              AND r.candidate = g.matched
        )
    """).fetchone()[0])

    r1 = int(con.execute(
        "SELECT COALESCE(SUM(top1_tp),0) FROM s1_summary"
    ).fetchone()[0])
    r3 = int(con.execute(
        "SELECT COALESCE(SUM(top3_tp),0) FROM s1_summary"
    ).fetchone()[0])
    r5 = int(con.execute(
        "SELECT COALESCE(SUM(top5_tp),0) FROM s1_summary"
    ).fetchone()[0])
    r10 = int(con.execute(
        "SELECT COALESCE(SUM(top10_tp),0) FROM s1_summary"
    ).fetchone()[0])
    r20 = int(con.execute(
        "SELECT COALESCE(SUM(top20_tp),0) FROM s1_summary"
    ).fetchone()[0])

    accepted = int(con.execute(f"""
        SELECT COUNT(*)
        FROM s1_summary
        WHERE top1_utility >= {threshold}
    """).fetchone()[0])

    accepted_tp = int(con.execute(f"""
        SELECT COUNT(*)
        FROM s1_summary
        WHERE top1_utility >= {threshold}
          AND top1_tp = 1
    """).fetchone()[0])

    accepted_fp = accepted - accepted_tp

    rejected_top1_tp = int(con.execute(f"""
        SELECT COUNT(*)
        FROM s1_summary
        WHERE top1_tp = 1
          AND top1_utility < {threshold}
    """).fetchone()[0])

    top1_wrong_with_candidate = int(con.execute("""
        SELECT COUNT(*)
        FROM s1_summary
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND top1_tp = 0
          AND best_true_rank IS NOT NULL
          AND best_true_rank <= 20
    """).fetchone()[0])

    top1_wrong_rank_gt20 = int(con.execute("""
        SELECT COUNT(*)
        FROM s1_summary
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND top1_tp = 0
          AND best_true_rank > 20
    """).fetchone()[0])

    top1_wrong_candidate_missing = int(con.execute("""
        SELECT COUNT(*)
        FROM s1_summary
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND top1_tp = 0
          AND best_true_rank IS NULL
    """).fetchone()[0])

    # S1-level score gaps where a true candidate exists.
    gap_rows = con.execute("""
        SELECT
            top1_utility,
            best_true_utility,
            best_true_rank,
            top1_utility - best_true_utility AS utility_gap
        FROM s1_summary
        WHERE truth_count > 0
          AND top1_utility IS NOT NULL
          AND best_true_utility IS NOT NULL
          AND top1_tp = 0
    """).fetchall()

    gaps = [
        float(r[3])
        for r in gap_rows
        if r[3] is not None and math.isfinite(float(r[3]))
    ]

    ranks = [
        int(r[2])
        for r in gap_rows
        if r[2] is not None
    ]

    summary = {
        "mode": mode,
        "target": target,
        "threshold": threshold,
        "s1_entities": int(con.execute(
            "SELECT COUNT(*) FROM eval_ids"
        ).fetchone()[0]),
        "ground_truth_pairs": int(con.execute(
            "SELECT COUNT(*) FROM gt_pairs"
        ).fetchone()[0]),
        "candidate_covered_gt_pairs": candidate_covered,
        "candidate_missing_gt_pairs": int(
            con.execute("SELECT COUNT(*) FROM gt_pairs").fetchone()[0]
        ) - candidate_covered,
        "recall_at_1_pairs": r1,
        "recall_at_3_pairs": r3,
        "recall_at_5_pairs": r5,
        "recall_at_10_pairs": r10,
        "recall_at_20_pairs": r20,
        "accepted_top1_predictions": accepted,
        "accepted_top1_tp": accepted_tp,
        "accepted_top1_fp": accepted_fp,
        "accepted_precision": accepted_tp / accepted if accepted else 0.0,
        "rejected_top1_true_positive": rejected_top1_tp,
        "wrong_top1_true_candidate_rank_le20": top1_wrong_with_candidate,
        "wrong_top1_true_rank_gt20": top1_wrong_rank_gt20,
        "wrong_top1_candidate_missing": top1_wrong_candidate_missing,
        "wrong_top1_mean_utility_gap": (
            statistics.mean(gaps) if gaps else None
        ),
        "wrong_top1_median_utility_gap": (
            statistics.median(gaps) if gaps else None
        ),
        "wrong_top1_median_best_true_rank": (
            statistics.median(ranks) if ranks else None
        ),
    }

    # Accepted false-positive shape: exact-field collision / similarity bins.
    fp_shape = con.execute(f"""
        SELECT
            COALESCE(top1_name_exact, -1),
            COALESCE(top1_address_exact, -1),
            COALESCE(top1_country_exact, -1),
            COUNT(*) AS n
        FROM s1_summary
        WHERE top1_utility >= {threshold}
          AND top1_tp = 0
        GROUP BY 1,2,3
        ORDER BY n DESC
        LIMIT 12
    """).fetchall()

    summary["accepted_fp_exactness_patterns"] = [
        {
            "name_exact": int(r[0]),
            "address_exact": int(r[1]),
            "country_exact": int(r[2]),
            "count": int(r[3]),
        }
        for r in fp_shape
    ]

    # Score distribution around the threshold.
    bins = con.execute(f"""
        SELECT
            CASE
                WHEN top1_utility IS NULL THEN 'NO_TOP1'
                WHEN top1_utility < {threshold} - 5 THEN 'LOW'
                WHEN top1_utility < {threshold} THEN 'JUST_BELOW'
                WHEN top1_utility < {threshold} + 5 THEN 'JUST_ABOVE'
                ELSE 'HIGH'
            END AS bin,
            COUNT(*) AS n,
            SUM(CASE WHEN top1_tp = 1 THEN 1 ELSE 0 END) AS tp
        FROM s1_summary
        WHERE top1_utility IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchall()

    summary["top1_utility_bins"] = [
        {
            "bin": str(r[0]),
            "count": int(r[1]),
            "tp": int(r[2] or 0),
        }
        for r in bins
    ]

    return summary


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

    header("AMAZON ML CHALLENGE — DEEP ERROR ANALYSIS V1")
    print(f"Memory        : {MEMORY}")
    print(f"Threads       : {THREADS}")
    print("Old validation: diagnostic/development")
    print("Fresh dev     : blind confirmation")
    print("Final Holdout : NOT READ")
    print("Test data     : NOT READ")
    print("Training      : NOT RUN")
    print("Candidate gen : NOT RUN")
    print("Final outputs : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    try:
        report = {
            "version": "DEEP_ERROR_ANALYSIS_V1",
            "threshold_policy": THRESHOLD,
            "datasets": {},
        }

        for mode, id_path, ranked_paths in [
            ("OLD", S1_PATH, OLD_RANKED),
            ("FRESH", FRESH_S1, FRESH_RANKED),
        ]:
            header(f"{mode} ANALYSIS")

            n, gt = build_eval_ids(
                con,
                mode,
            )

            print(f"{mode} S1 entities : {n:,}")
            print(f"{mode} GT pairs    : {gt:,}")

            report["datasets"][mode] = {}

            for target in ["S2", "S3"]:
                summary = analyze_target(
                    con,
                    ranked_paths[target],
                    target,
                    mode,
                )

                report["datasets"][mode][target] = summary

                print(f"\n{target}:")
                print(
                    f"  GT pairs                  : {summary['ground_truth_pairs']:,}"
                )
                print(
                    f"  candidate-covered GT      : {summary['candidate_covered_gt_pairs']:,}"
                )
                print(
                    f"  candidate-missing GT      : {summary['candidate_missing_gt_pairs']:,}"
                )
                print(
                    f"  Recall@1                  : {summary['recall_at_1_pairs']:,}"
                )
                print(
                    f"  Recall@5                  : {summary['recall_at_5_pairs']:,}"
                )
                print(
                    f"  Recall@20                 : {summary['recall_at_20_pairs']:,}"
                )
                print(
                    f"  accepted / TP / FP        : "
                    f"{summary['accepted_top1_predictions']:,} / "
                    f"{summary['accepted_top1_tp']:,} / "
                    f"{summary['accepted_top1_fp']:,}"
                )
                print(
                    f"  rejected true top1       : "
                    f"{summary['rejected_top1_true_positive']:,}"
                )
                print(
                    f"  wrong top1, true rank<=20: "
                    f"{summary['wrong_top1_true_candidate_rank_le20']:,}"
                )
                print(
                    f"  wrong top1, true rank>20 : "
                    f"{summary['wrong_top1_true_rank_gt20']:,}"
                )
                print(
                    f"  wrong top1, GT absent    : "
                    f"{summary['wrong_top1_candidate_missing']:,}"
                )
                print(
                    f"  mean utility gap         : "
                    f"{summary['wrong_top1_mean_utility_gap']}"
                )

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("ANALYSIS COMPLETE")
        print(f"Report: {REPORT}")
        print("SAFE: no final candidate/output files were changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
