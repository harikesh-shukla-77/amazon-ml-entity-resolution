#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
MARGIN-ABSTENTION AUDIT V3 — CORRECTED

This is a READ-ONLY experiment after the source-specific threshold policy
was confirmed on fresh development data.

Base policy:
    S2 utility >= 34.710811614990234
    S3 utility >= 22.125261306762695

Additional ambiguity gate:
    top1_utility - top2_utility >= margin[target]

A target with no top-2 row gets +infinity margin.

STRICT SAFETY
-------------
- OLD V7 validation is used for margin selection.
- FRESH development split is used only for blind confirmation.
- Final Holdout V2 ground truth is NEVER read.
- Test data is NEVER read.
- No model training.
- No candidate generation.
- No final output modification.

WHY V3
------
V2 had two bugs:
  1. load_top12 referenced g.matched_entity_id from truth_local,
     but truth_local intentionally contains only (s1, truth_count).
  2. OLD ranked files use v10_utility/v10_rank while the fresh ranked
     files use utility/rank.

V3 explicitly creates a gt_pairs_local table and detects the utility/rank
column names for each ranked file.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import duckdb
import numpy as np


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
    / "f05_margin_abstention_audit_v3"
)
REPORT = OUT / "f05_margin_abstention_audit_v3_report.json"

MEMORY = os.environ.get("MARGIN_AUDIT_MEMORY", "8GB")
THREADS = int(os.environ.get("MARGIN_AUDIT_THREADS", "2"))

OLD_VALIDATION_MOD = 5

THRESHOLD = {
    "S2": 34.710811614990234,
    "S3": 22.125261306762695,
}

MARGINS = [
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
]


def header(s: str) -> None:
    print("\n" + "=" * 112)
    print(s)
    print("=" * 112, flush=True)


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


def get_columns(con, path: Path) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({qp(path)})"
    ).fetchall()
    return [str(r[0]) for r in rows]


def detect_score_columns(con, path: Path) -> tuple[str, str]:
    cols = set(get_columns(con, path))

    utility_candidates = [
        "v10_utility",
        "utility",
        "v8_utility",
        "v7_probability",
    ]
    rank_candidates = [
        "v10_rank",
        "rank",
        "final_holdout_rank",
    ]

    utility = next((c for c in utility_candidates if c in cols), None)
    rank = next((c for c in rank_candidates if c in cols), None)

    if utility is None or rank is None:
        raise RuntimeError(
            f"Could not detect utility/rank columns in {path}. "
            f"Available columns: {sorted(cols)}"
        )

    if "source1_entity_id" not in cols:
        raise RuntimeError(f"{path} missing source1_entity_id")
    if "candidate_entity_id" not in cols:
        raise RuntimeError(f"{path} missing candidate_entity_id")

    return utility, rank


def load_eval_universe(con, id_path: Path, mode: str):
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
            FROM read_parquet({qp(id_path)})
        """)
    else:
        raise ValueError(mode)

    # Keep the pair-level GT table separate from truth counts.
    # This fixes the V2 bug and lets top1 candidates be checked directly.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_pairs_local AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)}) g
        INNER JOIN eval_ids e
          ON CAST(g.source1_entity_id AS VARCHAR) = e.s1
        WHERE COALESCE(g.label, 1) = 1
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_local AS
        SELECT
            e.s1,
            COUNT(g.matched)::INTEGER AS truth_count
        FROM eval_ids e
        LEFT JOIN gt_pairs_local g
          ON e.s1 = g.s1
        GROUP BY e.s1
    """)

    rows = con.execute("""
        SELECT s1, truth_count
        FROM truth_local
        ORDER BY s1
    """).fetchall()

    ids = [str(r[0]) for r in rows]
    truth = np.asarray(
        [int(r[1]) for r in rows],
        dtype=np.int32,
    )

    return ids, truth


def load_top12(
    con,
    path: Path,
    target: str,
    ids_count: int,
):
    utility_col, rank_col = detect_score_columns(con, path)

    # Filter to top 2 before joining GT.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ranked_local AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1,
            CAST(candidate_entity_id AS VARCHAR) AS candidate,
            CAST({utility_col} AS DOUBLE) AS utility,
            CAST({rank_col} AS BIGINT) AS rnk
        FROM read_parquet({qp(path)})
        WHERE CAST({rank_col} AS BIGINT) <= 2
    """)

    rows = con.execute(f"""
        WITH piv AS (
            SELECT
                s1,
                MAX(CASE WHEN rnk = 1 THEN utility END) AS u1,
                MAX(CASE WHEN rnk = 2 THEN utility END) AS u2,
                MAX(CASE WHEN rnk = 1 THEN candidate END) AS c1
            FROM ranked_local
            GROUP BY s1
        )
        SELECT
            e.s1,
            p.u1,
            p.u2,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM gt_pairs_local g
                    WHERE g.s1 = e.s1
                      AND g.matched = p.c1
                )
                THEN 1 ELSE 0
            END AS tp1
        FROM eval_ids e
        LEFT JOIN piv p
          ON e.s1 = p.s1
        ORDER BY e.s1
    """).fetchall()

    utility1 = np.full(ids_count, -np.inf, dtype=np.float64)
    margin = np.full(ids_count, -np.inf, dtype=np.float64)
    tp1 = np.zeros(ids_count, dtype=np.int8)

    for i, (_, u1, u2, tp) in enumerate(rows):
        if u1 is None:
            continue
        utility1[i] = float(u1)
        margin[i] = (
            np.inf if u2 is None
            else float(u1) - float(u2)
        )
        tp1[i] = int(tp)

    return utility1, margin, tp1, utility_col, rank_col


def evaluate_arrays(
    truth: np.ndarray,
    s2_u: np.ndarray,
    s2_m: np.ndarray,
    s2_tp: np.ndarray,
    s3_u: np.ndarray,
    s3_m: np.ndarray,
    s3_tp: np.ndarray,
    margin_s2: float,
    margin_s3: float,
):
    accept2 = (
        (s2_u >= THRESHOLD["S2"])
        & (s2_m >= margin_s2)
    )
    accept3 = (
        (s3_u >= THRESHOLD["S3"])
        & (s3_m >= margin_s3)
    )

    pred = accept2.astype(np.int8) + accept3.astype(np.int8)
    tp = (
        accept2.astype(np.int8) * s2_tp
        + accept3.astype(np.int8) * s3_tp
    )

    score = np.zeros_like(truth, dtype=np.float64)

    zero_truth = truth == 0
    score[zero_truth] = (pred[zero_truth] == 0).astype(np.float64)

    nonzero = ~zero_truth
    with_pred = nonzero & (pred > 0)
    score[with_pred] = (
        1.25 * tp[with_pred]
        / (0.25 * truth[with_pred] + pred[with_pred])
    )

    total_pred = int(pred.sum())
    total_tp = int(tp.sum())
    truth_total = int(truth.sum())

    return {
        "threshold_s2": THRESHOLD["S2"],
        "threshold_s3": THRESHOLD["S3"],
        "margin_s2": float(margin_s2),
        "margin_s3": float(margin_s3),
        "macro_f05": float(score.mean()),
        "precision": (
            total_tp / total_pred if total_pred else 0.0
        ),
        "recall": (
            total_tp / truth_total if truth_total else 0.0
        ),
        "predicted": total_pred,
        "tp": total_tp,
        "correctly_empty": int(
            np.sum(zero_truth & (pred == 0))
        ),
        "fp_singletons": int(
            np.sum(pred[zero_truth])
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for p in [
        S1_PATH,
        GT_PATH,
        OLD_RANKED["S2"],
        OLD_RANKED["S3"],
        FRESH_S1,
        FRESH_RANKED["S2"],
        FRESH_RANKED["S3"],
    ]:
        require(p)

    header("AMAZON ML CHALLENGE — MARGIN ABSTENTION AUDIT V3")

    print(f"Memory        : {MEMORY}")
    print(f"Threads       : {THREADS}")
    print(f"S2 threshold  : {THRESHOLD['S2']}")
    print(f"S3 threshold  : {THRESHOLD['S3']}")
    print("Old validation: SELECT MARGIN")
    print("Fresh dev     : BLIND CONFIRM")
    print("Final holdout : NOT READ")
    print("Test data     : NOT READ")
    print("Model         : NOT TRAINED")
    print("Candidate gen : NOT RUN")
    print("Final outputs : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    try:
        # ---------------- OLD VALIDATION ----------------
        header("1. OLD VALIDATION")

        old_ids, old_truth = load_eval_universe(
            con,
            S1_PATH,
            "OLD",
        )

        (
            old_s2_u,
            old_s2_m,
            old_s2_tp,
            old_s2_util_col,
            old_s2_rank_col,
        ) = load_top12(
            con,
            OLD_RANKED["S2"],
            "S2",
            len(old_ids),
        )

        (
            old_s3_u,
            old_s3_m,
            old_s3_tp,
            old_s3_util_col,
            old_s3_rank_col,
        ) = load_top12(
            con,
            OLD_RANKED["S3"],
            "S3",
            len(old_ids),
        )

        print(f"Old validation S1 : {len(old_ids):,}")
        print(f"Old validation GT : {int(old_truth.sum()):,}")
        print(
            f"S2 columns: utility={old_s2_util_col}, rank={old_s2_rank_col}"
        )
        print(
            f"S3 columns: utility={old_s3_util_col}, rank={old_s3_rank_col}"
        )

        old_base = evaluate_arrays(
            old_truth,
            old_s2_u,
            old_s2_m,
            old_s2_tp,
            old_s3_u,
            old_s3_m,
            old_s3_tp,
            0.0,
            0.0,
        )

        one_dim = []

        for m2 in MARGINS:
            one_dim.append(
                evaluate_arrays(
                    old_truth,
                    old_s2_u,
                    old_s2_m,
                    old_s2_tp,
                    old_s3_u,
                    old_s3_m,
                    old_s3_tp,
                    m2,
                    0.0,
                )
            )

        for m3 in MARGINS:
            one_dim.append(
                evaluate_arrays(
                    old_truth,
                    old_s2_u,
                    old_s2_m,
                    old_s2_tp,
                    old_s3_u,
                    old_s3_m,
                    old_s3_tp,
                    0.0,
                    m3,
                )
            )

        s2_ranked = sorted(
            [r for r in one_dim if r["margin_s3"] == 0.0],
            key=lambda x: x["macro_f05"],
            reverse=True,
        )[:5]

        s3_ranked = sorted(
            [r for r in one_dim if r["margin_s2"] == 0.0],
            key=lambda x: x["macro_f05"],
            reverse=True,
        )[:5]

        joint_m2 = sorted(
            {r["margin_s2"] for r in s2_ranked} | {0.0}
        )
        joint_m3 = sorted(
            {r["margin_s3"] for r in s3_ranked} | {0.0}
        )

        joint_old = []

        for m2 in joint_m2:
            for m3 in joint_m3:
                joint_old.append(
                    evaluate_arrays(
                        old_truth,
                        old_s2_u,
                        old_s2_m,
                        old_s2_tp,
                        old_s3_u,
                        old_s3_m,
                        old_s3_tp,
                        m2,
                        m3,
                    )
                )

        joint_old.sort(
            key=lambda x: (
                x["macro_f05"],
                x["precision"],
                -x["predicted"],
            ),
            reverse=True,
        )

        best_old_margin = joint_old[0]

        print("\nOLD VALIDATION — NO MARGIN")
        for k, v in old_base.items():
            print(f"{k:18s}: {v}")

        print("\nOLD VALIDATION — TOP JOINT MARGIN POLICIES")
        for i, r in enumerate(joint_old[:10], 1):
            print(
                f"{i:2d}. "
                f"M2={r['margin_s2']:<7g} "
                f"M3={r['margin_s3']:<7g} | "
                f"F0.5={r['macro_f05']:.12f} | "
                f"P={r['precision']:.6f} | "
                f"R={r['recall']:.6f}"
            )

        # ---------------- FRESH DEV ----------------
        header("2. FRESH DEV — BLIND CONFIRMATION")

        fresh_ids, fresh_truth = load_eval_universe(
            con,
            FRESH_S1,
            "FRESH",
        )

        (
            fresh_s2_u,
            fresh_s2_m,
            fresh_s2_tp,
            fresh_s2_util_col,
            fresh_s2_rank_col,
        ) = load_top12(
            con,
            FRESH_RANKED["S2"],
            "S2",
            len(fresh_ids),
        )

        (
            fresh_s3_u,
            fresh_s3_m,
            fresh_s3_tp,
            fresh_s3_util_col,
            fresh_s3_rank_col,
        ) = load_top12(
            con,
            FRESH_RANKED["S3"],
            "S3",
            len(fresh_ids),
        )

        fresh_base = evaluate_arrays(
            fresh_truth,
            fresh_s2_u,
            fresh_s2_m,
            fresh_s2_tp,
            fresh_s3_u,
            fresh_s3_m,
            fresh_s3_tp,
            0.0,
            0.0,
        )

        fresh_confirm = evaluate_arrays(
            fresh_truth,
            fresh_s2_u,
            fresh_s2_m,
            fresh_s2_tp,
            fresh_s3_u,
            fresh_s3_m,
            fresh_s3_tp,
            best_old_margin["margin_s2"],
            best_old_margin["margin_s3"],
        )

        print(f"Fresh dev S1 : {len(fresh_ids):,}")
        print(f"Fresh dev GT : {int(fresh_truth.sum()):,}")

        print("\nFRESH DEV — NO MARGIN")
        for k, v in fresh_base.items():
            print(f"{k:18s}: {v}")

        print("\nFRESH DEV — OLD-SELECTED MARGIN")
        for k, v in fresh_confirm.items():
            print(f"{k:18s}: {v}")

        delta = (
            fresh_confirm["macro_f05"]
            - fresh_base["macro_f05"]
        )

        print(
            f"\nFresh margin delta vs no-margin: "
            f"{delta:+.12f}"
        )

        report = {
            "version": "F05_MARGIN_ABSTENTION_AUDIT_V3",
            "threshold_policy": THRESHOLD,
            "old_validation_s1": len(old_ids),
            "old_validation_gt_pairs": int(old_truth.sum()),
            "fresh_dev_s1": len(fresh_ids),
            "fresh_dev_gt_pairs": int(fresh_truth.sum()),
            "old_validation_no_margin": old_base,
            "old_validation_top_joint_margin": joint_old[:10],
            "old_validation_selected_margin_policy": best_old_margin,
            "fresh_dev_no_margin": fresh_base,
            "fresh_dev_confirmed_margin_policy": fresh_confirm,
            "fresh_dev_margin_delta": delta,
            "old_column_detection": {
                "S2": {
                    "utility": old_s2_util_col,
                    "rank": old_s2_rank_col,
                },
                "S3": {
                    "utility": old_s3_util_col,
                    "rank": old_s3_rank_col,
                },
            },
            "fresh_column_detection": {
                "S2": {
                    "utility": fresh_s2_util_col,
                    "rank": fresh_s2_rank_col,
                },
                "S3": {
                    "utility": fresh_s3_util_col,
                    "rank": fresh_s3_rank_col,
                },
            },
            "final_holdout_gt_read": False,
            "test_data_read": False,
            "model_trained": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
            "v2_bugs_fixed": [
                "truth pair table is now separate from truth counts",
                "utility/rank column names detected per ranked file",
            ],
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("3. AUDIT COMPLETE")
        print(f"Report: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
