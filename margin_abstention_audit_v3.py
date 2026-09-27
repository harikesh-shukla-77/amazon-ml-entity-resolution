#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
MARGIN-ABSTENTION AUDIT V2

Deep, read-only experiment after confirming the source-specific threshold
policy on a fresh development split.

Confirmed policy from OLD validation:
    S2 >= 34.710811614990234
    S3 >= 22.125261306762695

Question:
Can we safely improve precision further by abstaining when the top-1 score
is too close to the top-2 score?

Policy:
    accept target T iff
      top1_utility >= threshold[T]
      AND (top1_utility - top2_utility) >= margin[T]

A missing top-2 candidate gives margin = +infinity.

Protocol:
1. Choose margin(s) on OLD V7 validation only.
2. Apply the selected margin policy to the already-scored FRESH DEV split.
3. Do NOT tune on fresh dev.
4. Never read Final Holdout GT.
5. Never read test data.
6. Never retrain, regenerate candidates, or modify final outputs.

This V2 implementation is vectorized. It avoids the expensive per-entity
Python/SQL loop used in the earlier draft.
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

FRESH_DIR = PROJECT / "validation_leakage_audit" / \
    "f05_fresh_dev_joint_confirmation_v1"

FRESH_S1 = FRESH_DIR / "fresh_dev_s1.parquet"
FRESH_RANKED = {
    "S2": FRESH_DIR / "fresh_ranked_s2.parquet",
    "S3": FRESH_DIR / "fresh_ranked_s3.parquet",
}

OUT = PROJECT / "validation_leakage_audit" / \
    "f05_margin_abstention_audit_v3"
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
        raise FileNotFoundError(f"Missing: {path}")


def make_ids_and_truth(con, mode: str):
    """
    Returns:
        ids: np.ndarray[str]
        truth: np.ndarray[int]
    """
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
        CREATE OR REPLACE TEMP TABLE truth_local AS
        SELECT
            e.s1,
            COUNT(g.matched_entity_id)::INTEGER AS truth_count
        FROM eval_ids e
        LEFT JOIN read_parquet({qp(GT_PATH)}) g
          ON CAST(g.source1_entity_id AS VARCHAR) = e.s1
         AND COALESCE(g.label, 1) = 1
        GROUP BY e.s1
    """)

    rows = con.execute("""
        SELECT s1, truth_count
        FROM truth_local
        ORDER BY s1
    """).fetchall()

    ids = np.array([str(r[0]) for r in rows], dtype=object)
    truth = np.array([int(r[1]) for r in rows], dtype=np.int32)

    return ids, truth


def load_top12(
    con,
    path: Path,
    target: str,
    ids,
):
    ids_count = len(ids)
    # Detect schema because OLD validation ranking files use
    # v10_utility / v10_rank, while fresh-development ranking files use
    # utility / rank.
    cols = {
        str(r[0])
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({qp(path)})"
        ).fetchall()
    }

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

    utility_col = next(
        (c for c in utility_candidates if c in cols),
        None,
    )

    rank_col = next(
        (c for c in rank_candidates if c in cols),
        None,
    )

    if utility_col is None or rank_col is None:
        raise RuntimeError(
            f"Could not detect utility/rank columns in {path}. "
            f"Available columns: {sorted(cols)}"
        )

    if "source1_entity_id" not in cols:
        raise RuntimeError(
            f"{path} missing source1_entity_id"
        )

    if "candidate_entity_id" not in cols:
        raise RuntimeError(
            f"{path} missing candidate_entity_id"
        )

    # First reduce each ranking file to top-2 candidates.
    # TP is checked directly against the challenge GT parquet instead of
    # truth-count aggregates. This fixes the V2 alias/table bug.
    rows = con.execute(
        f"""
        WITH piv AS (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                MAX(
                    CASE
                        WHEN CAST({rank_col} AS BIGINT) = 1
                        THEN CAST({utility_col} AS DOUBLE)
                    END
                ) AS u1,
                MAX(
                    CASE
                        WHEN CAST({rank_col} AS BIGINT) = 2
                        THEN CAST({utility_col} AS DOUBLE)
                    END
                ) AS u2,
                MAX(
                    CASE
                        WHEN CAST({rank_col} AS BIGINT) = 1
                        THEN CAST(candidate_entity_id AS VARCHAR)
                    END
                ) AS c1
            FROM read_parquet({qp(path)})
            WHERE CAST({rank_col} AS BIGINT) <= 2
            GROUP BY source1_entity_id
        )
        SELECT
            e.s1,
            p.u1,
            p.u2,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM read_parquet({qp(GT_PATH)}) gg
                    WHERE CAST(gg.source1_entity_id AS VARCHAR) = e.s1
                      AND CAST(gg.matched_entity_id AS VARCHAR) = p.c1
                      AND COALESCE(gg.label, 1) = 1
                )
                THEN 1
                ELSE 0
            END AS tp1
        FROM eval_ids e
        LEFT JOIN piv p
          ON e.s1 = p.s1
        ORDER BY e.s1
        """
    ).fetchall()

    utility1 = np.full(
        ids_count,
        -np.inf,
        dtype=np.float64,
    )

    margin = np.full(
        ids_count,
        -np.inf,
        dtype=np.float64,
    )

    tp1 = np.zeros(
        ids_count,
        dtype=np.int8,
    )

    for i, (_, u1, u2, tp) in enumerate(rows):
        if u1 is None:
            continue

        utility1[i] = float(u1)

        margin[i] = (
            np.inf
            if u2 is None
            else float(u1) - float(u2)
        )

        tp1[i] = int(tp)

    return utility1, margin, tp1

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
    print("Old validation: SELECTION SET")
    print("Fresh dev     : CONFIRMATION SET")
    print("Final holdout : NOT READ")
    print("Test data     : NOT READ")
    print("Model training: NOT RUN")
    print("Candidate gen : NOT RUN")
    print("Final outputs : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    try:
        # ---------------------- OLD VALIDATION -----------------------
        header("1. LOAD OLD VALIDATION TOP1/TOP2")

        old_ids, old_truth = make_ids_and_truth(con, "OLD")

        old_s2_u, old_s2_m, old_s2_tp = load_top12(
            con, OLD_RANKED["S2"], "S2", old_ids
        )
        old_s3_u, old_s3_m, old_s3_tp = load_top12(
            con, OLD_RANKED["S3"], "S3", old_ids
        )

        print(f"Old validation S1 : {len(old_ids):,}")
        print(f"Old validation GT : {int(old_truth.sum()):,}")

        # No-margin confirmed threshold policy.
        old_base = evaluate_arrays(
            old_truth,
            old_s2_u, old_s2_m, old_s2_tp,
            old_s3_u, old_s3_m, old_s3_tp,
            0.0, 0.0,
        )

        one_dim = []

        for m2 in MARGINS:
            one_dim.append(
                (
                    "S2_ONLY",
                    evaluate_arrays(
                        old_truth,
                        old_s2_u, old_s2_m, old_s2_tp,
                        old_s3_u, old_s3_m, old_s3_tp,
                        m2, 0.0,
                    ),
                )
            )

        for m3 in MARGINS:
            one_dim.append(
                (
                    "S3_ONLY",
                    evaluate_arrays(
                        old_truth,
                        old_s2_u, old_s2_m, old_s2_tp,
                        old_s3_u, old_s3_m, old_s3_tp,
                        0.0, m3,
                    ),
                )
            )

        s2_ranked = sorted(
            [r for kind, r in one_dim if kind == "S2_ONLY"],
            key=lambda x: x["macro_f05"],
            reverse=True,
        )
        s3_ranked = sorted(
            [r for kind, r in one_dim if kind == "S3_ONLY"],
            key=lambda x: x["macro_f05"],
            reverse=True,
        )

        joint_m2 = sorted(
            {r["margin_s2"] for r in s2_ranked[:5]} | {0.0}
        )
        joint_m3 = sorted(
            {r["margin_s3"] for r in s3_ranked[:5]} | {0.0}
        )

        joint_old = []
        for m2 in joint_m2:
            for m3 in joint_m3:
                joint_old.append(
                    evaluate_arrays(
                        old_truth,
                        old_s2_u, old_s2_m, old_s2_tp,
                        old_s3_u, old_s3_m, old_s3_tp,
                        m2, m3,
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

        # ----------------------- FRESH DEV --------------------------
        header("2. FRESH DEV — BLIND CONFIRMATION")

        fresh_ids, fresh_truth = make_ids_and_truth(con, "FRESH")

        fresh_s2_u, fresh_s2_m, fresh_s2_tp = load_top12(
            con, FRESH_RANKED["S2"], "S2", fresh_ids
        )
        fresh_s3_u, fresh_s3_m, fresh_s3_tp = load_top12(
            con, FRESH_RANKED["S3"], "S3", fresh_ids
        )

        fresh_base = evaluate_arrays(
            fresh_truth,
            fresh_s2_u, fresh_s2_m, fresh_s2_tp,
            fresh_s3_u, fresh_s3_m, fresh_s3_tp,
            0.0, 0.0,
        )

        fresh_confirm = evaluate_arrays(
            fresh_truth,
            fresh_s2_u, fresh_s2_m, fresh_s2_tp,
            fresh_s3_u, fresh_s3_m, fresh_s3_tp,
            best_old_margin["margin_s2"],
            best_old_margin["margin_s3"],
        )

        print(f"Fresh dev S1 : {len(fresh_ids):,}")
        print(f"Fresh dev GT : {int(fresh_truth.sum()):,}")

        print("\nFRESH DEV — NO MARGIN")
        for k, v in fresh_base.items():
            print(f"{k:18s}: {v}")

        print("\nFRESH DEV — OLD-VALIDATION SELECTED MARGIN")
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
            "version": "F05_MARGIN_ABSTENTION_AUDIT_V2",
            "threshold_policy": THRESHOLD,
            "old_validation_s1": len(old_ids),
            "old_validation_gt_pairs": int(old_truth.sum()),
            "fresh_dev_s1": len(fresh_ids),
            "fresh_dev_gt_pairs": int(fresh_truth.sum()),
            "old_validation_no_margin": old_base,
            "old_validation_top10_joint_margin": joint_old[:10],
            "old_validation_selected_margin_policy": best_old_margin,
            "fresh_dev_no_margin": fresh_base,
            "fresh_dev_confirmed_margin_policy": fresh_confirm,
            "fresh_dev_margin_delta": delta,
            "final_holdout_gt_read": False,
            "test_data_read": False,
            "model_retrained": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
            "note": (
                "Fresh-dev exact threshold search from the prior script "
                "was not used because its returned optimum was dominated by "
                "the directly evaluated old-validation policy. The old "
                "candidate threshold pair is used here as a pre-selected "
                "policy and is tested blindly on fresh development data."
            ),
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
