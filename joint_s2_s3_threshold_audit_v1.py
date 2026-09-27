#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
Exact joint S2/S3 threshold sweep for the frozen V10 Top-1 policy.

READ-ONLY validation experiment:
- old V7 validation split only
- no Final Holdout V2
- no test data
- no retraining
- no candidate generation
- no final output changes

The script finds the best pair of thresholds for S2 and S3 jointly.
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
    "S2": PROJECT / "candidate_output_v10_v8_leakage_safe" / "baseline_true_validation_ranked_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10_v8_leakage_safe" / "baseline_true_validation_ranked_s1_s3.parquet",
}
OUT_DIR = PROJECT / "validation_leakage_audit" / "f05_joint_threshold_audit_v1"
REPORT = OUT_DIR / "f05_joint_threshold_audit_v1_report.json"
MEMORY = os.environ.get("F05_JOINT_MEMORY", "8GB")
THREADS = int(os.environ.get("F05_JOINT_THREADS", "2"))
VALIDATION_MOD = 5
EXPECTED_S1 = 441_094
EXPECTED_GT = 1_527_077


def header(s: str) -> None:
    print("\n" + "=" * 110)
    print(s)
    print("=" * 110, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def score_f05(truth: int, pred: int, tp: int) -> float:
    if truth == 0 and pred == 0:
        return 1.0
    if truth == 0 or pred == 0:
        return 0.0
    return 1.25 * tp / (0.25 * truth + pred)


class SegmentTree:
    """Range-add / global-max segment tree."""
    def __init__(self, values):
        n = len(values)
        size = 1
        while size < n:
            size <<= 1
        self.n = n
        self.size = size
        self.mx = [-math.inf] * (2 * size)
        self.lz = [0.0] * (2 * size)
        for i, v in enumerate(values):
            self.mx[size + i] = float(v)
        for i in range(size - 1, 0, -1):
            self.mx[i] = max(self.mx[2 * i], self.mx[2 * i + 1])

    def _apply(self, p: int, delta: float) -> None:
        self.mx[p] += delta
        self.lz[p] += delta

    def _push(self, p: int) -> None:
        z = self.lz[p]
        if z:
            self._apply(2 * p, z)
            self._apply(2 * p + 1, z)
            self.lz[p] = 0.0

    def _add(self, p, lo, hi, ql, qh, delta) -> None:
        if ql > hi or qh < lo:
            return
        if ql <= lo and hi <= qh:
            self._apply(p, delta)
            return
        self._push(p)
        mid = (lo + hi) // 2
        self._add(2 * p, lo, mid, ql, qh, delta)
        self._add(2 * p + 1, mid + 1, hi, ql, qh, delta)
        self.mx[p] = max(self.mx[2 * p], self.mx[2 * p + 1])

    def add(self, ql: int, qh: int, delta: float) -> None:
        if ql > qh or self.n == 0:
            return
        ql = max(0, ql)
        qh = min(self.n - 1, qh)
        if ql <= qh:
            self._add(1, 0, self.size - 1, ql, qh, float(delta))

    def max_value(self) -> float:
        return float(self.mx[1])

    def _argmax(self, p, lo, hi, target):
        if lo == hi:
            return lo
        self._push(p)
        mid = (lo + hi) // 2
        if abs(self.mx[2 * p] - target) <= 1e-12:
            return self._argmax(2 * p, lo, mid, target)
        return self._argmax(2 * p + 1, mid + 1, hi, target)

    def argmax(self) -> int:
        return self._argmax(1, 0, self.size - 1, self.mx[1])


def load_validation_data(con):
    header("1. LOAD OLD V7 VALIDATION UNIVERSE")
    for p in (S1_PATH, GT_PATH, RANKED["S2"], RANKED["S3"]):
        require(p)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_s1 AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(S1_PATH)})
        WHERE MOD(ABS(HASH(CAST(entity_id AS VARCHAR))), {VALIDATION_MOD}) = 0
    """)
    n_s1 = int(con.execute("SELECT COUNT(*) FROM validation_s1").fetchone()[0])

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_all AS
        SELECT CAST(source1_entity_id AS VARCHAR) AS s1,
               CAST(matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)})
    """)
    gt_total = int(con.execute("""
        SELECT COUNT(*) FROM gt_all g JOIN validation_s1 v ON g.s1=v.s1
    """).fetchone()[0])

    if n_s1 != EXPECTED_S1:
        raise RuntimeError(f"Validation S1 mismatch: expected {EXPECTED_S1:,}, got {n_s1:,}")
    if gt_total != EXPECTED_GT:
        raise RuntimeError(f"Validation GT mismatch: expected {EXPECTED_GT:,}, got {gt_total:,}")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_counts AS
        SELECT v.s1, COUNT(g.matched)::INTEGER AS truth_count
        FROM validation_s1 v
        LEFT JOIN gt_all g ON g.s1=v.s1
        GROUP BY v.s1
    """)
    rows = con.execute("SELECT s1, truth_count FROM truth_counts ORDER BY s1").fetchall()
    truth = {str(s1): int(cnt) for s1, cnt in rows}
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
            SELECT CAST(source1_entity_id AS VARCHAR) AS s1,
                   CAST(candidate_entity_id AS VARCHAR) AS candidate,
                   CAST(v10_utility AS DOUBLE) AS utility
            FROM read_parquet({qp(path)})
            WHERE v10_rank=1
        )
        SELECT t.s1, t.utility,
               CASE WHEN g.matched IS NULL THEN 0 ELSE 1 END AS tp
        FROM top1 t
        LEFT JOIN {gt_table} g ON t.s1=g.s1 AND t.candidate=g.matched
    """).fetchall()
    out = {str(s1): (float(u), int(tp), True) for s1, u, tp in rows}
    print(f"{target} top1 rows : {len(out):,}")
    return out


def utility_positions(s3):
    vals = sorted({x[0] for x in s3.values()}, reverse=True)
    pos = {u: i + 1 for i, u in enumerate(vals)}
    return vals, pos


def build_initial_tree(ids, truth, s3, s3_values, s3_pos):
    base_sum = sum(score_f05(truth[s1], 0, 0) for s1 in ids)
    tree = SegmentTree([base_sum] * (len(s3_values) + 1))
    end = len(s3_values)
    for s1 in ids:
        row = s3.get(s1)
        if row is None:
            continue
        u, tp, _ = row
        delta = score_f05(truth[s1], 1, tp) - score_f05(truth[s1], 0, 0)
        tree.add(s3_pos[u], end, delta)
    return tree


def final_stats(ids, truth, s2, s3, t2, t3):
    pred_total = tp_total = score_sum = correctly_empty = fp_singletons = 0
    truth_total = sum(truth.values())
    for s1 in ids:
        tc = truth[s1]
        p = tp = 0
        r2 = s2.get(s1)
        if r2 is not None and r2[0] >= t2:
            p += 1; tp += r2[1]
        r3 = s3.get(s1)
        if r3 is not None and r3[0] >= t3:
            p += 1; tp += r3[1]
        score_sum += score_f05(tc, p, tp)
        pred_total += p
        tp_total += tp
        if tc == 0 and p == 0:
            correctly_empty += 1
        if tc == 0:
            fp_singletons += p
    return {
        "threshold_s2": t2,
        "threshold_s3": t3,
        "macro_f05": score_sum / len(ids),
        "precision": tp_total / pred_total if pred_total else 0.0,
        "recall": tp_total / truth_total if truth_total else 0.0,
        "predicted": pred_total,
        "tp": tp_total,
        "correctly_empty": correctly_empty,
        "fp_singletons": fp_singletons,
    }


def main():
    header("AMAZON ML CHALLENGE — EXACT JOINT S2/S3 THRESHOLD AUDIT V1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Memory        : {MEMORY}")
    print(f"Threads       : {THREADS}")
    print("Validation    : OLD V7 ONLY")
    print("Final holdout : NOT READ")
    print("Model         : NOT TRAINED")
    print("Candidate gen : NOT RUN")
    print("Final outputs : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    started = time.time()

    try:
        ids, truth = load_validation_data(con)
        s2 = load_top1(con, RANKED["S2"], "S2")
        s3 = load_top1(con, RANKED["S3"], "S3")
        s3_values, s3_pos = utility_positions(s3)

        tree = build_initial_tree(ids, truth, s3, s3_values, s3_pos)
        leaf = tree.argmax()
        best = {
            "threshold_s2": math.inf,
            "threshold_s3": math.inf if leaf == 0 else s3_values[leaf - 1],
            "macro_f05": tree.max_value() / len(ids),
        }

        s2_events = sorted(
            [(u, s1, tp) for s1, (u, tp, _) in s2.items()],
            key=lambda x: x[0], reverse=True,
        )

        header("2. EXACT JOINT THRESHOLD SWEEP")
        i = 0
        n = len(s2_events)
        while i < n:
            u = s2_events[i][0]
            j = i + 1
            while j < n and s2_events[j][0] == u:
                j += 1

            for _, s1, tp2 in s2_events[i:j]:
                tc = truth[s1]
                d0 = score_f05(tc, 1, tp2) - score_f05(tc, 0, 0)
                r3 = s3.get(s1)
                if r3 is None:
                    tree.add(0, len(s3_values), d0)
                else:
                    u3, tp3, _ = r3
                    d1 = score_f05(tc, 2, tp2 + tp3) - score_f05(tc, 1, tp3)
                    pos = s3_pos[u3]
                    tree.add(0, pos - 1, d0)
                    tree.add(pos, len(s3_values), d1)

            cur_score = tree.max_value() / len(ids)
            leaf = tree.argmax()
            cur_t3 = math.inf if leaf == 0 else s3_values[leaf - 1]
            if cur_score > best["macro_f05"]:
                best = {"threshold_s2": u, "threshold_s3": cur_t3, "macro_f05": cur_score}
            i = j
            if i % 100_000 == 0 or i == n:
                print(f"S2 events processed: {i:,}/{n:,} | current best={best['macro_f05']:.12f}", flush=True)

        header("3. VERIFY BEST POLICY")
        verified = final_stats(ids, truth, s2, s3, best["threshold_s2"], best["threshold_s3"])
        current_threshold = 20.945884704589844
        current = final_stats(ids, truth, s2, s3, current_threshold, current_threshold)
        delta = verified["macro_f05"] - current["macro_f05"]

        print("\nCURRENT FROZEN POLICY")
        for k, v in current.items(): print(f"{k:18s}: {v}")
        print("\nBEST JOINT THRESHOLD")
        for k, v in verified.items(): print(f"{k:18s}: {v}")
        print(f"\nMacro F0.5 delta vs current: {delta:+.12f}")

        report = {
            "version": "F05_JOINT_THRESHOLD_AUDIT_V1",
            "validation_mod": VALIDATION_MOD,
            "validation_s1": len(ids),
            "validation_gt_pairs": sum(truth.values()),
            "current_frozen_policy": current,
            "best_joint_policy": verified,
            "delta_macro_f05_vs_current": delta,
            "final_holdout_read": False,
            "test_data_read": False,
            "model_retrained": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        header("4. AUDIT COMPLETE")
        print(f"Report: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
