from pathlib import Path
import duckdb
import json
import os
import math

PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"

S1_PATH = TRAIN / "train_source1.parquet"
GT_PATH = TRAIN / "ground_truth_pairs.parquet"

RANKED = {
    "BASELINE": {
        "S2": PROJECT / "candidate_output_v10_v8_leakage_safe"
              / "baseline_true_validation_ranked_s1_s2.parquet",
        "S3": PROJECT / "candidate_output_v10_v8_leakage_safe"
              / "baseline_true_validation_ranked_s1_s3.parquet",
    },
    "V8_EXPANDED": {
        "S2": PROJECT / "candidate_output_v10_v8_leakage_safe"
              / "v8_expanded_true_validation_ranked_s1_s2.parquet",
        "S3": PROJECT / "candidate_output_v10_v8_leakage_safe"
              / "v8_expanded_true_validation_ranked_s1_s3.parquet",
    },
}

OUT_DIR = (
    PROJECT
    / "validation_leakage_audit"
    / "f05_top1_threshold_audit_v3"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / "f05_top1_threshold_audit_v3_report.json"

MEMORY = os.environ.get("F05_AUDIT_MEMORY", "8GB")
THREADS = int(os.environ.get("F05_AUDIT_THREADS", "2"))

VALIDATION_MOD = 5


def header(s):
    print("\n" + "=" * 110)
    print(s)
    print("=" * 110, flush=True)


def qp(path):
    return "'" + str(path).replace("'", "''") + "'"


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def score_f05(truth, pred, tp):
    if truth == 0 and pred == 0:
        return 1.0

    if truth == 0 or pred == 0:
        return 0.0

    return 1.25 * tp / (0.25 * truth + pred)


def make_validation_truth(con):
    header("1. BUILD EXACT VALIDATION UNIVERSE")

    require(S1_PATH)
    require(GT_PATH)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_s1 AS
        SELECT DISTINCT
            CAST(entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(S1_PATH)})
        WHERE MOD(
            ABS(HASH(CAST(entity_id AS VARCHAR))),
            {VALIDATION_MOD}
        ) = 0
    """)

    n_s1 = int(
        con.execute(
            "SELECT COUNT(*) FROM validation_s1"
        ).fetchone()[0]
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_all AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1,
            CAST(matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)})
    """)

    gt_total = int(
        con.execute("""
            SELECT COUNT(*)
            FROM gt_all g
            INNER JOIN validation_s1 v
              ON g.s1 = v.s1
        """).fetchone()[0]
    )

    if n_s1 != 441_094:
        raise RuntimeError(
            f"Validation S1 mismatch: expected 441,094, got {n_s1:,}"
        )

    if gt_total != 1_527_077:
        raise RuntimeError(
            f"Validation GT mismatch: expected 1,527,077, got {gt_total:,}"
        )

    con.execute("""
        CREATE OR REPLACE TEMP TABLE truth_counts AS
        SELECT
            v.s1,
            COUNT(g.matched)::BIGINT AS truth_count
        FROM validation_s1 v
        LEFT JOIN gt_all g
          ON v.s1 = g.s1
        GROUP BY v.s1
    """)

    rows = con.execute("""
        SELECT s1, truth_count
        FROM truth_counts
        ORDER BY s1
    """).fetchall()

    truth_counts = {
        str(s1): int(cnt)
        for s1, cnt in rows
    }

    validation_ids = list(truth_counts.keys())

    zero_truth = sum(
        1 for s1 in validation_ids
        if truth_counts[s1] == 0
    )

    print(f"Validation S1 entities : {n_s1:,}")
    print(f"Validation GT pairs    : {gt_total:,}")
    print(f"Validation zero-truth S1: {zero_truth:,}")

    return validation_ids, truth_counts


def load_top1(con, ranked_path, gt_prefix):
    require(ranked_path)

    gt_table = "gt_s2" if gt_prefix == "S2" else "gt_s3"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {gt_table} AS
        SELECT
            s1,
            matched
        FROM gt_all
        WHERE matched LIKE '{gt_prefix}-%'
    """)

    q = f"""
        WITH top1 AS (
            SELECT
                CAST(source1_entity_id AS VARCHAR) AS s1,
                CAST(candidate_entity_id AS VARCHAR) AS candidate,
                CAST(v10_utility AS DOUBLE) AS utility
            FROM read_parquet({qp(ranked_path)})
            WHERE v10_rank = 1
        )
        SELECT
            t.s1,
            t.utility,
            CASE
                WHEN g.matched IS NULL THEN 0
                ELSE 1
            END AS is_tp
        FROM top1 t
        LEFT JOIN {gt_table} g
          ON t.s1 = g.s1
         AND t.candidate = g.matched
    """

    rows = con.execute(q).fetchall()

    out = {}

    for s1, utility, is_tp in rows:
        out[str(s1)] = (
            float(utility),
            int(is_tp),
        )

    return out


def build_events(con, paths):
    s2 = load_top1(con, paths["S2"], "S2")
    s3 = load_top1(con, paths["S3"], "S3")

    events = []

    for s1, (utility, is_tp) in s2.items():
        events.append((
            utility,
            s1,
            "S2",
            is_tp,
        ))

    for s1, (utility, is_tp) in s3.items():
        events.append((
            utility,
            s1,
            "S3",
            is_tp,
        ))

    events.sort(key=lambda x: x[0], reverse=True)

    return s2, s3, events


def exact_global_threshold(events, validation_ids, truth_counts):
    """
    Exact threshold sweep using incremental state.

    Every unique observed utility is tested.
    No O(events × validation_s1) rescans.
    """

    pred = {}
    tp = {}

    # No predictions initially.
    current_sum = 0.0

    zero_truth_count = 0

    for s1 in validation_ids:
        truth = truth_counts[s1]

        if truth == 0:
            zero_truth_count += 1

        current_sum += 1.0 if truth == 0 else 0.0

    total_pred = 0
    total_tp = 0
    correctly_empty = zero_truth_count
    fp_singletons = 0

    best = {
        "threshold": math.inf,
        "macro_f05": current_sum / len(validation_ids),
        "precision": 0.0,
        "recall": 0.0,
        "predicted": 0,
        "tp": 0,
        "correctly_empty": correctly_empty,
        "fp_singletons": 0,
    }

    total_truth = sum(truth_counts.values())

    i = 0
    n_events = len(events)

    while i < n_events:
        utility = events[i][0]
        j = i + 1

        while j < n_events and events[j][0] == utility:
            j += 1

        for _, s1, _, is_tp in events[i:j]:
            truth = truth_counts[s1]

            old_pred = pred.get(s1, 0)
            old_tp = tp.get(s1, 0)

            old_score = score_f05(
                truth,
                old_pred,
                old_tp,
            )

            new_pred = old_pred + 1
            new_tp = old_tp + is_tp

            new_score = score_f05(
                truth,
                new_pred,
                new_tp,
            )

            current_sum += new_score - old_score

            pred[s1] = new_pred
            tp[s1] = new_tp

            total_pred += 1
            total_tp += is_tp

            if truth == 0:
                fp_singletons += 1

                if old_pred == 0:
                    correctly_empty -= 1

        macro = current_sum / len(validation_ids)

        precision = (
            total_tp / total_pred
            if total_pred
            else 0.0
        )

        recall = (
            total_tp / total_truth
            if total_truth
            else 0.0
        )

        candidate = {
            "threshold": utility,
            "macro_f05": macro,
            "precision": precision,
            "recall": recall,
            "predicted": total_pred,
            "tp": total_tp,
            "correctly_empty": correctly_empty,
            "fp_singletons": fp_singletons,
        }

        if candidate["macro_f05"] > best["macro_f05"]:
            best = candidate

        i = j

    return best


def evaluate_k1(validation_ids, truth_counts, s2, s3):
    pred = {}
    tp = {}

    for s1 in validation_ids:
        p = 0
        t = 0

        if s1 in s2:
            p += 1
            t += s2[s1][1]

        if s1 in s3:
            p += 1
            t += s3[s1][1]

        pred[s1] = p
        tp[s1] = t

    current_sum = sum(
        score_f05(
            truth_counts[s1],
            pred[s1],
            tp[s1],
        )
        for s1 in validation_ids
    )

    total_pred = sum(pred.values())
    total_tp = sum(tp.values())
    total_truth = sum(truth_counts.values())

    correctly_empty = sum(
        1
        for s1 in validation_ids
        if truth_counts[s1] == 0 and pred[s1] == 0
    )

    fp_singletons = sum(
        pred[s1]
        for s1 in validation_ids
        if truth_counts[s1] == 0
    )

    return {
        "macro_f05": current_sum / len(validation_ids),
        "precision": total_tp / total_pred
        if total_pred else 0.0,
        "recall": total_tp / total_truth
        if total_truth else 0.0,
        "predicted": total_pred,
        "tp": total_tp,
        "correctly_empty": correctly_empty,
        "fp_singletons": fp_singletons,
    }


def single_direction_threshold(
    events_for_direction,
    fixed_direction,
    validation_ids,
    truth_counts,
):
    """
    One direction uses threshold, other direction remains K1.
    """

    pred = {}
    tp = {}

    for s1 in validation_ids:
        p = 0
        t = 0

        fixed = fixed_direction.get(s1)

        if fixed is not None:
            p += 1
            t += fixed[1]

        pred[s1] = p
        tp[s1] = t

    current_sum = sum(
        score_f05(
            truth_counts[s1],
            pred[s1],
            tp[s1],
        )
        for s1 in validation_ids
    )

    total_pred = sum(pred.values())
    total_tp = sum(tp.values())
    total_truth = sum(truth_counts.values())

    correctly_empty = sum(
        1
        for s1 in validation_ids
        if truth_counts[s1] == 0 and pred[s1] == 0
    )

    fp_singletons = sum(
        pred[s1]
        for s1 in validation_ids
        if truth_counts[s1] == 0
    )

    best = None

    i = 0
    n_events = len(events_for_direction)

    while i < n_events:
        utility = events_for_direction[i][0]
        j = i + 1

        while j < n_events and events_for_direction[j][0] == utility:
            j += 1

        for _, s1, _, is_tp in events_for_direction[i:j]:
            truth = truth_counts[s1]

            old_pred = pred.get(s1, 0)
            old_tp = tp.get(s1, 0)

            old_score = score_f05(
                truth,
                old_pred,
                old_tp,
            )

            new_pred = old_pred + 1
            new_tp = old_tp + is_tp

            new_score = score_f05(
                truth,
                new_pred,
                new_tp,
            )

            current_sum += new_score - old_score

            pred[s1] = new_pred
            tp[s1] = new_tp

            total_pred += 1
            total_tp += is_tp

            if truth == 0:
                fp_singletons += 1

                if old_pred == 0:
                    correctly_empty -= 1

        macro = current_sum / len(validation_ids)

        precision = (
            total_tp / total_pred
            if total_pred
            else 0.0
        )

        recall = (
            total_tp / total_truth
            if total_truth
            else 0.0
        )

        candidate = {
            "threshold": utility,
            "macro_f05": macro,
            "precision": precision,
            "recall": recall,
            "predicted": total_pred,
            "tp": total_tp,
            "correctly_empty": correctly_empty,
            "fp_singletons": fp_singletons,
        }

        if best is None or (
            candidate["macro_f05"] > best["macro_f05"]
        ):
            best = candidate

        i = j

    return best


def main():
    header(
        "AMAZON ML CHALLENGE — OPTIMIZED TOP1 + "
        "ABSTENTION F0.5 AUDIT V3"
    )

    print(f"Memory        : {MEMORY}")
    print(f"Threads       : {THREADS}")
    print("Validation    : OLD V7 ONLY")
    print("Final holdout : NOT READ")
    print("Model         : NOT TRAINED")
    print("Candidate gen : NOT RUN")
    print("Algorithm     : INCREMENTAL EXACT SWEEP")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")

    try:
        validation_ids, truth_counts = make_validation_truth(con)

        results = {}

        for label in ["BASELINE", "V8_EXPANDED"]:
            header(f"2. ANALYZE {label}")

            s2, s3, events = build_events(
                con,
                RANKED[label],
            )

            print(f"S2 top1 rows : {len(s2):,}")
            print(f"S3 top1 rows : {len(s3):,}")
            print(f"Total events : {len(events):,}")

            k1 = evaluate_k1(
                validation_ids,
                truth_counts,
                s2,
                s3,
            )

            print("\nK1/K1 REFERENCE")
            for k, v in k1.items():
                print(f"{k:18s}: {v}")

            # Exact global threshold.
            global_events = events

            best_global = exact_global_threshold(
                global_events,
                validation_ids,
                truth_counts,
            )

            print("\nBEST GLOBAL TOP1 THRESHOLD")
            for k, v in best_global.items():
                print(f"{k:18s}: {v}")

            # Separate direction threshold sweeps.
            s2_events = [
                e for e in events
                if e[2] == "S2"
            ]

            s3_events = [
                e for e in events
                if e[2] == "S3"
            ]

            best_s2 = single_direction_threshold(
                s2_events,
                s3,
                validation_ids,
                truth_counts,
            )

            best_s3 = single_direction_threshold(
                s3_events,
                s2,
                validation_ids,
                truth_counts,
            )

            print("\nBEST S2 THRESHOLD + S3 K1")
            for k, v in best_s2.items():
                print(f"{k:18s}: {v}")

            print("\nBEST S2 K1 + S3 THRESHOLD")
            for k, v in best_s3.items():
                print(f"{k:18s}: {v}")

            results[label] = {
                "k1_k1": k1,
                "global_threshold": best_global,
                "s2_threshold_s3_k1": best_s2,
                "s2_k1_s3_threshold": best_s3,
            }

        # Historical references already independently validated.
        expected = {
            "BASELINE": 0.594112,
            "V8_EXPANDED": 0.577989,
        }

        reference_checks = {}

        for label in results:
            measured = results[label]["k1_k1"]["macro_f05"]
            exp = expected[label]
            delta = measured - exp

            reference_checks[label] = {
                "measured": measured,
                "expected": exp,
                "delta": delta,
                "pass": abs(delta) < 1e-6,
            }

        deltas = {}

        for label in results:
            k1 = results[label]["k1_k1"]["macro_f05"]

            deltas[label] = {
                "global_threshold":
                    results[label]["global_threshold"]["macro_f05"] - k1,
                "s2_threshold_s3_k1":
                    results[label]["s2_threshold_s3_k1"]["macro_f05"] - k1,
                "s2_k1_s3_threshold":
                    results[label]["s2_k1_s3_threshold"]["macro_f05"] - k1,
            }

        summary = {
            "validation_s1": len(validation_ids),
            "validation_gt_pairs": sum(truth_counts.values()),
            "results": results,
            "reference_checks": reference_checks,
            "deltas_vs_k1": deltas,
            "report": str(REPORT),
        }

        REPORT.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        header("FINAL V3 AUDIT RESULT")

        print("REFERENCE CHECKS")

        for label, x in reference_checks.items():
            print(
                f"{label}: "
                f"measured={x['measured']:.12f} | "
                f"expected={x['expected']:.6f} | "
                f"delta={x['delta']:+.12f} | "
                f"{'PASS' if x['pass'] else 'FAIL'}"
            )

        print("\nDELTA VS K1/K1")

        for label, x in deltas.items():
            print(f"\n{label}")

            print(
                f"  global threshold       : "
                f"{x['global_threshold']:+.6f}"
            )

            print(
                f"  S2 threshold + S3 K1 : "
                f"{x['s2_threshold_s3_k1']:+.6f}"
            )

            print(
                f"  S2 K1 + S3 threshold : "
                f"{x['s2_k1_s3_threshold']:+.6f}"
            )

        if not all(
            x["pass"]
            for x in reference_checks.values()
        ):
            raise RuntimeError(
                "REFERENCE CHECK FAILED. "
                "Do not use this audit for final policy."
            )

        print(f"\nReport: {REPORT}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
