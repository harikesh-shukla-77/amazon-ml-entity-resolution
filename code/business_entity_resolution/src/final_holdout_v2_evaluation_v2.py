#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# AMAZON ML CHALLENGE
# FINAL LOCKED HOLDOUT V2 — FROZEN BASELINE vs V8 EXPANDED
#
# This is the FINAL evaluation stage for the currently frozen
# candidate architecture.
#
# It does NOT:
#   - generate candidates
#   - choose blocks
#   - tune hyperparameters
#   - read final holdout GT during model fitting
#
# It evaluates two already-frozen candidate pools:
#
#   BASELINE  = existing V10 V6+V9.1 pool
#   EXPANDED  = existing V10 pool + V8-new pool
#
# BOTH models:
#   train only on S1 entities NOT in FINAL_HOLDOUT_S1
#   score only FINAL_HOLDOUT_S1
#   read FINAL_HOLDOUT_GT only inside evaluate_final()
#
# FINAL HOLDOUT V2 must not be used for any later tuning/selection.
# ============================================================

PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"
FINAL_DIR = PROJECT / "validation_leakage_audit" / "final_holdout_v2"

FINAL_S1 = FINAL_DIR / "final_holdout_s1.parquet"
FINAL_GT = FINAL_DIR / "final_holdout_ground_truth.parquet"
FINAL_MANIFEST = FINAL_DIR / "final_holdout_v2_manifest.json"

V10_DIR = PROJECT / "candidate_output_v10"

# Existing V10 V6+V9.1 baseline.
BASELINE_LABELED = {
    "S2": V10_DIR / "train_labeled_base_s1_s2.parquet",
    "S3": V10_DIR / "train_labeled_base_s1_s3.parquet",
}

# IMPORTANT:
# The previous leakage-safe V2 run wrote the expanded labeled pool under
# candidate_output_v10_v8_leakage_safe/ with the same basename.
EXPANDED_DIR = PROJECT / "candidate_output_v10_v8_leakage_safe"
EXPANDED_LABELED = {
    "S2": EXPANDED_DIR / "train_labeled_base_s1_s2.parquet",
    "S3": EXPANDED_DIR / "train_labeled_base_s1_s3.parquet",
}

OUT = PROJECT / "validation_leakage_audit" / "final_holdout_eval_v2"
TMP = OUT / "tmp"

MEMORY = os.environ.get("FINAL_EVAL_MEMORY", "8GB")
THREADS = int(os.environ.get("FINAL_EVAL_THREADS", "4"))
BATCH = int(os.environ.get("FINAL_EVAL_BATCH", "100_000"))

PAIR_POS_PER_S1 = 1
PAIR_NEG_RANK = 100
PAIR_NEG_PER_S1 = 2


FEATURES = [
    "name_similarity",
    "address_similarity",
    "name_exact",
    "address_exact",
    "country_exact",
    "name_length_ratio",
    "address_length_ratio",
    "evidence_rows",
    "evidence_file_count",
    "exact_key_count",
    "base_score",
    "log_base_rank",
    "name_x_address",
    "name_minus_address",
    "similarity_mean",
    "similarity_min",
    "exact_field_count",
]


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def header(title: str) -> None:
    print("\n" + "=" * 116)
    print(title)
    print("=" * 116, flush=True)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def setup_connection() -> duckdb.DuckDBPyConnection:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory={qp(TMP)}")
    return con


def verify_manifest() -> dict:
    require_file(FINAL_MANIFEST)
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))

    if manifest.get("status") != "LOCKED":
        raise RuntimeError("Final holdout manifest is not LOCKED.")

    required_false = [
        "model_tuning_run",
        "final_holdout_gt_used_for_selection",
        "final_holdout_gt_used_for_training",
    ]
    for key in required_false:
        if manifest.get(key) is not False:
            raise RuntimeError(
                f"Final holdout manifest guard failed: {key} must be false."
            )

    if not manifest.get("disjoint_old_validation", False):
        raise RuntimeError("Manifest does not certify old-validation disjointness.")
    if not manifest.get("disjoint_previous_holdout", False):
        raise RuntimeError("Manifest does not certify previous-holdout disjointness.")

    return manifest


def assert_training_exclusion(
    con: duckdb.DuckDBPyConnection,
    labeled_path: Path,
    target: str,
) -> int:
    overlap_rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({qp(labeled_path)}) l
            INNER JOIN read_parquet({qp(FINAL_S1)}) h
              ON l.source1_entity_id = h.source1_entity_id
            """
        ).fetchone()[0]
    )

    eligible_overlap = int(
        con.execute(
            f"""
            WITH eligible AS (
                SELECT l.source1_entity_id
                FROM read_parquet({qp(labeled_path)}) l
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM read_parquet({qp(FINAL_S1)}) h
                    WHERE h.source1_entity_id = l.source1_entity_id
                )
            )
            SELECT COUNT(*)
            FROM eligible e
            INNER JOIN read_parquet({qp(FINAL_S1)}) h
              ON e.source1_entity_id = h.source1_entity_id
            """
        ).fetchone()[0]
    )

    print(
        f"{target}: labeled rows belonging to final holdout S1: "
        f"{overlap_rows:,}"
    )
    print(
        f"{target}: eligible training rows overlapping holdout: "
        f"{eligible_overlap:,}"
    )

    if eligible_overlap != 0:
        raise RuntimeError(
            f"{target}: final holdout crossed training boundary."
        )

    return overlap_rows


def make_training_sql(labeled_path: Path) -> str:
    labeled = qp(labeled_path)
    holdout = qp(FINAL_S1)

    # FINAL_HOLDOUT_GT deliberately does not occur anywhere in this SQL.
    return f"""
        WITH
        eligible AS (
            SELECT l.*
            FROM read_parquet({labeled}) l
            WHERE NOT EXISTS (
                SELECT 1
                FROM read_parquet({holdout}) h
                WHERE h.source1_entity_id = l.source1_entity_id
            )
        ),
        positives AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY base_rank, candidate_entity_id
                    ) AS p_rank
                FROM eligible
                WHERE label = 1
            )
            WHERE p_rank <= {PAIR_POS_PER_S1}
        ),
        negatives AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY base_rank, candidate_entity_id
                    ) AS n_rank
                FROM eligible
                WHERE label = 0
                  AND base_rank <= {PAIR_NEG_RANK}
            )
            WHERE n_rank <= {PAIR_NEG_PER_S1}
        ),
        joined AS (
            SELECT
                p.name_similarity AS pn,
                n.name_similarity AS nn,
                p.address_similarity AS pa,
                n.address_similarity AS na,
                p.name_exact AS pe,
                n.name_exact AS ne,
                p.address_exact AS pae,
                n.address_exact AS nae,
                p.country_exact AS pc,
                n.country_exact AS nc,
                p.name_length_ratio AS pl,
                n.name_length_ratio AS nl,
                p.address_length_ratio AS pal,
                n.address_length_ratio AS nal,
                p.evidence_rows AS per,
                n.evidence_rows AS ner,
                p.evidence_file_count AS pef,
                n.evidence_file_count AS nef,
                p.exact_key_count AS pek,
                n.exact_key_count AS nek,
                p.base_score AS ps,
                n.base_score AS ns,
                p.base_rank AS pr,
                n.base_rank AS nr
            FROM positives p
            INNER JOIN negatives n
              ON p.source1_entity_id = n.source1_entity_id
        ),
        f AS (
            SELECT
                pn - nn AS name_similarity,
                pa - na AS address_similarity,
                pe - ne AS name_exact,
                pae - nae AS address_exact,
                pc - nc AS country_exact,
                pl - nl AS name_length_ratio,
                pal - nal AS address_length_ratio,
                per - ner AS evidence_rows,
                pef - nef AS evidence_file_count,
                pek - nek AS exact_key_count,
                ps - ns AS base_score,
                LN(1 + pr) - LN(1 + nr) AS log_base_rank,
                pn * pa - nn * na AS name_x_address,
                (pn - pa) - (nn - na) AS name_minus_address,
                ((pn + pa) / 2) - ((nn + na) / 2) AS similarity_mean,
                LEAST(pn, pa) - LEAST(nn, na) AS similarity_min,
                (pe + pae + pc) - (ne + nae + nc) AS exact_field_count,
                1 AS pair_label
            FROM joined
        )
        SELECT * FROM f
        UNION ALL
        SELECT
            -name_similarity,
            -address_similarity,
            -name_exact,
            -address_exact,
            -country_exact,
            -name_length_ratio,
            -address_length_ratio,
            -evidence_rows,
            -evidence_file_count,
            -exact_key_count,
            -base_score,
            -log_base_rank,
            -name_x_address,
            -name_minus_address,
            -similarity_mean,
            -similarity_min,
            -exact_field_count,
            0 AS pair_label
        FROM f
    """


def collect_training(
    con: duckdb.DuckDBPyConnection,
    labeled_path: Path,
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reader = con.execute(
        make_training_sql(labeled_path)
    ).to_arrow_reader(batch_size=BATCH)

    frames: list[pd.DataFrame] = []
    for batch in reader:
        d = batch.to_pandas()
        if not d.empty:
            frames.append(d)

    if not frames:
        raise RuntimeError(f"No training rows for {target}.")

    df = pd.concat(frames, ignore_index=True)

    X = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
    y = df["pair_label"].astype("int8").to_numpy()

    stats = {
        "training_rows": int(len(df)),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
    }

    print(
        f"{target} training rows: {stats['training_rows']:,} | "
        f"positive={stats['positive_rows']:,} | "
        f"negative={stats['negative_rows']:,}",
        flush=True,
    )

    return X, y, stats


def fit_model(X: np.ndarray, y: np.ndarray, target: str, label: str):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype("float32")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    t0 = time.time()
    model.fit(Xs, y)
    print(
        f"{target} {label} model fit: {time.time() - t0:.1f}s",
        flush=True,
    )

    return scaler, model


def make_score_frame(d: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=d.index)

    numeric_cols = [
        "name_similarity",
        "address_similarity",
        "name_exact",
        "address_exact",
        "country_exact",
        "name_length_ratio",
        "address_length_ratio",
        "evidence_rows",
        "evidence_file_count",
        "exact_key_count",
        "base_score",
    ]

    for col in numeric_cols:
        x[col] = (
            pd.to_numeric(d[col], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )

    rank = (
        pd.to_numeric(d["base_rank"], errors="coerce")
        .fillna(1_000_000.0)
        .astype("float64")
    )

    x["log_base_rank"] = np.log1p(rank).astype("float32")
    x["name_x_address"] = (
        x["name_similarity"] * x["address_similarity"]
    ).astype("float32")
    x["name_minus_address"] = (
        x["name_similarity"] - x["address_similarity"]
    ).astype("float32")
    x["similarity_mean"] = (
        (x["name_similarity"] + x["address_similarity"]) / 2.0
    ).astype("float32")
    x["similarity_min"] = np.minimum(
        x["name_similarity"],
        x["address_similarity"],
    ).astype("float32")
    x["exact_field_count"] = (
        x["name_exact"]
        + x["address_exact"]
        + x["country_exact"]
    ).astype("float32")

    return x[FEATURES]


def score_holdout(
    con: duckdb.DuckDBPyConnection,
    labeled_path: Path,
    target: str,
    scaler,
    model,
    tag: str,
) -> Path:
    parts_dir = OUT / f"{tag}_parts_{target.lower()}"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Only holdout S1 IDs are referenced here. Final GT is not referenced.
    query = f"""
        SELECT
            l.source1_entity_id,
            l.candidate_entity_id,
            l.base_rank,
            l.name_similarity,
            l.address_similarity,
            l.name_exact,
            l.address_exact,
            l.country_exact,
            l.name_length_ratio,
            l.address_length_ratio,
            l.evidence_rows,
            l.evidence_file_count,
            l.exact_key_count,
            l.base_score
        FROM read_parquet({qp(labeled_path)}) l
        INNER JOIN read_parquet({qp(FINAL_S1)}) h
          ON l.source1_entity_id = h.source1_entity_id
    """

    reader = con.execute(query).to_arrow_reader(batch_size=BATCH)

    total = 0
    idx = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue

        x = make_score_frame(d)
        Xs = scaler.transform(
            x.to_numpy(dtype="float32")
        ).astype("float32")

        d["final_holdout_utility"] = model.decision_function(Xs)

        out = parts_dir / f"part_{idx:05d}.parquet"
        d[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "final_holdout_utility",
            ]
        ].to_parquet(
            out,
            index=False,
            compression="zstd",
        )

        total += len(d)
        idx += 1

        if total and total % 1_000_000 < len(d):
            print(
                f"{target} {tag} scored: {total:,}",
                flush=True,
            )

    if total == 0:
        raise RuntimeError(
            f"No final-holdout candidates found for {target} / {tag}."
        )

    output = OUT / f"{tag}_ranked_s1_{target.lower()}.parquet"
    output.unlink(missing_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                final_holdout_utility,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY final_holdout_utility DESC, candidate_entity_id
                ) AS final_holdout_rank
            FROM read_parquet({qp(parts_dir / "*.parquet")})
        )
        TO {qp(output)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    shutil.rmtree(parts_dir, ignore_errors=True)
    return output


def evaluate_final(
    con: duckdb.DuckDBPyConnection,
    target: str,
    ranked_path: Path,
    labeled_path: Path,
) -> dict:
    header(f"FINAL HOLDOUT GT EVALUATION — {target}")

    # FINAL_HOLDOUT_GT is first read here.
    prefix = f"{target}-"

    gt_pairs = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({qp(FINAL_GT)})
            WHERE starts_with(matched_entity_id, '{prefix}')
            """
        ).fetchone()[0]
    )

    gt_s1s = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT source1_entity_id)
            FROM read_parquet({qp(FINAL_GT)})
            WHERE starts_with(matched_entity_id, '{prefix}')
            """
        ).fetchone()[0]
    )

    candidate_rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet({qp(labeled_path)}) c
            INNER JOIN read_parquet({qp(FINAL_S1)}) h
              ON c.source1_entity_id = h.source1_entity_id
            WHERE starts_with(c.candidate_entity_id, '{prefix}')
            """
        ).fetchone()[0]
    )

    candidate_covered = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT
                    g.source1_entity_id,
                    g.matched_entity_id
                FROM read_parquet({qp(FINAL_GT)}) g
                INNER JOIN read_parquet({qp(labeled_path)}) c
                  ON g.source1_entity_id = c.source1_entity_id
                 AND g.matched_entity_id = c.candidate_entity_id
                WHERE starts_with(g.matched_entity_id, '{prefix}')
            )
            """
        ).fetchone()[0]
    )

    result = {
        "target": target,
        "ground_truth_pairs": gt_pairs,
        "ground_truth_s1s": gt_s1s,
        "candidate_rows": candidate_rows,
        "candidate_covered_gt_pairs": candidate_covered,
        "candidate_coverage_pct": (
            100.0 * candidate_covered / gt_pairs if gt_pairs else 0.0
        ),
        "ranking": {},
    }

    for k in [1, 3, 5, 10, 20]:
        covered = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        g.source1_entity_id,
                        g.matched_entity_id
                    FROM read_parquet({qp(FINAL_GT)}) g
                    INNER JOIN read_parquet({qp(ranked_path)}) r
                      ON g.source1_entity_id = r.source1_entity_id
                     AND g.matched_entity_id = r.candidate_entity_id
                    WHERE starts_with(g.matched_entity_id, '{prefix}')
                      AND r.final_holdout_rank <= {k}
                )
                """
            ).fetchone()[0]
        )

        result["ranking"][f"covered_at_{k}"] = covered
        result["ranking"][f"recall_at_{k}"] = (
            100.0 * covered / gt_pairs if gt_pairs else 0.0
        )

    result["ranking"]["missed_at_20_total"] = (
        gt_pairs - result["ranking"]["covered_at_20"]
    )
    result["ranking"]["missed_at_20_candidate_missing"] = (
        gt_pairs - candidate_covered
    )
    result["ranking"]["missed_at_20_rank_gt_20"] = (
        candidate_covered - result["ranking"]["covered_at_20"]
    )

    zero_candidate_s1 = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet({qp(FINAL_GT)})
                WHERE starts_with(matched_entity_id, '{prefix}')
            ) g
            LEFT JOIN (
                SELECT DISTINCT c.source1_entity_id
                FROM read_parquet({qp(labeled_path)}) c
                INNER JOIN read_parquet({qp(FINAL_S1)}) h
                  ON c.source1_entity_id = h.source1_entity_id
                WHERE starts_with(c.candidate_entity_id, '{prefix}')
            ) c
              ON g.source1_entity_id = c.source1_entity_id
            WHERE c.source1_entity_id IS NULL
            """
        ).fetchone()[0]
    )

    result["gt_s1s_with_zero_candidates"] = zero_candidate_s1

    print(f"GT pairs                     : {gt_pairs:,}")
    print(f"GT S1s                       : {gt_s1s:,}")
    print(f"Candidate rows               : {candidate_rows:,}")
    print(
        f"Candidate-covered GT pairs  : {candidate_covered:,} "
        f"({result['candidate_coverage_pct']:.4f}%)"
    )
    print(f"GT S1s with zero candidates  : {zero_candidate_s1:,}")

    for k in [1, 3, 5, 10, 20]:
        rr = result["ranking"][f"recall_at_{k}"]
        cc = result["ranking"][f"covered_at_{k}"]
        print(
            f"Recall@{k:<2}                    : "
            f"{rr:.4f}% ({cc:,})"
        )

    print(
        f"Missed @20 — candidate missing: "
        f"{result['ranking']['missed_at_20_candidate_missing']:,}"
    )
    print(
        f"Missed @20 — rank >20         : "
        f"{result['ranking']['missed_at_20_rank_gt_20']:,}"
    )

    return result


def run_system(
    con: duckdb.DuckDBPyConnection,
    target: str,
    labeled_path: Path,
    tag: str,
) -> tuple[dict, dict]:
    header(f"{tag.upper()} — FINAL LOCKED HOLDOUT V2 — S1 -> {target}")

    require_file(labeled_path)
    labeled_rows = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({qp(labeled_path)})"
        ).fetchone()[0]
    )

    print(f"Candidate pool rows (all S1): {labeled_rows:,}")

    holdout_labeled_rows = assert_training_exclusion(
        con,
        labeled_path,
        target,
    )
    print(f"Labeled candidate rows on holdout S1: {holdout_labeled_rows:,}")

    X, y, train_stats = collect_training(
        con,
        labeled_path,
        target,
    )

    scaler, model = fit_model(
        X,
        y,
        target,
        f"FINAL_{tag}",
    )

    ranked = score_holdout(
        con,
        labeled_path,
        target,
        scaler,
        model,
        tag.lower(),
    )

    metrics = evaluate_final(
        con,
        target,
        ranked,
        labeled_path,
    )

    return (
        {
            "candidate_pool_rows": labeled_rows,
            "holdout_labeled_rows_expected_excluded_from_training": holdout_labeled_rows,
            "training": train_stats,
            "ranked_output": str(ranked),
        },
        metrics,
    )


def main() -> None:
    header("AMAZON ML CHALLENGE — FINAL LOCKED HOLDOUT V2 EVALUATION")

    print(f"Project                    : {PROJECT}")
    print(f"Memory                     : {MEMORY}")
    print(f"Threads                    : {THREADS}")
    print(f"Batch                      : {BATCH:,}")
    print("Final holdout status       : LOCKED")
    print("Candidate generation       : NOT RUN")
    print("Block selection            : NOT RUN")
    print("Tuning                     : NOT RUN")
    print("Final holdout GT during fit: NOT READ")

    manifest = verify_manifest()

    for p in [
        FINAL_S1,
        FINAL_GT,
        BASELINE_LABELED["S2"],
        BASELINE_LABELED["S3"],
        EXPANDED_LABELED["S2"],
        EXPANDED_LABELED["S3"],
    ]:
        require_file(p)

    OUT.mkdir(parents=True, exist_ok=True)
    con = setup_connection()
    holdout_s1_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet({qp(FINAL_S1)})"
        ).fetchone()[0]
    )
    holdout_s1_unique = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT source1_entity_id)
            FROM read_parquet({qp(FINAL_S1)})
            """
        ).fetchone()[0]
    )

    if holdout_s1_count != holdout_s1_unique:
        con.close()
        raise RuntimeError("Final holdout S1 is not unique.")

    print(f"Final holdout S1 rows      : {holdout_s1_count:,}")

    t0 = time.time()

    try:
        results = {}

        # Evaluate both frozen systems on the same locked holdout.
        results["S2_baseline"] = run_system(
            con,
            "S2",
            BASELINE_LABELED["S2"],
            "BASELINE",
        )
        results["S2_expanded"] = run_system(
            con,
            "S2",
            EXPANDED_LABELED["S2"],
            "EXPANDED_V8",
        )
        results["S3_baseline"] = run_system(
            con,
            "S3",
            BASELINE_LABELED["S3"],
            "BASELINE",
        )
        results["S3_expanded"] = run_system(
            con,
            "S3",
            EXPANDED_LABELED["S3"],
            "EXPANDED_V8",
        )
    finally:
        con.close()

    comparison = {}
    for target in ("S2", "S3"):
        b = results[f"{target}_baseline"][1]
        e = results[f"{target}_expanded"][1]

        comparison[target] = {}
        for k in [1, 3, 5, 10, 20]:
            comparison[target][f"recall_at_{k}"] = {
                "baseline": b["ranking"][f"recall_at_{k}"],
                "expanded": e["ranking"][f"recall_at_{k}"],
                "delta_pp": (
                    e["ranking"][f"recall_at_{k}"]
                    - b["ranking"][f"recall_at_{k}"]
                ),
            }

    report = {
        "version": "FINAL_LOCKED_HOLDOUT_V2_EVAL_V2",
        "holdout_manifest": str(FINAL_MANIFEST),
        "holdout_s1": str(FINAL_S1),
        "holdout_gt": str(FINAL_GT),
        "holdout_s1_rows": holdout_s1_count,
        "final_holdout_gt_used_for_training": False,
        "candidate_generation_run": False,
        "block_selection_run": False,
        "tuning_run": False,
        "historical_caveat": (
            "This is a post-selection holdout check because earlier "
            "TRAIN-GT-informed experiments influenced the frozen candidate "
            "architecture. No further selection or tuning is permitted "
            "using FINAL HOLDOUT V2."
        ),
        "manifest": manifest,
        "results": results,
        "comparison": comparison,
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    report_path = OUT / "final_holdout_v2_evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    header("FINAL HOLDOUT V2 EVALUATION COMPLETE")

    print(f"Report: {report_path}")

    for target in ("S2", "S3"):
        print(
            f"{target} R@20: "
            f"{comparison[target]['recall_at_20']['baseline']:.4f}% -> "
            f"{comparison[target]['recall_at_20']['expanded']:.4f}% "
            f"("
            f"{comparison[target]['recall_at_20']['delta_pp']:+.4f} pp)"
        )

    print("\nFINAL HOLDOUT STATUS: EVALUATED")
    print("DO NOT USE FINAL HOLDOUT V2 TO TUNE OR SELECT ANYTHING.")


if __name__ == "__main__":
    main()
