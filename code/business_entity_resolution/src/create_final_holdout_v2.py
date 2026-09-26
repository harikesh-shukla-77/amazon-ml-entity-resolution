#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb

# ============================================================
# AMAZON ML CHALLENGE
# FINAL LOCKED HOLDOUT V2 — CREATION + INTEGRITY ONLY
#
# Purpose:
#   Create a BRAND-NEW S1 holdout AFTER V8 architecture selection.
#
# This script:
#   - reads only TRAIN source1 + TRAIN ground truth
#   - does NOT read V6/V7/V8/V9/V10 candidate/ranking outputs
#   - does NOT evaluate any model
#   - does NOT tune any block/model parameter
#   - creates a new holdout S1 boundary and its GT snapshot
#   - proves disjointness from:
#       1) old V7 validation split
#       2) the consumed prospective_holdout_v1
#
# IMPORTANT:
#   The resulting holdout is to be treated as LOCKED.
#   Do not use its GT for model/block selection after this point.
# ============================================================

PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"
SOURCE1 = TRAIN / "train_source1.parquet"
GT = TRAIN / "ground_truth_pairs.parquet"

AUDIT = PROJECT / "validation_leakage_audit"
PREVIOUS_HOLDOUT_DIR = AUDIT / "holdout"
PREVIOUS_HOLDOUT_S1 = PREVIOUS_HOLDOUT_DIR / "prospective_holdout_s1.parquet"
PREVIOUS_HOLDOUT_GT = PREVIOUS_HOLDOUT_DIR / "prospective_holdout_ground_truth.parquet"

FINAL_DIR = AUDIT / "final_holdout_v2"
FINAL_S1 = FINAL_DIR / "final_holdout_s1.parquet"
FINAL_GT = FINAL_DIR / "final_holdout_ground_truth.parquet"
MANIFEST = FINAL_DIR / "final_holdout_v2_manifest.json"
TMP = FINAL_DIR / "duckdb_tmp"

MEMORY = os.environ.get("FINAL_HOLDOUT_MEMORY", "4GB")
THREADS = int(os.environ.get("FINAL_HOLDOUT_THREADS", "4"))

OLD_VALIDATION_MOD = 5
PREVIOUS_HOLDOUT_SALT = "PROSPECTIVE_HOLDOUT_V1"
FINAL_HOLDOUT_SALT = "FINAL_HOLDOUT_V2"
FINAL_HOLDOUT_MOD = 10


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
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory={qp(TMP)}")
    return con


def main() -> None:
    header("AMAZON ML CHALLENGE — FINAL LOCKED HOLDOUT V2")

    print(f"Project                   : {PROJECT}")
    print(f"Memory                    : {MEMORY}")
    print(f"Threads                   : {THREADS}")
    print(f"Old validation modulus    : {OLD_VALIDATION_MOD}")
    print(f"Previous holdout salt     : {PREVIOUS_HOLDOUT_SALT}")
    print(f"Final holdout salt        : {FINAL_HOLDOUT_SALT}")
    print(f"Final holdout modulus     : {FINAL_HOLDOUT_MOD}")
    print("Candidate outputs         : NOT READ")
    print("V6/V7/V8/V9/V10 outputs  : NOT READ")
    print("Model/ranking evaluation  : NOT RUN")
    print("Post-creation tuning      : MUST NOT USE FINAL HOLDOUT GT")

    require_file(SOURCE1)
    require_file(GT)
    require_file(PREVIOUS_HOLDOUT_S1)
    require_file(PREVIOUS_HOLDOUT_GT)

    # Never overwrite an existing final holdout. The point of this stage is
    # to establish a lock; accidental regeneration with a changed recipe
    # would undermine provenance.
    if FINAL_S1.exists() or FINAL_GT.exists() or MANIFEST.exists():
        raise RuntimeError(
            f"Final holdout already exists under {FINAL_DIR}. "
            "Refusing to overwrite a locked holdout."
        )

    con = setup_connection()
    t0 = time.time()

    try:
        header("1. SOURCE1 POPULATION")

        total_s1 = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({qp(SOURCE1)})"
            ).fetchone()[0]
        )

        unique_s1 = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT CAST(entity_id AS VARCHAR))
                FROM read_parquet({qp(SOURCE1)})
                """
            ).fetchone()[0]
        )

        old_validation_s1 = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT CAST(entity_id AS VARCHAR) AS entity_id
                    FROM read_parquet({qp(SOURCE1)})
                    WHERE MOD(
                        ABS(HASH(CAST(entity_id AS VARCHAR))),
                        {OLD_VALIDATION_MOD}
                    ) = 0
                )
                """
            ).fetchone()[0]
        )

        previous_holdout_s1 = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT CAST(source1_entity_id AS VARCHAR))
                FROM read_parquet({qp(PREVIOUS_HOLDOUT_S1)})
                """
            ).fetchone()[0]
        )

        print(f"Total source1 rows          : {total_s1:,}")
        print(f"Unique source1 entities     : {unique_s1:,}")
        print(f"Old V7 validation S1s       : {old_validation_s1:,}")
        print(f"Previous holdout V1 S1s     : {previous_holdout_s1:,}")

        if total_s1 != unique_s1:
            raise RuntimeError("source1 entity_id is not unique.")

        header("2. BUILD NEW HOLDOUT BOUNDARY")

        # Exclusion order:
        #   A) old V7 validation is excluded
        #   B) consumed prospective_holdout_v1 is excluded
        #   C) final holdout is a fresh salted hash sample from the remainder
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE source1_base AS
            SELECT DISTINCT
                CAST(entity_id AS VARCHAR) AS source1_entity_id
            FROM read_parquet({qp(SOURCE1)})
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE previous_holdout_ids AS
            SELECT DISTINCT
                CAST(source1_entity_id AS VARCHAR) AS source1_entity_id
            FROM read_parquet({qp(PREVIOUS_HOLDOUT_S1)})
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE final_holdout_ids AS
            SELECT source1_entity_id
            FROM source1_base
            WHERE MOD(
                ABS(HASH(source1_entity_id)),
                {OLD_VALIDATION_MOD}
            ) <> 0
              AND source1_entity_id NOT IN (
                  SELECT source1_entity_id
                  FROM previous_holdout_ids
              )
              AND MOD(
                  ABS(HASH(
                      source1_entity_id || '|{FINAL_HOLDOUT_SALT}|'
                  )),
                  {FINAL_HOLDOUT_MOD}
              ) = 0
            """
        )

        final_s1_count = int(
            con.execute(
                "SELECT COUNT(*) FROM final_holdout_ids"
            ).fetchone()[0]
        )

        print(f"New final holdout S1s      : {final_s1_count:,}")
        print(
            f"Fraction of all source1    : "
            f"{100.0 * final_s1_count / total_s1:.6f}%"
        )

        if final_s1_count == 0:
            raise RuntimeError("Final holdout is empty.")

        con.execute(
            f"""
            COPY (
                SELECT source1_entity_id
                FROM final_holdout_ids
                ORDER BY source1_entity_id
            )
            TO {qp(FINAL_S1)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        header("3. DISJOINTNESS CHECKS")

        old_overlap = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM final_holdout_ids f
                INNER JOIN source1_base s
                  ON f.source1_entity_id = s.source1_entity_id
                WHERE MOD(
                    ABS(HASH(f.source1_entity_id)),
                    {OLD_VALIDATION_MOD}
                ) = 0
                """
            ).fetchone()[0]
        )

        previous_overlap = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM final_holdout_ids f
                INNER JOIN previous_holdout_ids p
                  ON f.source1_entity_id = p.source1_entity_id
                """
            ).fetchone()[0]
        )

        saved_duplicate_rows = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({qp(FINAL_S1)})
                """
            ).fetchone()[0]
        )

        saved_unique_rows = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT source1_entity_id)
                FROM read_parquet({qp(FINAL_S1)})
                """
            ).fetchone()[0]
        )

        print(f"Overlap with old V7 validation: {old_overlap:,}")
        print(f"Overlap with previous holdout: {previous_overlap:,}")
        print(f"Saved S1 rows                : {saved_duplicate_rows:,}")
        print(f"Saved unique S1              : {saved_unique_rows:,}")

        if old_overlap != 0:
            raise RuntimeError("Final holdout overlaps old V7 validation.")
        if previous_overlap != 0:
            raise RuntimeError("Final holdout overlaps consumed previous holdout.")
        if saved_duplicate_rows != saved_unique_rows:
            raise RuntimeError("Final holdout contains duplicate S1 IDs.")

        header("4. EXTRACT FINAL HOLDOUT GROUND TRUTH")

        # This creates the evaluation-only label snapshot.
        # It is NOT used by this script for model training/evaluation.
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT
                    CAST(g.source1_entity_id AS VARCHAR) AS source1_entity_id,
                    CAST(g.matched_entity_id AS VARCHAR) AS matched_entity_id,
                    COALESCE(g.label, 1)::BIGINT AS label
                FROM read_parquet({qp(GT)}) g
                INNER JOIN final_holdout_ids h
                  ON CAST(g.source1_entity_id AS VARCHAR) =
                     h.source1_entity_id
                WHERE COALESCE(g.label, 1) = 1
            )
            TO {qp(FINAL_GT)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        final_gt_rows = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet({qp(FINAL_GT)})"
            ).fetchone()[0]
        )

        final_gt_pairs_unique = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT
                        source1_entity_id,
                        matched_entity_id
                    FROM read_parquet({qp(FINAL_GT)})
                )
                """
            ).fetchone()[0]
        )

        final_gt_labels_nonpositive = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({qp(FINAL_GT)})
                WHERE COALESCE(label, 1) <> 1
                """
            ).fetchone()[0]
        )

        gt_s1s = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT source1_entity_id)
                FROM read_parquet({qp(FINAL_GT)})
                """
            ).fetchone()[0]
        )

        print(f"Final holdout GT rows       : {final_gt_rows:,}")
        print(f"Final GT unique pairs       : {final_gt_pairs_unique:,}")
        print(f"Final GT S1s with positives : {gt_s1s:,}")
        print(f"Non-positive GT rows        : {final_gt_labels_nonpositive:,}")

        if final_gt_rows != final_gt_pairs_unique:
            raise RuntimeError("Final holdout GT contains duplicate pairs.")
        if final_gt_labels_nonpositive != 0:
            raise RuntimeError("Final holdout GT contains non-positive labels.")

        header("5. LOCK MANIFEST")

        elapsed = round(time.time() - t0, 3)

        manifest = {
            "version": "FINAL_LOCKED_HOLDOUT_V2",
            "status": "LOCKED",
            "purpose": (
                "Fresh final evaluation boundary created after V8 "
                "architecture selection and before final end-to-end evaluation."
            ),
            "source1_total_rows": total_s1,
            "source1_unique_entities": unique_s1,
            "old_v7_validation_s1": old_validation_s1,
            "previous_prospective_holdout_v1_s1": previous_holdout_s1,
            "final_holdout_s1": final_s1_count,
            "final_holdout_fraction_of_all_s1": final_s1_count / total_s1,
            "final_holdout_gt_rows": final_gt_rows,
            "final_holdout_gt_unique_pairs": final_gt_pairs_unique,
            "final_holdout_gt_s1s": gt_s1s,
            "final_holdout_salt": FINAL_HOLDOUT_SALT,
            "final_holdout_modulus": FINAL_HOLDOUT_MOD,
            "old_validation_modulus": OLD_VALIDATION_MOD,
            "previous_holdout_excluded": True,
            "old_validation_excluded": True,
            "candidate_outputs_read": False,
            "ranking_evaluation_run": False,
            "model_tuning_run": False,
            "final_holdout_gt_used_for_selection": False,
            "final_holdout_gt_used_for_training": False,
            "disjoint_old_validation": old_overlap == 0,
            "disjoint_previous_holdout": previous_overlap == 0,
            "s1_unique": saved_duplicate_rows == saved_unique_rows,
            "elapsed_seconds": elapsed,
            "files": {
                "s1": str(FINAL_S1),
                "ground_truth": str(FINAL_GT),
            },
        }

        MANIFEST.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        header("FINAL HOLDOUT V2 CREATED")

        print(f"S1 holdout : {FINAL_S1}")
        print(f"GT         : {FINAL_GT}")
        print(f"Manifest   : {MANIFEST}")
        print("\nLOCK CHECK: PASS")
        print("DO NOT USE final_holdout_ground_truth.parquet for tuning.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
