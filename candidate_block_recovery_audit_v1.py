#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
CANDIDATE-BLOCK RECOVERY AUDIT V1

Purpose
-------
The comprehensive error audit showed that the dominant remaining problem is
candidate generation, not thresholding:

  OLD/FRESH candidate-missing GT is ~67-69% in both S2 and S3.

This script therefore evaluates a library of tighter blocking keys.

IMPORTANT:
  This is a DEVELOPMENT/DIAGNOSTIC audit.
  OLD V7 validation is used to select a small set of promising blocks.
  FRESH development is used only for blind confirmation.

NO:
  - Final Holdout V2 GT
  - test data
  - model training
  - candidate generation files
  - final output modifications
  - ZIP modification

The script does NOT materialize candidate-pair files. It only estimates:
  - candidate pair cost
  - GT pairs recovered
  - NEW GT beyond the existing V6+V9.1 baseline
  - recovery efficiency

Then it greedily selects up to MAX_BLOCKS_PER_TARGET blocks under a pair
budget on OLD validation. Those selected blocks are evaluated blindly on the
FRESH development split.

This is intended to answer:
  "Which small blocking additions are worth implementing next?"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb


PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"

S1_PATH = TRAIN / "train_source1.parquet"
S2_PATH = TRAIN / "train_source2.parquet"
S3_PATH = TRAIN / "train_source3.parquet"
GT_PATH = TRAIN / "ground_truth_pairs.parquet"

OLD_RANKED = {
    "S2": PROJECT / "candidate_output_v10_v8_leakage_safe"
          / "baseline_true_validation_ranked_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10_v8_leakage_safe"
          / "baseline_true_validation_ranked_s1_s3.parquet",
}

FRESH_DIR = (
    PROJECT
    / "validation_leakage_audit"
    / "f05_fresh_dev_joint_confirmation_v1"
)

FRESH_S1 = FRESH_DIR / "fresh_dev_s1.parquet"
FRESH_RANKED = {
    "S2": FRESH_DIR / "fresh_ranked_s2.parquet",
    "S3": FRESH_DIR / "fresh_ranked_s3.parquet",
}

OUT = (
    PROJECT
    / "validation_leakage_audit"
    / "candidate_block_recovery_audit_v1"
)
REPORT = OUT / "candidate_block_recovery_audit_v1_report.json"

MEMORY = os.environ.get("BLOCK_AUDIT_MEMORY", "8GB")
THREADS = int(os.environ.get("BLOCK_AUDIT_THREADS", "2"))

OLD_VALIDATION_MOD = 5

# Prevent huge, low-quality blocks from dominating the experiment.
MAX_BUCKET_SIZE = int(
    os.environ.get("BLOCK_AUDIT_MAX_BUCKET_SIZE", "100")
)

OLD_PAIR_BUDGET = int(
    os.environ.get("BLOCK_AUDIT_OLD_PAIR_BUDGET", "12000000")
)

MAX_BLOCKS_PER_TARGET = int(
    os.environ.get("BLOCK_AUDIT_MAX_BLOCKS", "3")
)

MIN_NEW_GT = int(
    os.environ.get("BLOCK_AUDIT_MIN_NEW_GT", "100")
)


BLOCKS = {
    # ------------------------------------------------------------
    # Tight name-shape blocks
    # ------------------------------------------------------------
    "name_f3_l2_len5": {
        "expr": (
            "substr(name_compact,1,3) || '|' "
            "|| right(name_compact,2) || '|' "
            "|| CAST(floor(length(name_compact)/5) AS BIGINT)"
        ),
        "known": "new_candidate",
    },
    "name_f2_l3_len5": {
        "expr": (
            "substr(name_compact,1,2) || '|' "
            "|| right(name_compact,3) || '|' "
            "|| CAST(floor(length(name_compact)/5) AS BIGINT)"
        ),
        "known": "new_candidate",
    },
    "name_f4_l2_len5": {
        "expr": (
            "substr(name_compact,1,4) || '|' "
            "|| right(name_compact,2) || '|' "
            "|| CAST(floor(length(name_compact)/5) AS BIGINT)"
        ),
        "known": "new_candidate",
    },
    "name_f3_l2_postal": {
        "expr": (
            "postal || '|' || substr(name_compact,1,3) "
            "|| '|' || right(name_compact,2)"
        ),
        "known": "new_candidate",
    },
    "name_f2_l3_postal": {
        "expr": (
            "postal || '|' || substr(name_compact,1,2) "
            "|| '|' || right(name_compact,3)"
        ),
        "known": "new_candidate",
    },
    "name_f1_l2_postal": {
        "expr": (
            "postal || '|' || substr(name_compact,1,1) "
            "|| '|' || right(name_compact,2)"
        ),
        "known": "new_candidate",
    },

    # ------------------------------------------------------------
    # Name/address mixed blocks
    # ------------------------------------------------------------
    "name_f2_addr_f3_len5": {
        "expr": (
            "substr(name_compact,1,2) || '|' "
            "|| substr(address_compact,1,3) || '|' "
            "|| CAST(floor(length(address_compact)/5) AS BIGINT)"
        ),
        "known": "new_candidate",
    },
    "name_f3_addr_tail3_len5": {
        "expr": (
            "substr(name_compact,1,3) || '|' "
            "|| right(address_compact,3) || '|' "
            "|| CAST(floor(length(address_compact)/5) AS BIGINT)"
        ),
        "known": "new_candidate",
    },
    "addr_f3_tail3_name_f1": {
        "expr": (
            "substr(address_compact,1,3) || '|' "
            "|| right(address_compact,3) || '|' "
            "|| substr(name_compact,1,1)"
        ),
        "known": "new_candidate",
    },
    "addr_f4_tail4_name_f1": {
        "expr": (
            "substr(address_compact,1,4) || '|' "
            "|| right(address_compact,4) || '|' "
            "|| substr(name_compact,1,1)"
        ),
        "known": "new_candidate",
    },

    # ------------------------------------------------------------
    # Postal combinations
    # ------------------------------------------------------------
    "postal_name_f2_l2": {
        "expr": (
            "postal || '|' || substr(name_compact,1,2) "
            "|| '|' || right(name_compact,2)"
        ),
        "known": "new_candidate",
    },
    "postal_name_f3_l1": {
        "expr": (
            "postal || '|' || substr(name_compact,1,3) "
            "|| '|' || right(name_compact,1)"
        ),
        "known": "new_candidate",
    },
    "postal_addr_f3_tail2_name_f1": {
        "expr": (
            "postal || '|' || substr(address_compact,1,3) "
            "|| '|' || right(address_compact,2) "
            "|| '|' || substr(name_compact,1,1)"
        ),
        "known": "new_candidate",
    },
    "postal_addr_f2_name_f2": {
        "expr": (
            "postal || '|' || substr(address_compact,1,2) "
            "|| '|' || substr(name_compact,1,2)"
        ),
        "known": "new_candidate",
    },

    # ------------------------------------------------------------
    # Numeric-address blocks
    # ------------------------------------------------------------
    "num_addr_name_prefix2": {
        "expr": (
            "addr_num || '|' || substr(name_compact,1,2)"
        ),
        "known": "V8_known",
    },
    "num_addr_name_prefix3": {
        "expr": (
            "addr_num || '|' || substr(name_compact,1,3)"
        ),
        "known": "V9.1_known",
    },
    "num_addr_name_prefix4": {
        "expr": (
            "addr_num || '|' || substr(name_compact,1,4)"
        ),
        "known": "new_candidate",
    },
    "num_addr_name_f1_l2": {
        "expr": (
            "addr_num || '|' || substr(name_compact,1,1) "
            "|| '|' || right(name_compact,2)"
        ),
        "known": "new_candidate",
    },

    # ------------------------------------------------------------
    # Cross-signature blocks
    # ------------------------------------------------------------
    "name_f3_addr_f2": {
        "expr": (
            "substr(name_compact,1,3) || '|' "
            "|| substr(address_compact,1,2)"
        ),
        "known": "new_candidate",
    },
    "name_f2_addr_tail3": {
        "expr": (
            "substr(name_compact,1,2) || '|' "
            "|| right(address_compact,3)"
        ),
        "known": "new_candidate",
    },
    "name_f1_addr_f4_tail2": {
        "expr": (
            "substr(name_compact,1,1) || '|' "
            "|| substr(address_compact,1,4) || '|' "
            "|| right(address_compact,2)"
        ),
        "known": "new_candidate",
    },
}


def header(s: str) -> None:
    print("\n" + "=" * 118)
    print(s)
    print("=" * 118, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")


def build_source_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    path: Path,
) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            COALESCE(country_norm,'') AS country_norm,
            regexp_replace(
                COALESCE(name_norm,''),
                '[^a-z0-9]',
                '',
                'g'
            ) AS name_compact,
            regexp_replace(
                COALESCE(address_norm,''),
                '[^a-z0-9]',
                '',
                'g'
            ) AS address_compact,
            regexp_extract(
                COALESCE(address_norm,''),
                '[0-9]{{5,6}}',
                0
            ) AS postal,
            regexp_replace(
                regexp_replace(
                    COALESCE(address_norm,''),
                    '[^0-9]',
                    '',
                    'g'
                ),
                '^0+$',
                '',
                'g'
            ) AS addr_num
        FROM read_parquet({qp(path)})
        """
    )


def build_eval_ids(
    con: duckdb.DuckDBPyConnection,
    mode: str,
) -> int:
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
            SELECT DISTINCT
                CAST(source1_entity_id AS VARCHAR) AS s1
            FROM read_parquet({qp(FRESH_S1)})
        """)
    else:
        raise ValueError(mode)

    return int(
        con.execute("SELECT COUNT(*) FROM eval_ids").fetchone()[0]
    )


def build_gt_eval(
    con: duckdb.DuckDBPyConnection,
    target_table: str,
    target: str,
) -> int:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_eval AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)}) g
        INNER JOIN eval_ids e
            ON CAST(g.source1_entity_id AS VARCHAR) = e.s1
        WHERE COALESCE(g.label,1)=1
          AND CAST(g.matched_entity_id AS VARCHAR)
              LIKE '{target}-%'
    """)
    return int(
        con.execute("SELECT COUNT(*) FROM gt_eval").fetchone()[0]
    )


def build_baseline_cov(
    con: duckdb.DuckDBPyConnection,
    ranked_path: Path,
) -> int:
    utility_cols = {
        str(r[0])
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({qp(ranked_path)})"
        ).fetchall()
    }
    # Ranking files are complete baseline candidate pools. Only IDs are
    # needed for coverage.
    rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM gt_eval g
            INNER JOIN read_parquet({qp(ranked_path)}) r
                ON CAST(r.source1_entity_id AS VARCHAR)=g.s1
               AND CAST(r.candidate_entity_id AS VARCHAR)=g.matched
            """
        ).fetchone()[0]
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE baseline_cov AS
        SELECT DISTINCT
            g.s1,
            g.matched
        FROM gt_eval g
        INNER JOIN read_parquet({qp(ranked_path)}) r
            ON CAST(r.source1_entity_id AS VARCHAR)=g.s1
           AND CAST(r.candidate_entity_id AS VARCHAR)=g.matched
    """)

    return rows


def make_key_view(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    view_name: str,
    expression: str,
) -> None:
    # Country is always part of the key to preserve open-set country handling.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW {view_name} AS
        SELECT
            entity_id,
            country_norm || '|' || ({expression}) AS block_key
        FROM {source_table}
        """
    )


def evaluate_block(
    con: duckdb.DuckDBPyConnection,
    target: str,
    target_table: str,
    block_name: str,
    expr: str,
) -> dict:
    left_view = f"left_{block_name}"
    right_view = f"right_{block_name}"

    make_key_view(
        con,
        "s1_eval_source",
        left_view,
        expr,
    )
    make_key_view(
        con,
        target_table,
        right_view,
        expr,
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE left_counts AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {left_view}
        WHERE block_key <> ''
          AND NOT ends_with(block_key, '|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE right_counts AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {right_view}
        WHERE block_key <> ''
          AND NOT ends_with(block_key, '|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    cost = int(
        con.execute("""
            SELECT COALESCE(
                SUM(l.n * r.n),
                0
            )
            FROM left_counts l
            INNER JOIN right_counts r
                ON l.block_key = r.block_key
        """).fetchone()[0]
    )

    # Build keys for the GT universe without materializing candidate pairs.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_block AS
        SELECT
            g.s1,
            g.matched,
            l.block_key AS left_key,
            r.block_key AS right_key
        FROM gt_eval g
        INNER JOIN {left_view} l
            ON l.entity_id = g.s1
        INNER JOIN {right_view} r
            ON r.entity_id = g.matched
    """)

    covered = int(
        con.execute("""
            SELECT COUNT(*)
            FROM gt_block g
            INNER JOIN left_counts l
                ON l.block_key = g.left_key
            INNER JOIN right_counts r
                ON r.block_key = g.right_key
            WHERE g.left_key = g.right_key
        """).fetchone()[0]
    )

    new_gt = int(
        con.execute("""
            SELECT COUNT(*)
            FROM gt_block g
            INNER JOIN left_counts l
                ON l.block_key = g.left_key
            INNER JOIN right_counts r
                ON r.block_key = g.right_key
            LEFT JOIN baseline_cov b
                ON b.s1 = g.s1
               AND b.matched = g.matched
            WHERE g.left_key = g.right_key
              AND b.s1 IS NULL
        """).fetchone()[0]
    )

    return {
        "block": block_name,
        "known": BLOCKS[block_name]["known"],
        "estimated_pairs": cost,
        "truth_covered": covered,
        "new_gt_vs_baseline": new_gt,
        "new_gt_per_million_pairs": (
            1_000_000.0 * new_gt / cost
            if cost else 0.0
        ),
        "within_budget": cost <= OLD_PAIR_BUDGET,
    }


def selected_union_coverage(
    con: duckdb.DuckDBPyConnection,
    selected: list[str],
    target_table: str,
    target: str,
) -> tuple[int, int]:
    """
    Recompute union coverage of selected blocks and incremental new GT vs
    baseline. No candidate pairs are materialized.
    """
    if not selected:
        return 0, 0

    selects = []
    for b in selected:
        expr = BLOCKS[b]["expr"]
        left_view = f"left_{b}"
        right_view = f"right_{b}"

        make_key_view(
            con,
            "s1_eval_source",
            left_view,
            expr,
        )
        make_key_view(
            con,
            target_table,
            right_view,
            expr,
        )

        selects.append(
            f"""
            SELECT
                g.s1,
                g.matched
            FROM gt_eval g
            INNER JOIN {left_view} l
                ON l.entity_id=g.s1
            INNER JOIN {right_view} r
                ON r.entity_id=g.matched
            INNER JOIN left_counts lc
                ON lc.block_key=l.block_key
            INNER JOIN right_counts rc
                ON rc.block_key=r.block_key
            WHERE l.block_key=r.block_key
            """
        )

    # Recreate counts correctly for each selected block inside this union
    # using a separate pair list of GT-only matches. This remains tiny
    # (bounded by GT count) compared with candidate-pair materialization.
    recovered = []
    for b in selected:
        expr = BLOCKS[b]["expr"]

        make_key_view(
            con,
            "s1_eval_source",
            f"sel_l_{b}",
            expr,
        )
        make_key_view(
            con,
            target_table,
            f"sel_r_{b}",
            expr,
        )

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE sel_lc_{b} AS
            SELECT block_key, COUNT(*)::BIGINT AS n
            FROM sel_l_{b}
            WHERE block_key <> ''
              AND NOT ends_with(block_key,'|')
            GROUP BY block_key
            HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE sel_rc_{b} AS
            SELECT block_key, COUNT(*)::BIGINT AS n
            FROM sel_r_{b}
            WHERE block_key <> ''
              AND NOT ends_with(block_key,'|')
            GROUP BY block_key
            HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
        """)

        recovered.append(
            f"""
            SELECT DISTINCT
                g.s1,
                g.matched
            FROM gt_eval g
            INNER JOIN sel_l_{b} l
                ON l.entity_id=g.s1
            INNER JOIN sel_r_{b} r
                ON r.entity_id=g.matched
            INNER JOIN sel_lc_{b} lc
                ON lc.block_key=l.block_key
            INNER JOIN sel_rc_{b} rc
                ON rc.block_key=r.block_key
            WHERE l.block_key=r.block_key
            """
        )

    union_sql = "\nUNION\n".join(recovered)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE selected_cov AS
        {union_sql}
    """)

    total_selected = int(
        con.execute(
            "SELECT COUNT(*) FROM selected_cov"
        ).fetchone()[0]
    )

    new_selected = int(
        con.execute("""
            SELECT COUNT(*)
            FROM selected_cov s
            LEFT JOIN baseline_cov b
                ON b.s1=s.s1
               AND b.matched=s.matched
            WHERE b.s1 IS NULL
        """).fetchone()[0]
    )

    return total_selected, new_selected


def run_mode(
    con: duckdb.DuckDBPyConnection,
    mode: str,
    s1_eval_path: Path | None,
) -> dict:
    header(f"{mode} BLOCK RECOVERY AUDIT")

    n = build_eval_ids(con, mode)

    # Materialize only the S1 evaluation source, never candidate pairs.
    if mode == "OLD":
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE s1_eval_source AS
            SELECT s.*
            FROM s1_source s
            INNER JOIN eval_ids e
                ON s.entity_id=e.s1
        """)
    else:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE s1_eval_source AS
            SELECT s.*
            FROM s1_source s
            INNER JOIN eval_ids e
                ON s.entity_id=e.s1
        """)

    result = {
        "s1_entities": n,
        "targets": {},
    }

    for target, target_table in [
        ("S2", "s2_source"),
        ("S3", "s3_source"),
    ]:
        header(f"{mode} / S1 -> {target}")

        gt_total = build_gt_eval(con, target_table, target)
        baseline_cov = build_baseline_cov(
            con,
            OLD_RANKED[target] if mode == "OLD"
            else FRESH_RANKED[target],
        )

        print(f"S1 entities          : {n:,}")
        print(f"GT pairs             : {gt_total:,}")
        print(f"Baseline-covered GT  : {baseline_cov:,}")
        print(
            f"Baseline-missing GT  : "
            f"{gt_total - baseline_cov:,}"
        )

        block_results = []

        for block_name, meta in BLOCKS.items():
            r = evaluate_block(
                con,
                target,
                target_table,
                block_name,
                meta["expr"],
            )
            block_results.append(r)

        block_results.sort(
            key=lambda x: (
                x["new_gt_vs_baseline"],
                x["new_gt_per_million_pairs"],
            ),
            reverse=True,
        )

        # OLD: greedy selection using only old validation evidence.
        selected = []
        budget_used = 0

        if mode == "OLD":
            remaining = block_results.copy()

            while (
                remaining
                and len(selected) < MAX_BLOCKS_PER_TARGET
            ):
                candidates = [
                    x for x in remaining
                    if x["new_gt_vs_baseline"] >= MIN_NEW_GT
                    and x["estimated_pairs"] > 0
                    and budget_used + x["estimated_pairs"]
                        <= OLD_PAIR_BUDGET
                ]

                if not candidates:
                    break

                candidates.sort(
                    key=lambda x: (
                        x["new_gt_per_million_pairs"],
                        x["new_gt_vs_baseline"],
                    ),
                    reverse=True,
                )

                pick = candidates[0]
                selected.append(pick["block"])
                budget_used += pick["estimated_pairs"]
                remaining = [
                    x for x in remaining
                    if x["block"] != pick["block"]
                ]

            sel_total, sel_new = selected_union_coverage(
                con,
                selected,
                target_table,
                target,
            )

            print("\nGREEDY OLD-VALIDATION SELECTION")
            print(
                f"Selected blocks     : {selected}"
            )
            print(
                f"Estimated cost      : {budget_used:,}"
            )
            print(
                f"Selected union GT   : {sel_total:,}"
            )
            print(
                f"Selected NEW GT     : {sel_new:,}"
            )
        else:
            # Fresh is confirmation-only. Use the exact blocks selected from
            # OLD; never select new blocks from fresh.
            selected = result.get("_selected_by_old", [])
            sel_total = sel_new = 0

        result["targets"][target] = {
            "ground_truth_pairs": gt_total,
            "baseline_covered_gt": baseline_cov,
            "baseline_missing_gt": gt_total - baseline_cov,
            "blocks_ranked": block_results,
            "selected_blocks_old": selected,
            "selected_union_gt": sel_total,
            "selected_union_new_gt": sel_new,
            "selected_union_cost_budget": budget_used,
        }

    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    required = [
        S1_PATH,
        S2_PATH,
        S3_PATH,
        GT_PATH,
        OLD_RANKED["S2"],
        OLD_RANKED["S3"],
        FRESH_S1,
        FRESH_RANKED["S2"],
        FRESH_RANKED["S3"],
    ]
    for p in required:
        require(p)

    header("AMAZON ML CHALLENGE — CANDIDATE-BLOCK RECOVERY AUDIT V1")

    print(f"Memory             : {MEMORY}")
    print(f"Threads            : {THREADS}")
    print(f"Max bucket size    : {MAX_BUCKET_SIZE}")
    print(f"Old pair budget    : {OLD_PAIR_BUDGET:,}")
    print(f"Max selected blocks: {MAX_BLOCKS_PER_TARGET}")
    print("Old validation     : BLOCK SELECTION")
    print("Fresh development  : BLIND CONFIRMATION")
    print("Final Holdout GT   : NOT READ")
    print("Test data          : NOT READ")
    print("Training           : NOT RUN")
    print("Candidate material : NOT RUN")
    print("Final outputs      : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")

    started = time.time()

    try:
        header("1. BUILD SOURCE TABLES")

        build_source_table(con, "s1_source", S1_PATH)
        build_source_table(con, "s2_source", S2_PATH)
        build_source_table(con, "s3_source", S3_PATH)

        # OLD selection.
        old = run_mode(con, "OLD", None)

        selected_old = {
            target: old["targets"][target]["selected_blocks_old"]
            for target in ("S2", "S3")
        }

        # FRESH confirmation. We need to reuse the exact selected blocks, not
        # select anything based on fresh GT. We therefore set a transient
        # marker used only by the printing/reporting stage.
        fresh = run_mode(con, "FRESH", None)

        # Replace fresh selected-block fields with the OLD selections and
        # compute their union coverage on fresh.
        for target in ("S2", "S3"):
            fresh["targets"][target]["selected_blocks_old"] = selected_old[target]

            # Recompute fresh union coverage for the old-selected blocks.
            # baseline_cov/gt_eval are the current target-specific tables.
            sel_total, sel_new = selected_union_coverage(
                con,
                selected_old[target],
                "s2_source" if target == "S2" else "s3_source",
                target,
            )

            fresh["targets"][target]["selected_union_gt"] = sel_total
            fresh["targets"][target]["selected_union_new_gt"] = sel_new

        report = {
            "version": "CANDIDATE_BLOCK_RECOVERY_AUDIT_V1",
            "parameters": {
                "max_bucket_size": MAX_BUCKET_SIZE,
                "old_pair_budget": OLD_PAIR_BUDGET,
                "max_blocks_per_target": MAX_BLOCKS_PER_TARGET,
                "min_new_gt": MIN_NEW_GT,
            },
            "old_validation": old,
            "fresh_confirmation": fresh,
            "safety": {
                "final_holdout_gt_read": False,
                "test_data_read": False,
                "model_training_run": False,
                "candidate_materialization_run": False,
                "final_outputs_modified": False,
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("2. FINAL BLOCK RECOMMENDATION SUMMARY")

        for target in ("S2", "S3"):
            o = old["targets"][target]
            f = fresh["targets"][target]

            print(f"\nTARGET {target}")

            print(
                f"OLD baseline missing GT : "
                f"{o['baseline_missing_gt']:,}"
            )
            print(
                f"OLD selected blocks      : "
                f"{o['selected_blocks_old']}"
            )
            print(
                f"OLD selected new GT     : "
                f"{o['selected_union_new_gt']:,}"
            )

            print(
                f"FRESH baseline missing GT: "
                f"{f['baseline_missing_gt']:,}"
            )
            print(
                f"FRESH selected new GT    : "
                f"{f['selected_union_new_gt']:,}"
            )

            print("\nTOP 10 BLOCKS BY NEW-GT RECOVERY")
            for i, r in enumerate(
                o["blocks_ranked"][:10],
                1,
            ):
                print(
                    f"{i:2d}. {r['block']:32s} | "
                    f"newGT={r['new_gt_vs_baseline']:,} | "
                    f"pairs={r['estimated_pairs']:,} | "
                    f"newGT/M={r['new_gt_per_million_pairs']:.2f} | "
                    f"{r['known']}"
                )

        print(f"\nReport: {REPORT}")
        print("SAFE: no final candidate/output files were changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
