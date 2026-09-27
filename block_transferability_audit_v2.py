#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
BLOCK TRANSFERABILITY AUDIT V2

Why this exists
---------------
The previous block audit selected OLD-validation blocks greedily, but its
greedy rule scored every block against the baseline only; it did not update
marginal gain after each selected block. More importantly, the selected S2
union recovered 0 new GT pairs on FRESH DEV, so we must NOT promote those
blocks.

This audit therefore does the right next experiment:

1) Evaluate a curated set of the strongest OLD-validation blocks individually
   on OLD and FRESH.
2) Measure:
      - candidate cost
      - GT covered
      - NEW GT beyond V6+V9.1 baseline
      - NEW GT / million candidate pairs
3) For each target, evaluate unions of the strongest transferable blocks on
   FRESH, without materializing candidate pairs.
4) Produce a transferability ranking.

No:
  - Final Holdout V2 GT
  - test data
  - model training
  - candidate materialization
  - final output changes

This is diagnostic only.
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
    PROJECT / "validation_leakage_audit"
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
    / "block_transferability_audit_v2"
)
REPORT = OUT / "block_transferability_audit_v2_report.json"

MEMORY = os.environ.get("BLOCK_TRANSFER_MEMORY", "8GB")
THREADS = int(os.environ.get("BLOCK_TRANSFER_THREADS", "2"))

OLD_VALIDATION_MOD = 5
MAX_BUCKET_SIZE = int(
    os.environ.get("BLOCK_TRANSFER_MAX_BUCKET_SIZE", "100")
)

# These are the highest-value / informative blocks observed in V1, plus
# known V8/V9.1-style blocks for comparison.
BLOCKS = {
    "postal_addr_f3_tail2_name_f1": (
        "postal || '|' || substr(address_compact,1,3) "
        "|| '|' || right(address_compact,2) "
        "|| '|' || substr(name_compact,1,1)"
    ),
    "name_f2_addr_tail3": (
        "substr(name_compact,1,2) || '|' || right(address_compact,3)"
    ),
    "addr_f3_tail3_name_f1": (
        "substr(address_compact,1,3) || '|' "
        "|| right(address_compact,3) || '|' "
        "|| substr(name_compact,1,1)"
    ),
    "name_f3_addr_f2": (
        "substr(name_compact,1,3) || '|' "
        "|| substr(address_compact,1,2)"
    ),
    "name_f1_addr_f4_tail2": (
        "substr(name_compact,1,1) || '|' "
        "|| substr(address_compact,1,4) || '|' "
        "|| right(address_compact,2)"
    ),
    "addr_f4_tail4_name_f1": (
        "substr(address_compact,1,4) || '|' "
        "|| right(address_compact,4) || '|' "
        "|| substr(name_compact,1,1)"
    ),
    "name_f3_addr_tail3_len5": (
        "substr(name_compact,1,3) || '|' "
        "|| right(address_compact,3) || '|' "
        "|| CAST(floor(length(address_compact)/5) AS BIGINT)"
    ),
    "postal_addr_f2_name_f2": (
        "postal || '|' || substr(address_compact,1,2) "
        "|| '|' || substr(name_compact,1,2)"
    ),
    "name_f4_l2_len5": (
        "substr(name_compact,1,4) || '|' || right(name_compact,2) "
        "|| '|' || CAST(floor(length(name_compact)/5) AS BIGINT)"
    ),
    "name_f3_l2_len5": (
        "substr(name_compact,1,3) || '|' || right(name_compact,2) "
        "|| '|' || CAST(floor(length(name_compact)/5) AS BIGINT)"
    ),
    "name_f3_l2_postal": (
        "postal || '|' || substr(name_compact,1,3) "
        "|| '|' || right(name_compact,2)"
    ),
    "postal_name_f3_l1": (
        "postal || '|' || substr(name_compact,1,3) "
        "|| '|' || right(name_compact,1)"
    ),
    "name_f2_l3_len5": (
        "substr(name_compact,1,2) || '|' || right(name_compact,3) "
        "|| '|' || CAST(floor(length(name_compact)/5) AS BIGINT)"
    ),
    "num_addr_name_prefix4": (
        "addr_num || '|' || substr(name_compact,1,4)"
    ),
    "num_addr_name_prefix2": (
        "addr_num || '|' || substr(name_compact,1,2)"
    ),
}


def header(s: str) -> None:
    print("\n" + "=" * 118)
    print(s)
    print("=" * 118, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def build_sources(con):
    def one(table, path):
        con.execute(f"""
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
        """)

    one("s1_source", S1_PATH)
    one("s2_source", S2_PATH)
    one("s3_source", S3_PATH)


def build_eval_ids(con, mode):
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
    else:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE eval_ids AS
            SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS s1
            FROM read_parquet({qp(FRESH_S1)})
        """)

    return int(
        con.execute("SELECT COUNT(*) FROM eval_ids").fetchone()[0]
    )


def build_gt(con, target):
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_eval AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT_PATH)}) g
        INNER JOIN eval_ids e
          ON CAST(g.source1_entity_id AS VARCHAR)=e.s1
        WHERE COALESCE(g.label,1)=1
          AND CAST(g.matched_entity_id AS VARCHAR)
              LIKE '{target}-%'
    """)

    return int(
        con.execute("SELECT COUNT(*) FROM gt_eval").fetchone()[0]
    )


def baseline_coverage(con, ranked_path):
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
    return int(con.execute(
        "SELECT COUNT(*) FROM baseline_cov"
    ).fetchone()[0])


def key_view(con, source_table, view_name, expr, eval_s1=False):
    source = f"""
        SELECT
            entity_id,
            country_norm || '|' || ({expr}) AS block_key
        FROM {source_table}
    """

    con.execute(
        f"CREATE OR REPLACE TEMP VIEW {view_name} AS {source}"
    )


def block_metrics(con, target, target_table, block_name, expr):
    lv = f"bl_{block_name}"
    rv = f"br_{block_name}"

    key_view(con, "s1_eval_source", lv, expr)
    key_view(con, target_table, rv, expr)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE lcnt AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {lv}
        WHERE block_key <> ''
          AND NOT ends_with(block_key,'|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE rcnt AS
        SELECT block_key, COUNT(*)::BIGINT AS n
        FROM {rv}
        WHERE block_key <> ''
          AND NOT ends_with(block_key,'|')
        GROUP BY block_key
        HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
    """)

    cost = int(con.execute("""
        SELECT COALESCE(SUM(l.n*r.n),0)
        FROM lcnt l
        JOIN rcnt r USING(block_key)
    """).fetchone()[0])

    block_gt = int(con.execute(f"""
        SELECT COUNT(*)
        FROM gt_eval g
        JOIN {lv} l ON l.entity_id=g.s1
        JOIN {rv} r ON r.entity_id=g.matched
        JOIN lcnt lc ON lc.block_key=l.block_key
        JOIN rcnt rc ON rc.block_key=r.block_key
        WHERE l.block_key=r.block_key
    """).fetchone()[0])

    new_gt = int(con.execute(f"""
        SELECT COUNT(*)
        FROM gt_eval g
        JOIN {lv} l ON l.entity_id=g.s1
        JOIN {rv} r ON r.entity_id=g.matched
        JOIN lcnt lc ON lc.block_key=l.block_key
        JOIN rcnt rc ON rc.block_key=r.block_key
        LEFT JOIN baseline_cov b
          ON b.s1=g.s1 AND b.matched=g.matched
        WHERE l.block_key=r.block_key
          AND b.s1 IS NULL
    """).fetchone()[0])

    return {
        "block": block_name,
        "estimated_pairs": cost,
        "gt_covered_by_block": block_gt,
        "new_gt_vs_baseline": new_gt,
        "new_gt_per_million_pairs": (
            1_000_000.0 * new_gt / cost if cost else 0.0
        ),
    }


def union_metrics(con, target, target_table, block_names):
    recovered_queries = []

    for b in block_names:
        expr = BLOCKS[b]

        lv = f"ul_{b}"
        rv = f"ur_{b}"

        key_view(con, "s1_eval_source", lv, expr)
        key_view(con, target_table, rv, expr)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE ulcnt_{b} AS
            SELECT block_key, COUNT(*)::BIGINT AS n
            FROM {lv}
            WHERE block_key <> ''
              AND NOT ends_with(block_key,'|')
            GROUP BY block_key
            HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE urcnt_{b} AS
            SELECT block_key, COUNT(*)::BIGINT AS n
            FROM {rv}
            WHERE block_key <> ''
              AND NOT ends_with(block_key,'|')
            GROUP BY block_key
            HAVING COUNT(*) <= {MAX_BUCKET_SIZE}
        """)

        recovered_queries.append(f"""
            SELECT DISTINCT g.s1, g.matched
            FROM gt_eval g
            JOIN {lv} l ON l.entity_id=g.s1
            JOIN {rv} r ON r.entity_id=g.matched
            JOIN ulcnt_{b} lc ON lc.block_key=l.block_key
            JOIN urcnt_{b} rc ON rc.block_key=r.block_key
            WHERE l.block_key=r.block_key
        """)

    if not recovered_queries:
        return {
            "union_gt": 0,
            "union_new_gt": 0,
        }

    con.execute(
        "CREATE OR REPLACE TEMP TABLE union_cov AS "
        + "\nUNION\n".join(recovered_queries)
    )

    total = int(con.execute(
        "SELECT COUNT(*) FROM union_cov"
    ).fetchone()[0])

    new = int(con.execute("""
        SELECT COUNT(*)
        FROM union_cov u
        LEFT JOIN baseline_cov b
          ON b.s1=u.s1 AND b.matched=u.matched
        WHERE b.s1 IS NULL
    """).fetchone()[0])

    return {
        "union_gt": total,
        "union_new_gt": new,
    }


def run_mode(con, mode):
    header(f"{mode} BLOCK TRANSFERABILITY")

    n = build_eval_ids(con, mode)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s1_eval_source AS
        SELECT s.*
        FROM s1_source s
        INNER JOIN eval_ids e
          ON s.entity_id=e.s1
    """)

    report = {
        "s1_entities": n,
        "targets": {},
    }

    selected_candidates = {}

    for target, target_table, ranked_path in [
        ("S2", "s2_source", OLD_RANKED["S2"] if mode == "OLD" else FRESH_RANKED["S2"]),
        ("S3", "s3_source", OLD_RANKED["S3"] if mode == "OLD" else FRESH_RANKED["S3"]),
    ]:
        gt_total = build_gt(con, target)
        baseline = baseline_coverage(con, ranked_path)

        print(
            f"\n{target}: GT={gt_total:,} | "
            f"baseline covered={baseline:,} | "
            f"baseline missing={gt_total-baseline:,}"
        )

        rows = []

        for block_name, expr in BLOCKS.items():
            r = block_metrics(
                con,
                target,
                target_table,
                block_name,
                expr,
            )
            rows.append(r)

        rows.sort(
            key=lambda x: (
                x["new_gt_vs_baseline"],
                x["new_gt_per_million_pairs"],
            ),
            reverse=True,
        )

        report["targets"][target] = {
            "gt_total": gt_total,
            "baseline_covered": baseline,
            "baseline_missing": gt_total - baseline,
            "blocks": rows,
        }

        # On OLD, retain top blocks by NEW-GT recovery and efficiency.
        # Fresh never selects; it only confirms the OLD shortlist.
        selected_candidates[target] = [
            r["block"]
            for r in rows[:8]
            if r["new_gt_vs_baseline"] > 0
        ]

    return report, selected_candidates


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for p in [
        S1_PATH, S2_PATH, S3_PATH, GT_PATH,
        OLD_RANKED["S2"], OLD_RANKED["S3"],
        FRESH_S1, FRESH_RANKED["S2"], FRESH_RANKED["S3"],
    ]:
        require(p)

    header("AMAZON ML CHALLENGE — BLOCK TRANSFERABILITY AUDIT V2")

    print(f"Memory          : {MEMORY}")
    print(f"Threads         : {THREADS}")
    print(f"Max bucket size : {MAX_BUCKET_SIZE}")
    print(f"Blocks tested   : {len(BLOCKS)}")
    print("OLD             : selection/diagnostic")
    print("FRESH           : blind confirmation")
    print("Final Holdout   : NOT READ")
    print("Test            : NOT READ")
    print("Training        : NOT RUN")
    print("Candidate files : NOT MATERIALIZED")
    print("Final outputs   : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")

    start = time.time()

    try:
        build_sources(con)

        old_report, old_shortlists = run_mode(con, "OLD")

        # Fresh: compute EVERY individual block again, but never use its
        # results to change the shortlist. This is the blind transfer test.
        fresh_report, _ = run_mode(con, "FRESH")

        # Evaluate unions formed from OLD shortlist only on both splits.
        union_report = {}

        for mode, eval_report, id_label in [
            ("OLD", old_report, "OLD"),
            ("FRESH", fresh_report, "FRESH"),
        ]:
            # Rebuild eval universe for each mode so gt_eval/baseline_cov
            # correspond to that split before union evaluation.
            build_eval_ids(con, mode)
            con.execute("""
                CREATE OR REPLACE TEMP TABLE s1_eval_source AS
                SELECT s.*
                FROM s1_source s
                INNER JOIN eval_ids e ON s.entity_id=e.s1
            """)

            union_report[mode] = {}

            for target, target_table, ranked_path in [
                ("S2", "s2_source", OLD_RANKED["S2"] if mode == "OLD" else FRESH_RANKED["S2"]),
                ("S3", "s3_source", OLD_RANKED["S3"] if mode == "OLD" else FRESH_RANKED["S3"]),
            ]:
                build_gt(con, target)
                baseline_coverage(con, ranked_path)

                # Test prefix-unions of the OLD shortlist:
                # first 1, 2, 3 ... up to 5 blocks.
                shortlist = old_shortlists[target]
                prefixes = {}
                for k in range(1, min(5, len(shortlist)) + 1):
                    chosen = shortlist[:k]
                    prefixes[str(k)] = {
                        "blocks": chosen,
                        **union_metrics(
                            con,
                            target,
                            target_table,
                            chosen,
                        ),
                    }

                union_report[mode][target] = prefixes

        # Make a concise transferability comparison.
        transfer_summary = {}

        for target in ("S2", "S3"):
            old_blocks = {
                r["block"]: r
                for r in old_report["targets"][target]["blocks"]
            }
            fresh_blocks = {
                r["block"]: r
                for r in fresh_report["targets"][target]["blocks"]
            }

            rows = []

            for block in BLOCKS:
                o = old_blocks[block]
                f = fresh_blocks[block]

                rows.append({
                    "block": block,
                    "old_new_gt": o["new_gt_vs_baseline"],
                    "old_pairs": o["estimated_pairs"],
                    "old_efficiency": o["new_gt_per_million_pairs"],
                    "fresh_new_gt": f["new_gt_vs_baseline"],
                    "fresh_pairs": f["estimated_pairs"],
                    "fresh_efficiency": f["new_gt_per_million_pairs"],
                    "transfer_ratio": (
                        f["new_gt_vs_baseline"]
                        / o["new_gt_vs_baseline"]
                        if o["new_gt_vs_baseline"] > 0 else None
                    ),
                })

            rows.sort(
                key=lambda x: (
                    x["fresh_new_gt"],
                    x["fresh_efficiency"],
                ),
                reverse=True,
            )

            transfer_summary[target] = rows

        report = {
            "version": "BLOCK_TRANSFERABILITY_AUDIT_V2",
            "parameters": {
                "max_bucket_size": MAX_BUCKET_SIZE,
                "block_count": len(BLOCKS),
            },
            "old_validation": old_report,
            "fresh_confirmation": fresh_report,
            "old_shortlists": old_shortlists,
            "union_prefix_tests": union_report,
            "transfer_summary": transfer_summary,
            "safety": {
                "final_holdout_gt_read": False,
                "test_data_read": False,
                "model_training_run": False,
                "candidate_materialization_run": False,
                "final_outputs_modified": False,
            },
            "elapsed_seconds": round(time.time() - start, 3),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("TRANSFERABILITY SUMMARY")

        for target in ("S2", "S3"):
            print(f"\nTARGET {target}")
            print("Top blocks by FRESH new-GT recovery:")

            for i, r in enumerate(
                transfer_summary[target][:10],
                1,
            ):
                print(
                    f"{i:2d}. {r['block']:32s} | "
                    f"OLD newGT={r['old_new_gt']:,} | "
                    f"FRESH newGT={r['fresh_new_gt']:,} | "
                    f"FRESH eff={r['fresh_efficiency']:.2f}/M | "
                    f"transfer={r['transfer_ratio']}"
                )

            print("\nOLD-shortlist union prefixes:")
            for k, r in union_report["OLD"][target].items():
                print(
                    f"  k={k}: newGT={r['union_new_gt']:,} "
                    f"blocks={r['blocks']}"
                )

            print("\nFRESH-shortlist union prefixes:")
            for k, r in union_report["FRESH"][target].items():
                print(
                    f"  k={k}: newGT={r['union_new_gt']:,} "
                    f"blocks={r['blocks']}"
                )

        print(f"\nReport: {REPORT}")
        print("SAFE: no final candidate/output files were changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
