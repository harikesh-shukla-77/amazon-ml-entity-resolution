#!/usr/bin/env python3
"""
AMAZON ML CHALLENGE 2026
FRESH DEVELOPMENT JOINT-THRESHOLD CONFIRMATION V1

Purpose
-------
The old V7 validation found a strong joint policy:
    S2 = 34.710811614990234
    S3 = 22.125261306762695

Robustness around that point was excellent (72/81 nearby states within
0.002 Macro F0.5). The remaining question is whether the gain generalizes
to a completely fresh development split.

This script creates a NEW development split from TRAIN Source-1 IDs and
then:
  1. excludes that split from model training;
  2. trains fresh S2 and S3 pairwise LogisticRegression models on the
     existing frozen V6+V9.1 labeled candidate pools;
  3. scores only the fresh development S1 entities;
  4. evaluates the CURRENT threshold policy and the JOINT threshold
     policy on this fresh split;
  5. performs an exact joint threshold sweep over observed top-1 utility
     values on this fresh split.

Safety
------
- Does NOT read Final Holdout V2 ground truth.
- Does NOT use Final Holdout V2 labels.
- Does NOT generate candidates.
- Does NOT modify final outputs or the final ZIP.
- Uses Final Holdout V2 S1 IDs only as an exclusion guard, never its GT.
- Uses an S1 split different from old V7 validation and excludes the
  previously consumed prospective holdout.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/Users/harikeshshukla/mla")
TRAIN = PROJECT / "processed_dataset" / "train"
GT = TRAIN / "ground_truth_pairs.parquet"
S1 = TRAIN / "train_source1.parquet"

BASELINE_LABELED = {
    "S2": PROJECT / "candidate_output_v10" / "train_labeled_base_s1_s2.parquet",
    "S3": PROJECT / "candidate_output_v10" / "train_labeled_base_s1_s3.parquet",
}

AUDIT = PROJECT / "validation_leakage_audit"
PREVIOUS_HOLDOUT_S1 = AUDIT / "holdout" / "prospective_holdout_s1.parquet"
FINAL_HOLDOUT_S1 = AUDIT / "final_holdout_v2" / "final_holdout_s1.parquet"

OUT = AUDIT / "f05_fresh_dev_joint_confirmation_v1"
TMP = OUT / "tmp"
PARTS = OUT / "score_parts"

REPORT = OUT / "f05_fresh_dev_joint_confirmation_v1_report.json"

MEMORY = os.environ.get("FRESH_DEV_MEMORY", "8GB")
THREADS = int(os.environ.get("FRESH_DEV_THREADS", "2"))
BATCH = int(os.environ.get("FRESH_DEV_BATCH", "100000"))

# Split guards.
OLD_VALIDATION_MOD = 5
FRESH_MOD = 11
FRESH_SALT = "FRESH_DEV_JOINT_CONFIRMATION_V1"

# Frozen policy from the original final pipeline.
CURRENT_S2 = 20.945884704589844
CURRENT_S3 = 20.945884704589844

# Candidate discovered on OLD V7 validation.
CANDIDATE_S2 = 34.710811614990234
CANDIDATE_S3 = 22.125261306762695

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

# Same pairwise recipe used by the V10 final training:
POS_PER_S1 = 1
NEG_RANK = 100
NEG_PER_S1 = 2


def header(s: str) -> None:
    print("\n" + "=" * 112)
    print(s)
    print("=" * 112, flush=True)


def qp(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")


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
        self.lazy = [0.0] * (2 * size)

        for i, v in enumerate(values):
            self.mx[size + i] = float(v)
        for i in range(size - 1, 0, -1):
            self.mx[i] = max(self.mx[2 * i], self.mx[2 * i + 1])

    def _apply(self, p: int, delta: float) -> None:
        self.mx[p] += delta
        self.lazy[p] += delta

    def _push(self, p: int) -> None:
        z = self.lazy[p]
        if z:
            self._apply(2 * p, z)
            self._apply(2 * p + 1, z)
            self.lazy[p] = 0.0

    def _add(self, p: int, lo: int, hi: int, ql: int, qh: int, d: float) -> None:
        if ql > hi or qh < lo:
            return
        if ql <= lo and hi <= qh:
            self._apply(p, d)
            return
        self._push(p)
        mid = (lo + hi) // 2
        self._add(2 * p, lo, mid, ql, qh, d)
        self._add(2 * p + 1, mid + 1, hi, ql, qh, d)
        self.mx[p] = max(self.mx[2 * p], self.mx[2 * p + 1])

    def add(self, ql: int, qh: int, d: float) -> None:
        if ql > qh or self.n == 0:
            return
        ql = max(0, ql)
        qh = min(self.n - 1, qh)
        if ql <= qh:
            self._add(1, 0, self.size - 1, ql, qh, d)

    def max_value(self) -> float:
        return float(self.mx[1])

    def _argmax(self, p: int, lo: int, hi: int, target: float) -> int:
        if lo == hi:
            return lo
        self._push(p)
        mid = (lo + hi) // 2
        if abs(self.mx[2 * p] - target) <= 1e-12:
            return self._argmax(2 * p, lo, mid, target)
        return self._argmax(2 * p + 1, mid + 1, hi, target)

    def argmax(self) -> int:
        return self._argmax(1, 0, self.size - 1, self.mx[1])


def build_fresh_split(con):
    header("1. BUILD FRESH DEVELOPMENT SPLIT")

    for p in [S1, GT, PREVIOUS_HOLDOUT_S1, FINAL_HOLDOUT_S1]:
        require(p)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE source1_base AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(S1)})
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE old_validation AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(S1)})
        WHERE MOD(
            ABS(HASH(CAST(entity_id AS VARCHAR))),
            {OLD_VALIDATION_MOD}
        ) = 0
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE previous_holdout AS
        SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(PREVIOUS_HOLDOUT_S1)})
    """)

    # FINAL_HOLDOUT_S1 is used only as a boundary exclusion guard.
    # FINAL_HOLDOUT_GT is never read anywhere in this script.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE final_holdout_boundary AS
        SELECT DISTINCT CAST(source1_entity_id AS VARCHAR) AS s1
        FROM read_parquet({qp(FINAL_HOLDOUT_S1)})
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fresh_dev_ids AS
        SELECT b.s1
        FROM source1_base b
        WHERE NOT EXISTS (
            SELECT 1 FROM old_validation v WHERE v.s1 = b.s1
        )
        AND NOT EXISTS (
            SELECT 1 FROM previous_holdout p WHERE p.s1 = b.s1
        )
        AND NOT EXISTS (
            SELECT 1 FROM final_holdout_boundary f WHERE f.s1 = b.s1
        )
        AND MOD(
            ABS(HASH(b.s1 || '|{FRESH_SALT}|')),
            {FRESH_MOD}
        ) = 0
    """)

    n = int(con.execute("SELECT COUNT(*) FROM fresh_dev_ids").fetchone()[0])

    if n < 1000:
        raise RuntimeError(f"Fresh development split unexpectedly small: {n:,}")

    old_overlap = int(con.execute("""
        SELECT COUNT(*)
        FROM fresh_dev_ids f
        JOIN old_validation v ON v.s1 = f.s1
    """).fetchone()[0])

    prev_overlap = int(con.execute("""
        SELECT COUNT(*)
        FROM fresh_dev_ids f
        JOIN previous_holdout p ON p.s1 = f.s1
    """).fetchone()[0])

    final_overlap = int(con.execute("""
        SELECT COUNT(*)
        FROM fresh_dev_ids f
        JOIN final_holdout_boundary h ON h.s1 = f.s1
    """).fetchone()[0])

    print(f"Fresh development S1 : {n:,}")
    print(f"Old validation overlap: {old_overlap:,}")
    print(f"Previous holdout overlap: {prev_overlap:,}")
    print(f"Final holdout S1 overlap: {final_overlap:,}")

    if old_overlap or prev_overlap or final_overlap:
        raise RuntimeError("Fresh split boundary overlap detected.")

    con.execute(f"""
        COPY (
            SELECT s1 AS source1_entity_id
            FROM fresh_dev_ids
            ORDER BY s1
        )
        TO {qp(OUT / "fresh_dev_s1.parquet")}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    return n


def make_training_sql(path: Path) -> str:
    fresh = qp(OUT / "fresh_dev_s1.parquet")
    p = qp(path)

    return f"""
    WITH eligible AS (
        SELECT *
        FROM read_parquet({p}) t
        WHERE NOT EXISTS (
            SELECT 1
            FROM read_parquet({fresh}) f
            WHERE CAST(f.source1_entity_id AS VARCHAR)
                = CAST(t.source1_entity_id AS VARCHAR)
        )
    ),
    positives AS (
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY source1_entity_id
                       ORDER BY base_rank, candidate_entity_id
                   ) AS rn
            FROM eligible
            WHERE label = 1
        )
        WHERE rn <= {POS_PER_S1}
    ),
    negatives AS (
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY source1_entity_id
                       ORDER BY base_rank, candidate_entity_id
                   ) AS rn
            FROM eligible
            WHERE label = 0
              AND base_rank <= {NEG_RANK}
        )
        WHERE rn <= {NEG_PER_S1}
    ),
    joined AS (
        SELECT
            p.name_similarity pn, n.name_similarity nn,
            p.address_similarity pa, n.address_similarity na,
            p.name_exact pe, n.name_exact ne,
            p.address_exact pae, n.address_exact nae,
            p.country_exact pc, n.country_exact nc,
            p.name_length_ratio pl, n.name_length_ratio nl,
            p.address_length_ratio pal, n.address_length_ratio nal,
            p.evidence_rows per, n.evidence_rows ner,
            p.evidence_file_count pef, n.evidence_file_count nef,
            p.exact_key_count pek, n.exact_key_count nek,
            p.base_score ps, n.base_score ns,
            p.base_rank pr, n.base_rank nr
        FROM positives p
        JOIN negatives n USING (source1_entity_id)
    ),
    f AS (
        SELECT
            pn-nn AS name_similarity,
            pa-na AS address_similarity,
            pe-ne AS name_exact,
            pae-nae AS address_exact,
            pc-nc AS country_exact,
            pl-nl AS name_length_ratio,
            pal-nal AS address_length_ratio,
            per-ner AS evidence_rows,
            pef-nef AS evidence_file_count,
            pek-nek AS exact_key_count,
            ps-ns AS base_score,
            LN(1+pr)-LN(1+nr) AS log_base_rank,
            pn*pa-nn*na AS name_x_address,
            (pn-pa)-(nn-na) AS name_minus_address,
            ((pn+pa)/2)-((nn+na)/2) AS similarity_mean,
            LEAST(pn,pa)-LEAST(nn,na) AS similarity_min,
            (pe+pae+pc)-(ne+nae+nc) AS exact_field_count,
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


def fit_one(con, target: str):
    path = BASELINE_LABELED[target]

    header(f"2. TRAIN FRESH {target} MODEL")

    require(path)

    reader = con.execute(make_training_sql(path)).to_arrow_reader(
        batch_size=BATCH
    )

    frames = []
    total = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue
        frames.append(d)
        total += len(d)
        if total and total % 1_000_000 < len(d):
            print(f"{target} training pairs loaded: {total:,}", flush=True)

    if not frames:
        raise RuntimeError(f"No training data for {target}.")

    d = pd.concat(frames, ignore_index=True)

    X = (
        d[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
        .to_numpy()
    )
    y = d["pair_label"].astype("int8").to_numpy()

    print(
        f"{target}: pair rows={len(d):,} | "
        f"positive={int((y==1).sum()):,} | "
        f"negative={int((y==0).sum()):,}",
        flush=True,
    )

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

    print(f"{target} model fit: {time.time() - t0:.1f}s", flush=True)

    del d, X, Xs, y, frames
    return scaler, model


def score_fresh(con, target: str, scaler, model) -> Path:
    header(f"3. SCORE FRESH DEVELOPMENT SPLIT — {target}")

    path = BASELINE_LABELED[target]
    target_parts = PARTS / target.lower()

    if target_parts.exists():
        shutil.rmtree(target_parts)
    target_parts.mkdir(parents=True, exist_ok=True)

    q = f"""
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
        FROM read_parquet({qp(path)}) l
        INNER JOIN read_parquet({qp(OUT / "fresh_dev_s1.parquet")}) h
          ON CAST(l.source1_entity_id AS VARCHAR)
           = CAST(h.source1_entity_id AS VARCHAR)
    """

    reader = con.execute(q).to_arrow_reader(batch_size=BATCH)
    paths = []
    total = 0
    idx = 0

    for batch in reader:
        d = batch.to_pandas()
        if d.empty:
            continue

        x = pd.DataFrame(index=d.index)

        numeric = [
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

        for c in numeric:
            x[c] = (
                pd.to_numeric(d[c], errors="coerce")
                .fillna(0)
                .astype("float32")
            )

        rank = pd.to_numeric(
            d["base_rank"],
            errors="coerce",
        ).fillna(1_000_000).astype("float64")

        x["log_base_rank"] = np.log1p(rank).astype("float32")
        x["name_x_address"] = (
            x["name_similarity"] * x["address_similarity"]
        ).astype("float32")
        x["name_minus_address"] = (
            x["name_similarity"] - x["address_similarity"]
        ).astype("float32")
        x["similarity_mean"] = (
            (x["name_similarity"] + x["address_similarity"]) / 2
        ).astype("float32")
        x["similarity_min"] = np.minimum(
            x["name_similarity"],
            x["address_similarity"],
        ).astype("float32")
        x["exact_field_count"] = (
            x["name_exact"] + x["address_exact"] + x["country_exact"]
        ).astype("float32")

        X = x[FEATURES].to_numpy(dtype="float32")
        Xs = scaler.transform(X).astype("float32")

        d["utility"] = model.decision_function(Xs)

        p = target_parts / f"part_{idx:05d}.parquet"
        d[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "utility",
            ]
        ].to_parquet(
            p,
            index=False,
            compression="zstd",
        )

        paths.append(p)
        total += len(d)
        idx += 1

        if total and total % 1_000_000 < len(d):
            print(f"{target} fresh candidates scored: {total:,}", flush=True)

    if total == 0:
        raise RuntimeError(f"No fresh candidates for {target}.")

    ranked = OUT / f"fresh_ranked_{target.lower()}.parquet"
    ranked.unlink(missing_ok=True)

    glob = qp(target_parts / "*.parquet")

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id,
                utility,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY utility DESC, candidate_entity_id
                ) AS rank
            FROM read_parquet({glob})
        )
        TO {qp(ranked)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    shutil.rmtree(target_parts, ignore_errors=True)

    print(f"{target} fresh scored rows: {total:,}", flush=True)
    return ranked


def load_truth(con):
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_fresh AS
        SELECT
            CAST(g.source1_entity_id AS VARCHAR) AS s1,
            CAST(g.matched_entity_id AS VARCHAR) AS matched
        FROM read_parquet({qp(GT)}) g
        INNER JOIN read_parquet({qp(OUT / "fresh_dev_s1.parquet")}) f
          ON CAST(g.source1_entity_id AS VARCHAR)
           = CAST(f.source1_entity_id AS VARCHAR)
        WHERE COALESCE(g.label, 1) = 1
    """)

    return {
        str(s1): int(cnt)
        for s1, cnt in con.execute("""
            SELECT f.source1_entity_id, COUNT(g.matched_entity_id)
            FROM read_parquet(
                '""" + str(OUT / "fresh_dev_s1.parquet").replace("'", "''") + """'
            ) f
            LEFT JOIN gt_fresh g
              ON CAST(f.source1_entity_id AS VARCHAR) = g.s1
            GROUP BY f.source1_entity_id
        """).fetchall()
    }


def load_top1(con, ranked: Path, target: str):
    prefix = f"{target}-"

    rows = con.execute(f"""
        SELECT
            CAST(r.source1_entity_id AS VARCHAR) AS s1,
            CAST(r.utility AS DOUBLE) AS utility,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM gt_fresh g
                    WHERE g.s1 = CAST(r.source1_entity_id AS VARCHAR)
                      AND g.matched = CAST(r.candidate_entity_id AS VARCHAR)
                )
                THEN 1 ELSE 0
            END AS tp
        FROM read_parquet({qp(ranked)}) r
        WHERE r.rank = 1
          AND CAST(r.candidate_entity_id AS VARCHAR) LIKE '{prefix}%'
    """).fetchall()

    return {
        str(s1): (float(u), int(tp))
        for s1, u, tp in rows
    }


def evaluate_policy(ids, truth, s2, s3, t2, t3):
    score_sum = 0.0
    predicted = 0
    tp_total = 0
    correctly_empty = 0
    fp_singletons = 0
    truth_total = sum(truth.values())

    for s1 in ids:
        tc = truth[s1]
        p = 0
        tp = 0

        r2 = s2.get(s1)
        if r2 is not None and r2[0] >= t2:
            p += 1
            tp += r2[1]

        r3 = s3.get(s1)
        if r3 is not None and r3[0] >= t3:
            p += 1
            tp += r3[1]

        score_sum += score_f05(tc, p, tp)
        predicted += p
        tp_total += tp

        if tc == 0 and p == 0:
            correctly_empty += 1
        if tc == 0:
            fp_singletons += p

    return {
        "threshold_s2": t2,
        "threshold_s3": t3,
        "macro_f05": score_sum / len(ids),
        "precision": tp_total / predicted if predicted else 0.0,
        "recall": tp_total / truth_total if truth_total else 0.0,
        "predicted": predicted,
        "tp": tp_total,
        "correctly_empty": correctly_empty,
        "fp_singletons": fp_singletons,
    }


def exact_joint_sweep(ids, truth, s2, s3):
    s3_values = sorted({v[0] for v in s3.values()}, reverse=True)
    s3_pos = {u: i + 1 for i, u in enumerate(s3_values)}

    # Leaf 0 = no S3. Leaf k+1 = S3 threshold equal to s3_values[k].
    base = sum(score_f05(truth[s1], 0, 0) for s1 in ids)
    tree = SegmentTree([base] * (len(s3_values) + 1))

    # Initially S2 excluded. Turn S3 on at its observed threshold.
    for s1 in ids:
        r3 = s3.get(s1)
        if r3 is None:
            continue
        u3, tp3 = r3
        delta = (
            score_f05(truth[s1], 1, tp3)
            - score_f05(truth[s1], 0, 0)
        )
        tree.add(
            s3_pos[u3],
            len(s3_values),
            delta,
        )

    best = {
        "threshold_s2": math.inf,
        "threshold_s3": (
            math.inf
            if tree.argmax() == 0
            else s3_values[tree.argmax() - 1]
        ),
        "macro_f05": tree.max_value() / len(ids),
    }

    events = sorted(
        [(u, s1, tp) for s1, (u, tp) in s2.items()],
        key=lambda z: z[0],
        reverse=True,
    )

    i = 0
    while i < len(events):
        u2 = events[i][0]
        j = i + 1
        while j < len(events) and events[j][0] == u2:
            j += 1

        for _, s1, tp2 in events[i:j]:
            tc = truth[s1]

            d_no_s3 = (
                score_f05(tc, 1, tp2)
                - score_f05(tc, 0, 0)
            )

            r3 = s3.get(s1)

            if r3 is None:
                tree.add(0, len(s3_values), d_no_s3)
            else:
                u3, tp3 = r3
                pos = s3_pos[u3]

                d_with_s3 = (
                    score_f05(tc, 2, tp2 + tp3)
                    - score_f05(tc, 1, tp3)
                )

                tree.add(0, pos - 1, d_no_s3)
                tree.add(pos, len(s3_values), d_with_s3)

        cur_score = tree.max_value() / len(ids)
        leaf = tree.argmax()
        cur_t3 = math.inf if leaf == 0 else s3_values[leaf - 1]

        if cur_score > best["macro_f05"]:
            best = {
                "threshold_s2": u2,
                "threshold_s3": cur_t3,
                "macro_f05": cur_score,
            }

        i = j

    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)

    header("AMAZON ML CHALLENGE — FRESH DEV JOINT THRESHOLD CONFIRMATION")

    print(f"Memory          : {MEMORY}")
    print(f"Threads         : {THREADS}")
    print(f"Fresh modulus   : {FRESH_MOD}")
    print(f"Fresh salt      : {FRESH_SALT}")
    print("Final Holdout GT: NOT READ")
    print("Test data       : NOT READ")
    print("Final outputs   : NOT MODIFIED")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")

    started = time.time()

    try:
        fresh_n = build_fresh_split(con)

        s2_scaler, s2_model = fit_one(con, "S2")
        s3_scaler, s3_model = fit_one(con, "S3")

        ranked_s2 = score_fresh(con, "S2", s2_scaler, s2_model)
        ranked_s3 = score_fresh(con, "S3", s3_scaler, s3_model)

        truth = load_truth(con)
        ids = sorted(truth)

        s2 = load_top1(con, ranked_s2, "S2")
        s3 = load_top1(con, ranked_s3, "S3")

        header("4. FRESH DEV POLICY COMPARISON")

        current = evaluate_policy(
            ids, truth, s2, s3,
            CURRENT_S2, CURRENT_S3
        )

        old_candidate = evaluate_policy(
            ids, truth, s2, s3,
            CANDIDATE_S2, CANDIDATE_S3
        )

        joint = exact_joint_sweep(
            ids, truth, s2, s3
        )

        best = evaluate_policy(
            ids, truth, s2, s3,
            joint["threshold_s2"],
            joint["threshold_s3"],
        )

        print("\nCURRENT FROZEN POLICY")
        for k, v in current.items():
            print(f"{k:18s}: {v}")

        print("\nOLD VALIDATION CANDIDATE")
        for k, v in old_candidate.items():
            print(f"{k:18s}: {v}")

        print("\nFRESH-DEV EXACT JOINT BEST")
        for k, v in best.items():
            print(f"{k:18s}: {v}")

        print(
            f"\nFresh-dev candidate delta vs current: "
            f"{best['macro_f05'] - current['macro_f05']:+.12f}"
        )

        report = {
            "version": "F05_FRESH_DEV_JOINT_CONFIRMATION_V1",
            "fresh_dev_s1": fresh_n,
            "fresh_dev_gt_pairs": sum(truth.values()),
            "old_validation_mod": OLD_VALIDATION_MOD,
            "fresh_mod": FRESH_MOD,
            "fresh_salt": FRESH_SALT,
            "final_holdout_s1_used_only_for_exclusion": True,
            "final_holdout_gt_read": False,
            "test_data_read": False,
            "candidate_generation_run": False,
            "final_outputs_modified": False,
            "models_retrained_on_fresh_dev_exclusion": True,
            "current_frozen_policy": current,
            "old_validation_candidate_policy": old_candidate,
            "fresh_dev_exact_joint_best": best,
            "fresh_dev_delta_vs_current": best["macro_f05"] - current["macro_f05"],
            "elapsed_seconds": round(time.time() - started, 3),
        }

        REPORT.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        header("5. CONFIRMATION COMPLETE")
        print(f"Report: {REPORT}")
        print("SAFE: final candidate/output files were not changed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
