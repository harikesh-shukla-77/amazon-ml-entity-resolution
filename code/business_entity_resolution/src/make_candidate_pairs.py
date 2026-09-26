#!/usr/bin/env python3
from pathlib import Path
import os
import time
import json
import duckdb

ROOT = Path("/Users/harikeshshukla/mla")
OUT = ROOT / "candidate_output_final_v6"
S2_CAND = OUT / "test_candidates_s1_s2.parquet"
S3_CAND = OUT / "test_candidates_s1_s3.parquet"
S1_TEST = ROOT / "processed_dataset/test/test_source1.parquet"

WORK = OUT / "candidate_pairs_build"
BUCKETS = 256
MEMORY = os.environ.get("CANDIDATE_PAIRS_MEMORY", "2GB")
THREADS = int(os.environ.get("CANDIDATE_PAIRS_THREADS", "1"))

# Final locations.
OUTPUT_DIR = ROOT / "student_resource/output"
CANDIDATE_TSV = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_TSV = OUT / "matching_results.tsv"

def qp(p: Path) -> str:
    return "'" + p.as_posix().replace("'", "''") + "'"

def rows(con, p: Path) -> int:
    return int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({qp(p)})"
    ).fetchone()[0])

def main():
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in (S1_TEST, S2_CAND, S3_CAND, MATCHING_TSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print("=" * 100)
    print("AMAZON ML CHALLENGE — BUILD EXACT candidate_pairs.tsv")
    print("=" * 100)
    check_con = duckdb.connect()
    try:
        print(f"S1 test rows : {rows(check_con, S1_TEST):,}")
        print(f"S2 candidates: {rows(check_con, S2_CAND):,}")
        print(f"S3 candidates: {rows(check_con, S3_CAND):,}")
    finally:
        check_con.close()
    print(f"Buckets      : {BUCKETS}")
    print(f"Memory       : {MEMORY}")
    print(f"Threads      : {THREADS}")

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{MEMORY}'")
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory={qp(WORK / 'duckdb_tmp')}")
    (WORK / "duckdb_tmp").mkdir(parents=True, exist_ok=True)

    try:
        # Each bucket contains a disjoint subset of S1 entities. We union the
        # exact S2/S3 candidate pools that were actually fed to the V10 model.
        for b in range(BUCKETS):
            out = WORK / f"bucket_{b:03d}.parquet"
            if not out.exists() or out.stat().st_size == 0:
                con.execute(f"""
                    COPY (
                        SELECT
                            source1_entity_id,
                            string_agg(
                                candidate_entity_id::VARCHAR,
                                ',' ORDER BY candidate_entity_id
                            ) AS candidate_entity_ids
                        FROM (
                            SELECT source1_entity_id, candidate_entity_id
                            FROM read_parquet({qp(S2_CAND)})
                            WHERE (hash(source1_entity_id::VARCHAR) % {BUCKETS}) = {b}

                            UNION ALL

                            SELECT source1_entity_id, candidate_entity_id
                            FROM read_parquet({qp(S3_CAND)})
                            WHERE (hash(source1_entity_id::VARCHAR) % {BUCKETS}) = {b}
                        ) u
                        GROUP BY source1_entity_id
                    )
                    TO {qp(out)}
                    (FORMAT PARQUET, COMPRESSION ZSTD)
                """)

            if b % 16 == 0 or b == BUCKETS - 1:
                print(f"Built bucket {b + 1:03d}/{BUCKETS}")

        # Remove an old zero-byte/partial final file before writing.
        if CANDIDATE_TSV.exists() and CANDIDATE_TSV.stat().st_size == 0:
            CANDIDATE_TSV.unlink()

        print("\nWriting exact candidate_pairs.tsv...")
        con.execute(f"""
            COPY (
                SELECT
                    t.entity_id AS source1_entity_id,
                    coalesce(c.candidate_entity_ids, '') AS candidate_entity_ids
                FROM read_parquet({qp(S1_TEST)}) t
                LEFT JOIN read_parquet({qp(WORK / 'bucket_*.parquet')}) c
                  ON c.source1_entity_id = t.entity_id
                ORDER BY t.entity_id
            )
            TO {qp(CANDIDATE_TSV)}
            (HEADER, DELIMITER '\t')
        """)

        print(f"Created: {CANDIDATE_TSV}")
        print(f"Size: {CANDIDATE_TSV.stat().st_size / (1024**2):.1f} MB")

        manifest = {
            "status": "candidate_pairs_created",
            "source": {
                "s2_candidate_parquet": str(S2_CAND),
                "s3_candidate_parquet": str(S3_CAND),
            },
            "candidate_pairs": str(CANDIDATE_TSV),
            "buckets": BUCKETS,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        (OUT / "candidate_pairs_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    finally:
        con.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        raise

