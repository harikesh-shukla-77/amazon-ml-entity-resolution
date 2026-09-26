
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_DIR = ROOT / "processed_dataset" / "train"
OUTPUT_DIR = ROOT / "candidate_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

SOURCE1 = TRAIN_DIR / "train_source1.parquet"
SOURCE2 = TRAIN_DIR / "train_source2.parquet"
SOURCE3 = TRAIN_DIR / "train_source3.parquet"

GROUND_TRUTH = (
    TRAIN_DIR / "ground_truth_pairs.parquet"
)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 100)
print("AMAZON ML - CANDIDATE GENERATION")
print("=" * 100)

print("\nLoading Source 1...")

s1 = pd.read_parquet(SOURCE1)

print(f"Source 1 rows: {len(s1):,}")


# ============================================================
# LOAD SOURCE 2 + SOURCE 3
# ============================================================

print("\nLoading Source 2...")

s2 = pd.read_parquet(SOURCE2)

print(f"Source 2 rows: {len(s2):,}")


print("\nLoading Source 3...")

s3 = pd.read_parquet(SOURCE3)

print(f"Source 3 rows: {len(s3):,}")


# ============================================================
# FUNCTION: EXACT CANDIDATES
# ============================================================

def generate_exact_candidates(
    source1,
    source23,
    source_name,
):
    """
    Generate candidates using multiple exact normalized keys.

    Priority:
        1. full_key
        2. name_address_key
        3. name_country_key
        4. address_country_key
        5. name_key
    """

    print("\n" + "=" * 100)
    print(f"GENERATING CANDIDATES: {source_name}")
    print("=" * 100)

    candidate_parts = []

    # --------------------------------------------------------
    # Keys to use
    # --------------------------------------------------------

    keys = [
        "full_key",
        "name_address_key",
        "name_country_key",
        "address_country_key",
        "name_key",
    ]

    # --------------------------------------------------------
    # Create lookup tables
    # --------------------------------------------------------

    for key in keys:

        print(f"\nKey: {key}")

        left = source1[
            [
                "entity_id",
                key,
            ]
        ].copy()

        right = source23[
            [
                "entity_id",
                key,
            ]
        ].copy()

        # Remove empty keys
        left = left[
            left[key].notna()
            & (left[key] != "")
        ]

        right = right[
            right[key].notna()
            & (right[key] != "")
        ]

        # ----------------------------------------------------
        # Merge exact key
        # ----------------------------------------------------

        merged = left.merge(
            right,
            on=key,
            how="inner",
            suffixes=(
                "_source1",
                "_candidate",
            ),
        )

        if len(merged) == 0:

            print("  Candidates: 0")
            continue

        # ----------------------------------------------------
        # Rename
        # ----------------------------------------------------

        merged = merged[
            [
                "entity_id_source1",
                "entity_id_candidate",
            ]
        ].copy()

        merged["match_key"] = key

        candidate_parts.append(merged)

        print(
            f"  Candidates: {len(merged):,}"
        )

    # --------------------------------------------------------
    # Combine all keys
    # --------------------------------------------------------

    if not candidate_parts:

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "match_keys",
            ]
        )

    candidates = pd.concat(
        candidate_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Rename IDs
    # --------------------------------------------------------

    candidates = candidates.rename(
        columns={
            "entity_id_source1":
                "source1_entity_id",

            "entity_id_candidate":
                "candidate_entity_id",
        }
    )

    # --------------------------------------------------------
    # Combine multiple matching keys
    # --------------------------------------------------------

    candidates = (
        candidates
        .groupby(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ],
            as_index=False,
        )
        .agg(
            match_keys=(
                "match_key",
                lambda x: "|".join(
                    sorted(set(x))
                )
            )
        )
    )

    # --------------------------------------------------------
    # Number of exact keys
    # --------------------------------------------------------

    candidates["exact_key_count"] = (
        candidates["match_keys"]
        .str.count(r"\|")
        + 1
    )

    # --------------------------------------------------------
    # Sort strongest candidates first
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        [
            "source1_entity_id",
            "exact_key_count",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return candidates


# ============================================================
# SOURCE 1 -> SOURCE 2
# ============================================================

candidates_s2 = generate_exact_candidates(
    s1,
    s2,
    "SOURCE 1 → SOURCE 2",
)


# ============================================================
# SOURCE 1 -> SOURCE 3
# ============================================================

candidates_s3 = generate_exact_candidates(
    s1,
    s3,
    "SOURCE 1 → SOURCE 3",
)


# ============================================================
# SAVE CANDIDATES
# ============================================================

print("\n" + "=" * 100)
print("SAVING CANDIDATES")
print("=" * 100)


s2_output = (
    OUTPUT_DIR
    / "train_candidates_s1_s2.parquet"
)

s3_output = (
    OUTPUT_DIR
    / "train_candidates_s1_s3.parquet"
)


candidates_s2.to_parquet(
    s2_output,
    index=False,
)

candidates_s3.to_parquet(
    s3_output,
    index=False,
)


print(
    f"\nS1 → S2 candidates: "
    f"{len(candidates_s2):,}"
)

print(
    f"S1 → S3 candidates: "
    f"{len(candidates_s3):,}"
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 100)
print("CANDIDATE STATISTICS")
print("=" * 100)


def print_statistics(
    candidates,
    name,
):

    print(f"\n{name}")

    if candidates.empty:

        print("No candidates found.")

        return

    unique_s1 = (
        candidates[
            "source1_entity_id"
        ]
        .nunique()
    )

    print(
        f"Unique Source1 entities with candidates: "
        f"{unique_s1:,}"
    )

    print(
        f"Total candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Average candidates per Source1: "
        f"{len(candidates) / unique_s1:.2f}"
    )

    print("\nExact-key strength:")

    print(
        candidates[
            "exact_key_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


print_statistics(
    candidates_s2,
    "SOURCE 1 → SOURCE 2",
)

print_statistics(
    candidates_s3,
    "SOURCE 1 → SOURCE 3",
)


# ============================================================
# CHECK GROUND TRUTH COVERAGE
# ============================================================

print("\n" + "=" * 100)
print("GROUND TRUTH COVERAGE")
print("=" * 100)


gt = pd.read_parquet(
    GROUND_TRUTH
)

# ------------------------------------------------------------
# Separate S2 and S3 ground truth
# ------------------------------------------------------------

gt_s2 = gt[
    gt["matched_entity_id"]
    .str.startswith("S2-", na=False)
].copy()

gt_s3 = gt[
    gt["matched_entity_id"]
    .str.startswith("S3-", na=False)
].copy()


# ------------------------------------------------------------
# Coverage function
# ------------------------------------------------------------

def calculate_coverage(
    candidates,
    ground_truth,
    label,
):

    candidate_pairs = set(
        zip(
            candidates[
                "source1_entity_id"
            ],
            candidates[
                "candidate_entity_id"
            ],
        )
    )

    truth_pairs = set(
        zip(
            ground_truth[
                "source1_entity_id"
            ],
            ground_truth[
                "matched_entity_id"
            ],
        )
    )

    if not truth_pairs:

        print(
            f"\n{label}: no ground truth"
        )

        return

    found = len(
        truth_pairs
        & candidate_pairs
    )

    coverage = (
        found
        / len(truth_pairs)
        * 100
    )

    print(
        f"\n{label}"
    )

    print(
        f"Ground-truth pairs : "
        f"{len(truth_pairs):,}"
    )

    print(
        f"Candidates covering truth: "
        f"{found:,}"
    )

    print(
        f"Recall / coverage: "
        f"{coverage:.2f}%"
    )


calculate_coverage(
    candidates_s2,
    gt_s2,
    "S1 → S2",
)

calculate_coverage(
    candidates_s3,
    gt_s3,
    "S1 → S3",
)


# ============================================================
# SHOW STRONGEST EXAMPLES
# ============================================================

print("\n" + "=" * 100)
print("SAMPLE CANDIDATES")
print("=" * 100)


print("\nS1 → S2")

print(
    candidates_s2
    .head(20)
    .to_string(index=False)
)


print("\nS1 → S3")

print(
    candidates_s3
    .head(20)
    .to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 100)
print("✅ CANDIDATE GENERATION COMPLETED")
print("=" * 100)

print(
    f"\nOutput directory:"
    f"\n{OUTPUT_DIR}"
)

