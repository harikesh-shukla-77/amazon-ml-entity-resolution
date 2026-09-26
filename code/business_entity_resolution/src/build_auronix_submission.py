#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import json

ROOT = Path("/Users/harikeshshukla/mla")
STAGE = ROOT / "_Auronix_submission_build"
ZIP_PATH = ROOT / "Auronix_submission.zip"

MATCHING = ROOT / "student_resource/output/matching_results.tsv"
CANDIDATE = ROOT / "student_resource/output/candidate_pairs.tsv"
VALIDATOR = ROOT / "student_resource/utils/validate_submission.py"

# Use the renamed final inference script if present.
FINAL_SCRIPT_CANDIDATES = [
    ROOT / "final_one_shot_v6.py",
    ROOT / "amazon_ml_final_one_shot_v6.py",
]

REQUIREMENTS = """duckdb==1.5.5
numpy==2.3.2
pandas==3.0.6
pyarrow==25.0.1
RapidFuzz==3.14.6
scikit-learn==1.9.1
scipy==1.18.1
joblib==1.6.0
cloudpickle==3.1.2
Unidecode==1.4.0
"""

README = r"""# Auronix — Amazon ML Challenge 2026

## Overview
This submission solves the Business Entity Resolution task using only the supplied
training/test data and derived features. Source 1 is matched against Source 2 and
Source 3 independently, then the ranked IDs are combined per Source-1 entity.

## Final pipeline
1. Normalize business name/address/country fields in the provided data.
2. Generate candidates using multiple blocking families validated on training data:
   - V1 exact composite keys
   - V2 safe postal blocking
   - V4 selected address/postal blocks
   - V5 name-prefix + postal and address-tail + postal blocks
   - V9.1 numeric-address + name-prefix blocking
3. Compute transparent pair features, including exact matches and RapidFuzz
   string similarities.
4. Build a V6-compatible base score:
   30*name_exact + 25*address_exact + 8*country_exact
   + 22*name_similarity + 12*address_similarity
   + 2*name_length_ratio + 1*address_length_ratio
   + 0.75*evidence_file_count(clipped at 4)
   + 0.25*exact_key_count(clipped at 5).
5. Train a separate pairwise LogisticRegression ranker for S1->S2 and S1->S3
   using standardized V10 features. Training uses one positive pair and two
   negatives sampled at a distant rank per Source-1 entity.
6. Score the complete final candidate pool and rank candidates within each
   Source-1 entity.
7. Produce `matching_results.tsv`.

## Feature set
The final ranker uses:
name_similarity, address_similarity, name_exact, address_exact,
country_exact, name_length_ratio, address_length_ratio, evidence_rows,
evidence_file_count, exact_key_count, base_score, log_base_rank,
name_x_address, name_minus_address, similarity_mean, similarity_min,
exact_field_count.

## Reproduction
The main inference/training implementation is in:
`code/business_entity_resolution/src/final_one_shot_v6.py`

Run it from the original project root after the provided data has been prepared:

```bash
/Users/harikeshshukla/mla/.venv/bin/python -u final_one_shot_v6.py
```

The script expects the processed Parquet inputs and V10 labeled training pools
used by the final pipeline.

## Final outputs
`output/matching_results.tsv` is the leaderboard-scored file.
`output/candidate_pairs.tsv` is the final candidate set supplied to the model
for test inference.

## Validation
The challenge-provided validator was run with ID checking enabled and returned:

`PASS — no blocking issues found. Safe to submit.`

Final test-set sizes used by the pipeline:
- Source 1: 1,732,544
- Source 2 candidates: 25,448,679
- Source 3 candidates: 44,404,686

The final submission contains one row for every Source-1 test entity. No external
business-identity lookup or external data augmentation was used.
"""

METHODOLOGY = r"""# Amazon ML Challenge 2026 — Auronix Methodology

## 1. Methodology used
We treated the task as large-scale cross-source entity resolution. Source 1 was
matched independently to Source 2 and Source 3. Text fields were normalized and
candidate pairs were generated with deterministic blocking. The final ranking
stage combined exact-field evidence, string similarity, candidate evidence counts,
and a pairwise LogisticRegression model.

The solution was designed around precision-heavy evaluation: candidates are first
restricted by blocking, then ranked using increasingly informative signals rather
than comparing every possible cross-source pair.

## 2. Candidate generation / blocking
The final test candidate pool combines blocking families validated on the training
data:

- V1 exact composite keys using normalized name/address/country combinations.
- V2 safe postal blocking.
- V4 selected postal/address blocks, including address fragments and a bounded
  numeric-address block.
- V5 name-first-3 + postal and postal + address-tail-5 blocks.
- V9.1 numeric-address + first-three-name blocking.

Candidate evidence from the different blocks is consolidated by
`source1_entity_id` and `candidate_entity_id`. The final candidate set is the
exact set scored by the inference model.

On the final test set, the candidate pool contained 25,448,679 S1->S2 pairs and
44,404,686 S1->S3 pairs.

## 3. Features and model architecture
For each candidate pair we computed:

- exact normalized name, address and country indicators;
- RapidFuzz name and address similarity;
- name/address length ratios;
- number of evidence rows and distinct blocking evidence groups;
- number of exact blocking keys;
- a transparent base score and rank-derived transformations;
- interactions and summary similarity statistics.

The base score used was:

30*name_exact + 25*address_exact + 8*country_exact
+ 22*name_similarity + 12*address_similarity
+ 2*name_length_ratio + 1*address_length_ratio
+ 0.75*evidence_file_count (clipped at 4)
+ 0.25*exact_key_count (clipped at 5).

Two separate pairwise LogisticRegression rankers were trained, one for S1->S2
and one for S1->S3. Before fitting, the V10 feature vectors were standardized.
The pair-training construction used one positive pair and two negative pairs per
Source-1 entity, with negatives drawn from a distant rank in the candidate list.

## 4. Training experiments
The blocking strategy was expanded iteratively using training ground truth.
Earlier V1/V2/V4/V5 experiments established complementary exact and structured
blocks. A V9.1 numeric-address/name-prefix block added substantial additional
training coverage without using unrestricted high-cardinality joins.

The final V10 validation on the complete expanded training candidate pools gave:

| Direction | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|
| S1->S2 | 38.44% | 64.12% | 64.40% | 64.53% |
| S1->S3 | 36.74% | 62.33% | 62.82% | 63.05% |

These are internal training/validation retrieval measurements, not test-set scores.

## 5. Final inference and outputs
The final inference pipeline scored every candidate in the final test pools and
ranked candidates separately for Source 2 and Source 3. The top ranked IDs were
combined per Source-1 entity into `matching_results.tsv`.

The final test file contains 1,732,544 Source-1 rows. The output was validated
against the supplied test IDs with the challenge validator using ID-existence
checking. The validation result was:

`PASS — no blocking issues found. Safe to submit.`

A total of 50,241 Source-1 entities have empty match lists in the final output.

## 6. Data use and reproducibility
The pipeline uses the supplied challenge data and internally derived features
only. No external business-identity databases, lookup APIs, geocoding services, or
external data augmentation are part of the method.

The final code, pinned Python dependencies, `matching_results.tsv`, and
`candidate_pairs.tsv` are included in the submission package.
"""

def copy_tree_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    print("=" * 100)
    print("BUILDING Auronix_submission.zip")
    print("=" * 100)

    required = [MATCHING, CANDIDATE, VALIDATOR]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(map(str, missing)))

    final_script = next((p for p in FINAL_SCRIPT_CANDIDATES if p.exists()), None)
    if final_script is None:
        raise FileNotFoundError("Could not find final_one_shot_v6.py or amazon_ml_final_one_shot_v6.py")

    # Validate the originals one final time before packaging.
    print("\nRunning final validator...")
    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--matching", str(MATCHING),
        "--candidate", str(CANDIDATE),
        "--test-dir", str(ROOT / "student_resource/dataset/test"),
        "--check-ids",
    ]
    subprocess.run(cmd, check=True)

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # Required output files.
    copy_tree_file(MATCHING, STAGE / "output/matching_results.tsv")
    copy_tree_file(CANDIDATE, STAGE / "output/candidate_pairs.tsv")

    # Reproducible source tree. Include all top-level Python scripts used during
    # experiments plus the final inference code, without including datasets or
    # generated Parquet/CSV artifacts.
    src_dir = STAGE / "code/business_entity_resolution/src"
    src_dir.mkdir(parents=True)
    py_files = sorted(ROOT.glob("*.py"))
    for p in py_files:
        copy_tree_file(p, src_dir / p.name)

    # Include the challenge validator as a useful audit utility.
    copy_tree_file(VALIDATOR, STAGE / "code/business_entity_resolution/src/validate_submission.py")

    (STAGE / "code/business_entity_resolution/README.md").write_text(README, encoding="utf-8")
    (STAGE / "code/business_entity_resolution/requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (STAGE / "Documentation_template.md").write_text(METHODOLOGY, encoding="utf-8")

    # Include final model metadata when available.
    coeff = ROOT / "candidate_output_final_v6/final_model_coefficients.json"
    if coeff.exists():
        copy_tree_file(coeff, STAGE / "code/business_entity_resolution/final_model_coefficients.json")

    manifest = {
        "team_name": "Auronix",
        "matching_rows": 1732544,
        "candidate_rows": 1732544,
        "test_candidates_s2": 25448679,
        "test_candidates_s3": 44404686,
        "main_inference_script": str(final_script.name),
        "validation": "PASS — no blocking issues found. Safe to submit.",
    }
    (STAGE / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print("\nCreating ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(STAGE))

    zip_size_gb = ZIP_PATH.stat().st_size / (1024**3)
    print("\n" + "=" * 100)
    print("✅ AURONIX SUBMISSION PACKAGE CREATED")
    print("=" * 100)
    print(f"ZIP : {ZIP_PATH}")
    print(f"Size: {zip_size_gb:.2f} GB")
    print("\nPackage structure:")
    for item in sorted(STAGE.rglob("*")):
        if item.is_file():
            print(" ", item.relative_to(STAGE))

if __name__ == "__main__":
    main()
