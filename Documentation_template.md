# Amazon ML Challenge 2026 — Auronix Methodology

## 1. Methodology used

The task is treated as large-scale cross-source business entity
resolution. Source 1 is matched independently to Source 2 and Source 3.

The production pipeline uses deterministic candidate generation followed
by transparent feature extraction and learned pairwise ranking.

The challenge metric is macro F0.5 per Source-1 entity, including
singletons. The metric is precision-heavy, so incorrect merges are
especially costly.

## 2. Candidate generation / blocking

Candidate generation was developed iteratively using the supplied
training data.

The production candidate pool combines complementary blocking families:

- V1 exact normalized composite keys
- V2 postal blocking
- V4 selected address/postal structures
- V5 structured name/postal and address/postal blocks
- V9.1 numeric-address + name-prefix blocking

For final test inference the resulting exact candidate pools contain:

- S1 -> S2: 25,448,679 candidate pairs
- S1 -> S3: 44,404,686 candidate pairs

The candidate pool recorded in candidate_pairs.tsv is the exact candidate
set supplied to the inference ranking stage.

## 3. Feature engineering

For every candidate pair the pipeline computes:

- normalized name similarity
- normalized address similarity
- exact normalized name indicator
- exact normalized address indicator
- exact country indicator
- name length ratio
- address length ratio
- number of evidence rows
- number of distinct evidence groups
- exact-key count
- transparent base score
- logarithmic base-rank feature
- name/address interaction
- name/address difference
- mean similarity
- minimum similarity
- exact-field count

RapidFuzz is used for fuzzy name and address similarity.

## 4. Model architecture

Two independent pairwise LogisticRegression rankers are trained:

- S1 -> S2
- S1 -> S3

Training examples are constructed by comparing a positive candidate
against selected negative candidates within the same Source-1 entity.
The resulting difference features are standardized before fitting.

No external pretrained entity-resolution model is used.

## 5. Prediction policy

The model produces `v10_utility` for every candidate.

For each Source-1 entity and each target source:

- sort candidates by descending `v10_utility`;
- consider only rank 1;
- accept the rank-1 candidate only if
  `v10_utility >= 20.945884704589844`;
- otherwise emit no match for that target source.

Thus the final output can contain zero, one, or two predicted IDs
per Source-1 entity: at most one from Source 2 and one from Source 3.

## 6. Validation evidence

Using the old validation protocol with validation entities excluded
from model fitting:

Baseline K1/K1:

- macro F0.5 = 0.594111908498
- precision = 0.726289916951
- recall = 0.375394953889

Frozen Top-1 + abstention:

- threshold = 20.945884704589844
- macro F0.5 = 0.617515903939
- precision = 0.775931545288
- recall = 0.373745397252

The policy therefore trades a small amount of recall for substantially
higher precision, which is aligned with the challenge's F0.5 metric.

These are internal validation measurements, not hidden test-set scores.

## 7. V8 investigation

A four-block V8 expansion was evaluated separately:

- num_addr_name_prefix2
- addr_last2_postal
- name_last_token_postal
- name_first1_postal

The expansion increased retrieval Recall@20 in the audited validation
experiments, but under the precision-heavy F0.5 prediction audit its
K1/K1 and optimized threshold results remained below the corresponding
baseline policy.

Therefore the final test candidate pool remains the already-scored
V6 + V9.1 production pool.

## 8. Data integrity and fair play

The pipeline uses only the supplied challenge data and derived features.

No external business-identity lookup, commercial entity-resolution API,
government registry lookup, geocoding API, or external business-data
augmentation is used.

Country is handled as an open-set string field and is not restricted to
US/India.

## 9. Reproducibility

The submission package includes the final inference source, the final
thresholding logic, dependency versions, output files, and methodology.

The challenge-provided validator is executed before packaging.
