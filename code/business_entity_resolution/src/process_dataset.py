
from pathlib import Path
import re
import unicodedata

import pandas as pd
from unidecode import unidecode


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "student_resource" / "dataset"

OUTPUT = ROOT / "processed_dataset"

TRAIN_DIR = DATASET / "train"
TEST_DIR = DATASET / "test"

OUTPUT_TRAIN = OUTPUT / "train"
OUTPUT_TEST = OUTPUT / "test"

# Large files -> process in chunks
CHUNK_SIZE = 200_000


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUTPUT_TEST.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize business names and addresses.

    Steps:
    1. Handle missing values
    2. Convert Unicode to transliterated text
    3. Lowercase
    4. Remove punctuation
    5. Normalize whitespace
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode transliteration
    value = unidecode(value)

    # Lowercase
    value = value.lower()

    # Normalize unicode
    value = unicodedata.normalize("NFKD", value)

    # Replace punctuation/special characters with spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_country(value):
    """
    Normalize country values.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    country_map = {
        "india": "india",
        "ind": "india",
        "us": "us",
        "usa": "us",
        "united states": "us",
        "france": "france",
    }

    return country_map.get(value, value)


# ============================================================
# ADD PROCESSED FEATURES
# ============================================================

def process_chunk(df):
    """
    Clean a single dataframe chunk and create
    matching features.
    """

    # Make sure expected columns exist
    expected = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    for column in expected:

        if column not in df.columns:
            df[column] = ""

    # --------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_text)
    )

    df["country_norm"] = (
        df["country"]
        .map(normalize_country)
    )

    # --------------------------------------------------------
    # Remove duplicate spaces
    # --------------------------------------------------------

    df["name_norm"] = (
        df["name_norm"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["address_norm"] = (
        df["address_norm"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # --------------------------------------------------------
    # Matching keys
    # --------------------------------------------------------

    # Exact normalized name
    df["name_key"] = df["name_norm"]

    # Name + country
    df["name_country_key"] = (
        df["name_norm"]
        + "||"
        + df["country_norm"]
    )

    # Address + country
    df["address_country_key"] = (
        df["address_norm"]
        + "||"
        + df["country_norm"]
    )

    # Name + address
    df["name_address_key"] = (
        df["name_norm"]
        + "||"
        + df["address_norm"]
    )

    # Name + address + country
    df["full_key"] = (
        df["name_norm"]
        + "||"
        + df["address_norm"]
        + "||"
        + df["country_norm"]
    )

    return df


# ============================================================
# PROCESS ONE TSV FILE
# ============================================================

def process_file(input_file, output_file):

    print("\n" + "=" * 100)
    print(f"PROCESSING: {input_file}")
    print("=" * 100)

    print(f"Output: {output_file}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    first_chunk = True

    total_rows = 0

    reader = pd.read_csv(
        input_file,
        sep="\t",
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, 1):

        print(
            f"\nChunk {chunk_number:,} "
            f"| rows = {len(chunk):,}"
        )

        # Process
        chunk = process_chunk(chunk)

        # Remove completely empty entity IDs
        chunk = chunk[
            chunk["entity_id"]
            .notna()
        ]

        total_rows += len(chunk)

        # Write parquet
        if first_chunk:

            chunk.to_parquet(
                output_file,
                index=False,
                engine="pyarrow",
            )

            first_chunk = False

        else:

            # Append using PyArrow
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(
                chunk,
                preserve_index=False,
            )

            parquet_file = pq.ParquetFile(
                output_file
            )

            # Existing parquet cannot be safely
            # appended with pandas directly.
            # Therefore this branch is handled below.
            #
            # We intentionally stop here so the file
            # is not corrupted.

            raise RuntimeError(
                "Parquet append requires the batch writer. "
                "Use process_file_streaming() instead."
            )

    print(
        f"\nProcessed rows: {total_rows:,}"
    )


# ============================================================
# STREAMING PARQUET WRITER
# ============================================================

def process_file_streaming(input_file, output_file):

    print("\n" + "=" * 100)
    print(f"PROCESSING: {input_file.name}")
    print("=" * 100)

    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pd.read_csv(
        input_file,
        sep="\t",
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    writer = None

    total_rows = 0

    try:

        for chunk_number, chunk in enumerate(reader, 1):

            print(
                f"Chunk {chunk_number:,}"
                f" | {len(chunk):,} rows"
            )

            chunk = process_chunk(chunk)

            chunk = chunk[
                chunk["entity_id"]
                .notna()
            ]

            total_rows += len(chunk)

            table = pa.Table.from_pandas(
                chunk,
                preserve_index=False,
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

    finally:

        if writer is not None:
            writer.close()

    print(
        f"Completed: {total_rows:,} rows"
    )


# ============================================================
# PROCESS TRAIN SOURCES
# ============================================================

def process_train_sources():

    print("\n" + "#" * 100)
    print("# TRAIN DATA")
    print("#" * 100)

    for source in [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]:

        input_file = TRAIN_DIR / source

        output_file = (
            OUTPUT_TRAIN
            / source.replace(".tsv", ".parquet")
        )

        process_file_streaming(
            input_file,
            output_file,
        )


# ============================================================
# PROCESS TEST SOURCES
# ============================================================

def process_test_sources():

    print("\n" + "#" * 100)
    print("# TEST DATA")
    print("#" * 100)

    for source in [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]:

        input_file = TEST_DIR / source

        output_file = (
            OUTPUT_TEST
            / source.replace(".tsv", ".parquet")
        )

        process_file_streaming(
            input_file,
            output_file,
        )


# ============================================================
# PROCESS GROUND TRUTH
# ============================================================

def process_ground_truth():

    input_file = (
        TRAIN_DIR
        / "train_ground_truth.tsv"
    )

    output_file = (
        OUTPUT_TRAIN
        / "ground_truth_pairs.parquet"
    )

    print("\n" + "#" * 100)
    print("# GROUND TRUTH")
    print("#" * 100)

    print(
        f"\nReading: {input_file}"
    )

    gt = pd.read_csv(
        input_file,
        sep="\t",
        dtype="string",
    )

    # --------------------------------------------------------
    # Remove rows without matches
    # --------------------------------------------------------

    gt = gt[
        gt["matched_entity_ids"]
        .notna()
    ].copy()

    # --------------------------------------------------------
    # Split multiple matched IDs
    # --------------------------------------------------------

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .str.split(",")
    )

    # One source1 -> one matched entity per row
    gt = gt.explode(
        "matched_entity_ids"
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .str.strip()
    )

    # Rename
    gt = gt.rename(
        columns={
            "source1_entity_id":
                "source1_entity_id",
            "matched_entity_ids":
                "matched_entity_id",
        }
    )

    # Positive label
    gt["label"] = 1

    # Keep only useful columns
    gt = gt[
        [
            "source1_entity_id",
            "matched_entity_id",
            "label",
        ]
    ]

    print(
        f"\nPositive matching pairs: "
        f"{len(gt):,}"
    )

    print("\nFirst 10 pairs:")

    print(
        gt.head(10)
        .to_string(index=False)
    )

    gt.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 100)
    print("AMAZON ML DATA PROCESSING PIPELINE")
    print("=" * 100)

    print(
        f"\nDataset: {DATASET}"
    )

    print(
        f"Output: {OUTPUT}"
    )

    # Process source files
    process_train_sources()

    process_test_sources()

    # Process labels
    process_ground_truth()

    print("\n" + "=" * 100)
    print("✅ DATA PROCESSING COMPLETED")
    print("=" * 100)

    print("\nProcessed files are available at:")

    print(OUTPUT)

