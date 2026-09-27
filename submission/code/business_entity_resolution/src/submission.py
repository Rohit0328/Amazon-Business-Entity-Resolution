import pandas as pd
import numpy as np
import joblib
import os
import gc

# ============================================================
# FINISH SUBMISSION FROM EXISTING CANDIDATES
# ============================================================

CANDIDATE_FILE = "candidate_pairs.tsv"
MODEL_FILE = "xgboost_entity_match_model.joblib"

S1_FILE = "processed_data/test_source1.tsv"
S2_FILE = "processed_data/test_source2.tsv"
S3_FILE = "processed_data/test_source3.tsv"

OUTPUT_FILE = "matching_results.tsv"

THRESHOLD = 0.95
CHUNK_SIZE = 100_000

print("=" * 70)
print("FINISHING MODEL-BASED SUBMISSION")
print("=" * 70)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")
model = joblib.load(MODEL_FILE)
print("Model loaded.")

# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

s1 = s1.rename(columns={
    "entity_id": "source1_entity_id",
    "business_name": "name1",
    "business_address": "address1",
    "country": "country1"
})

print("S1:", len(s1))

print("\nLoading Source 2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

s2 = s2.rename(columns={
    "entity_id": "candidate_entity_id",
    "business_name": "name2",
    "business_address": "address2",
    "country": "country2"
})

print("S2:", len(s2))

print("\nLoading Source 3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

s3 = s3.rename(columns={
    "entity_id": "candidate_entity_id",
    "business_name": "name2",
    "business_address": "address2",
    "country": "country2"
})

print("S3:", len(s3))

# Combine S2 + S3 lookup

print("\nCreating candidate lookup...")

candidates_source = pd.concat(
    [s2, s3],
    ignore_index=True
)

del s2
del s3
gc.collect()

# ============================================================
# INDEX DATA FOR FAST MERGE
# ============================================================

s1_lookup = s1.set_index("source1_entity_id")

candidate_lookup = candidates_source.set_index(
    "candidate_entity_id"
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "name_exact",
    "name_length1",
    "name_length2",
    "name_length_diff",
    "name_words1",
    "name_words2",
    "name_word_diff",
    "name_prefix3_match",
    "name_suffix3_match",
    "name_length_ratio",
    "address_exact",
    "address_length1",
    "address_length2",
    "address_length_diff",
    "address_words1",
    "address_words2",
    "address_word_diff",
    "address_prefix3_match",
    "address_suffix3_match",
    "address_length_ratio",
    "country_match",
    "has_name_anchor",
    "has_address_anchor",
    "has_exact_name",
    "has_exact_address"
]

# ============================================================
# FEATURE FUNCTION
# ============================================================

def make_features(df):

    for col in [
        "name1",
        "name2",
        "address1",
        "address2",
        "country1",
        "country2"
    ]:
        df[col] = df[col].fillna("").astype(str)

    df["name_exact"] = (
        df["name1"] == df["name2"]
    ).astype("int8")

    df["name_length1"] = (
        df["name1"].str.len()
    ).astype("int32")

    df["name_length2"] = (
        df["name2"].str.len()
    ).astype("int32")

    df["name_length_diff"] = (
        abs(
            df["name_length1"]
            - df["name_length2"]
        )
    ).astype("int32")

    df["name_words1"] = (
        df["name1"].str.count(" ") + 1
    ).astype("int16")

    df["name_words2"] = (
        df["name2"].str.count(" ") + 1
    ).astype("int16")

    df["name_word_diff"] = (
        abs(
            df["name_words1"]
            - df["name_words2"]
        )
    ).astype("int16")

    df["name_prefix3_match"] = (
        df["name1"].str[:3]
        ==
        df["name2"].str[:3]
    ).astype("int8")

    df["name_suffix3_match"] = (
        df["name1"].str[-3:]
        ==
        df["name2"].str[-3:]
    ).astype("int8")

    df["name_length_ratio"] = (
        df["name_length1"]
        /
        df["name_length2"].replace(0, 1)
    ).clip(0, 5).astype("float32")

    df["address_exact"] = (
        df["address1"] == df["address2"]
    ).astype("int8")

    df["address_length1"] = (
        df["address1"].str.len()
    ).astype("int32")

    df["address_length2"] = (
        df["address2"].str.len()
    ).astype("int32")

    df["address_length_diff"] = (
        abs(
            df["address_length1"]
            - df["address_length2"]
        )
    ).astype("int32")

    df["address_words1"] = (
        df["address1"].str.count(" ") + 1
    ).astype("int16")

    df["address_words2"] = (
        df["address2"].str.count(" ") + 1
    ).astype("int16")

    df["address_word_diff"] = (
        abs(
            df["address_words1"]
            - df["address_words2"]
        )
    ).astype("int16")

    df["address_prefix3_match"] = (
        df["address1"].str[:3]
        ==
        df["address2"].str[:3]
    ).astype("int8")

    df["address_suffix3_match"] = (
        df["address1"].str[-3:]
        ==
        df["address2"].str[-3:]
    ).astype("int8")

    df["address_length_ratio"] = (
        df["address_length1"]
        /
        df["address_length2"].replace(0, 1)
    ).clip(0, 5).astype("float32")

    df["country_match"] = (
        df["country1"] == df["country2"]
    ).astype("int8")

    rules = df["blocking_rules"].fillna("")

    df["has_name_anchor"] = (
        rules.str.contains("name", case=False)
    ).astype("int8")

    df["has_address_anchor"] = (
        rules.str.contains("address", case=False)
    ).astype("int8")

    df["has_exact_name"] = (
        rules.str.contains("exact_name", case=False)
    ).astype("int8")

    df["has_exact_address"] = (
        rules.str.contains("exact_address", case=False)
    ).astype("int8")

    return df


# ============================================================
# OUTPUT
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

results = {}

# Initialize ALL S1 records.
# This guarantees one row per S1.

for sid in s1["source1_entity_id"]:
    results[sid] = []

# ============================================================
# PROCESS CANDIDATES
# ============================================================

print("\nProcessing candidate_pairs.tsv...")

total = 0
accepted = 0

for chunk_no, candidates in enumerate(
    pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    # --------------------------------------------------------
    # Merge S1 information
    # --------------------------------------------------------

    candidates = candidates.merge(
        s1.reset_index(drop=True),
        on="source1_entity_id",
        how="left"
    )

    # --------------------------------------------------------
    # Merge candidate information
    # --------------------------------------------------------

    candidates = candidates.merge(
        candidates_source,
        on="candidate_entity_id",
        how="left"
    )

    # Remove accidental duplicate columns if present

    candidates = candidates.loc[
        :,
        ~candidates.columns.duplicated()
    ]

    if len(candidates) == 0:
        continue

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    candidates = make_features(candidates)

    X = candidates[FEATURES]

    probabilities = model.predict_proba(X)[:, 1]

    candidates["probability"] = probabilities

    # --------------------------------------------------------
    # MODEL DECISION
    # --------------------------------------------------------

    accepted_rows = candidates[
        candidates["probability"] >= THRESHOLD
    ]

    total += len(candidates)
    accepted += len(accepted_rows)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    for row in accepted_rows.itertuples(index=False):

        sid = row.source1_entity_id
        cid = row.candidate_entity_id

        if cid not in results[sid]:
            results[sid].append(cid)

    if chunk_no % 10 == 0:
        print(
            f"Processed {total:,} candidates | "
            f"model accepted {accepted:,}"
        )

    del candidates
    gc.collect()

# ============================================================
# WRITE RESULTS
# ============================================================

print("\nWriting matching_results.tsv...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    for sid in s1["source1_entity_id"]:

        ids = results[sid]

        f.write(
            sid
            + "\t"
            + ",".join(ids)
            + "\n"
        )

# ============================================================
# SUMMARY
# ============================================================

matched_s1 = sum(
    len(v) > 0
    for v in results.values()
)

total_matches = sum(
    len(v)
    for v in results.values()
)

print("\n" + "=" * 70)
print("SUBMISSION READY")
print("=" * 70)

print("Total candidates processed:", f"{total:,}")
print("Model accepted:", f"{accepted:,}")
print("S1 records:", f"{len(s1):,}")
print("S1 with matches:", f"{matched_s1:,}")
print("Total matched IDs:", f"{total_matches:,}")

print("\nCreated:")
print(OUTPUT_FILE)

print("\nThreshold:", THRESHOLD)

print("\nDONE.")