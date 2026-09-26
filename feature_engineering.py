import pandas as pd
import os

print("Loading candidate pairs...")

candidates = pd.read_csv(
    "processed_data/candidate_pairs_s2_new.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "candidate_entity_id",
        "blocking_rules"
    ]
)

print("Candidate pairs:", candidates.shape)


print("Loading Source 1...")

s1 = pd.read_csv(
    "processed_data/train_source1_processed.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Source 1:", s1.shape)


print("Loading Source 2...")

s2 = pd.read_csv(
    "processed_data/train_source2_processed.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Source 2:", s2.shape)


print("Loading Ground Truth...")

gt = pd.read_csv(
    "processed_data/train_ground_truth_processed.tsv",
    sep="\t"
)

print("Ground Truth:", gt.shape)


# ==========================================
# SOURCE 1
# ==========================================

print("Merging Source 1...")

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "name1",
        "business_address": "address1",
        "country": "country1"
    }
)

data = candidates.merge(
    s1,
    on="source1_entity_id",
    how="left"
)

print("After Source1 merge:", data.shape)


# ==========================================
# SOURCE 2
# ==========================================

print("Merging Source 2...")

s2 = s2.rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "name2",
        "business_address": "address2",
        "country": "country2"
    }
)

data = data.merge(
    s2,
    on="candidate_entity_id",
    how="left"
)

print("After Source2 merge:", data.shape)


# ==========================================
# BASIC CLEANING
# ==========================================

data["name1"] = data["name1"].fillna("")
data["name2"] = data["name2"].fillna("")

data["address1"] = data["address1"].fillna("")
data["address2"] = data["address2"].fillna("")

data["country1"] = data["country1"].fillna("")
data["country2"] = data["country2"].fillna("")


# ==========================================
# NAME FEATURES
# ==========================================

print("Creating name features...")

data["name_exact"] = (
    data["name1"] == data["name2"]
).astype("int8")

data["name_length1"] = (
    data["name1"].str.len()
).astype("int32")

data["name_length2"] = (
    data["name2"].str.len()
).astype("int32")

data["name_length_diff"] = (
    abs(
        data["name_length1"]
        - data["name_length2"]
    )
).astype("int32")

data["name_words1"] = (
    data["name1"]
    .str.count(" ")
    + 1
).astype("int16")

data["name_words2"] = (
    data["name2"]
    .str.count(" ")
    + 1
).astype("int16")

data["name_word_diff"] = (
    abs(
        data["name_words1"]
        - data["name_words2"]
    )
).astype("int16")


# ==========================================
# ADDRESS FEATURES
# ==========================================

print("Creating address features...")

data["address_exact"] = (
    data["address1"] == data["address2"]
).astype("int8")

data["address_length1"] = (
    data["address1"].str.len()
).astype("int32")

data["address_length2"] = (
    data["address2"].str.len()
).astype("int32")

data["address_length_diff"] = (
    abs(
        data["address_length1"]
        - data["address_length2"]
    )
).astype("int32")

data["address_words1"] = (
    data["address1"]
    .str.count(" ")
    + 1
).astype("int16")

data["address_words2"] = (
    data["address2"]
    .str.count(" ")
    + 1
).astype("int16")

data["address_word_diff"] = (
    abs(
        data["address_words1"]
        - data["address_words2"]
    )
).astype("int16")


# ==========================================
# COUNTRY
# ==========================================

print("Creating country feature...")

data["country_match"] = (
    data["country1"] == data["country2"]
).astype("int8")


# ==========================================
# BLOCKING FEATURES
# ==========================================

print("Creating blocking features...")

data["has_name_anchor"] = (
    data["blocking_rules"]
    .fillna("")
    .str.contains("name", case=False)
).astype("int8")

data["has_address_anchor"] = (
    data["blocking_rules"]
    .fillna("")
    .str.contains("address", case=False)
).astype("int8")

data["has_exact_name"] = (
    data["blocking_rules"]
    .fillna("")
    .str.contains("exact_name", case=False)
).astype("int8")

data["has_exact_address"] = (
    data["blocking_rules"]
    .fillna("")
    .str.contains("exact_address", case=False)
).astype("int8")


# ==========================================
# CREATE TRUE PAIRS
# ==========================================

print("Creating labels...")

true_pairs = []

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for candidate_id in ids:

        candidate_id = candidate_id.strip()

        if candidate_id:
            true_pairs.append(
                (
                    source1_id,
                    candidate_id
                )
            )


true_pairs = pd.DataFrame(
    true_pairs,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

true_pairs["label"] = 1

print("True match pairs:", true_pairs.shape)


# ==========================================
# MERGE LABELS
# ==========================================

print("Merging labels...")

data = data.merge(
    true_pairs,
    on=[
        "source1_entity_id",
        "candidate_entity_id"
    ],
    how="left"
)

data["label"] = (
    data["label"]
    .fillna(0)
    .astype("int8")
)


# ==========================================
# FINAL FEATURES
# ==========================================

features = [

    "name_exact",
    "name_length1",
    "name_length2",
    "name_length_diff",
    "name_words1",
    "name_words2",
    "name_word_diff",

    "address_exact",
    "address_length1",
    "address_length2",
    "address_length_diff",
    "address_words1",
    "address_words2",
    "address_word_diff",

    "country_match",

    "has_name_anchor",
    "has_address_anchor",
    "has_exact_name",
    "has_exact_address",

    "label"
]


final_data = data[features]


# ==========================================
# SAVE
# ==========================================

os.makedirs(
    "processed_data",
    exist_ok=True
)

output_file = (
    "processed_data/training_features.csv"
)

print("Saving...")

final_data.to_csv(
    output_file,
    index=False
)


print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print("Final shape:", final_data.shape)

print("\nLabel distribution:")

print(
    final_data["label"].value_counts()
)

print("\nSaved:", output_file)