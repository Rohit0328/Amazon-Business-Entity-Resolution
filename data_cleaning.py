import numpy as np 
import pandas as pd

train_source1 = pd.read_csv("train_source1.tsv", sep="\t")
train_source2 = pd.read_csv("train_source2.tsv", sep="\t")
train_source3 = pd.read_csv("train_source3.tsv", sep="\t")
train_ground_truth = pd.read_csv("train_ground_truth.tsv", sep="\t")

print("Source 1:", train_source1.shape)
print("Source 2:", train_source2.shape)
print("Source 3:", train_source3.shape)
print("Ground Truth:", train_ground_truth.shape)

print(train_source1.head())

print("Source 1:", train_source1.columns.tolist())
print("Source 2:", train_source2.columns.tolist())
print("Source 3:", train_source3.columns.tolist())
print("Ground Truth:", train_ground_truth.columns.tolist())

print("\nGround Truth sample:")
print(train_ground_truth.head())

print("SOURCE 1")
print(train_source1.isna().sum())
print("Duplicate entity IDs:", train_source1["entity_id"].duplicated().sum())

print("\nSOURCE 2")
print(train_source2.isna().sum())
print("Duplicate entity IDs:", train_source2["entity_id"].duplicated().sum())

print("\nSOURCE 3")
print(train_source3.isna().sum())
print("Duplicate entity IDs:", train_source3["entity_id"].duplicated().sum())

print("\nGROUND TRUTH")
print(train_ground_truth.isna().sum())
print("Duplicate Source 1 IDs:",
      train_ground_truth["source1_entity_id"].duplicated().sum())

empty_gt = train_ground_truth[
    train_ground_truth["matched_entity_ids"].isna() |
    (train_ground_truth["matched_entity_ids"].astype(str).str.strip() == "")
]

print("Empty/no-match ground truth rows:", len(empty_gt))

print("\nExamples:")
print(empty_gt.head(10))

import re

def normalize_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


for df in [train_source1, train_source2, train_source3]:
    df["name_normalized"] = df["business_name"].apply(normalize_text)
    df["address_normalized"] = df["business_address"].apply(normalize_text)
    df["country_normalized"] = df["country"].apply(normalize_text)

print(train_source1.head())

print("SOURCE 1")
print("Missing names:", (train_source1["name_normalized"] == "").sum())
print("Missing addresses:", (train_source1["address_normalized"] == "").sum())

print("\nSOURCE 2")
print("Missing names:", (train_source2["name_normalized"] == "").sum())
print("Missing addresses:", (train_source2["address_normalized"] == "").sum())

print("\nSOURCE 3")
print("Missing names:", (train_source3["name_normalized"] == "").sum())
print("Missing addresses:", (train_source3["address_normalized"] == "").sum())


import os
import shutil

os.makedirs("processed_data", exist_ok=True)

train_source1.to_csv(
    "processed_data/train_source1_processed.tsv",
    sep="\t",
    index=False
)

train_source2.to_csv(
    "processed_data/train_source2_processed.tsv",
    sep="\t",
    index=False
)

train_source3.to_csv(
    "processed_data/train_source3_processed.tsv",
    sep="\t",
    index=False
)

train_ground_truth.to_csv(
    "processed_data/train_ground_truth_processed.tsv",
    sep="\t",
    index=False
)

shutil.make_archive(
    "amazon_ml_processed_data",
    "zip",
    "processed_data"
)

print("DONE")
print("amazon_ml_processed_data.zip created")