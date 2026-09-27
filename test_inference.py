import pandas as pd
import re
import gc
import os

BASE = "processed_data"

S1_FILE = os.path.join(BASE, "test_source1.tsv")
S2_FILE = os.path.join(BASE, "test_source2.tsv")
S3_FILE = os.path.join(BASE, "test_source3.tsv")

OUTPUT = os.path.join(BASE, "matching_results.tsv")


def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Loading Source1...")
s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str)

print("Normalizing Source1...")
s1["name_norm"] = s1["business_name"].map(normalize)
s1["address_norm"] = s1["business_address"].map(normalize)
s1["country_norm"] = s1["country"].map(normalize)

print("Source1 ready:", s1.shape)

# ---------------------------------------------------------
# PROCESS ONE SOURCE AT A TIME
# ---------------------------------------------------------

def process_source(source_file, source_name):

    print("\n" + "=" * 50)
    print("Processing", source_name)
    print("=" * 50)

    src = pd.read_csv(source_file, sep="\t", dtype=str)

    print(source_name, "loaded:", src.shape)

    print("Normalizing", source_name + "...")

    src["name_norm"] = src["business_name"].map(normalize)
    src["address_norm"] = src["business_address"].map(normalize)
    src["country_norm"] = src["country"].map(normalize)

    results = []

    # -----------------------------------------------------
    # EXACT NAME + COUNTRY
    # -----------------------------------------------------

    print("Matching exact NAME + COUNTRY...")

    name_src = src[
        (src["name_norm"] != "") &
        (src["country_norm"] != "")
    ][
        ["entity_id", "name_norm", "country_norm"]
    ].drop_duplicates()

    name_s1 = s1[
        (s1["name_norm"] != "") &
        (s1["country_norm"] != "")
    ][
        ["entity_id", "name_norm", "country_norm"]
    ]

    name_match = name_s1.merge(
        name_src,
        on=["name_norm", "country_norm"],
        how="inner",
        suffixes=("_s1", "_src")
    )

    if len(name_match) > 0:
        name_result = name_match[
            ["entity_id_s1", "entity_id_src"]
        ].copy()

        name_result.columns = [
            "source1_entity_id",
            "matched_entity_id"
        ]

        name_result["source"] = source_name
        name_result["match_type"] = "exact_name_country"

        results.append(name_result)

    print("Name matches:", len(name_match))

    del name_src, name_s1, name_match
    gc.collect()

    # -----------------------------------------------------
    # EXACT ADDRESS + COUNTRY
    # -----------------------------------------------------

    print("Matching exact ADDRESS + COUNTRY...")

    addr_src = src[
        (src["address_norm"] != "") &
        (src["country_norm"] != "")
    ][
        ["entity_id", "address_norm", "country_norm"]
    ].drop_duplicates()

    addr_s1 = s1[
        (s1["address_norm"] != "") &
        (s1["country_norm"] != "")
    ][
        ["entity_id", "address_norm", "country_norm"]
    ]

    addr_match = addr_s1.merge(
        addr_src,
        on=["address_norm", "country_norm"],
        how="inner",
        suffixes=("_s1", "_src")
    )

    if len(addr_match) > 0:
        addr_result = addr_match[
            ["entity_id_s1", "entity_id_src"]
        ].copy()

        addr_result.columns = [
            "source1_entity_id",
            "matched_entity_id"
        ]

        addr_result["source"] = source_name
        addr_result["match_type"] = "exact_address_country"

        results.append(addr_result)

    print("Address matches:", len(addr_match))

    del addr_src, addr_s1, addr_match
    gc.collect()

    # -----------------------------------------------------
    # SAVE SOURCE RESULTS
    # -----------------------------------------------------

    if results:
        final_source = pd.concat(results, ignore_index=True)

        final_source.drop_duplicates(
            subset=[
                "source1_entity_id",
                "matched_entity_id",
                "source"
            ],
            inplace=True
        )

        print(
            source_name,
            "final matches:",
            len(final_source)
        )

        return final_source

    return pd.DataFrame(
        columns=[
            "source1_entity_id",
            "matched_entity_id",
            "source",
            "match_type"
        ]
    )


# ---------------------------------------------------------
# SOURCE 2
# ---------------------------------------------------------

s2_results = process_source(S2_FILE, "source2")

print("\nSaving Source2 results...")
s2_results.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print("Source2 results saved.")

del s2_results
gc.collect()


# ---------------------------------------------------------
# SOURCE 3
# ---------------------------------------------------------

s3_results = process_source(S3_FILE, "source3")

print("\nAppending Source3 results...")

s3_results.to_csv(
    OUTPUT,
    sep="\t",
    index=False,
    mode="a",
    header=False
)

print("Source3 results appended.")

del s3_results
del s1
gc.collect()


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("Output:", OUTPUT)

if os.path.exists(OUTPUT):
    size = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"File size: {size:.2f} MB")