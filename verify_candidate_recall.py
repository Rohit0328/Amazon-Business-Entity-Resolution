import pandas as pd
import csv
import time

S1_FILE = "processed_data/train_source1_processed.tsv"
GT_FILE = "processed_data/train_ground_truth_processed.tsv"
CANDIDATE_FILE = "processed_data/candidate_pairs_final_train.tsv"

VALIDATION_LIMIT = 10_000
CHUNK_SIZE = 250_000

print("=" * 80)
print("FINAL CANDIDATE RECALL VALIDATION")
print("=" * 80)

start = time.time()

# ============================================================
# 1. EXACT VALIDATION S1 POPULATION
# ============================================================

print("\nLoading EXACT validation population from Source1...")

s1_ids = []

with open(
    S1_FILE,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for i, row in enumerate(reader):

        if i >= VALIDATION_LIMIT:
            break

        s1_ids.append(str(row["entity_id"]).strip())

validation_s1 = set(s1_ids)

print("Validation S1 entities:", len(validation_s1))


# ============================================================
# 2. GROUND TRUTH
# ============================================================

print("\nReading ground truth...")

true_pairs = set()
true_by_s1 = {}

for chunk in pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE,
    keep_default_na=False
):

    chunk["source1_entity_id"] = (
        chunk["source1_entity_id"].str.strip()
    )

    subset = chunk[
        chunk["source1_entity_id"].isin(validation_s1)
    ][
        ["source1_entity_id", "matched_entity_ids"]
    ]

    for s1_id, raw_matches in zip(
        subset["source1_entity_id"],
        subset["matched_entity_ids"]
    ):

        if not raw_matches:
            matches = set()
        else:
            matches = {
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            }

        true_by_s1[s1_id] = matches

        for entity_id in matches:
            true_pairs.add(
                (s1_id, entity_id)
            )


# Ensure every S1 exists
for s1_id in validation_s1:
    if s1_id not in true_by_s1:
        true_by_s1[s1_id] = set()


print("S1 with true matches:",
      sum(bool(x) for x in true_by_s1.values()))

print("True match pairs:",
      len(true_pairs))


# ============================================================
# 3. SCAN CANDIDATES
# ============================================================

print("\nScanning candidate pairs...")

found_pairs = set()
candidate_pair_count = 0
candidate_s1 = set()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ),
    start=1
):

    chunk["source1_entity_id"] = (
        chunk["source1_entity_id"].str.strip()
    )

    chunk["candidate_entity_id"] = (
        chunk["candidate_entity_id"].str.strip()
    )

    subset = chunk[
        chunk["source1_entity_id"].isin(validation_s1)
    ]

    if len(subset) == 0:
        continue

    candidate_pair_count += len(subset)

    candidate_s1.update(
        subset["source1_entity_id"].unique()
    )

    # Convert only this small chunk to pairs
    pairs = zip(
        subset["source1_entity_id"].values,
        subset["candidate_entity_id"].values
    )

    for pair in pairs:

        if pair in true_pairs:
            found_pairs.add(pair)

    if chunk_number % 4 == 0:

        print(
            f"  processed ~"
            f"{chunk_number * CHUNK_SIZE:,} "
            f"candidate rows | "
            f"true matches found: "
            f"{len(found_pairs):,}"
        )


# ============================================================
# 4. RESULTS
# ============================================================

missed_pairs = true_pairs - found_pairs

s1_with_all = 0
s1_with_missed = 0

missed_examples = []

for s1_id in s1_ids:

    truth = true_by_s1[s1_id]

    if not truth:
        continue

    found = {
        entity_id
        for sid, entity_id in found_pairs
        if sid == s1_id
    }

    missed = truth - found

    if not missed:
        s1_with_all += 1
    else:
        s1_with_missed += 1

        if len(missed_examples) < 20:
            missed_examples.append(
                (s1_id, sorted(truth), sorted(missed))
            )


recall = (
    len(found_pairs) / len(true_pairs)
    if true_pairs
    else 0
)

elapsed = (time.time() - start) / 60


print("\n")
print("=" * 80)
print("RECALL RESULT")
print("=" * 80)

print(
    f"Source1 checked          : {len(validation_s1):,}"
)

print(
    f"S1 with true matches     : "
    f"{sum(bool(x) for x in true_by_s1.values()):,}"
)

print(
    f"S1 without true matches  : "
    f"{sum(not x for x in true_by_s1.values()):,}"
)

print(
    f"True match pairs         : "
    f"{len(true_pairs):,}"
)

print(
    f"Found in candidates      : "
    f"{len(found_pairs):,}"
)

print(
    f"Missed true pairs        : "
    f"{len(missed_pairs):,}"
)

print(
    f"Candidate recall         : "
    f"{recall:.4%}"
)

print(
    f"S1 with all matches found: "
    f"{s1_with_all:,}"
)

print(
    f"S1 with missed matches   : "
    f"{s1_with_missed:,}"
)

print(
    f"Candidate pairs checked  : "
    f"{candidate_pair_count:,}"
)

print(
    f"Average candidates/S1    : "
    f"{candidate_pair_count / len(validation_s1):.2f}"
)

print(
    f"Time                     : "
    f"{elapsed:.2f} minutes"
)


# ============================================================
# MISSES
# ============================================================

print("\n")
print("=" * 80)
print("FIRST MISSED MATCHES")
print("=" * 80)

for s1_id, truth, missed in missed_examples:

    print("\nS1:", s1_id)
    print("True IDs:", truth)
    print("Missed:", missed)


print("\n")
print("=" * 80)

if recall >= 0.95:
    print("STATUS: STRONG CANDIDATE RECALL")

elif recall >= 0.90:
    print("STATUS: USABLE BUT NEEDS REVIEW")

else:
    print("STATUS: CANDIDATE RECALL TOO LOW")

print("=" * 80)