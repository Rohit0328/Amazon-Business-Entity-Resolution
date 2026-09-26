import duckdb
import time

S1_FILE = "processed_data/train_source1_processed.tsv"
GT_FILE = "processed_data/train_ground_truth_processed.tsv"
CANDIDATE_FILE = "processed_data/candidate_pairs_s2_new.tsv"

VALIDATION_LIMIT = 10000

start = time.time()

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")

print("=" * 80)
print("FINAL CANDIDATE RECALL CHECK")
print("=" * 80)

# ------------------------------------------------------------
# Validation Source1
# ------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE TEMP TABLE validation_s1 AS

SELECT entity_id AS source1_entity_id

FROM read_csv(
    '{S1_FILE}',
    delim='\t',
    header=true,
    auto_detect=true
)

LIMIT {VALIDATION_LIMIT};
""")


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

print("\nLoading ground truth...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE true_pairs AS

SELECT DISTINCT

    g.source1_entity_id,
    TRIM(t.s2_id) AS candidate_entity_id

FROM read_csv(
    '{GT_FILE}',
    delim='\t',
    header=true,
    auto_detect=true
) g

JOIN validation_s1 v
    ON g.source1_entity_id = v.source1_entity_id

CROSS JOIN UNNEST(
    string_split(
        COALESCE(g.matched_entity_ids, ''),
        ','
    )
) AS t(s2_id)

WHERE
    TRIM(t.s2_id) LIKE 'S2-%'
    AND TRIM(t.s2_id) <> '';
""")


true_count = con.execute("""
SELECT COUNT(*)
FROM true_pairs;
""").fetchone()[0]


# ------------------------------------------------------------
# Candidates
# ------------------------------------------------------------

print("Loading new candidates...")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE candidates AS

SELECT DISTINCT

    c.source1_entity_id,
    c.candidate_entity_id

FROM read_csv(
    '{CANDIDATE_FILE}',
    delim='\t',
    header=true,
    auto_detect=true
) c

JOIN validation_s1 v
    ON c.source1_entity_id = v.source1_entity_id;
""")


candidate_count = con.execute("""
SELECT COUNT(*)
FROM candidates;
""").fetchone()[0]


# ------------------------------------------------------------
# Recall
# ------------------------------------------------------------

found_count = con.execute("""
SELECT COUNT(*)

FROM true_pairs t

JOIN candidates c

    ON t.source1_entity_id =
       c.source1_entity_id

    AND t.candidate_entity_id =
        c.candidate_entity_id;
""").fetchone()[0]


missed_count = true_count - found_count

recall = (
    found_count / true_count * 100
    if true_count > 0
    else 0
)

avg_candidates = candidate_count / VALIDATION_LIMIT


print("\n")
print("=" * 80)
print("RESULT")
print("=" * 80)

print(f"Validation Source1 : {VALIDATION_LIMIT:,}")
print(f"True S2 pairs      : {true_count:,}")
print(f"Candidate pairs    : {candidate_count:,}")
print(f"Found true pairs   : {found_count:,}")
print(f"Missed true pairs  : {missed_count:,}")
print(f"Recall             : {recall:.4f}%")
print(f"Avg candidates/S1  : {avg_candidates:,.2f}")

print("\nTime:")
print(f"{(time.time() - start) / 60:.2f} minutes")

print("=" * 80)

con.close()