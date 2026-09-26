import duckdb
import time
import os

# ============================================================
# CONFIG
# ============================================================

S1_FILE = "processed_data/train_source1_processed.tsv"
S2_FILE = "processed_data/train_source2_processed.tsv"

OUTPUT_FILE = "processed_data/candidate_pairs_s2_new.tsv"

# IMPORTANT:
# First run only 10,000 Source1 records.
# After validation succeeds, change to None for full dataset.
VALIDATION_LIMIT = 10000

# Number of rare token anchors used from each field
ANCHORS_PER_FIELD = 3

# A token is used as a blocking anchor only when it appears
# in at most this many Source2 records.
#
# This is NOT removing common words from matching.
# It only controls which tokens are used to retrieve candidates.
MAX_TOKEN_FREQUENCY = 2000

start = time.time()

os.makedirs("processed_data", exist_ok=True)

print("=" * 80)
print("AMAZON ML - CANDIDATE GENERATION")
print("=" * 80)

print("\nConfiguration:")
print(f"Validation limit       : {VALIDATION_LIMIT}")
print(f"Anchors per field      : {ANCHORS_PER_FIELD}")
print(f"Maximum token frequency: {MAX_TOKEN_FREQUENCY}")


# ============================================================
# CONNECT DUCKDB
# ============================================================

con = duckdb.connect("candidate_generation_final.duckdb")

con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\n[1/10] Loading Source1...")

limit_sql = (
    f"LIMIT {VALIDATION_LIMIT}"
    if VALIDATION_LIMIT is not None
    else ""
)

con.execute(f"""
CREATE OR REPLACE TABLE s1 AS

SELECT
    entity_id,
    country,
    name_normalized,
    address_normalized

FROM read_csv(
    '{S1_FILE}',
    delim='\t',
    header=true,
    auto_detect=true
)

{limit_sql};
""")

s1_count = con.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

print(f"Source1 records: {s1_count:,}")


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("\n[2/10] Loading Source2...")

con.execute(f"""
CREATE OR REPLACE TABLE s2 AS

SELECT
    entity_id,
    country,
    name_normalized,
    address_normalized

FROM read_csv(
    '{S2_FILE}',
    delim='\t',
    header=true,
    auto_detect=true
);
""")

s2_count = con.execute(
    "SELECT COUNT(*) FROM s2"
).fetchone()[0]

print(f"Source2 records: {s2_count:,}")


# ============================================================
# EXACT NAME BLOCK
# ============================================================

print("\n[3/10] Exact name blocking...")

con.execute("""
CREATE OR REPLACE TABLE exact_name_candidates AS

SELECT DISTINCT

    s1.entity_id AS source1_entity_id,
    s2.entity_id AS candidate_entity_id,
    'exact_name' AS blocking_rule

FROM s1

JOIN s2

    ON s1.country = s2.country

    AND s1.name_normalized <> ''

    AND s1.name_normalized =
        s2.name_normalized;
""")

exact_name_count = con.execute("""
SELECT COUNT(*)
FROM exact_name_candidates;
""").fetchone()[0]

print(
    f"Exact-name candidates: "
    f"{exact_name_count:,}"
)


# ============================================================
# EXACT ADDRESS BLOCK
# ============================================================

print("\n[4/10] Exact address blocking...")

con.execute("""
CREATE OR REPLACE TABLE exact_address_candidates AS

SELECT DISTINCT

    s1.entity_id AS source1_entity_id,
    s2.entity_id AS candidate_entity_id,
    'exact_address' AS blocking_rule

FROM s1

JOIN s2

    ON s1.country = s2.country

    AND s1.address_normalized <> ''

    AND s1.address_normalized =
        s2.address_normalized;
""")

exact_address_count = con.execute("""
SELECT COUNT(*)
FROM exact_address_candidates;
""").fetchone()[0]

print(
    f"Exact-address candidates: "
    f"{exact_address_count:,}"
)


# ============================================================
# SOURCE2 NAME TOKEN FREQUENCIES
# ============================================================

print("\n[5/10] Building Source2 name token frequencies...")

con.execute("""
CREATE OR REPLACE TABLE s2_name_tokens AS

SELECT DISTINCT

    s2.entity_id,
    s2.country,
    LOWER(TRIM(token)) AS token

FROM s2

CROSS JOIN UNNEST(
    string_split(
        COALESCE(s2.name_normalized, ''),
        ' '
    )
) AS t(token)

WHERE
    TRIM(token) <> '';
""")


con.execute("""
CREATE OR REPLACE TABLE s2_name_token_frequency AS

SELECT

    country,
    token,
    COUNT(*) AS frequency

FROM s2_name_tokens

GROUP BY
    country,
    token;
""")


# ============================================================
# SOURCE1 NAME TOKENS
# ============================================================

print("[6/10] Selecting rare name anchors...")

con.execute(f"""
CREATE OR REPLACE TABLE s1_name_tokens AS

SELECT DISTINCT

    s1.entity_id,
    s1.country,
    LOWER(TRIM(token)) AS token

FROM s1

CROSS JOIN UNNEST(
    string_split(
        COALESCE(s1.name_normalized, ''),
        ' '
    )
) AS t(token)

WHERE
    TRIM(token) <> '';
""")


con.execute(f"""
CREATE OR REPLACE TABLE s1_name_anchors AS

SELECT
    entity_id,
    country,
    token

FROM (

    SELECT

        s.entity_id,
        s.country,
        s.token,
        f.frequency,

        ROW_NUMBER() OVER (
            PARTITION BY s.entity_id
            ORDER BY f.frequency ASC, s.token
        ) AS rn

    FROM s1_name_tokens s

    JOIN s2_name_token_frequency f

        ON s.country = f.country
        AND s.token = f.token

    WHERE
        f.frequency <= {MAX_TOKEN_FREQUENCY}

)

WHERE rn <= {ANCHORS_PER_FIELD};
""")


name_anchor_count = con.execute("""
SELECT COUNT(*)
FROM s1_name_anchors;
""").fetchone()[0]

print(
    f"Source1 name anchors: "
    f"{name_anchor_count:,}"
)


# ============================================================
# NAME ANCHOR CANDIDATES
# ============================================================

print("\n[7/10] Generating name-anchor candidates...")

con.execute(f"""
CREATE OR REPLACE TABLE name_anchor_candidates AS

SELECT DISTINCT

    a.entity_id AS source1_entity_id,
    t.entity_id AS candidate_entity_id,
    'name_anchor' AS blocking_rule

FROM s1_name_anchors a

JOIN s2_name_tokens t

    ON a.country = t.country
    AND a.token = t.token

JOIN s2_name_token_frequency f

    ON t.country = f.country
    AND t.token = f.token

WHERE
    f.frequency <= {MAX_TOKEN_FREQUENCY};
""")

name_candidate_count = con.execute("""
SELECT COUNT(*)
FROM name_anchor_candidates;
""").fetchone()[0]

print(
    f"Name-anchor candidates: "
    f"{name_candidate_count:,}"
)


# ============================================================
# SOURCE2 ADDRESS TOKEN FREQUENCIES
# ============================================================

print("\n[8/10] Building Source2 address token frequencies...")

con.execute("""
CREATE OR REPLACE TABLE s2_address_tokens AS

SELECT DISTINCT

    s2.entity_id,
    s2.country,
    LOWER(TRIM(token)) AS token

FROM s2

CROSS JOIN UNNEST(
    string_split(
        COALESCE(s2.address_normalized, ''),
        ' '
    )
) AS t(token)

WHERE
    TRIM(token) <> '';
""")


con.execute("""
CREATE OR REPLACE TABLE s2_address_token_frequency AS

SELECT

    country,
    token,
    COUNT(*) AS frequency

FROM s2_address_tokens

GROUP BY
    country,
    token;
""")


# ============================================================
# SOURCE1 ADDRESS TOKENS
# ============================================================

print("[9/10] Selecting rare address anchors...")

con.execute(f"""
CREATE OR REPLACE TABLE s1_address_tokens AS

SELECT DISTINCT

    s1.entity_id,
    s1.country,
    LOWER(TRIM(token)) AS token

FROM s1

CROSS JOIN UNNEST(
    string_split(
        COALESCE(s1.address_normalized, ''),
        ' '
    )
) AS t(token)

WHERE
    TRIM(token) <> '';
""")


con.execute(f"""
CREATE OR REPLACE TABLE s1_address_anchors AS

SELECT
    entity_id,
    country,
    token

FROM (

    SELECT

        s.entity_id,
        s.country,
        s.token,
        f.frequency,

        ROW_NUMBER() OVER (
            PARTITION BY s.entity_id
            ORDER BY f.frequency ASC, s.token
        ) AS rn

    FROM s1_address_tokens s

    JOIN s2_address_token_frequency f

        ON s.country = f.country
        AND s.token = f.token

    WHERE
        f.frequency <= {MAX_TOKEN_FREQUENCY}

)

WHERE rn <= {ANCHORS_PER_FIELD};
""")


address_anchor_count = con.execute("""
SELECT COUNT(*)
FROM s1_address_anchors;
""").fetchone()[0]

print(
    f"Source1 address anchors: "
    f"{address_anchor_count:,}"
)


# ============================================================
# ADDRESS ANCHOR CANDIDATES
# ============================================================

print("\n[10/10] Generating address-anchor candidates...")

con.execute(f"""
CREATE OR REPLACE TABLE address_anchor_candidates AS

SELECT DISTINCT

    a.entity_id AS source1_entity_id,
    t.entity_id AS candidate_entity_id,
    'address_anchor' AS blocking_rule

FROM s1_address_anchors a

JOIN s2_address_tokens t

    ON a.country = t.country
    AND a.token = t.token

JOIN s2_address_token_frequency f

    ON t.country = f.country
    AND t.token = f.token

WHERE
    f.frequency <= {MAX_TOKEN_FREQUENCY};
""")


address_candidate_count = con.execute("""
SELECT COUNT(*)
FROM address_anchor_candidates;
""").fetchone()[0]

print(
    f"Address-anchor candidates: "
    f"{address_candidate_count:,}"
)


# ============================================================
# COMBINE EVERYTHING
# ============================================================

print("\nCombining candidate sources...")

con.execute("""
CREATE OR REPLACE TABLE final_candidates AS

SELECT
    source1_entity_id,
    candidate_entity_id,
    STRING_AGG(
        DISTINCT blocking_rule,
        ','
    ) AS blocking_rules

FROM (

    SELECT *
    FROM exact_name_candidates

    UNION ALL

    SELECT *
    FROM exact_address_candidates

    UNION ALL

    SELECT *
    FROM name_anchor_candidates

    UNION ALL

    SELECT *
    FROM address_anchor_candidates

)

GROUP BY

    source1_entity_id,
    candidate_entity_id;
""")


final_count = con.execute("""
SELECT COUNT(*)
FROM final_candidates;
""").fetchone()[0]


avg_candidates = (
    final_count / s1_count
    if s1_count
    else 0
)


print("\n")
print("=" * 80)
print("CANDIDATE GENERATION RESULT")
print("=" * 80)

print(
    f"Source1 records       : {s1_count:,}"
)

print(
    f"Source2 records       : {s2_count:,}"
)

print(
    f"Exact name            : {exact_name_count:,}"
)

print(
    f"Exact address         : {exact_address_count:,}"
)

print(
    f"Name anchor           : {name_candidate_count:,}"
)

print(
    f"Address anchor        : {address_candidate_count:,}"
)

print(
    f"FINAL candidates      : {final_count:,}"
)

print(
    f"Average candidates/S1 : {avg_candidates:,.2f}"
)


# ============================================================
# WRITE OUTPUT
# ============================================================

print("\nWriting candidate file...")

con.execute(f"""
COPY final_candidates
TO '{OUTPUT_FILE}'
(
    HEADER,
    DELIMITER '\t'
);
""")


elapsed = (time.time() - start) / 60

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Time: {elapsed:.2f} minutes"
)

print("=" * 80)

con.close()