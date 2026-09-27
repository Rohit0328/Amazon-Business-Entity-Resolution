import csv
import gc
import os
import time
from collections import Counter, defaultdict

# ============================================================
# AMAZON ML CHALLENGE
# V-FINAL CANDIDATE GENERATION
# NO DUCKDB / NO SQLITE
# ============================================================

S1_FILE = "processed_data/train_source1_processed.tsv"
S2_FILE = "processed_data/train_source2_processed.tsv"
S3_FILE = "processed_data/train_source3_processed.tsv"

OUTPUT_FILE = "processed_data/candidate_pairs_final_train.tsv"

# ------------------------------------------------------------
# SAFETY
# ------------------------------------------------------------
# First run only 10,000 S1 records.
#
# After validation, change to:
#
# VALIDATION_LIMIT = None
#
# ------------------------------------------------------------

VALIDATION_LIMIT = 10000

# Only tokens appearing in <= this many source records
# are used as blocking anchors.
MAX_TOKEN_FREQUENCY = 500

# Number of rare tokens used from each field.
ANCHORS_PER_FIELD = 4

# Character-signature blocking
CHAR_SIGNATURE_LENGTH = 5
MAX_CHAR_SIGNATURE_FREQUENCY = 1000

START = time.time()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Unicode-safe normalization.

    Keeps letters/numbers from non-English languages.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    result = []

    for ch in value:
        if ch.isalnum():
            result.append(ch)
        else:
            result.append(" ")

    return " ".join("".join(result).split())


def normalize_country(value):

    if value is None:
        return ""

    return str(value).strip().lower()


def unique_tokens(text):

    if not text:
        return []

    return list(dict.fromkeys(text.split()))


def character_signatures(text):
    """Return bounded character prefix/suffix signatures."""
    if not text:
        return []

    compact = text.replace(" ", "")

    if len(compact) < CHAR_SIGNATURE_LENGTH:
        return [compact]

    prefix = compact[:CHAR_SIGNATURE_LENGTH]
    suffix = compact[-CHAR_SIGNATURE_LENGTH:]

    if prefix == suffix:
        return [prefix]

    return [prefix, suffix]


# ============================================================
# TSV READER
# ============================================================

def read_source(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            entity_id = row.get(
                "entity_id",
                ""
            ).strip()

            if not entity_id:
                continue

            country = normalize_country(
                row.get("country", "")
            )

            name = normalize_text(
                row.get(
                    "name_normalized",
                    row.get(
                        "business_name",
                        ""
                    )
                )
            )

            address = normalize_text(
                row.get(
                    "address_normalized",
                    row.get(
                        "business_address",
                        ""
                    )
                )
            )

            yield (
                entity_id,
                country,
                name,
                address
            )


# ============================================================
# SOURCE1 LOADER
# ============================================================

def load_source1():

    rows = []

    count = 0

    for row in read_source(S1_FILE):

        rows.append(row)

        count += 1

        if (
            VALIDATION_LIMIT is not None
            and count >= VALIDATION_LIMIT
        ):
            break

    return rows


# ============================================================
# BUILD TOKEN FREQUENCY
# ============================================================

def build_token_frequency(path, field_index):

    """
    First pass over one source.

    field_index:
        2 = name
        3 = address

    Returns Counter of:
        (country, token)
    """

    frequencies = Counter()

    count = 0

    print(
        f"    Building token frequencies..."
    )

    for entity_id, country, name, address in read_source(path):

        text = (
            name
            if field_index == 2
            else address
        )

        for token in unique_tokens(text):

            frequencies[
                (country, token)
            ] += 1

        count += 1

        if count % 500000 == 0:

            print(
                f"      processed "
                f"{count:,} rows"
            )

    print(
        f"    rows processed: {count:,}"
    )

    print(
        f"    unique tokens : "
        f"{len(frequencies):,}"
    )

    return frequencies


# ============================================================
# BUILD EXACT INDEX
# ============================================================

def build_exact_index(path, field_index):

    """
    Creates:

        (country, normalized_value)
              ->
        list of entity IDs
    """

    index = defaultdict(list)

    count = 0

    print(
        "    Building exact index..."
    )

    for entity_id, country, name, address in read_source(path):

        value = (
            name
            if field_index == 2
            else address
        )

        if value:

            index[
                (country, value)
            ].append(entity_id)

        count += 1

        if count % 500000 == 0:

            print(
                f"      processed "
                f"{count:,} rows"
            )

    print(
        f"    exact keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# BUILD RARE TOKEN INDEX
# ============================================================

def build_rare_token_index(
    path,
    field_index,
    frequencies
):

    """
    Only stores tokens whose frequency is <= threshold.

        (country, token)
              ->
        list of entity IDs
    """

    index = defaultdict(list)

    count = 0

    for entity_id, country, name, address in read_source(path):

        text = (
            name
            if field_index == 2
            else address
        )

        for token in unique_tokens(text):

            frequency = frequencies.get(
                (country, token),
                0
            )

            if (
                0 < frequency
                <= MAX_TOKEN_FREQUENCY
            ):

                index[
                    (country, token)
                ].append(entity_id)

        count += 1

        if count % 500000 == 0:

            print(
                f"      rare-index pass: "
                f"{count:,} rows"
            )

    print(
        f"    rare token keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# BUILD CHARACTER SIGNATURE INDEX
# ============================================================

def build_char_signature_index(path, field_index):
    """Build a bounded character prefix/suffix index."""

    frequencies = Counter()

    print("    Building character-signature frequencies...")

    count = 0

    for entity_id, country, name, address in read_source(path):

        text = name if field_index == 2 else address

        for signature in set(character_signatures(text)):
            if signature:
                frequencies[(country, signature)] += 1

        count += 1

        if count % 500000 == 0:
            print(f"      char frequency pass: {count:,} rows")

    print(f"    character signatures: {len(frequencies):,}")

    index = defaultdict(list)

    print("    Building character-signature index...")

    count = 0

    for entity_id, country, name, address in read_source(path):

        text = name if field_index == 2 else address

        for signature in set(character_signatures(text)):

            frequency = frequencies.get((country, signature), 0)

            if 0 < frequency <= MAX_CHAR_SIGNATURE_FREQUENCY:
                index[(country, signature)].append(entity_id)

        count += 1

        if count % 500000 == 0:
            print(f"      char index pass: {count:,} rows")

    print(f"    usable character keys: {len(index):,}")

    del frequencies
    gc.collect()

    return index


# ============================================================
# FIND SOURCE1 ANCHORS
# ============================================================

def choose_anchors(
    country,
    text,
    frequencies
):

    candidates = []

    for token in unique_tokens(text):

        frequency = frequencies.get(
            (country, token)
        )

        if frequency is None:
            continue

        if (
            0 < frequency
            <= MAX_TOKEN_FREQUENCY
        ):

            candidates.append(
                (
                    frequency,
                    token
                )
            )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    return [
        token
        for _, token in candidates[
            :ANCHORS_PER_FIELD
        ]
    ]


# ============================================================
# GENERATE CANDIDATES FOR SOURCE1
# ============================================================

def generate_for_source(
    source_path,
    source_name,
    s1_rows,
    writer
):

    print("\n" + "=" * 80)

    print(
        f"PROCESSING {source_name}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # NAME FREQUENCY
    # --------------------------------------------------------

    print(
        "\n[1/5] "
        "Building name token frequencies..."
    )

    name_frequency = build_token_frequency(
        source_path,
        2
    )

    # --------------------------------------------------------
    # ADDRESS FREQUENCY
    # --------------------------------------------------------

    print(
        "\n[2/5] "
        "Building address token frequencies..."
    )

    address_frequency = build_token_frequency(
        source_path,
        3
    )

    # --------------------------------------------------------
    # EXACT NAME
    # --------------------------------------------------------

    print(
        "\n[3/5] "
        "Building exact name index..."
    )

    exact_name = build_exact_index(
        source_path,
        2
    )

    # --------------------------------------------------------
    # EXACT ADDRESS
    # --------------------------------------------------------

    print(
        "\n[4/5] "
        "Building exact address index..."
    )

    exact_address = build_exact_index(
        source_path,
        3
    )

    # --------------------------------------------------------
    # RARE TOKEN INDEXES
    # --------------------------------------------------------

    print(
        "\n[5/5] "
        "Building rare-token indexes..."
    )

    rare_name = build_rare_token_index(
        source_path,
        2,
        name_frequency
    )

    rare_address = build_rare_token_index(
        source_path,
        3,
        address_frequency
    )

    # --------------------------------------------------------
    # CHARACTER SIGNATURE INDEXES
    # --------------------------------------------------------

    print("\n[6/7] Building character-signature indexes...")

    char_name = build_char_signature_index(source_path, 2)
    char_address = build_char_signature_index(source_path, 3)

    print("\nAll indexes ready.")

    print(f"Character names    : {len(char_name):,}")
    print(f"Character addresses: {len(char_address):,}")

    print(
        f"Exact names    : "
        f"{len(exact_name):,}"
    )

    print(
        f"Exact addresses: "
        f"{len(exact_address):,}"
    )

    print(
        f"Rare names     : "
        f"{len(rare_name):,}"
    )

    print(
        f"Rare addresses : "
        f"{len(rare_address):,}"
    )

    # --------------------------------------------------------
    # PROCESS SOURCE1
    # --------------------------------------------------------

    total_candidates = 0
    processed_s1 = 0

    for (
        s1_id,
        country,
        name,
        address
    ) in s1_rows:

        candidate_rules = defaultdict(set)

        # ----------------------------------------------------
        # EXACT NAME
        # ----------------------------------------------------

        if name:

            ids = exact_name.get(
                (country, name),
                []
            )

            for candidate_id in ids:

                candidate_rules[
                    candidate_id
                ].add("exact_name")

        # ----------------------------------------------------
        # EXACT ADDRESS
        # ----------------------------------------------------

        if address:

            ids = exact_address.get(
                (country, address),
                []
            )

            for candidate_id in ids:

                candidate_rules[
                    candidate_id
                ].add("exact_address")

        # ----------------------------------------------------
        # NAME ANCHORS
        # ----------------------------------------------------

        name_anchors = choose_anchors(
            country,
            name,
            name_frequency
        )

        for token in name_anchors:

            ids = rare_name.get(
                (country, token),
                []
            )

            for candidate_id in ids:

                candidate_rules[
                    candidate_id
                ].add("name_anchor")

        # ----------------------------------------------------
        # ADDRESS ANCHORS
        # ----------------------------------------------------

        address_anchors = choose_anchors(
            country,
            address,
            address_frequency
        )

        for token in address_anchors:

            ids = rare_address.get(
                (country, token),
                []
            )

            for candidate_id in ids:

                candidate_rules[
                    candidate_id
                ].add("address_anchor")

        # ----------------------------------------------------
        # CHARACTER NAME SIGNATURES
        # ----------------------------------------------------

        for signature in character_signatures(name):

            ids = char_name.get((country, signature), [])

            for candidate_id in ids:
                candidate_rules[candidate_id].add("char_name")

        # ----------------------------------------------------
        # CHARACTER ADDRESS SIGNATURES
        # ----------------------------------------------------

        for signature in character_signatures(address):

            ids = char_address.get((country, signature), [])

            for candidate_id in ids:
                candidate_rules[candidate_id].add("char_address")

        # ----------------------------------------------------
        # WRITE
        # ----------------------------------------------------

        for candidate_id, rules in (
            candidate_rules.items()
        ):

            writer.writerow([
                s1_id,
                candidate_id,
                ",".join(
                    sorted(rules)
                )
            ])

            total_candidates += 1

        processed_s1 += 1

        if processed_s1 % 500 == 0:

            elapsed = (
                time.time() - START
            ) / 60

            average = (
                total_candidates
                / processed_s1
            )

            print(
                f"  {source_name} | "
                f"S1 {processed_s1:,} | "
                f"candidates {total_candidates:,} | "
                f"avg/S1 {average:,.2f} | "
                f"time {elapsed:.1f} min"
            )

    # --------------------------------------------------------
    # FREE MEMORY
    # --------------------------------------------------------

    del name_frequency
    del address_frequency

    del exact_name
    del exact_address

    del rare_name
    del rare_address

    del char_name
    del char_address

    gc.collect()

    print(
        f"\n{source_name} complete."
    )

    print(
        f"Candidates generated: "
        f"{total_candidates:,}"
    )

    return total_candidates


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("AMAZON ML - V-FINAL CANDIDATE GENERATION")
print("=" * 80)

print("\nConfiguration")
print("----------------------------------------")

print(
    f"S1 validation limit : "
    f"{VALIDATION_LIMIT}"
)

print(
    f"Max token frequency : "
    f"{MAX_TOKEN_FREQUENCY}"
)

print(
    f"Anchors per field   : "
    f"{ANCHORS_PER_FIELD}"
)

# ============================================================
# LOAD S1 ONLY
# ============================================================

print(
    "\nLoading Source1..."
)

s1_rows = load_source1()

print(
    f"Source1 loaded: "
    f"{len(s1_rows):,}"
)

# ============================================================
# OUTPUT
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print(
        f"\nRemoving previous output:"
    )

    print(OUTPUT_FILE)

    os.remove(OUTPUT_FILE)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as output_file:

    writer = csv.writer(
        output_file,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_id",
        "blocking_rules"
    ])

    # --------------------------------------------------------
    # SOURCE2
    # --------------------------------------------------------

    s2_candidates = generate_for_source(
        S2_FILE,
        "SOURCE2",
        s1_rows,
        writer
    )

    # --------------------------------------------------------
    # SOURCE3
    # --------------------------------------------------------

    s3_candidates = generate_for_source(
        S3_FILE,
        "SOURCE3",
        s1_rows,
        writer
    )


# ============================================================
# RESULT
# ============================================================

total = (
    s2_candidates
    + s3_candidates
)

elapsed = (
    time.time() - START
) / 60

print("\n" + "=" * 80)
print("V-FINAL CANDIDATE GENERATION COMPLETE")
print("=" * 80)

print(
    f"Source1 records       : "
    f"{len(s1_rows):,}"
)

print(
    f"Source2 candidates    : "
    f"{s2_candidates:,}"
)

print(
    f"Source3 candidates    : "
    f"{s3_candidates:,}"
)

print(
    f"TOTAL candidates      : "
    f"{total:,}"
)

print(
    f"Average candidates/S1: "
    f"{total / len(s1_rows):,.2f}"
)

print(
    f"Output                : "
    f"{OUTPUT_FILE}"
)

print(
 n   f"Time                  : "
    f"{elapsed:.2f} minutes"
)

print("=" * 80)