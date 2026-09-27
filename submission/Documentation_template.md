
# Documentation — Business Entity Resolution

## 1. Problem Statement

The objective is to resolve business entities from Source 1 against Source 2 and Source 3 and identify the corresponding entity IDs.

## 2. Data Preparation

The source datasets were processed using data cleaning and normalization of business names, addresses and country information.

The preprocessing pipeline handles missing values and creates normalized representations used during candidate generation and matching.

## 3. Candidate Generation

Candidate pairs were generated using blocking strategies based on normalized business names, normalized addresses and candidate anchors.

The purpose of blocking is to reduce the number of possible comparisons before applying the final matching model.

## 4. Feature Engineering

Candidate pairs were converted into matching features based on business name, address and country information.

The feature set includes:

- Name exact-match indicator
- Name length features
- Name word-count features
- Name prefix and suffix comparisons
- Name length ratio
- Address exact-match indicator
- Address length features
- Address word-count features
- Address prefix and suffix comparisons
- Address length ratio
- Country match
- Name anchor indicator
- Address anchor indicator
- Exact name and address indicators

## 5. Matching Model

An XGBoost binary classification model was trained to classify candidate pairs as matching or non-matching entities.

The trained model is included with the submission package.

## 6. Inference

During inference, candidate pairs are transformed using the same feature structure used during model training.

The XGBoost model assigns a match probability to each candidate pair. A decision threshold is then applied to determine the predicted matches.

## 7. Submission Outputs

The submission contains two output files:

### matching_results.tsv

Contains one row for every Source 1 entity and the predicted matched Source 2 and/or Source 3 entity IDs.

### candidate_pairs.tsv

Contains one row for every Source 1 entity and its generated candidate entity IDs.

## 8. Validation

The official submission validator was executed against the test data.

Validation result:

`PASS — no blocking issues found.`

The submission contains the required Source 1 rows in both output files.

## 9. Reproducibility

The submission package contains:

- Source code
- Trained XGBoost model
- Python dependency list
- Matching results
- Candidate pairs
- Project documentation

The Python dependencies are listed in `requirements.txt`.