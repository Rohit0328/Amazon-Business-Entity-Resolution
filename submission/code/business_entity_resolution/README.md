@"
# Amazon ML Challenge 2026
## Business Entity Resolution

This project implements a model-based business entity resolution pipeline for matching business records across multiple source datasets.

## Pipeline

1. Source data preprocessing and normalization.
2. Candidate generation using blocking strategies.
3. Candidate-pair feature engineering.
4. XGBoost binary classification for entity matching.
5. Model-based candidate scoring.
6. Threshold-based prediction.
7. Generation of `matching_results.tsv` and `candidate_pairs.tsv`.

## Features

The matching model uses features based on:

- Business name similarity
- Business address similarity
- Country agreement
- Exact name matching
- Exact address matching
- Name and address length differences
- Word-count differences
- Prefix and suffix comparisons
- Length ratios
- Candidate blocking/anchor indicators

## Model

The final matching model is an XGBoost binary classification model trained on generated candidate pairs.

The trained model is included in:

`code/business_entity_resolution/xgboost_entity_match_model.joblib`

## Output

### matching_results.tsv

Contains the predicted matched entity IDs for every Source 1 entity.

### candidate_pairs.tsv

Contains the candidate entity set generated for every Source 1 entity before final matching decisions.

## Validation

The official ML Challenge 2026 submission validator was executed successfully with:

`PASS — no blocking issues found.`

## Project Structure

```text
submission/
├── README.md
├── requirements.txt
├── Documentation_template.md
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── code/
    └── business_entity_resolution/
        ├── xgboost_entity_match_model.joblib
        └── src/
            ├── candidate_generation.py
            ├── data_cleaning.py
            ├── feature_engineering_v2.py
            ├── model_training.py
            └── submission.py