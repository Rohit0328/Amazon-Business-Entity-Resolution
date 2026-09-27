import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading training features...")

data = pd.read_csv(
    "processed_data/training_features_v2.csv"
)

print("Dataset shape:", data.shape)


# ==========================================
# 2. X AND Y
# ==========================================

X = data.drop(columns=["label"])
y = data["label"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nClass distribution:")
print(y.value_counts())


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ==========================================
# 4. XGBOOST
# ==========================================

print("\nTraining XGBoost...")

positive = (y_train == 1).sum()
negative = (y_train == 0).sum()

scale_pos_weight = negative / positive

print("Scale pos weight:", scale_pos_weight)


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)

print("\nXGBoost training completed!")


# ==========================================
# 5. PREDICT
# ==========================================

print("\nGenerating probabilities...")

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 6. THRESHOLD SEARCH
# ==========================================

print("\n================================")
print("THRESHOLD SEARCH")
print("================================")

best_threshold = 0.50
best_f05 = 0.0

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_test,
        predictions,
        beta=0.5,
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F0.5 {f05:.4f}"
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# ==========================================
# 7. SAVE MODEL
# ==========================================

print("\n================================")
print("SAVING MODEL")
print("================================")

joblib.dump(
    model,
    "xgboost_entity_match_model.joblib"
)

with open(
    "best_threshold.txt",
    "w"
) as f:

    f.write(str(best_threshold))


print(
    "Model saved: "
    "xgboost_entity_match_model.joblib"
)

print(
    "Threshold saved:",
    best_threshold
)


# ==========================================
# 8. FINAL RESULT
# ==========================================

print("\n================================")
print("FINAL MODEL RESULT")
print("================================")

print("Best threshold:", best_threshold)
print("Best F0.5:", best_f05)


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)

print("\nTraining complete.")