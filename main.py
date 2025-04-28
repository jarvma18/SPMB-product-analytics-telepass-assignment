import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
  accuracy_score,
  roc_auc_score,
  classification_report,
  confusion_matrix,
  roc_curve
)

from typing import Tuple

def print_model_evaluation_stats(name: str, y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
  print(f"\n{name} Accuracy:", accuracy_score(y_test, y_pred))
  print(f"{name} ROC AUC:", roc_auc_score(y_test, y_proba))
  print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))
  print(f"{name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def load_telepass_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
  quotes = pd.read_excel(filepath, sheet_name="Insurance Quotes")
  transactions = pd.read_excel(filepath, sheet_name="Transactions")
  return quotes, transactions

def preprocess_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
  tx_pivot = transactions.pivot_table(
    index="client_id",
    columns="service_type",
    values=["number_transactions", "expenditures"],
    aggfunc="sum",
    fill_value=0
  )
  tx_pivot.columns = [f"{stat}_{stype}" for stat, stype in tx_pivot.columns]
  return tx_pivot.reset_index()

def merge_quotes_and_transactions(quotes: pd.DataFrame, transactions_pivot: pd.DataFrame) -> pd.DataFrame:
  return quotes.merge(transactions_pivot, on="client_id", how="left")

def get_feature_lists(df: pd.DataFrame) -> Tuple[list[str], list[str], str]:
  target = "issued"
  categorical = [
    "driving_type", "gender", "county", "car_brand", "car_model",
    "base_type", "operating_system"
  ]
  pivot_features = [col for col in df.columns if col.startswith("number_transactions_") or col.startswith("expenditures_")]
  numeric = [
    "roadside_assistance", "driver_injury", "basic_coverage", "legal_protection",
    "waive_right_compensation", "uninsured_vehicles", "protected_bonus", "windows",
    "natural_events", "theft_fire", "kasko", "license_revoked", "collision", "vandalism",
    "key_loss", "price_sale", "price_full", "discount_percent"
  ] + pivot_features
  return categorical, numeric, target

def build_preprocessing_pipeline(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
  categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])
  numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
  ])
  return ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
  ])

quotes, transactions = load_telepass_data("./data/Telepass.xlsx")
tx_pivot = preprocess_transactions(transactions)
df = merge_quotes_and_transactions(quotes, tx_pivot)
categorical, numeric, target = get_feature_lists(df)

df = df[df[target].notna()]
X = df[categorical + numeric].copy()
y = df[target].astype(int)
X[categorical] = X[categorical].astype(str)

preprocessor = build_preprocessing_pipeline(categorical, numeric)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 1. Train Random Forest for Feature Importance ---
rf_model_full = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier", RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
  ))
])

rf_model_full.fit(X_train, y_train)

# Haetaan feature importances
importances = rf_model_full.named_steps["classifier"].feature_importances_
feature_names = rf_model_full.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(by="importance", ascending=False)

# Valitaan Top 40 featurea
top_features = importance_df.head(40)["feature"].tolist()

# --- 2. Build Reduced Dataset ---
X_train_transformed = rf_model_full.named_steps["preprocessor"].transform(X_train)
X_test_transformed = rf_model_full.named_steps["preprocessor"].transform(X_test)

X_train_reduced = pd.DataFrame(X_train_transformed.toarray(), columns=feature_names)[top_features]
X_test_reduced = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)[top_features]

# --- 3. Train Models with Reduced Features ---
# Logistic Regression
reduced_lr = LogisticRegression(max_iter=3000, class_weight="balanced")
reduced_lr.fit(X_train_reduced, y_train)
y_pred_lr = reduced_lr.predict(X_test_reduced)
y_proba_lr = reduced_lr.predict_proba(X_test_reduced)[:, 1]

# Decision Tree
reduced_dt = DecisionTreeClassifier(
  max_depth=12,
  min_samples_split=10,
  min_samples_leaf=5,
  random_state=42
)
reduced_dt.fit(X_train_reduced, y_train)
y_pred_dt = reduced_dt.predict(X_test_reduced)
y_proba_dt = reduced_dt.predict_proba(X_test_reduced)[:, 1]

# Random Forest
reduced_rf = RandomForestClassifier(
  n_estimators=300,
  max_depth=15,
  min_samples_split=10,
  min_samples_leaf=5,
  class_weight="balanced",
  random_state=42,
  n_jobs=-1
)
reduced_rf.fit(X_train_reduced, y_train)
y_pred_rf = reduced_rf.predict(X_test_reduced)
y_proba_rf = reduced_rf.predict_proba(X_test_reduced)[:, 1]

plt.figure(figsize=(10, 8))

print_model_evaluation_stats("Reduced Logistic Regression", y_test, y_pred_lr, y_proba_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")

print_model_evaluation_stats("Reduced Decision Tree", y_test, y_pred_dt, y_proba_dt)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
plt.plot(fpr_dt, tpr_dt, label="Decision Tree")

print_model_evaluation_stats("Reduced Random Forest", y_test, y_pred_rf, y_proba_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.plot(fpr_rf, tpr_rf, label="Random Forest")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Reduced Features")
plt.legend()
plt.grid()
plt.show()