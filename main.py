import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

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
  confusion_matrix,
  classification_report,
  roc_curve
)

# Load data
quotes: pd.DataFrame = pd.read_excel("./data/Telepass.xlsx", sheet_name="Insurance Quotes")
transactions: pd.DataFrame = pd.read_excel("./data/Telepass.xlsx", sheet_name="Transactions")

# Pivot transaction data
tx_pivot: pd.DataFrame = transactions.pivot_table(
  index="client_id",
  columns="service_type",
  values=["number_transactions", "expenditures"],
  aggfunc="sum",
  fill_value=0
)

# Flatten pivoted columns
tx_pivot.columns = [f"{stat}_{stype}" for stat, stype in tx_pivot.columns]
tx_pivot = tx_pivot.reset_index()

# Merge datasets
df: pd.DataFrame = quotes.merge(tx_pivot, on="client_id", how="left")

# Target
target: str = "issued"

# Features
categorical: List[str] = [
  "driving_type", "gender", "county", "car_brand", "car_model", "base_type", "operating_system"
]

pivot_features: List[str] = [
  col for col in df.columns if col.startswith("number_transactions_") or col.startswith("expenditures_")
]

numeric: List[str] = [
  "roadside_assistance", "driver_injury", "basic_coverage", "legal_protection",
  "waive_right_compensation", "uninsured_vehicles", "protected_bonus", "windows",
  "natural_events", "theft_fire", "kasko", "license_revoked", "collision", "vandalism",
  "key_loss", "price_sale", "price_full", "discount_percent"
] + pivot_features

# Filter and prepare features
df = df[df[target].notna()]
X: pd.DataFrame = df[categorical + numeric].copy()
y: pd.Series = df[target].astype(int)

# Ensure categorical columns are strings
X[categorical] = X[categorical].astype(str)

# Preprocessing pipelines
categorical_transformer: Pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
  ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer: Pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="median")),
  ("scaler", StandardScaler())
])

preprocessor: ColumnTransformer = ColumnTransformer([
  ("num", numeric_transformer, numeric),
  ("cat", categorical_transformer, categorical)
])

# Model pipeline
model: Pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression
model.fit(X_train, y_train)

# Predictions
y_pred: np.ndarray = model.predict(X_test)
y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
fpr: np.ndarray
tpr: np.ndarray
thresholds: np.ndarray
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# --- Decision Tree

# Tree Preprocessing (no scaling)
categorical_transformer_tree: Pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
  ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer_tree: Pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="median"))
])

preprocessor_tree: ColumnTransformer = ColumnTransformer([
  ("num", numeric_transformer_tree, numeric),
  ("cat", categorical_transformer_tree, categorical)
])

# Decision Tree Model
tree_model: Pipeline = Pipeline([
  ("preprocessor", preprocessor_tree),
  ("classifier", DecisionTreeClassifier(
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
  ))
])

# Train
tree_model.fit(X_train, y_train)

# Predictions
y_pred_tree: np.ndarray = tree_model.predict(X_test)
y_proba_tree: np.ndarray = tree_model.predict_proba(X_test)[:, 1]

# Metrics
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree ROC AUC:", roc_auc_score(y_test, y_proba_tree))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

# ROC Curve
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, y_proba_tree)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree, label="Decision Tree")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.grid()
plt.show()

# --- Random Forest

# RF Preprocessing (same as Decision Tree)
preprocessor_rf = preprocessor_tree

# Random Forest Model
rf_model: Pipeline = Pipeline([
  ("preprocessor", preprocessor_rf),
  ("classifier", RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
  ))
])

# Train
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf: np.ndarray = rf_model.predict(X_test)
y_proba_rf: np.ndarray = rf_model.predict_proba(X_test)[:, 1]

# Metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest ROC AUC:", roc_auc_score(y_test, y_proba_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ROC Curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid()
plt.show()