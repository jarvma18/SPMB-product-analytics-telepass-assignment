import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from typing import List, Tuple

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

# Flatten the column names
tx_pivot.columns = [f"{stat}_{stype}" for stat, stype in tx_pivot.columns]
tx_pivot = tx_pivot.reset_index()

# Merge pivoted transaction features into quotes
df: pd.DataFrame = quotes.merge(tx_pivot, on="client_id", how="left")

# Define target
target: str = "issued"

# Categorical and numeric features
categorical: List[str] = [
  "driving_type", "gender", "county", "car_brand", "car_model", "base_type", "operating_system"
]

pivot_features: List[str] = [
  col for col in df.columns
  if col.startswith("number_transactions_") or col.startswith("expenditures_")
]

numeric: List[str] = [
  "roadside_assistance", "driver_injury", "basic_coverage", "legal_protection",
  "waive_right_compensation", "uninsured_vehicles", "protected_bonus", "windows",
  "natural_events", "theft_fire", "kasko", "license_revoked", "collision", "vandalism",
  "key_loss", "price_sale", "price_full", "discount_percent"
] + pivot_features

# Prepare features
df = df[df[target].notna()]
X: pd.DataFrame = df[categorical + numeric]
y: pd.Series = df[target].astype(int)

# Ensure categorical columns are strings
for col in categorical:
  X.loc[:, col] = X[col].astype(str)

# Preprocessing pipelines
categorical_transformer: Pipeline = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
  ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer: Pipeline = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="median")),
  ("scaler", StandardScaler())
])

preprocessor: ColumnTransformer = ColumnTransformer(transformers=[
  ("num", numeric_transformer, numeric),
  ("cat", categorical_transformer, categorical)
])

# Full model pipeline
model: Pipeline = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

# Train-test split
X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.Series
y_test: pd.Series
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred: pd.Series = model.predict(X_test)
y_proba: pd.Series = model.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
fpr: List[float]
tpr: List[float]
_: List[float]
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Decision tree start here

# Pipelines
categorical_transformer_tree: Pipeline = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
  ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer_tree: Pipeline = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="median"))
  # No scaler here!
])

preprocessor_tree: ColumnTransformer = ColumnTransformer(transformers=[
  ("num", numeric_transformer_tree, numeric),
  ("cat", categorical_transformer_tree, categorical)
])

# Decision Tree model
tree_model: Pipeline = Pipeline(steps=[
  ("preprocessor", preprocessor_tree),
  ("classifier", DecisionTreeClassifier(
    max_depth=12,          # A tuned starting point, avoids overfitting
    min_samples_split=10,  # Prevents splits on very tiny groups
    min_samples_leaf=5,    # Ensures minimum samples in leaves
    random_state=42
  ))
])

# Train-test split (already done earlier, reuse X_train, X_test, y_train, y_test)

# Train
tree_model.fit(X_train, y_train)

# Predictions
y_pred_tree: pd.Series = tree_model.predict(X_test)
y_proba_tree: pd.Series = tree_model.predict_proba(X_test)[:, 1]

# Metrics
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Decision Tree ROC AUC:", roc_auc_score(y_test, y_proba_tree))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

# ROC Curve
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree, label="Decision Tree (New)")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Decision Tree (Properly Tuned)")
plt.legend()
plt.grid()
plt.show()

# Random forest starts here

# Pipelines (same as Decision Tree for preprocessing)
categorical_transformer_rf: Pipeline = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
  ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer_rf: Pipeline = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="median"))
  # No scaler here!
])

preprocessor_rf: ColumnTransformer = ColumnTransformer(transformers=[
  ("num", numeric_transformer_rf, numeric),
  ("cat", categorical_transformer_rf, categorical)
])

# Random Forest model
rf_model: Pipeline = Pipeline(steps=[
  ("preprocessor", preprocessor_rf),
  ("classifier", RandomForestClassifier(
    n_estimators=100,      # 100 trees (good starting point)
    max_depth=15,          # Reasonable limit to avoid huge trees
    min_samples_split=10,  # To avoid overfitting on noise
    min_samples_leaf=5,    # Ensure reasonable leaf size
    class_weight="balanced",  # Balance classes!!
    random_state=42,
    n_jobs=-1              # Use all CPU cores
  ))
])

# Train-test split (reuse X_train, X_test, y_train, y_test)

# Train
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf: pd.Series = rf_model.predict(X_test)
y_proba_rf: pd.Series = rf_model.predict_proba(X_test)[:, 1]

# Metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest ROC AUC:", roc_auc_score(y_test, y_proba_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (New)")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Random Forest (Properly Tuned)")
plt.legend()
plt.grid()
plt.show()