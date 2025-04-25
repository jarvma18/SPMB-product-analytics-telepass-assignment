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
