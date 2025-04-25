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

quotes = pd.read_excel("./data/Telepass.xlsx", sheet_name="Insurance Quotes")
transactions = pd.read_excel("./data/Telepass.xlsx", sheet_name="Transactions")

# Pivot transaction data: aggregates sums per service_type
tx_pivot = transactions.pivot_table(
    index="client_id",
    columns="service_type",
    values=["number_transactions", "expenditures"],
    aggfunc="sum",
    fill_value=0
)

# Flatten the column names (e.g., expenditures_TelepassPay, number_transactions_TelepassPay)
tx_pivot.columns = [f"{stat}_{stype}" for stat, stype in tx_pivot.columns]
tx_pivot = tx_pivot.reset_index()

# Merge pivoted transaction features into quotes
df = quotes.merge(tx_pivot, on="client_id", how="left")

# Define target
target = "issued"

# Categorical and Numeric Features
categorical = [
  "driving_type", "gender", "county", "car_brand", "car_model", "base_type", "operating_system"
]

# Dynamic detection of pivoted service features
pivot_features = [col for col in df.columns if col.startswith("number_transactions_") or col.startswith("expenditures_")]

numeric = [
  "roadside_assistance", "driver_injury", "basic_coverage", "legal_protection",
  "waive_right_compensation", "uninsured_vehicles", "protected_bonus", "windows",
  "natural_events", "theft_fire", "kasko", "license_revoked", "collision", "vandalism",
  "key_loss", "price_sale", "price_full", "discount_percent"
] + pivot_features

# Prepare features
df = df[df[target].notna()]
X = df[categorical + numeric]
y = df[target].astype(int)

# FIX: Make categorical columns string
for col in categorical:
  X.loc[:, col] = X[col].astype(str)

# Preprocessing
categorical_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
  ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="median")),
  ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
  ("num", numeric_transformer, numeric),
  ("cat", categorical_transformer, categorical)
])

# Full pipeline
model = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42, stratify=y
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
