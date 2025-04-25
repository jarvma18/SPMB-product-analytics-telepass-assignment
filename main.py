import pandas as pd

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

# Target
target = "issued"

# Select features
categorical = [
  "driving_type", "gender", "county", "car_brand", "car_model", "base_type", "operating_system"
]

numeric = [
  "roadside_assistance", "driver_injury", "basic_coverage", "legal_protection",
  "waive_right_compensation", "uninsured_vehicles", "protected_bonus", "windows", 
  "natural_events", "theft_fire", "kasko", "license_revoked", "collision", "vandalism", 
  "key_loss", "price_sale", "price_full", "discount_percent",
  "number_transactions_OBU", "number_transactions_TelepassPay",
  "expenditures_OBU", "expenditures_TelepassPay"
]

# Filter null target rows
df = df[df[target].notna()]

X = df[categorical + numeric]
y = df[target].astype(int)