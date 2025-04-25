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
