import pandas as pd

quotes = pd.read_excel("./data/Telepass.xlsx", sheet_name="Insurance Quotes")
transactions = pd.read_excel("./data/Telepass.xlsx", sheet_name="Transactions")
