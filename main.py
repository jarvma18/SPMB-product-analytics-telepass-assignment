import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

def load_excel_sheets(filepath: str) -> tuple[DataFrame, DataFrame]:
  transactions: DataFrame = pd.read_excel(filepath, sheet_name='Transactions')
  insurance_quotes: DataFrame = pd.read_excel(filepath, sheet_name='Insurance Quotes')
  return transactions, insurance_quotes

transactions_df.head()
transactions_df.info()
transactions_df.describe()

quotes_df.head()
quotes_df.info()

print("Quotes shape:", quotes_df.shape)
print("Transactions shape:", transactions_df.shape)

print("Insurance Purchase Rate:")
print(quotes_df['issued'].value_counts(normalize=True))

print("Unique clients in quotes:", quotes_df['client_id'].nunique())
print("Unique clients in transactions:", transactions_df['client_id'].nunique())