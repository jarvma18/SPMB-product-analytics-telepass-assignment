import pandas as pd

transactions_df = pd.read_excel('./data/Telepass.xlsx', sheet_name='Transactions')
quotes_df = pd.read_excel('./data/Telepass.xlsx', sheet_name='Insurance Quotes')

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