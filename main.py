import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

def load_excel_sheets(filepath: str) -> tuple[DataFrame, DataFrame]:
  transactions: DataFrame = pd.read_excel(filepath, sheet_name='Transactions')
  insurance_quotes: DataFrame = pd.read_excel(filepath, sheet_name='Insurance Quotes')
  return transactions, insurance_quotes

def generate_customer_behavior_features(transactions: DataFrame) -> DataFrame:
  spending_per_service: DataFrame = transactions.pivot_table(
    index='client_id',
    columns='service_type',
    values='expenditures',
    aggfunc='sum',
    fill_value=0
  ).add_prefix('spend_')

  usage_count_per_service: DataFrame = transactions.pivot_table(
    index='client_id',
    columns='service_type',
    values='number_transactions',
    aggfunc='sum',
    fill_value=0
  ).add_prefix('usage_count_')

  overall_behavior: DataFrame = transactions.groupby('client_id').agg({
    'number_transactions': 'sum',
    'expenditures': 'sum',
    'telepass_pay': 'mean',
    'service_type': pd.Series.nunique
  }).rename(columns={
    'number_transactions': 'total_transaction_count',
    'expenditures': 'total_expenditure',
    'telepass_pay': 'average_telepass_pay_usage',
    'service_type': 'distinct_services_used'
  })

  combined_behavior_features: DataFrame = overall_behavior.join([spending_per_service, usage_count_per_service])
  return combined_behavior_features

print("Unique clients in quotes:", quotes_df['client_id'].nunique())
print("Unique clients in transactions:", transactions_df['client_id'].nunique())