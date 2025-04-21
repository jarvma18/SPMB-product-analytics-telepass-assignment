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

def merge_quotes_with_behavior_features(
  customer_quotes: pd.DataFrame, customer_behavior: pd.DataFrame
) -> pd.DataFrame:
  merged_customer_data: pd.DataFrame = pd.merge(
    customer_quotes, customer_behavior, on='client_id', how='left'
  )

  numeric_feature_columns: pd.Index = merged_customer_data.select_dtypes(include='number').columns
  merged_customer_data[numeric_feature_columns] = merged_customer_data[numeric_feature_columns].fillna(0)

  return merged_customer_data

def encode_categorical_columns(dataset: pd.DataFrame) -> pd.DataFrame:
  non_id_columns = dataset.drop(columns=["client_id", "issued"])
  categorical_columns = non_id_columns.select_dtypes(include="object").columns

  encoded_dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
  return encoded_dataset

def split_features_and_target(encoded_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  feature_matrix: pd.DataFrame = encoded_data.drop(columns=["client_id", "issued"])
  target_vector: pd.Series = encoded_data["issued"]
  return feature_matrix, target_vector

def create_train_test_sets(
  features: pd.DataFrame, target: pd.Series, test_ratio: float = 0.2, random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  return train_test_split(
      features, target, test_size=test_ratio, stratify=target, random_state=random_seed
  )

transactions_data, quotes_data = load_excel_sheets('./data/Telepass.xlsx')
customer_behavior_features = generate_customer_behavior_features(transactions_data)
final_dataset = merge_quotes_with_behavior_features(quotes_data, customer_behavior_features)
encoded_dataset = encode_categorical_columns(final_dataset)
X, y = split_features_and_target(encoded_dataset)
X_train, X_test, y_train, y_test = create_train_test_sets(X, y)

print(y.value_counts(normalize=True))
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")