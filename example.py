import pandas as pd

train_path = "/Users/martino.orioligmail.com/data/churn-prediction-25-26/train.parquet"
test_path  = "/Users/martino.orioligmail.com/data/churn-prediction-25-26/test.parquet"

train = pd.read_parquet(train_path)
test  = pd.read_parquet(test_path)

print(train.shape)
print(test.head())

print(train.columns)
print(test.columns)
