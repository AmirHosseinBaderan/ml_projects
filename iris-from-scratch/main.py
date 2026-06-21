import pandas as pd

df = pd.read_csv("./data/iris.csv")

print(df.head())

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())