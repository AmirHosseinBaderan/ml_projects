import pandas as pd

df = pd.read_csv("./data/iris.csv")

# Features , Label
X = df.drop(columns=["variety"])
y = df["variety"]