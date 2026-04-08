import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns.tolist())
print(repr(df.columns[0]))
print(repr(df.columns[1]))