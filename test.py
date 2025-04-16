import pandas as pd

symbol = "SOLUSDT"
df = pd.read_csv(f"dataset/ohlcv/{symbol}.csv")
df = df.drop("time", axis=1)
print(df)