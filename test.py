import pandas as pd
from backtester import *
from strategy.simple_rsi import *
from data import *

# symbol = "SOLUSDT"
# df = pd.read_csv(f"dataset/ohlcv/{symbol}.csv")
# df = df.drop("time", axis=1)
# print(df)
strategy = SimpleRSIStrategy()

candles = get_data("ETHUSDT", "15m")
backtester = Backtester(strategy, candles)
backtester.run()
