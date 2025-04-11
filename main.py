import torch
import ccxt
from datetime import datetime, timezone
import pandas as pd
import time
from settings import binance_session
from data import *

tickers = binance_session.fetch_tickers()
ticker_names = []
for key in tickers.keys():
    ticker_names.append(key.split(":")[0])

for ticker_name in ticker_names:
    try:
        save_data(ticker_name, "15m")
    except Exception as e:
        print("error: ", e)
    finally:
        continue

#print(df)
