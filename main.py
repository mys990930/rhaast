import torch
import ccxt
from datetime import datetime, timezone
import pandas as pd
import time
import os
from dotenv import load_dotenv
from settings import binance_session

eth_ohlcv = binance_session.fetch_ohlcv(
        symbol='ETH/USDT',
        timeframe='15m',
        since=int(time.mktime(datetime.strptime("2025-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timetuple()))*1000,
        limit=2000
    )

df = pd.DataFrame(eth_ohlcv, columns= ['time', 'open', 'high', 'low', 'close', 'volume'])
print(df.loc[0, "time"])
df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
print(df.loc[0, "time"])
df.set_index('time', inplace=True)

#print(df)