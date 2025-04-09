import time
from datetime import datetime
import pandas as pd

from settings import binance_session

def save_data(symbol: str, timeframe: str):
    ohlcv = binance_session.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=int(time.mktime(
            datetime.strptime("2020-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ").timetuple()) + 32400) * 1000, #UTC+09:00 = 32400s
        limit=2000
    )
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    print(df.loc[0, "time"])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    print(df.loc[0, "time"])
    df.set_index('time', inplace=True)
    return

save_data("ETH/USDT", "15m")
