import time
from datetime import datetime
import pandas as pd

from settings import binance_session

def save_all_data(timeframe: str): #15m, 1h, 5m ...
    tickers = binance_session.fetch_tickers()
    ticker_names = []
    for key in tickers.keys():
        ticker_names.append(key.split(":")[0])

    for ticker_name in ticker_names:
        try:
            save_data(ticker_name, timeframe)
        except Exception as e:
            print("error: ", e)
        finally:
            continue


def save_data(symbol: str, timeframe: str) -> pd.DataFrame:
    print("symbol name: ", symbol)
    df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    start_time = int(time.mktime(
            datetime.strptime("2019-12-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ").timetuple()) + 32400) * 1000 #UTC+09:00 = 32400s
    while start_time <= int(time.mktime(datetime.now().timetuple()) - 6000)*1000:
        ohlcv = binance_session.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_time,
            limit=1001
        )
        new_df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        new_df = new_df.iloc[1:]
        df = pd.concat([df, new_df], axis=0, ignore_index=True)
        last_time = df.iloc[-1, 0]
        start_time = last_time
        time.sleep(0.2)
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    print("initial data at: ", df.iloc[0, 0])
    print("last data at:", df.iloc[-1, 0])
    symbol = symbol.replace("/", "")
    df.to_csv(f"dataset/ohlcv/{symbol}.csv")
    return df

def slice_data(data: pd.DataFrame, length: int) -> pd.DataFrame:
    #slices data to a sequence of length
    data = 
    return

