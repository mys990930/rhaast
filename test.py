import time
from datetime import datetime, timezone
from settings import binance_session
from pprint import pprint

# start_time = int(time.mktime(
#     datetime.strptime("2010-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ").timetuple()) + 32400) * 1000  # UTC+09:00 = 32400s
# ohlcv = binance_session.fetch_ohlcv(
#             symbol="ETH/USDT",
#             timeframe="1d",
#             since=start_time,
#             limit=1000
#         )

print(str(datetime.fromtimestamp(1574812800000/1000)))