import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = str(os.environ.get('apiKey'))
SECRET_KEY = str(os.environ.get('secret'))
FEE_RATE = 0.001  # 바이낸스 기준 기본 거래 수수료 0.1%
SLIPPAGE = 0.001  # 슬리피지 0.1%
RISK_PER_TRADE = 0.01  # 전체 자산 대비 한 트레이드당 리스크 1%

class BinanceAPI:
    def __init__(self):
        self.exchange = ccxt.binance(config={
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options':{
            'defaultType': 'future',
            'adjustForTimeDifference': True,
            }
        })

    def get_balance(self, asset):
        balance = self.exchange.fetch_balance()
        return balance["free"].get(asset, 0)

    def get_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def get_ohlcv(self, symbol, timeframe="1m", limit=100):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def create_market_order(self, symbol, side, amount):
        return self.exchange.create_market_order(symbol, side, amount)
