import pandas as pd
from strategy.base_strategy import BaseStrategy

class SimpleRSIStrategy(BaseStrategy):
    def __init__(self, period=14):
        self.period = period

    def get_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, candles):
        prices = candles['close']  # 'close' 컬럼 사용
        rsi = self.get_rsi(prices).iloc[-1]
        #print(f"현재 RSI: {rsi:.2f}")
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        return "HOLD"

    def get_stop_loss(self, candles):
        low = candles.iloc[-10:]['low'].min()  # 최근 10분 저점
        return low
