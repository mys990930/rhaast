from config import *

class Trader:
    def __init__(self, strategy, symbol="BTC/USDT"):
        self.symbol = symbol
        self.api = BinanceAPI()
        self.strategy = strategy

    def run(self):
        candles = self.api.get_ohlcv(self.symbol, timeframe="1m", limit=100)
        signal = self.strategy.generate_signal(candles)
        price = candles[-1][4]

        if signal == "BUY":
            usdt_balance = self.api.get_balance("USDT")
            stop_loss = self.strategy.get_stop_loss(candles)
            risk_amount = usdt_balance * RISK_PER_TRADE
            position_size = risk_amount / (price - stop_loss)
            position_value = position_size * price

            if usdt_balance > position_value:
                print(f"BUY 실행 | 포지션 가치: {position_value:.2f}")
                self.api.create_market_order(self.symbol, "buy", position_size)
        elif signal == "SELL":
            btc_balance = self.api.get_balance("BTC")
            if btc_balance > 0.0001:
                print("SELL 실행")
                self.api.create_market_order(self.symbol, "sell", btc_balance)
        else:
            print("신호 없음")