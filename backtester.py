from config import *

class Backtester:
    def __init__(self, strategy, historical_candles):
        self.strategy = strategy
        self.candles = historical_candles
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []

    def run(self):
        for i in range(100, len(self.candles)):
            window = self.candles.iloc[i - 100:i]
            signal = self.strategy.generate_signal(window)
            price = window.iloc[-1]['close']  # 마지막 close 가격
            stop_loss = self.strategy.get_stop_loss(window)
            fee = FEE_RATE
            slip = SLIPPAGE
            price_with_slip = price * (1 + slip)

            if signal == "BUY" and self.balance > 0:
                risk_amount = self.balance * RISK_PER_TRADE
                size = risk_amount / (price_with_slip - stop_loss)
                cost = size * price_with_slip * (1 + fee)
                print("BUY: " + str(self.balance))
                if self.balance >= cost:
                    self.position = size
                    self.entry_price = price_with_slip
                    self.balance -= cost
                    self.trades.append((self.candles.iloc[i]['time'], "BUY", price_with_slip))
            elif signal == "SELL" and self.position > 0:
                sell_price = price * (1 - slip)
                proceeds = self.position * sell_price * (1 - fee)
                print("SELL: " + str(self.balance))
                self.balance += proceeds
                self.trades.append((self.candles.iloc[i]['time'], "SELL", sell_price))
                self.position = 0

        final_value = self.balance + self.position * price
        print(f"초기 자산: {self.initial_balance} → 최종 자산: {final_value:.2f}")

    def get_trades(self):
        return self.trades