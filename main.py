from core.trader import Trader
from strategy.simple_rsi import SimpleRSIStrategy
import time

if __name__ == "__main__":
    strategy = SimpleRSIStrategy(period=14)
    trader = Trader(strategy)

    while True:
        try:
            #trader.run()
            print("test")
        except Exception as e:
            print(f"에러 발생: {e}")
        time.sleep(60)
