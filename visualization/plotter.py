import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

class Plotter:
    @staticmethod
    def plot_candles_with_trades(candles, trades=None):
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        apds = []
        if trades:
            trade_times = [pd.to_datetime(t[0], unit='ms') for t in trades]
            trade_prices = [t[2] for t in trades]
            trade_colors = ['g' if t[1] == 'BUY' else 'r' for t in trades]
            apds.append(mpf.make_addplot(trade_prices, scatter=True, markersize=100, marker='^', color=trade_colors, panel=0))

        mpf.plot(df, type='candle', addplot=apds, style='charles', volume=True)