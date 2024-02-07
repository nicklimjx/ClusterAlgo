import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, csv :str, start :int, interval :int) -> None:
        self.df = pd.read_csv(csv, delimiter=',')
        self.start_idx = self.current_idx = self.df.index[self.df['unix'] == start][0]
        self.interval = int(interval / 3600)
        self.owned = self.loss = 0

    def get_close(self, datapoints :int) -> float:
        return self.df.iloc[self.start_idx:self.start_idx + self.interval * datapoints:self.interval]['close']
    
    def get_price(self):
        return self.df.iloc[self.current_idx]['close']

    def buy(self, volume: float):
        if self.owned >= 10:
            pass
        else:
            self.owned += volume
            self.loss -= volume * self.get_price()

    def sell(self, volume: float):
        if self.owned >= volume:
            self.owned -= volume
            self.loss += volume * self.get_price()
        else:
            self.loss += self.owned * self.get_price()
            self.owned = 0

    def simple_rsi(self, datapoints :int, lookback=14):
        # lookback can be adjusted
        changes = self.get_close(datapoints).pct_change()
        simple_rsis = [np.nan]*lookback

        for window in changes.rolling(window=lookback):
    
            if len(window) != lookback: continue

            positives = window[window>0].sum()
            negatives = window[window<0].sum() * -1

            simple_rsis.append(100 - 100/(1 + positives/negatives))

        return pd.Series(index = self.get_close(datapoints).index, data=simple_rsis[:-1])
    
    def get_signals(self, datapoints, buy_thresh: int, sell_thresh: int):
        rsi_data = self.simple_rsi(datapoints)
        actions = []
        for rsi in rsi_data:
            if not rsi: actions.append(None)

            if rsi >= sell_thresh: actions.append('SELL')
            elif rsi <= buy_thresh: actions.append('BUY')
            else: actions.append(None)
        
        return pd.Series(index=rsi_data.index, data=actions)

    def calc_pnl(self):
        return self.loss + self.owned * self.get_price()
    
def run_algo():
    ETHUSD = Backtest('BacktestData.csv', 1586188800, 3600)
    signals = ETHUSD.get_signals(1000, 25, 75)
    for signal in signals:
        if signal == 'BUY':
            ETHUSD.buy(0.5)
        elif signal == 'SELL':
            ETHUSD.sell(ETHUSD.owned)
        else:
            pass
        print(ETHUSD.owned)
        ETHUSD.current_idx += ETHUSD.interval
    print(ETHUSD.calc_pnl())