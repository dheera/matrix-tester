#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

class Strategy:
    """Base Strategy class that extracts available tickers from data."""

    BUY = 1            # buy (inclding cover short) stock
    SELL = 2           # sell (including short sell) stock
    EXIT = 50          # exit a position for a ticker regardless of buy or sell
    END_TRADING = 100  # stop trading for the day (signal intention to StrategyTester that it can safely stop processing data for the day)

    LIFO = 1000
    FIFO = 1001

    def __init__(self, tester):
        self._tester = tester

        self.tickers = []  # List of tickers available

        self.request_stocks = []
        self.request_options = []

        # settings
        self.exit_on_market_close = True
        self.close_positions_order = self.LIFO

        # "ohlc" mode, gets minute aggregate data for 1024+ stocks; options not yet supported
        # can use slippage/commission parameters to simulate slippage
        self._data_mode = "ohlc"
        self._request_stocks = []
        self._request_options = []

        # "bidask" mode, only gets data for selected stocks and option chains for selected underlyings 
        # and gets gets bid/ask data for more accurate simulation
        # self._data_mode = "bidask"
        # self._request_stocks = ["NVDA"]
        # self._request_options = ["NVDA"]

        # for logging
        self._log = []
        os.makedirs("logs", exist_ok = True)

    def _set_tickers(self, tickers):
        self.tickers = tickers
        
    def get_cash(self):
        return self._tester.get_cash()

    def get_positions(self, ticker = None):
        return self._tester.get_positions(ticker)

    def get_positions_value(self, d):
        return self._tester.get_positions_value(d)

    # Override this
    def on_step(self, timestamp, d):
        return [] # Should return a list of trading actions

    # Override this
    def on_start_day(self):
        pass

    # Override this
    def on_end_day(self):
        # write logs if the strategy logged anything
        if len(self._log) > 0:
            df = pd.DataFrame(self._log)
            if "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)
            # TODO: do better logging and file naming later
            df.to_parquet(os.path.join("logs", f"{self.__class__.__name__}_log.parquet"))

    def log(self, datapoint = {}):
        self._log.append(datapoint)