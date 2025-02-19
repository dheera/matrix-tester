#!/usr/bin/env python3

import pandas as pd
import numpy as np

class Strategy:
    """Base Strategy class that extracts available tickers from data."""

    BUY = 1            # buy (inclding cover short) stock
    SELL = 2           # sell (including short sell) stock
    EXIT = 100         # exit a position for a ticker regardless of buy or sell
    END_TRADING = 1000 # stop trading for the rest of the day

    def __init__(self, tester):
        self._tester = tester

        self.tickers = []  # List of tickers available

        self.exit_on_market_close = True

    def _set_tickers(self, tickers):
        """
        Extracts available tickers by removing suffixes like _open, _close, _high, _low, _volume, _vwap.
        Stores unique tickers in self.tickers.
        """
        self.tickers = tickers
        
    def get_cash(self):
        return self._tester.get_cash()

    def get_positions(self, ticker = None):
        return self._tester.get_positions(ticker)

    def get_positions_value(self):
        return self._tester.get_positions_value()
   
    # Override this
    def on_step(self, timestamp, d):
        return [] # Should return a list of trading actions

    # Override this
    def on_start_day(self):
        pass

    # Override this
    def on_end_day(self):
        pass

