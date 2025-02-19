#!/usr/bin/env python3

import pandas as pd
import numpy as np

class Strategy:
    """Base Strategy class that extracts available tickers from data."""

    BUY = 1            # buy (inclding cover short) stock
    SELL = 2           # sell (including short sell) stock
    EXIT = 100         # exit a position for a ticker regardless of buy or sell
    END_TRADING = 1000 # stop trading for the rest of the day

    def __init__(self, initial_cash):
        self.initial_cash = initial_cash  # Strategy knows initial cash
        self.tickers = []  # List of tickers available

        self.exit_on_market_close = True

    def _set_tickers(self, tickers):
        """
        Extracts available tickers by removing suffixes like _open, _close, _high, _low, _volume, _vwap.
        Stores unique tickers in self.tickers.
        """
        self.tickers = tickers
        
    def on_step(self, timestamp, d):
        """
        This method should be overridden by subclasses to implement trading logic.
        """
        return []  # No trading actions by default

    def on_start_day(self):
        pass

    def on_end_day(self):
        pass