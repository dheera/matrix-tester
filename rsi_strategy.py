#!/usr/bin/env python3

# Stupid strategy to demonstrate how this library works

import pandas as pd
import numpy as np
from strategy import Strategy

class RSIStrategy(Strategy):
    """RSI-based trading strategy"""

    def __init__(self, tester, overbought=90, oversold=10):
        super().__init__(tester)
        self.overbought = overbought
        self.oversold = oversold

    def on_start_day(self):
        print("Day has started!")

    def on_end_day(self):
        print("Day has ended!")

    def on_step(self, timestamp, d):
        """Generates trading signals based on RSI column already available in the dataset."""
        actions = []

        # check all available tickers for overbought and oversold stocks
        for ticker in self.tickers:

            if d[ticker, "rsi"] > self.overbought:  # SHORT Signal
                num_shares = 1000 // d[ticker, "close"]
                actions.append({
                    "action": Strategy.SELL,
                    "ticker": ticker,
                    "shares": num_shares,
                })
                actions.append({
                    "action": Strategy.BUY,
                    "ticker": ticker,
                    "shares": num_shares,
                    "execution_delay": 10,
                })

            elif d[ticker, "rsi"] < self.oversold:  # LONG Signal
                num_shares = 1000 // d[ticker, "close"]
                actions.append({
                    "action": Strategy.BUY,
                    "ticker": ticker,
                    "shares": num_shares,  
                })
                actions.append({
                    "action": Strategy.SELL,
                    "ticker": ticker,
                    "shares": num_shares,
                    "execution_delay": 10,
                })

        return actions
