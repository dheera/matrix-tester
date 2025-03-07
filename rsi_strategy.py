#!/usr/bin/env python3

# Stupid strategy to demonstrate how this library works

import pandas as pd
import numpy as np
from strategy import Strategy

class RSIStrategy(Strategy):
    """RSI-based trading strategy"""

    def __init__(self, tester, overbought=70, oversold=30, targets=None):
        super().__init__(tester)
        self.overbought = overbought
        self.oversold = oversold
        self.targets = targets.split(",") if type(targets) is str else targets

        # parameters for tester
        self._data_mode = "ohlc" # this operates in ohlc ("candle") mode and gets aggregates for all tickers

    def on_step(self, timestamp, d):
        """Generates trading signals based on RSI column already available in the dataset."""
        actions = []

        # check all available tickers for overbought and oversold stocks
        for ticker in (self.targets if self.targets else self.tickers):
            if (ticker, "close") not in d:
                continue


            if len(self.get_positions(ticker)) > 0:
                # don't buy more if we already did
                continue

            if d[ticker, "rsi"] > self.overbought:  # SHORT Signal
                num_shares = 1000 // d[ticker, "close"]
                actions.append({
                    "action": Strategy.SELL,
                    "ticker": ticker,
                    "shares": num_shares,
                    "reason": f"shorted because rsi was overbought at {d[ticker, 'rsi']}",
                })
                actions.append({
                    "action": Strategy.BUY,
                    "ticker": ticker,
                    "shares": num_shares,
                    "execution_delay": 10,
                    "reason": "covered after 10 minutes",
                })

            elif d[ticker, "rsi"] < self.oversold:  # LONG Signal
                num_shares = 1000 // d[ticker, "close"]
                actions.append({
                    "action": Strategy.BUY,
                    "ticker": ticker,
                    "shares": num_shares,
                    "reason": f"bought because rsi was oversold at {d[ticker, 'rsi']}",
                })
                actions.append({
                    "action": Strategy.SELL,
                    "ticker": ticker,
                    "shares": num_shares,
                    "execution_delay": 10,
                    "reason": "sold after 10 minutes",
                })

        return actions
