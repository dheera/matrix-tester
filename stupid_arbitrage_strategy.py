#!/usr/bin/env python3
import numpy as np
from strategy import Strategy
import pandas as pd
import math

class StupidArbitrageStrategy(Strategy):
    """
    Stupid arbitrage strategy that doesn't really work without some creative changes
    but demos how this library works
    """

    def __init__(self, tester,
        x = "VOO", y = "VTI",
        expected_ratio = 1.86,
        enter_threshold = 0.001,
        exit_threshold = 0.0005,
        trade_notional=10000
    ):
        """
        Parameters:
            y=Ax+B arbitrage
        """
        super().__init__(tester)

        # parameters from initialization
        self.x = x
        self.y = y
        self.expected_ratio = expected_ratio
        self.trade_notional = trade_notional
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold

        # variables
        self.x_qty = None
        self.y_qty = None
        self.currently_in_arbitrage = False
        self.num_trades = 0

        # tell simulator we are in bidask mode
        self._data_mode = "bidask"
        # ask simulator for data for for these tickers, need to do this for bidask mode
        self._request_stocks = [x, y]
        # ask simulator to not close all trades at the end of day since this is a swing strategy
        self._exit_on_market_close = False


    def on_step(self, timestamp, d):
        actions = []

        x_price = (d[self.x, "bid"] + d[self.x, "ask"]) / 2
        y_price = (d[self.y, "bid"] + d[self.y, "ask"]) / 2

        deviation = x_price/y_price - self.expected_ratio

        # determine quantities if we have not already
        if self.x_qty is None or self.y_qty is None:
            self.x_qty = (self.trade_notional / x_price)
            self.y_qty = (self.trade_notional / x_price * self.expected_ratio)
            print(f"Decided x_qty: {self.x_qty}, y_qty: {self.y_qty}")
        
        # TODO: make short side whole numbers and long side fractional to match alpaca's capabilities

        if not self.currently_in_arbitrage and deviation > self.enter_threshold:
            actions.append({
                "action": Strategy.SELL, # short
                "ticker": self.x,
                "shares": self.x_qty,
            })
            actions.append({
                "action": Strategy.BUY, # long
                "ticker": self.y,
                "shares": self.y_qty,
            })
            self.num_trades += 1
            self.currently_in_arbitrage = True

        if not self.currently_in_arbitrage and deviation < -self.enter_threshold:
            actions.append({
                "action": Strategy.BUY, # long
                "ticker": self.x,
                "shares": self.x_qty,
            })
            actions.append({
                "action": Strategy.SELL, # sell
                "ticker": self.y,
                "shares": self.y_qty,
            })
            self.num_trades += 1
            self.currently_in_arbitrage = True

        if self.currently_in_arbitrage and abs(deviation) < self.exit_threshold:
            actions.append({
                "action": Strategy.EXIT,
                "ticker": self.x,
            })
            actions.append({
                "action": Strategy.EXIT,
                "ticker": self.y,
            })
            self.num_trades += 1
            self.currently_in_arbitrage = False

        positions_value = self.get_positions_value(d)
        cash_value = self.get_cash()
        total_value = positions_value + cash_value

        self.log({
            "timestamp": timestamp,
            "deviation": deviation,
            "x_price": x_price,
            "y_price": y_price,
            "positions": positions_value,
            "cash": cash_value,
            "total": total_value,
            "num_trades": self.num_trades,
        })

        return actions
