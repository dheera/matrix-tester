#!/usr/bin/env python3
import numpy as np
from strategy import Strategy
import pandas as pd
from ewm_utils import RollingStats
import math
import time
import copy
import json

def deepcopy(foo):
    return json.loads(json.dumps(foo))

class Arbitrageur:
    def __init__(self, formulas):
        self.formulas = formulas
        self.targets = []
        self.arbitrages = []

        self.targets = [{ "level_quantized": 0.0, "positions": {} }] * len(self.formulas)
        self.targets = list(map(copy.copy, self.targets)) # because dicts are refs

    def update_targets(self, quotes):
        self.targets = [
            self.get_target_for_formula(self.targets[idx], formula, quotes)
            for idx, formula in enumerate(self.formulas)
        ]
        return self.targets

    def get_target_for_formula(self, current_target, formula, quotes):
        target = {}

        if True: # lazy to remove indents in vi
            y = sum(quotes[formula["tickers"][0]])/2 #bidask mean
            x = [sum(quotes[ticker])/2 for ticker in formula["tickers"][1:]]
            A = formula["A"]
            B = formula["B"]
            level_size = formula["level_size"]

            y_pred = sum([A[i]*x[i] for i in range(len(x))]) + B

            level_quantized = current_target["level_quantized"]

            deviation = (y_pred - y) / ((y_pred + y) / 2)
            level = min(max(deviation/level_size, -formula["max_levels"]), formula["max_levels"])

            # TODO: disqualify the entire arbitrage if this deviation is too high

            # old logic, levels are not fixed
            # if abs(level - level_quantized) > 1.0:
            #    print(f"ARB LEVEL CHANGE: {level_quantized} -> {level}")
            #    level_quantized = level

            # level is the actual current float level (deviation divided by level size)
            # level_quantized is the current quantized level
            # level_quantized_candidate is the new candidate quantized level

            # quantized levels should be ints to induce hysteris to capture arbitrage
            # the quantized level determines the targets for the stock positions we acquire

            if level > 0:
                level_quantized_candidate = math.floor(level)
                if level_quantized_candidate > level_quantized:
                    print(f"ARB LEVEL UP: {level_quantized} -> {level_quantized_candidate}")
                    level_quantized = level_quantized_candidate

                level_quantized_candidate = math.ceil(level)
                if level_quantized_candidate < level_quantized:
                    print(f"ARB LEVEL DOWN: {level_quantized} -> {level_quantized_candidate}")
                    level_quantized = level_quantized_candidate

            if level < 0:
                level_quantized_candidate = math.ceil(level)
                if level_quantized_candidate < level_quantized:
                    print(f"ARB LEVEL DOWN: {level_quantized} -> {level_quantized_candidate}")
                    level_quantized = level_quantized_candidate

                level_quantized_candidate = math.floor(level)
                if level_quantized_candidate > level_quantized:
                    print(f"ARB LEVEL UP: {level_quantized} -> {level_quantized_candidate}")
                    level_quantized = level_quantized_candidate


            if level_quantized == 0 or "qty_scale" not in current_target:
                qty_scale = formula["notional"] / y
            else:
                qty_scale = current_target["qty_scale"]
            
            y_qty = level_quantized * qty_scale # level_quantized * formula["notional"] / y
            x_qtys = [-A[i] * level_quantized * qty_scale for i in range(len(x))]

            target = {
                "deviation": deviation,
                "level": level,
                "level_quantized": level_quantized,
                "positions": {
                    formula["tickers"][0]: y_qty,
                },
                "qty_scale": qty_scale,
            }
            
            for i in range(1, len(formula["tickers"])):
                target["positions"][formula["tickers"][i]] = x_qtys[i-1]


            #x_qtys = [
            #    -level_quantized * formula["notional"] / (A[i]*x[i])
            #    for i in range(len(x))
            #]

        return target

    def get_actions(self, portfolio):
        actions = []
        have_quantities = {}
        for ticker in portfolio:
            have_quantities[ticker] = sum([position["shares"] for position in portfolio[ticker]])

        want_quantities = {}
        for target in self.targets:
            for ticker in target["positions"]:
                if ticker not in want_quantities:
                    want_quantities[ticker] = 0.0
                want_quantities[ticker] += target["positions"][ticker]

        for ticker in want_quantities:
            cha = want_quantities[ticker] - have_quantities.get(ticker, 0.0)
            #print(f"{ticker} WANT {want_quantities[ticker]} HAVE {have_quantities.get(ticker, 0.0)}")
            if cha > 1.5:
                actions.append({
                    "action": Strategy.BUY,
                    "ticker": ticker,
                    "shares": cha,
                })
            elif cha < -1.5:
                actions.append({
                    "action": Strategy.SELL,
                    "ticker": ticker,
                    "shares": -cha,
                })

        return actions

class Ladder2ArbitrageStrategy(Strategy):
    """
    Arbitrage strategy
    """

    def __init__(self, tester, formulas=None):
        """
        Parameters:
            y=Ax+B arbitrage
        """
        super().__init__(tester)
        self._data_mode = "bidask"
        self._exit_on_market_close = False
        
        self.formulas = formulas

        if type(self.formulas) is str:
            self.formulas = json.loads(self.formulas)

        if self.formulas is None:
            self.formulas = [
                {
                    "tickers": ["SPXL", "TSM"], #y, x1, x2, ...
                    "A": [0.79], # y = A1*x1 + A2*x2 + ...
                    "B": 15.57,
                    "std": 0.5,
                    "level_size": 0.02,
                    "level_bound": 5,
                    "notional": 1000,
                    "max_levels": 5,
                }
            ]

        self.formulas = [
                {
                    "tickers": ["VST", "CEG", "CVNA"],
                    "A": [0.307649, 0.333727],
                    "B": -16.303406,
                    "level_size": 0.05,
                    "notional": 1000,
                    "max_levels": 2,
                }
        ]

        self.formulas = [
                {
                    "tickers": ["SOFI", "UAL", "SCCO"],
                    "A": [0.154905, -0.069257],
                    "B": 7.127092,
                    "level_size": 0.05,
                    "notional": 2000,
                    "max_levels": 5,
                }
        ]

        self.arb = Arbitrageur(self.formulas)
        
        for formula in self.formulas:
            self._request_stocks += formula["tickers"]

    def on_step(self, timestamp, d):
        actions = []

        quotes = {}
        for ticker in self._request_stocks:
            quotes[ticker] = (d[ticker, "bid"], d[ticker, "ask"])

        targets = self.arb.update_targets(quotes)
        actions = self.arb.get_actions(self.get_positions())

        have_quantities = {}
        positions = self.get_positions()
        for ticker in positions:
            have_quantities[ticker] = sum([position["shares"] for position in positions[ticker]])

        log_entry = {
            "timestamp": timestamp,
            "total": self.get_positions_value(d) + self.get_cash(),
            "positions": self.get_positions_value(d),
            "cash": self.get_cash(),
        }

        margin = 0.0
        for ticker in self._request_stocks:
            log_entry[ticker] = have_quantities.get(ticker, 0.0)
            margin += abs(log_entry[ticker] * d[ticker, "bid"]) # margin requirements

        for i, target in enumerate(self.arb.targets):
            log_entry[f"dev{i}"] = self.arb.targets[i]["deviation"]
            log_entry[f"level{i}"] = self.arb.targets[i]["level"]
            log_entry[f"levelq{i}"] = self.arb.targets[i]["level_quantized"]

        log_entry["margin"] = margin

        self.log(log_entry)

        return actions
