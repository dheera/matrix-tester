#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
from strategy import Strategy
from performance_analyzer import PerformanceAnalyzer

class StrategyTester:
    def __init__(self, strategy_class, reset_every_day=True, initial_cash=10000, slippage=0.0, commission=0.0, strategy_args={}):
        """
        :param strategy_class: The strategy class to instantiate.
        :param initial_cash: Starting account balance.
        :param slippage: Per-share slippage cost. e.g. 0.01 means +0.01 on buys, -0.01 on sells.
        :param commission: Fractional commission on notional. e.g. 0.0005 => 0.05%.
        """
        self.strategy = strategy_class(self, **strategy_args)
        self.initial_cash = initial_cash
        self.slippage = slippage        # e.g. 0.01 => add 0.01 to buy price, subtract 0.01 from sell price
        self.commission = commission    # e.g. 0.001 => 0.1% of notional

        self.reset_every_day = reset_every_day
        self._reset()

    def _reset(self):
        # Current account balance
        self.cash = self.initial_cash
        # Dictionary of positions: {ticker: [{"entry_price": float, "shares": int}, ...]}
        # Positive shares => long positions, negative => short positions.
        self.positions = {}
        # Trade history for performance analysis
        self.trades = []
        # Scheduled orders for delayed execution
        # each item is { "execute_at": pd.Timestamp, "action": {...} }
        self.scheduled_orders = []

    def _force_close_position(self, ticker, pos, close_price, timestamp=None, reason=""):
        """
        Force-close a single position (long or short) by calling self._sell(...) or self._buy(...):
         - If shares > 0 => SELL to exit
         - If shares < 0 => BUY to exit
        """
        share_count = abs(pos["shares"])
        if share_count == 0:
            return

        if pos["shares"] > 0:
            # It's a long position, so we SELL all shares to close
            self._sell(
                ticker=ticker,
                price=close_price,
                shares=share_count,
                timestamp=timestamp,
                reason=reason,
            )
        else:
            # It's a short position, so we BUY all shares to close
            self._buy(
                ticker=ticker,
                price=close_price,
                shares=share_count,
                timestamp=timestamp,
                reason=reason,
            )

    def _close_fifo_positions(self, ticker, price, shares_to_close, side_to_close, timestamp, reason):
        """
        Closes positions in FIFO order.
        :param ticker: Ticker symbol
        :param price: Current market price (should be ask for BUY or bid for SELL orders)
        :param shares_to_close: How many shares we need to close (always positive)
        :param side_to_close: 'short' => close negative positions, 'long' => close positive positions
        :param timestamp: Current timestamp
        :return: leftover_shares (int) how many shares remain to buy or sell (if any)
        
        This now applies slippage & commission:
          - If side_to_close=='short', we are effectively buying to close => fill_price = price + slippage
          - If side_to_close=='long', we are effectively selling to close => fill_price = price - slippage
          - Commission is fraction of notional => commission_amount = fill_price * shares_closed * self.commission
        """
        if ticker not in self.positions:
            return shares_to_close

        leftover = shares_to_close
        idx = 0
        while idx < len(self.positions[ticker]) and leftover > 0:
            pos = self.positions[ticker][idx]

            if side_to_close == "short" and pos["shares"] < 0:
                short_size = abs(pos["shares"])
                close_amt = min(short_size, leftover)
                fill_price = price + self.slippage
                notional = fill_price * close_amt
                comm_cost = notional * self.commission

                # profit from closing short = (entry_price - fill_price) * close_amt
                profit = (pos["entry_price"] - fill_price) * close_amt

                profit_color = "\033[92m" if profit >= 0 else "\033[91m"
                print(f"{profit_color}[EXIT] {timestamp} | {ticker} | "
                      f"Closed {close_amt} SHARES SHORT @ {fill_price:.2f} | Profit: {profit:.2f}, Comm: {comm_cost:.2f}\033[0m")
                
                self.cash -= notional      # add proceeds
                self.cash -= comm_cost     # subtract commission

                pos["shares"] += close_amt # reduce short
                leftover -= close_amt

                # Record separate EXIT trade
                self.trades.append({
                    "action": Strategy.EXIT,
                    "ticker": ticker,
                    "shares": close_amt,
                    "fill_price": fill_price,
                    "profit": profit,
                    "commission": comm_cost,
                    "timestamp": timestamp,
                    "reason_open": pos.get("reason", ""),
                    "reason_close": reason,
                })

                if pos["shares"] == 0:
                    self.positions[ticker].pop(idx)
                else:
                    idx += 1

            elif side_to_close == "long" and pos["shares"] > 0:
                long_size = pos["shares"]
                close_amt = min(long_size, leftover)
                fill_price = price - self.slippage
                notional = fill_price * close_amt
                comm_cost = notional * self.commission

                # profit from closing long = (fill_price - entry_price) * close_amt
                profit = (fill_price - pos["entry_price"]) * close_amt

                profit_color = "\033[92m" if profit >= 0 else "\033[91m"
                print(f"{profit_color}[EXIT] {timestamp} | {ticker} | "
                      f"Closed {close_amt} SHARES LONG @ {fill_price:.2f} | Profit: {profit:.2f}, Comm: {comm_cost:.2f}\033[0m")

                self.cash += notional
                self.cash -= comm_cost

                pos["shares"] -= close_amt
                leftover -= close_amt

                self.trades.append({
                    "action": Strategy.EXIT,
                    "ticker": ticker,
                    "shares": close_amt,
                    "fill_price": fill_price,
                    "profit": profit,
                    "commission": comm_cost,
                    "timestamp": timestamp,
                    "reason_open": pos.get("reason", ""),
                    "reason_close": reason,
                })

                if pos["shares"] == 0:
                    self.positions[ticker].pop(idx)
                else:
                    idx += 1
            else:
                idx += 1

        return leftover

    def _close_lifo_positions(self, ticker, price, shares_to_close, side_to_close, timestamp, reason):
        """Close positions using LIFO (last-in, first-out) order."""
        if ticker not in self.positions:
            return shares_to_close

        leftover = shares_to_close
        while self.positions[ticker] and leftover > 0:
            # Always process the last (most recent) position
            pos = self.positions[ticker][-1]

            if side_to_close == "short" and pos["shares"] < 0:
                short_size = abs(pos["shares"])
                close_amt = min(short_size, leftover)
                fill_price = price + self.slippage
                notional = fill_price * close_amt
                comm_cost = notional * self.commission
                profit = (pos["entry_price"] - fill_price) * close_amt
                profit_color = "\033[92m" if profit >= 0 else "\033[91m"
                print(f"{profit_color}[EXIT] {timestamp} | {ticker} | "
                      f"Closed {close_amt} SHARES SHORT @ {fill_price:.2f} | Profit: {profit:.2f}, Comm: {comm_cost:.2f}\033[0m")
                self.cash -= notional
                self.cash -= comm_cost
                pos["shares"] += close_amt  # note: pos["shares"] is negative
                leftover -= close_amt
                self.trades.append({
                    "action": self.strategy.EXIT,
                    "ticker": ticker,
                    "shares": close_amt,
                    "fill_price": fill_price,
                    "profit": profit,
                    "commission": comm_cost,
                    "timestamp": timestamp,
                    "reason_open": pos.get("reason", ""),
                    "reason_close": reason,
                })
                if pos["shares"] == 0:
                    self.positions[ticker].pop()  # remove the last element
            elif side_to_close == "long" and pos["shares"] > 0:
                long_size = pos["shares"]
                close_amt = min(long_size, leftover)
                fill_price = price - self.slippage
                notional = fill_price * close_amt
                comm_cost = notional * self.commission
                profit = (fill_price - pos["entry_price"]) * close_amt
                profit_color = "\033[92m" if profit >= 0 else "\033[91m"
                print(f"{profit_color}[EXIT] {timestamp} | {ticker} | "
                      f"Closed {close_amt} SHARES LONG @ {fill_price:.2f} | Profit: {profit:.2f}, Comm: {comm_cost:.2f}\033[0m")
                self.cash += notional
                self.cash -= comm_cost
                pos["shares"] -= close_amt
                leftover -= close_amt
                self.trades.append({
                    "action": self.strategy.EXIT,
                    "ticker": ticker,
                    "shares": close_amt,
                    "fill_price": fill_price,
                    "profit": profit,
                    "commission": comm_cost,
                    "timestamp": timestamp,
                    "reason_open": pos.get("reason", ""),
                    "reason_close": reason,
                })
                if pos["shares"] == 0:
                    self.positions[ticker].pop()
            else:
                break

        return leftover

    def _close_positions(self, ticker, price, shares_to_close, side_to_close, timestamp, reason):
        """Choose FIFO or LIFO closing based on strategy settings."""
        order = getattr(self.strategy, "_close_positions_order", "fifo")
        if order == Strategy.LIFO:
            return self._close_lifo_positions(ticker, price, shares_to_close, side_to_close, timestamp, reason)
        elif order == Strategy.FIFO:
            return self._close_fifo_positions(ticker, price, shares_to_close, side_to_close, timestamp, reason)
        else:
            raise ValueError(f"Unknown _close_positions_order value of {order}")

    def _get_market_price(self, ticker, row, side, fill_at_mid_price = False):
        """
        Helper: returns the market price for the ticker from the current row.
        If bid/ask columns exist then:
          - For 'buy' orders, returns the ask price.
          - For 'sell' orders, returns the bid price.
        Otherwise, returns the close price.
        """
        if (ticker, "bid") in row and (ticker, "ask") in row:
            if fill_at_mid_price:
                return (row[ticker, "bid"] + row[ticker, "ask"]) / 2
            if side == "buy":
                return row[ticker, "ask"]
            elif side == "sell":
                return row[ticker, "bid"]
            else:
                raise ValueError("side must be 'buy' or 'sell'")
        else:
            return row[ticker, "close"]

    def _buy(self, ticker, price, value=None, shares=None, timestamp=None, reason=""):
        """
        Executes a buy order by first closing any open short positions (FIFO)
        before opening new long positions (if leftover shares remain).
        The strategy can specify either 'value' or 'shares' (but not both).

        Slippage & commission:
          For new buy orders, fill_price = price + self.slippage
          Commission = fill_price * shares_bought * self.commission
        """
        # Check that data contains either a "close" column or both "bid" and "ask" columns.
        if not (((ticker, "close") in self.data.columns) or (((ticker, "bid") in self.data.columns) and ((ticker, "ask") in self.data.columns))):
            raise KeyError(f"Price data not found for {ticker}. Expected either close or bid/ask columns.")

        if value in (0, 0.0, None) and shares in (0, 0.0, None):
            return
        
        if not ticker.startswith("O:"):
            if price is None or price <= 0:
                raise ValueError(f"Invalid price={price} for {ticker}")
        
        if shares is not None and shares > 0:
            shares_to_buy = shares
        elif value is not None and value > 0:
            shares_to_buy = int(value // price)
        else:
            raise ValueError(f"Must specify either positive 'value' or positive 'shares', but not both (ticker: {ticker}, shares: {shares}, value: {value})")

        if shares_to_buy == 0:
            return
        
        if shares_to_buy < 0:
            raise ValueError(f"Skipping buy for {ticker}, computed shares_to_buy={shares_to_buy} <= 0.")

        print(f"[BUY] {timestamp} | {ticker} | Price={price:.2f} => fill={price + self.slippage:.2f}, Shares={shares_to_buy} | {reason}")

        leftover = self._close_positions(
            ticker=ticker,
            price=price,
            shares_to_close=shares_to_buy,
            side_to_close="short",
            timestamp=timestamp,
            reason=reason,
        )

        if leftover > 0:
            fill_price = price + self.slippage
            notional = fill_price * leftover
            comm_cost = notional * self.commission

            if ticker not in self.positions:
                self.positions[ticker] = []

            self.positions[ticker].append({
                "entry_price": fill_price,
                "shares": leftover,
                "reason": reason,
            })
            # subtract cost + commission
            self.cash -= notional
            self.cash -= comm_cost

            self.trades.append({
                "action": Strategy.BUY,
                "ticker": ticker,
                "shares": leftover,
                "fill_price": fill_price,
                "commission": comm_cost,
                "timestamp": timestamp,
            })

    def _sell(self, ticker, price, value=None, shares=None, timestamp=None, reason=""):
        """
        Executes a sell order by first closing any open long positions (FIFO)
        before opening new short positions (if leftover shares remain).
        Slippage & commission:
          For new short orders, fill_price = price - self.slippage
          Commission = fill_price * shares_sold * self.commission
        """
        # Check that data contains either a "close" column or both "bid" and "ask" columns.
        if not (((ticker, "close") in self.data.columns) or (((ticker, "bid") in self.data.columns) and ((ticker, "ask") in self.data.columns))):
            raise KeyError(f"Price data not found for {ticker}. Expected either close or bid/ask columns.")

        if not ticker.startswith("O:"):
            if price is None or price <= 0:
                raise ValueError(f"Invalid price={price} for {ticker}")

        if value in (0, 0.0, None) and shares in (0, 0.0, None):
            return

        if shares is not None and shares > 0:
            shares_to_sell = shares
        elif value is not None and value > 0:
            shares_to_sell = int(value // price)
        else:
            raise ValueError(f"Must specify either positive 'value' or positive 'shares', but not both (ticker: {ticker}, shares: {shares})")

        if shares_to_sell == 0:
            return

        if shares_to_sell < 0:
            raise ValueError(f"Skipping sell for {ticker}, computed shares_to_sell={shares_to_sell} <= 0.")
            
        print(f"[SELL] {timestamp} | {ticker} | Price={price:.2f} => fill={price - self.slippage:.2f}, Shares={shares_to_sell} | {reason}")

        leftover = self._close_positions(
            ticker=ticker,
            price=price,
            shares_to_close=shares_to_sell,
            side_to_close="long",
            timestamp=timestamp,
            reason=reason,
        )

        if leftover > 0:
            fill_price = price - self.slippage
            notional = fill_price * leftover
            comm_cost = notional * self.commission

            if ticker not in self.positions:
                self.positions[ticker] = []

            self.positions[ticker].append({
                "entry_price": fill_price,
                "shares": -leftover,
                "reason": reason,
            })

            self.cash += notional
            self.cash -= comm_cost

            self.trades.append({
                "action": Strategy.SELL,
                "ticker": ticker,
                "shares": -leftover,
                "fill_price": fill_price,
                "commission": comm_cost,
                "timestamp": timestamp,
            })

    def _exit(self, ticker, row, timestamp=None, reason=""):
        """
        Reuses buy/sell logic to close each position.
        For each open position, the appropriate market price is chosen based on its direction:
         - For a long position, we use the 'sell' price (bid)
         - For a short position, we use the 'buy' price (ask)
        """
        if ticker not in self.positions or not self.positions[ticker]:
            return

        while self.positions[ticker]:
            pos = self.positions[ticker][0]
            if pos["shares"] > 0:
                market_price = self._get_market_price(ticker, row, "sell")
            else:
                market_price = self._get_market_price(ticker, row, "buy")
            self._force_close_position(ticker, pos, market_price, timestamp, reason)

    def _execute_order(self, action, timestamp, row):
        """
        Executes a single action (BUY/SELL/EXIT) immediately using the current market prices.
        For data with bid/ask columns:
          - BUY orders use the ask price.
          - SELL orders use the bid price.
          - EXIT orders determine the price based on the position's direction.
        """
        action_type = action["action"]
        tkr = action["ticker"]
        reason = action.get("reason", "")
        if action_type == Strategy.BUY:
            market_price = self._get_market_price(tkr, row, "buy")
            val = action.get("value")
            num_shares = action.get("shares")
            self._buy(
                ticker=tkr,
                price=market_price,
                value=val,
                shares=num_shares,
                timestamp=timestamp,
                reason=reason,
            )
        elif action_type == Strategy.SELL:
            market_price = self._get_market_price(tkr, row, "sell")
            val = action.get("value")
            num_shares = action.get("shares")
            self._sell(
                ticker=tkr,
                price=market_price,
                value=val,
                shares=num_shares,
                timestamp=timestamp,
                reason=reason,
            )
        elif action_type == Strategy.EXIT:
            self._exit(tkr, row, timestamp, reason)

    def run(self, data, force_exit_on_market_close=False):
        """
        Runs the strategy over the provided DataFrame, minute by minute, applying slippage & commission.
        On the last minute (row), forcibly exits all positions and does not allow new trades.

        Allows delayed execution via:  action["execution_delay"] = X (in minutes).
        If present, we schedule the trade for future timestamp, else we execute immediately.
        """
        if self.reset_every_day:
            self._reset()
        
        self.trades = []
        action_log = []
        self.data = data

        self.strategy._set_tickers(list(self.data.columns.get_level_values(0).unique()))

        # This list holds pending orders: { "execute_at": pd.Timestamp, "action": {...} }
        self.scheduled_orders = []

        self.strategy.on_start_day()

        print(f"Starting cash:", self.get_cash())

        for i, (timestamp, d) in enumerate(tqdm(self.data.iterrows(), total=len(data))):
            # 1) Execute any scheduled orders whose 'execute_at' <= current timestamp
            due_orders = []
            keep_orders = []
            for so in self.scheduled_orders:
                if so["execute_at"] <= timestamp:
                    due_orders.append(so)
                else:
                    keep_orders.append(so)
            self.scheduled_orders = keep_orders

            # Execute all due orders
            for so in due_orders:
                self._execute_order(so["action"], timestamp, d)

            # If this is the last row: forcibly exit all trades, no new trades
            if self.strategy._exit_on_market_close or force_exit_on_market_close:
                if i == len(data) - 1:
                    for tkr in list(self.positions.keys()):
                        # Instead of a single close price, pass the entire row to _exit
                        self._exit(tkr, d, timestamp)
                    # Discard any remaining scheduled orders
                    self.scheduled_orders.clear()
                    continue

            # 2) Let the strategy decide new actions (buy, sell, exit) with optional 'execution_delay'
            actions = self.strategy.on_step(timestamp, d)
            end_trading = False

            for action in actions:
                action["timestamp"] = timestamp
                action_log.append(action)

                if "value" in action and action["value"] < 0:
                    raise ValueError(f"value cannot be negative: {action}")
                
                if "shares" in action and action["shares"] < 0:
                    raise ValueError(f"shares cannot be negative: {action}")

                if action["action"] == Strategy.END_TRADING:
                    end_trading = True
                    break

                delay_minutes = action.get("execution_delay", 0)
                if delay_minutes > 0:
                    execute_at = timestamp + pd.Timedelta(minutes=delay_minutes)
                    self.scheduled_orders.append({
                        "execute_at": execute_at,
                        "action": action
                    })
                else:
                    self._execute_order(action, timestamp, d)

            if end_trading:
                break
        
        self.strategy.on_end_day()

        # Analyze performance
        df_overall, df_by_ticker, df_by_hour = PerformanceAnalyzer(self.trades).get_performance()
        results = {
            "overall_performance": df_overall,
            "performance_by_ticker": df_by_ticker,
            "performance_by_hour": df_by_hour,
            "trades": pd.DataFrame(self.trades),
            "actions": pd.DataFrame(action_log),
            "cash": self.cash,
        }

        print("Performance summary:")
        print(df_overall)
        portfolio_value = self.get_positions_value(d)
        cash_value = self.get_cash()
        print(f"Ending portfolio value:", portfolio_value)
        print(f"Ending cash:", cash_value)
        print(f"Ending total value:", portfolio_value + cash_value)
        return results
    
    def get_cash(self):
        return self.cash
  
    def get_positions(self, ticker=None):
        if ticker is None:
            return self.positions
        else:
            return self.positions.get(ticker, [])

    def get_positions_value(self, d):
        """
        Returns the current total portfolio value (cash + market value of positions)
        using the row `d` (MultiIndex format) for current prices.
        """
        total_value = 0

        for ticker, pos_list in self.positions.items():
            for pos in pos_list:
                shares = pos["shares"]
                # If bid/ask data exists, use them appropriately:
                if (ticker, "bid") in d and (ticker, "ask") in d:
                    # For long positions, use bid (selling), for short, use ask (buying to cover)
                    price = d[ticker, "bid"] if shares > 0 else d[ticker, "ask"]
                else:
                    price = d[ticker, "close"]
                position_value = shares * price
                total_value += position_value

        return total_value
