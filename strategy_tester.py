import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
from strategy import Strategy
from performance_analyzer import PerformanceAnalyzer

class StrategyTester:
    def __init__(self, strategy_class, reset_every_day = True, initial_cash=10000, slippage=0.0, commission=0.0, strategy_args = [], strategy_kwargs = {}):
        """
        :param strategy_class: The strategy class to instantiate.
        :param initial_cash: Starting account balance.
        :param slippage: Per-share slippage cost. e.g. 0.01 means +0.01 on buys, -0.01 on sells.
        :param commission: Fractional commission on notional. e.g. 0.0005 => 0.05%.
        """
        self.strategy = strategy_class(self, *strategy_args, **strategy_kwargs)
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

    def _force_close_position(self, ticker, pos, close_price, timestamp=None):
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
                timestamp=timestamp
            )
        else:
            # It's a short position, so we BUY all shares to close
            self._buy(
                ticker=ticker,
                price=close_price,
                shares=share_count,
                timestamp=timestamp
            )

    def _close_fifo_positions(self, ticker, price, shares_to_close, side_to_close, timestamp):
        """
        Closes positions in FIFO order.
        :param ticker: Ticker symbol
        :param price: Current market price
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

                # Update cash
                print(f"close short {notional}")
                
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
                    "timestamp": timestamp
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

                print(f"close long {notional}")
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
                    "timestamp": timestamp
                })

                if pos["shares"] == 0:
                    self.positions[ticker].pop(idx)
                else:
                    idx += 1
            else:
                idx += 1

        return leftover

    def _buy(self, ticker, price, value=None, shares=None, timestamp=None):
        """
        Executes a buy order by first closing any open short positions (FIFO)
        before opening new long positions (if leftover shares remain).
        The strategy can specify either 'value' or 'shares' (but not both).

        Slippage & commission:
          fill_price = price + self.slippage for any new buy
          commission = fill_price * shares_bought * self.commission
        """
        
        if value in (0, 0.0, None) and shares in (0, 0.0, None):
            return
        
        if price is None or price <= 0:
            raise ValueError(f"Invalid price={price} for {ticker}")
        if (ticker, "close") not in self.data.columns:
            raise KeyError(f"{ticker}_close not found in DataFrame columns")

        if shares is not None and shares > 0:
            shares_to_buy = shares
        elif value is not None and value > 0:
            shares_to_buy = int(value // price)
        else:
            raise ValueError(f"Must specify either positive 'value' or positive 'shares', but not both (ticker: {ticker}, shares: {shares})")

        if shares_to_buy == 0:
            return
        
        if shares_to_buy < 0:
            raise ValueError(f"Skipping buy for {ticker}, computed shares_to_buy={shares_to_buy} <= 0.")
            return

        print(f"[BUY] {timestamp} | {ticker} | Price={price:.2f} => fill={price + self.slippage:.2f}, Shares={shares_to_buy}")

        leftover = self._close_fifo_positions(
            ticker=ticker,
            price=price,
            shares_to_close=shares_to_buy,
            side_to_close="short",
            timestamp=timestamp
        )

        if leftover > 0:
            fill_price = price + self.slippage
            notional = fill_price * leftover
            comm_cost = notional * self.commission

            if ticker not in self.positions:
                self.positions[ticker] = []

            self.positions[ticker].append({
                "entry_price": fill_price,
                "shares": leftover
            })
            # subtract cost + commission
            self.cash -= notional
            self.cash -= comm_cost

            # print(f"[OPEN LONG] {timestamp} | {ticker} | +{leftover} @ {fill_price:.2f}, Comm={comm_cost:.2f}")

            self.trades.append({
                "action": Strategy.BUY,
                "ticker": ticker,
                "shares": leftover,
                "fill_price": fill_price,
                "commission": comm_cost,
                "timestamp": timestamp
            })

    def _sell(self, ticker, price, value=None, shares=None, timestamp=None):
        """
        Executes a sell order by first closing any open long positions (FIFO)
        before opening new short positions (if leftover shares remain).
        Slippage & commission:
          fill_price = price - self.slippage for new short
          commission = fill_price * shares_sold * self.commission
        """
        if price is None or price <= 0:
            raise ValueError(f"Invalid price={price} for {ticker}")
        if (ticker, "close") not in self.data.columns:
            raise KeyError(f"{ticker}_close not found in DataFrame columns")

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
            
        print(f"[SELL] {timestamp} | {ticker} | Price={price:.2f} => fill={price - self.slippage:.2f}, Shares={shares_to_sell}")

        leftover = self._close_fifo_positions(
            ticker=ticker,
            price=price,
            shares_to_close=shares_to_sell,
            side_to_close="long",
            timestamp=timestamp
        )

        if leftover > 0:
            fill_price = price - self.slippage
            notional = fill_price * leftover
            comm_cost = notional * self.commission

            if ticker not in self.positions:
                self.positions[ticker] = []

            self.positions[ticker].append({
                "entry_price": fill_price,
                "shares": -leftover
            })

            # We'll credit the notional but also subtract commission?
            # Actually for short open, we might not do anything with self.cash if no margin is required.
            # If your logic requires, you could do self.cash += notional, then subtract comm_cost, etc.
            # We'll skip adding to self.cash here or do partial if you want.

            self.cash += notional
            self.cash -= comm_cost

            # print(f"[OPEN SHORT] {timestamp} | {ticker} | -{leftover} @ {fill_price:.2f}, Comm={comm_cost:.2f}")

            self.trades.append({
                "action": Strategy.SELL,
                "ticker": ticker,
                "shares": -leftover,
                "fill_price": fill_price,
                "commission": comm_cost,
                "timestamp": timestamp
            })

    def _exit(self, ticker, price, timestamp=None):
        """
        Reuses buy/sell logic to close each position.
        We'll keep looping until no positions remain.
        """
        if ticker not in self.positions or not self.positions[ticker]:
            return

        while self.positions[ticker]:
            pos = self.positions[ticker][0]
            self._force_close_position(ticker, pos, price, timestamp)

        # print(f"\033[96m[EXIT] {timestamp} | {ticker} | All positions closed @ {price:.2f}\033[0m")

    def _execute_order(self, action, timestamp, row):
        """
        Executes a single action (BUY/SELL/EXIT) immediately using the row's close_price
        for that ticker.
        """
        action_type = action["action"]
        tkr = action["ticker"]
        close_price = row[tkr, "close"]

        val = action.get("value")
        num_shares = action.get("shares")

        if action_type == Strategy.BUY:
            self._buy(
                ticker=tkr,
                price=close_price,
                value=val,
                shares=num_shares,
                timestamp=timestamp
            )
        elif action_type == Strategy.SELL:
            self._sell(
                ticker=tkr,
                price=close_price,
                value=val,
                shares=num_shares,
                timestamp=timestamp
            )
        elif action_type == Strategy.EXIT:
            self._exit(tkr, close_price, timestamp)

    def run(self, data):
        """
        Runs the strategy over the provided DataFrame, minute by minute, applying slippage & commission.
        On the last minute (row), forcibly exits all positions and does not allow new trades.

        Allows delayed execution via:  action["execution_delay"] = X (in minutes).
        If present, we schedule the trade for future timestamp, else we execute immediately.
        """
        if self.reset_every_day:
            self._reset()
        
        self.data = data
        self.strategy._set_tickers(sorted(self.data.columns.get_level_values(0).unique()))

        # This list holds pending orders: { "execute_at": pd.Timestamp, "action": {...} }
        self.scheduled_orders = []

        self.strategy.on_start_day()

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
            if self.strategy.exit_on_market_close:
                if i == len(data) - 1:
                    for tkr in list(self.positions.keys()):
                        close_price = d[tkr, "close"]
                        if close_price is None:
                            raise ValueError(f"{tkr} not found in row")
                        self._exit(tkr, close_price, timestamp)
                    # discard any remaining scheduled orders
                    self.scheduled_orders.clear()
                    continue

            # 2) Let the strategy decide new actions (buy, sell, exit) with optional 'execution_delay'
            actions = self.strategy.on_step(timestamp, d)
            end_trading = False

            for action in actions:
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
            "cash": self.cash,
        }

        print("Performance summary:")
        print(df_overall)
        print(f"Ending portfolio value:", self.get_positions_value(d) + self.get_cash())
        print(f"Ending cash:", self.get_cash())
        return results
    
    def get_cash(self):
        return self.cash
  
    def get_positions(self, ticker = None):
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
            # pos_list is a list of dicts like {"entry_price": ..., "shares": ...}
            # If you store multiple partial positions, sum them all
            for pos in pos_list:
                shares = pos["shares"]
                # Use the current close price from the multiindex row
                current_price = d[ticker, "close"]
                # Market value of this position
                position_value = shares * current_price
                total_value += position_value

        return total_value

