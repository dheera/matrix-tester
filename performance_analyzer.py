import pandas as pd
import numpy as np
from strategy import Strategy

class PerformanceAnalyzer:
    def __init__(self, trades):
        self.trades = trades

    def get_performance(self):
        """Calculates and returns performance metrics while accounting for multiple entry points."""
        df_trades = pd.DataFrame(self.trades)
        if df_trades.empty:
            return None, None, None  

        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
        df_trades["hour"] = df_trades["timestamp"].dt.hour
        df_trades["date"] = df_trades["timestamp"].dt.date

        df_trades = df_trades[df_trades["action"] == Strategy.EXIT]

        if len(df_trades) == 0:
            print("No exit trades to analyze")
            return None, None, None

        total_profit = df_trades["profit"].sum()
        num_trades = len(df_trades)
        num_wins = len(df_trades[df_trades["profit"] > 0])
        win_rate = num_wins / num_trades if num_trades > 0 else 0
        avg_profit = df_trades["profit"].mean() if num_trades > 0 else 0
        date_start = df_trades["date"].min()

        df_overall = pd.DataFrame([{
            "date_start": date_start,
            "num_trades": num_trades,
            "wins": num_wins,
            "total_profit": total_profit,
            "win_rate": win_rate,
            "avg_profit": avg_profit
        }])

        df_by_ticker = df_trades.groupby("ticker").agg(
            num_trades=("profit", "count"),
            total_profit=("profit", "sum"),
            wins=("profit", lambda x: (x > 0).sum()),
            win_rate=("profit", lambda x: (x > 0).mean()),
            avg_profit=("profit", "mean")
        ).reset_index()

        df_by_hour = df_trades.groupby("hour").agg(
            num_trades=("profit", "count"),
            total_profit=("profit", "sum"),
            wins=("profit", lambda x: (x > 0).sum()),
            win_rate=("profit", lambda x: (x > 0).mean()),
            avg_profit=("profit", "mean")
        ).reset_index()

        df_by_hour["win_rate"] = df_by_hour["wins"] / df_by_hour["num_trades"]
        df_by_hour["avg_profit"] = df_by_hour["total_profit"] / df_by_hour["num_trades"]

        return df_overall, df_by_ticker, df_by_hour
