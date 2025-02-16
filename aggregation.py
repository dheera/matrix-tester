import pandas as pd

def aggregate(all_results):
    # Collect results
    all_by_date = [res["overall_performance"] for res in all_results if res["overall_performance"] is not None]
    all_by_ticker = [res["performance_by_ticker"] for res in all_results if res["performance_by_ticker"] is not None]
    all_by_hour = [res["performance_by_hour"] for res in all_results if res["performance_by_hour"] is not None]

    # Convert lists to DataFrames
    df_by_date = pd.concat(all_by_date, ignore_index=True) if all_by_date else pd.DataFrame()
    df_by_ticker = aggregate_ticker_stats(pd.concat(all_by_ticker, ignore_index=True)) if all_by_ticker else pd.DataFrame()
    df_by_hour = aggregate_hourly_stats(pd.concat(all_by_hour, ignore_index=True)) if all_by_hour else pd.DataFrame()

    df_overall = compute_overall_performance(df_by_date)

    return {
        "overall": df_overall,
        "by_date": df_by_date,
        "by_ticker": df_by_ticker,
        "by_hour": df_by_hour,
    }

def aggregate_hourly_stats(df_by_hour):
    """Aggregates hourly stats across multiple days."""
    if df_by_hour.empty:
        return df_by_hour

    df_agg = df_by_hour.groupby("hour").agg(
        num_trades=("num_trades", "sum"),
        wins=("wins", "sum"),
        total_profit=("total_profit", "sum")
    ).reset_index()

    df_agg["win_rate"] = df_agg["wins"] / df_agg["num_trades"]
    df_agg["avg_profit"] = df_agg["total_profit"] / df_agg["num_trades"]

    return df_agg

def aggregate_ticker_stats(df_by_ticker):
    """Aggregates ticker stats across multiple days."""
    if df_by_ticker.empty:
        return df_by_ticker

    df_agg = df_by_ticker.groupby("ticker").agg(
        num_trades=("num_trades", "sum"),
        total_profit=("total_profit", "sum"),
        wins=("wins", "sum")
    ).reset_index()

    df_agg["win_rate"] = df_agg["wins"] / df_agg["num_trades"]
    df_agg["avg_profit"] = df_agg["total_profit"] / df_agg["num_trades"]

    return df_agg

def compute_overall_performance(df_by_date):
    """Computes an overall performance metric across all dates."""
    if df_by_date.empty:
        return pd.DataFrame()

    total_trades = df_by_date["num_trades"].sum()
    total_wins = df_by_date["wins"].sum()
    total_profit = df_by_date["total_profit"].sum()

    win_rate = total_wins / total_trades if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0

    return pd.DataFrame([{
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_profit": total_profit,
        "win_rate": win_rate,
        "avg_profit": avg_profit
    }])