#!/usr/bin/env python3

import pandas as pd
import os
import argparse
import importlib
import multiprocessing
import concurrent
from concurrent.futures import ProcessPoolExecutor
from strategy_tester import StrategyTester

DATA_DIR = "/ml/fin/us_stocks_sip/minute_aggs_matrix"
NUM_WORKERS = min(8, os.cpu_count())  # Up to 8 parallel workers

# Fix: Ensure multiprocessing works properly
multiprocessing.set_start_method("spawn", force=True)

def get_most_recent_file(directory):
    """Finds the most recent .parquet file in the given directory."""
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".parquet")],
        reverse=True
    )
    if not files:
        raise FileNotFoundError("No data files found.")
    return os.path.join(directory, files[0])

def get_files_in_range(directory, start_date, end_date):
    """Finds all .parquet files in the directory that fall within the date range."""
    files = sorted([f for f in os.listdir(directory) if f.endswith(".parquet")])
    
    filtered_files = [
        f for f in files if start_date <= f.replace(".parquet", "") <= end_date
    ]

    if not filtered_files:
        raise FileNotFoundError(f"No data files found in range {start_date} to {end_date}.")

    return [os.path.join(directory, f) for f in filtered_files]

def load_strategy(strategy_file):
    """Dynamically loads the Strategy class from a given Python file."""
    module_name = os.path.splitext(os.path.basename(strategy_file))[0]

    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    if spec is None:
        raise ImportError(f"Could not create a spec for {strategy_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    strategy_classes = [cls for cls in dir(module) if cls.endswith("Strategy") and cls != "Strategy"]
    if not strategy_classes:
        raise ImportError(f"No strategy class found in {strategy_file}")

    strategy_class = getattr(module, strategy_classes[0])
    return strategy_class

def run_strategy_on_file(data_file, strategy_file):
    """Runs a strategy on a single date file and returns results."""
    print(f"Processing: {data_file}")

    # Load strategy inside the worker process (Fix: avoid pickling issue)
    strategy_class = load_strategy(strategy_file)

    df = pd.read_parquet(data_file)

    # Extract tickers
    stock_symbols = set(col.split("_")[0] for col in df.columns if "_volume" in col)

    # Compute ranking metric
    stock_metrics = {
        stock: (df[f"{stock}_volume"] * df[f"{stock}_close"]).sum()
        for stock in stock_symbols
        if f"{stock}_volume" in df.columns and f"{stock}_close" in df.columns
    }

    # Rank and select top stocks
    df_ranking = pd.DataFrame(stock_metrics.items(), columns=["Stock", "RankMetric"])
    df_ranking = df_ranking.sort_values(by="RankMetric", ascending=False)
    top_stocks = df_ranking["Stock"].head(256).tolist()
    selected_columns = [col for col in df.columns if col.split("_")[0] in top_stocks]
    df_top = df[selected_columns].copy()

    # Run strategy
    tester = StrategyTester(strategy_class)
    results = tester.run(df_top)

    # Add date column for aggregation
    date = data_file.split("/")[-1].replace(".parquet", "")
    if results["overall_performance"] is not None:
        results["overall_performance"]["date"] = date

    if results["performance_by_ticker"] is not None:
        results["performance_by_ticker"]["date"] = date

    if results["performance_by_hour"] is not None:
        results["performance_by_hour"]["date"] = date

    if not results["trades"].empty:
        results["trades"]["date"] = date

    return results

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trading strategy on stock data.")
    parser.add_argument("strategy_file", help="Python file containing the Strategy class.")
    parser.add_argument("--date", type=str, help="Date to run the strategy on (YYYY-MM-DD).")
    parser.add_argument("--start", type=str, help="Start date for range (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, help="End date for range (YYYY-MM-DD).")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage per trade ($)")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade (%)")
    parser.add_argument("--output", type=str, default="aggregated_results.parquet", help="Output Parquet file name.")
    parser.add_argument("--mode", type=str, default="parallel", help="parallel|sequential")

    args = parser.parse_args()

    # Determine which files to process
    if args.date:
        data_files = [os.path.join(DATA_DIR, f"{args.date}.parquet")]
    elif args.start and args.end:
        data_files = get_files_in_range(DATA_DIR, args.start, args.end)
    else:
        print("No date range specified, using the most recent available data.")
        data_files = [get_most_recent_file(DATA_DIR)]

    # Run strategies in parallel (Fix: Load strategy inside workers)
    all_results = []

    if args.mode == "parallel":
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(run_strategy_on_file, file, args.strategy_file): file for file in data_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
    elif args.mode == "sequential":
        for file in data_files:
            print(f"Running on {file}")
            result = run_strategy_on_file(file, args.strategy_file)
            all_results.append(result)

    # Collect results
    all_by_date = [res["overall_performance"] for res in all_results if res["overall_performance"] is not None]
    all_by_ticker = [res["performance_by_ticker"] for res in all_results if res["performance_by_ticker"] is not None]
    all_by_hour = [res["performance_by_hour"] for res in all_results if res["performance_by_hour"] is not None]

    # Convert lists to DataFrames
    df_by_date = pd.concat(all_by_date, ignore_index=True) if all_by_date else pd.DataFrame()
    df_by_ticker = aggregate_ticker_stats(pd.concat(all_by_ticker, ignore_index=True)) if all_by_ticker else pd.DataFrame()
    df_by_hour = aggregate_hourly_stats(pd.concat(all_by_hour, ignore_index=True)) if all_by_hour else pd.DataFrame()

    df_overall = compute_overall_performance(df_by_date)

    # Save results
    df_by_date.to_parquet(args.output.replace(".parquet", "_by_date.parquet"))
    df_by_ticker.to_parquet(args.output.replace(".parquet", "_by_ticker.parquet"))
    df_by_hour.to_parquet(args.output.replace(".parquet", "_by_hour.parquet"))

    print(f"✅ Parallel processing complete! Results saved.")

    print("\n📊 **Overall Performance Metrics:**")
    print(df_overall)

    if not df_by_date.empty:
        print("\n🏆 **Top 10 Win Days**:")
        print(df_by_date.nlargest(10, "win_rate")[["date", "num_trades", "total_profit",  "win_rate", "avg_profit"]])

        print("\n💰 **Top 10 Profit Days**:")
        print(df_by_date.nlargest(10, "total_profit")[["date", "num_trades", "total_profit", "win_rate", "avg_profit"]])

    if not df_by_ticker.empty:
        print("\n🏆 **Top 10 Win Tickers**:")
        print(df_by_ticker.nlargest(10, "win_rate")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

        print("\n💰 **Top 10 Profit Tickers**:")
        print(df_by_ticker.nlargest(10, "total_profit")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

    if not df_by_hour.empty:
        print("\n🏆 **Top 3 Win Hours (across all days)**:")
        print("\n💰 **Top 3 Profit Hours (across all days)**:")
        print(df_by_hour)