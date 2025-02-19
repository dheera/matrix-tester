#!/usr/bin/env python3

import pandas as pd
import os
import argparse
import importlib
import multiprocessing
import concurrent
from concurrent.futures import ProcessPoolExecutor
from strategy_tester import StrategyTester
from aggregation import aggregate
from file_search import get_files_in_range, get_most_recent_file

# Fix: Ensure multiprocessing works properly
multiprocessing.set_start_method("spawn", force=True)

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

def run_strategy_on_file(data_file, strategy_file, output_dir):
    """Runs a strategy on a single date file and returns results."""
    print(f"Processing: {data_file}")

    # Load strategy inside the worker process (Fix: avoid pickling issue)
    strategy_class = load_strategy(strategy_file)

    df = pd.read_parquet(data_file)

    # Extract tickers
    stock_symbols = set(df.columns.get_level_values(0).unique())

    # Compute ranking metric
    stock_metrics = {
        stock: (df[stock, "volume"] * df[stock, "close"]).sum()
        for stock in stock_symbols
    }

    # Rank and select top stocks
    df_ranking = pd.DataFrame(stock_metrics.items(), columns=["Stock", "RankMetric"])
    df_ranking = df_ranking.sort_values(by="RankMetric", ascending=False)
    top_stocks = df_ranking["Stock"].head(256).tolist()
    df_top = df[top_stocks].copy()

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

    stem = os.path.basename(data_file).replace(".parquet", "")
    os.makedirs(os.path.join(output_dir, "daily", stem), exist_ok = True)
    results["overall_performance"].to_parquet(os.path.join(output_dir, "daily", stem, "overall.parquet"))
    results["performance_by_ticker"].to_parquet(os.path.join(output_dir, "daily", stem, "by_ticker.parquet"))
    results["performance_by_hour"].to_parquet(os.path.join(output_dir, "daily", stem, "by_hour.parquet"))
    results["trades"].to_parquet(os.path.join(output_dir, "daily", stem, "trades.parquet"))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trading strategy on stock data.")
    parser.add_argument("strategy_file", help="Python file containing the Strategy class.")
    parser.add_argument("--date", type=str, help="Date to run the strategy on (YYYY-MM-DD).")
    parser.add_argument("--start", type=str, help="Start date for range (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, help="End date for range (YYYY-MM-DD).")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage per trade ($)")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade (%)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory to put aggregated parquet output")
    parser.add_argument("--mode", type=str, default="parallel", help="parallel|sequential")
    parser.add_argument("--data-dir", type=str, default="/fin/matrix", help="Where is the data")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count()) , help="How many workers")

    args = parser.parse_args()

    # Determine which files to process
    if args.date:
        data_files = [os.path.join(args.data_dir, f"{args.date}.parquet")]
    elif args.start and args.end:
        data_files = get_files_in_range(args.data_dir, args.start, args.end)
    else:
        print("No date range specified, using the most recent available data.")
        data_files = [get_most_recent_file(args.data_dir)]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "daily"), exist_ok=True)

    # Run strategies in parallel (Fix: Load strategy inside workers)
    all_results = []

    if args.mode == "parallel":
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(run_strategy_on_file, file, args.strategy_file, args.output_dir): file for file in data_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
        print(f"✅ Parallel processing complete")

    elif args.mode == "sequential":
        for file in data_files:
            print(f"Running on {file}")
            result = run_strategy_on_file(file, args.strategy_file, args.output_dir)
            all_results.append(result)
        print(f"✅ Sequential processing complete")

    aggregated = aggregate(all_results)

    print("\n📊 **Overall Performance Metrics:**")
    print(aggregated["overall"])

    if not aggregated["by_date"].empty:
        print("\n🏆 **Top 10 Win Days**:")
        print(aggregated["by_date"].nlargest(10, "win_rate")[["date", "num_trades", "total_profit",  "win_rate", "avg_profit"]])

        print("\n💰 **Top 10 Profit Days**:")
        print(aggregated["by_date"].nlargest(10, "total_profit")[["date", "num_trades", "total_profit", "win_rate", "avg_profit"]])

    if not aggregated["by_ticker"].empty:
        print("\n🏆 **Top 10 Win Tickers**:")
        print(aggregated["by_ticker"].nlargest(10, "win_rate")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

        print("\n💰 **Top 10 Profit Tickers**:")
        print(aggregated["by_ticker"].nlargest(10, "total_profit")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

    if not aggregated["by_hour"].empty:
        print("\n🏆 **Top 3 Win Hours (across all days)**:")
        print("\n💰 **Top 3 Profit Hours (across all days)**:")
        print(aggregated["by_hour"])

    # Save results
    aggregated["overall"].to_parquet(os.path.join(args.output_dir, "overall.parquet"))
    aggregated["by_date"].to_parquet(os.path.join(args.output_dir, "by_date.parquet"))
    aggregated["by_ticker"].to_parquet(os.path.join(args.output_dir, "by_ticker.parquet"))
    aggregated["by_hour"].to_parquet(os.path.join(args.output_dir, "by_hour.parquet"))
