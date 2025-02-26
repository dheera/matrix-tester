#!/usr/bin/env python3

import pandas as pd
import shutil
import os
import argparse
import importlib
import multiprocessing
import concurrent
from concurrent.futures import ProcessPoolExecutor
from strategy_tester import StrategyTester
from strategy import Strategy
from aggregation import aggregate
from file_search import get_files_in_range, get_ticker_files_in_range, get_most_recent_date
import time

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

def read_merged(files):
    dfs = []
    for ticker in files:
        if ticker.startswith("_"): # ignore the _date key
            continue

        if "stocks" in files[ticker]:
            print(f'Reading stocks data from {files[ticker]["stocks"]}')
            df = pd.read_parquet(files[ticker]["stocks"])
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            dfs.append(df)

        if "options" in files[ticker]:
            print(f'Reading options data from {files[ticker]["options"]}')
            df = pd.read_parquet(files[ticker]["options"])
            
            df = df.reset_index()

            # Create the ticker column e.g. O:XSP25.....
            # Here, we assume that the underlying symbol is 'XSP'.
            # Multiply the strike by 1000 and format as an 8-digit number.
            df['ticker'] = (
                'O:' + ticker +
                df['expiry'].astype(str) +
                df['type'] +
                (df['strike'] * 1000).astype(int).apply(lambda x: f"{x:08d}")
            )

            # Pivot the DataFrame so that 'window_start' becomes the row index and
            # the columns are the metrics, grouped by ticker.
            pivot_df = df.pivot(
                index='window_start',
                columns='ticker',
                values=['last', 'last_size', 'volume', 'bid', 'bid_size', 'ask', 'ask_size']
            )

            # By default, this gives a MultiIndex on the columns with the metric as the first level.
            # Swap levels so that the ticker is the top level.
            pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

            volume_cols = [col for col in pivot_df.columns if col[1] == 'volume']
            non_volume_cols = [col for col in pivot_df.columns if col[1] != 'volume']

            # Forward fill the non-volume columns (bid, ask, bid_size, ask_size, last, last_size) along the index:
            pivot_df[non_volume_cols] = pivot_df[non_volume_cols].ffill()

            # For volume columns, instead of forward filling, replace NaNs with 0:
            pivot_df[volume_cols] = pivot_df[volume_cols].fillna(0)

            dfs.append(pivot_df)

    return pd.concat(dfs, axis = 1, join = "inner")

def run_strategy_on_file(data_files, strategy_file, output_dir, slippage, commission, strategy_args):
    """Runs a strategy on a single date file and returns results."""
    # Load strategy inside the worker process (Fix: avoid pickling issue)
    strategy_class = load_strategy(strategy_file)
    tester = StrategyTester(strategy_class, reset_every_day = False, slippage = slippage, commission = commission, strategy_args = strategy_args)

    results_collection = []

    for i, data_file in enumerate(data_files):
        print(f"Processing: {data_file}")

        if type(data_file) is dict:
            # multiple data files need to be merged first
            df = read_merged(data_file)
        else:
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
            top_stocks = df_ranking["Stock"].head(1024).tolist()
            df = df[top_stocks].copy()

        # Run strategy
        results = tester.run(
            df,
            force_exit_on_market_close = (i == len(data_files) - 1), # we need to force exit for the last day in sequential
        )
    
        # Add date column for aggregation
        if type(data_file) is dict:
            date = data_file["_date"]
        else:
            date = data_file.split("/")[-1].replace(".parquet", "")
        if results["overall_performance"] is not None:
            results["overall_performance"]["date"] = date
    
        if results["performance_by_ticker"] is not None:
            results["performance_by_ticker"]["date"] = date
    
        if results["performance_by_hour"] is not None:
            results["performance_by_hour"]["date"] = date
    
        if not results["trades"].empty:
            results["trades"]["date"] = date
    
        os.makedirs(os.path.join(output_dir, "daily", date), exist_ok = True)

        if results["overall_performance"] is not None:
            results["overall_performance"].to_parquet(os.path.join(output_dir, "daily", date, "overall.parquet"))
        if results["performance_by_ticker"] is not None:
            results["performance_by_ticker"].to_parquet(os.path.join(output_dir, "daily", date, "by_ticker.parquet"))
        if results["performance_by_hour"] is not None:
            results["performance_by_hour"].to_parquet(os.path.join(output_dir, "daily", date, "by_hour.parquet"))
        if results["trades"] is not None and len(results["trades"]) > 0:
            results["trades"][results["trades"]["action"]==Strategy.EXIT].to_parquet(os.path.join(output_dir, "daily", date, "exits.parquet"))
        if results["actions"] is not None:
            results["actions"].to_parquet(os.path.join(output_dir, "daily", date, "actions.parquet"))
   
        results_collection.append(results)

    return results_collection

def convert_value(v):
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v

def parse_strategy_args(s):
    # Split the string on commas, then on '=' for each pair.
    if not s:
        return {}
    return {
        key.strip(): convert_value(value)
        for part in s.split(' ')
        for key, value in [part.split('=')]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trading strategy on stock data.")
    parser.add_argument("strategy_file", help="Python file containing the Strategy class.")
    parser.add_argument("--date", type=str, help="Date to run the strategy on (YYYY-MM-DD).")
    parser.add_argument("--start", type=str, help="Start date for range (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, help="End date for range (YYYY-MM-DD).")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage per trade")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory to put aggregated parquet output")
    parser.add_argument("--mode", type=str, default="parallel", help="parallel|sequential")
    parser.add_argument("--matrix-dir", type=str, default="/fin/us_stocks_sip/minute_aggs_matrix_2048", help="Matrix data dir")
    parser.add_argument("--stocks-tq-aggs-dir", type=str, default="/fin/us_stocks_sip/tq_aggs", help="Stocks TQ aggs data dir")
    parser.add_argument("--options-tq-aggs-dir", type=str, default="/fin/us_options_opra/tq_aggs", help="Options TQ aggs data dir")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count()) , help="How many workers")
    parser.add_argument("--strategy-args", type=str, default = "", help="Pass args to strategy")

    args = parser.parse_args()

    # Load an example strategy instance so that we can see what kind of data it requests

    strategy_args = parse_strategy_args(args.strategy_args)

    example_strategy_instance = load_strategy(args.strategy_file)(tester=None, **strategy_args)
    data_mode = example_strategy_instance._data_mode
    request_stocks = example_strategy_instance._request_stocks
    request_options = example_strategy_instance._request_options
    exit_on_market_close = example_strategy_instance._exit_on_market_close

    if (not exit_on_market_close) and args.mode == "parallel":
        print("Warning: Because the selected strategy does not exit on market close, sequential mode is being forced since you cannot evaluate multiple days simultaneously")
        args.mode = "sequential"

    # Determine which files to process
    if args.date:
        start = args.date
        end = args.date
    elif args.start and args.end:
        start = args.start
        end = args.end
    else:
        start = None
        end = None

    if data_mode == "ohlc":
        if start is None and end is None:
            start = end = get_most_recent_date(args.matrix_dir)
            print(f"Warning: No date specified, processing on most recent date {start}")
        data_files = get_files_in_range(args.matrix_dir, start, end)
    elif data_mode == "bidask":
        if start is None and end is None:
            start = end = get_most_recent_date(args.stocks_tq_aggs_dir)
            print(f"Warning: No date specified, processing on most recent date {start}")
        data_files = get_ticker_files_in_range(args.stocks_tq_aggs_dir, args.options_tq_aggs_dir, start, end, stocks = request_stocks, options = request_options)

    if len(data_files) == 0:
        print("No data in range to process, exiting.")
        exit(1)

    print("**** arg summary ****")

    print("data mode:", data_mode)
    print("run mode:", args.mode)
    print("request stocks:", request_stocks)
    print("request options:", request_options)
    print("date range:", start, "->", end)
    print("data files:", data_files)

    print("*********************")

    os.makedirs(os.path.join(args.output_dir, "daily"), exist_ok=True)
    shutil.rmtree(os.path.join(args.output_dir, "daily"))
    os.makedirs(os.path.join(args.output_dir, "daily"), exist_ok=True)

    # Run strategies in parallel (Fix: Load strategy inside workers)
    all_results = []

    if args.mode == "parallel":
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(run_strategy_on_file, [file], args.strategy_file, args.output_dir, args.slippage, args.commission, strategy_args): file for file in data_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_results.append(future.result()[0])
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
        print(f"✅ Parallel processing complete")

    elif args.mode == "sequential":
        all_results = run_strategy_on_file(data_files, args.strategy_file, args.output_dir, args.slippage, args.commission, strategy_args)
        print(f"✅ Sequential processing complete")

    aggregated = aggregate(all_results)

    print("\n📊 **Overall Performance Metrics:**")
    print(aggregated["overall"])

    if not aggregated["by_date"].empty:
        print("\n🏆 **Top 10 Win Days**:")
        print(aggregated["by_date"].nlargest(10, "win_rate")[["date", "num_trades", "total_profit",  "win_rate", "avg_profit"]])

        print("\n🏆 **Worst 10 Win Days**:")
        print(aggregated["by_date"].nsmallest(10, "win_rate")[["date", "num_trades", "total_profit",  "win_rate", "avg_profit"]])

        print("\n💰 **Top 10 Profit Days**:")
        print(aggregated["by_date"].nlargest(10, "total_profit")[["date", "num_trades", "total_profit", "win_rate", "avg_profit"]])

        print("\n💰 **Worst 10 Profit Days**:")
        print(aggregated["by_date"].nsmallest(10, "total_profit")[["date", "num_trades", "total_profit", "win_rate", "avg_profit"]])

    if not aggregated["by_ticker"].empty:
        print("\n🏆 **Top 10 Win Tickers**:")
        print(aggregated["by_ticker"].nlargest(10, "win_rate")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

        print("\n🏆 **Worst 10 Win Tickers**:")
        print(aggregated["by_ticker"].nsmallest(10, "win_rate")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

        print("\n💰 **Top 10 Profit Tickers**:")
        print(aggregated["by_ticker"].nlargest(10, "total_profit")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

        print("\n💰 **Worst 10 Profit Tickers**:")
        print(aggregated["by_ticker"].nsmallest(10, "total_profit")[["ticker", "num_trades", "total_profit", "win_rate", "avg_profit"]])

    if not aggregated["by_hour"].empty:
        print("\n💰 **Results by hour (across all days)**:")
        print(aggregated["by_hour"])

    # Save results
    aggregated["overall"].to_parquet(os.path.join(args.output_dir, "overall.parquet"))
    aggregated["by_date"].to_parquet(os.path.join(args.output_dir, "by_date.parquet"))
    aggregated["by_ticker"].to_parquet(os.path.join(args.output_dir, "by_ticker.parquet"))
    aggregated["by_hour"].to_parquet(os.path.join(args.output_dir, "by_hour.parquet"))
