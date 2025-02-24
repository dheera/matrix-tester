import os

def get_most_recent_date(directory):
    """Finds the most recent .parquet file in the given directory."""
    files = sorted(
        [f for f in os.listdir(directory)],
        reverse=True
    )
    if not files:
        raise FileNotFoundError("No data files found.")

    return files[0].replace(".parquet", "")

def get_files_in_range(directory, start_date, end_date):
    """Finds all .parquet files in the directory that fall within the date range."""
    files = sorted([f for f in os.listdir(directory) if f.endswith(".parquet")])
    
    filtered_files = [
        f for f in files if start_date <= f.replace(".parquet", "") <= end_date
    ]

    if not filtered_files:
        raise FileNotFoundError(f"No data files found in range {start_date} to {end_date}.")

    return [os.path.join(directory, f) for f in filtered_files]

def get_ticker_files_in_range(stocks_directory, options_directory, start_date, end_date, stocks = [], options = []):
    """Finds all .parquet files in the directory that fall within the date range for the ticker."""

    if len(stocks)== 0:
        raise ValueError("Must request at least 1 stock ticker.")

    stocks_subdirs = sorted([f for f in os.listdir(stocks_directory) if os.path.isdir(os.path.join(stocks_directory, f))])
    options_subdirs = sorted([f for f in os.listdir(options_directory) if os.path.isdir(os.path.join(options_directory, f))])
    
    filtered_stocks_subdirs = [
        subdir for subdir in stocks_subdirs if start_date <= subdir <= end_date
    ]

    filtered_options_subdirs = [
        subdir for subdir in options_subdirs if start_date <= subdir <= end_date
    ]

    if not filtered_stocks_subdirs:
        raise FileNotFoundError(f"No data files found in range {start_date} to {end_date}.")

    if len(options) > 0 and len(filtered_stocks_subdirs) / len(stocks) != len(filtered_options_subdirs) / len(options): # TODO: super crude and bad, fix later
        print("Stocks or options missing data.")
        print("Found stocks data for:", filtered_stocks_subdirs)
        print("Found options data for:", filtered_options_subdirs)
        exit(1)

    files = []
    for subdir in filtered_stocks_subdirs:
        day_files = {"_date": subdir}
        for ticker in stocks:
            day_files[ticker] = {
                "stocks": os.path.join(stocks_directory, subdir, f"{subdir}-{ticker}.parquet")
            }
        for underlying in options:
            if underlying not in day_files:
                day_files[underlying] = {}
            day_files[underlying]["options"] = os.path.join(options_directory, subdir, f"{subdir}-{underlying}.parquet")

        files.append(day_files)
    
    return files
