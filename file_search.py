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

def get_ticker_files_in_range(directory, start_date, end_date, tickers = []):
    """Finds all .parquet files in the directory that fall within the date range for the ticker."""

    if len(tickers) == 0:
        raise ValueError("Must request at least 1 ticker.")

    files = sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])
    
    filtered_subdirs = [
        f for f in files if start_date <= f <= end_date
    ]

    if not filtered_subdirs:
        raise FileNotFoundError(f"No data files found in range {start_date} to {end_date}.")

    files = []
    for subdir in filtered_subdirs:
        files.append({ticker: os.path.join(directory, subdir, f"{subdir}-{ticker}.parquet") for ticker in tickers})

    return files

