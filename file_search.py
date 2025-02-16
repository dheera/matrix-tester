import os

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

