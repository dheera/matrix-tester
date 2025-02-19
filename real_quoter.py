#!/usr/bin/env python3

from datetime import datetime
import pandas as pd
from quoter import Quoter

class RealQuoter(Quoter):
    def __init__(self, data_path = "/fin/us_stocks_sip/quotes"):
        self.data_path = data_path
        self.df_by_ticker = {}

    def get_bid_ask(self, timestamp, ticker):
        """
        Returns the best bid and ask as of the given `timestamp`.
        """
        # Convert to date string for the parquet path
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # If not already loaded, read the parquet for that ticker & date
        if (ticker, date_str) not in self.df_by_ticker:
            # Build the file path with the date/ticker
            path = os.path.join(self.data_path, date_str, f"{date_str}-{ticker}.parquet"
            df = pd.read_parquet(path)
            # Make sip_timestamp the index to simplify lookups
            df = df.set_index('sip_timestamp')
            self.df_by_ticker[(ticker, date_str)] = df

        df = self.df_by_ticker[(ticker, date_str)]

        # “As-of” lookup:
        #   1) We find the largest sip_timestamp that is <= the requested timestamp
        #   2) If nothing is earlier, you may want to handle that case (return None?).
        # .asof() is a Pandas method on a DatetimeIndex
        # If your index is not exactly a DatetimeIndex, convert it:
        # df.index = pd.to_datetime(df.index)
        # df = df.sort_index()

        # Use .asof if you want the last valid timestamp up to `timestamp`:
        #   idx = df.index.asof(timestamp)
        #   if pd.isnull(idx):
        #       # Means no quote earlier or at that time. Handle as you wish.
        #       return None, None
        #   row = df.loc[idx]

        # If you only want an EXACT match, do something like:
        #   try:
        #       row = df.loc[timestamp]
        #   except KeyError:
        #       # No exact match. Handle as you wish.
        #       return None, None

        # Example with "as-of" logic:
        idx = df.index.asof(timestamp)
        if pd.isnull(idx):
            return None, None  # or raise an exception

        row = df.loc[idx]
        return (row['bid_price'], row['ask_price'])

