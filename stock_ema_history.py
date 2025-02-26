import numpy as np

class StockEMAHistory:
    def __init__(self, max_history_length, ema_windows, num_stocks, adjust=False):
        """
        Initialize the rolling history and extrema tracking.

        Parameters:
          max_history_length (int): Maximum number of updates to store.
          ema_windows (list[int]): List of window sizes for which to compute EMAs.
          num_stocks (int): The number of stocks (i.e. the length of each price vector).
          adjust (bool): Whether to use adjusted weighting (like pandas) when computing EMA.
                         If False, uses the recursive formula.
        """
        self.max_history = max_history_length
        self.ema_windows = ema_windows
        self.num_stocks = num_stocks
        self.adjust = adjust

        # Preallocate arrays for timestamps, prices, and update indices.
        self.timestamps = np.empty(max_history_length, dtype=object)
        self.prices = np.empty((max_history_length, num_stocks))
        self.update_indices = np.empty(max_history_length, dtype=int)

        # For each EMA window, preallocate an array for EMA values.
        self.ema = {window: np.empty((max_history_length, num_stocks))
                    for window in ema_windows}

        # Precompute smoothing factors: alpha = 2/(window+1)
        self.alpha = {window: 2 / (window + 1) for window in ema_windows}

        # Circular buffer pointer and total update count.
        self.index = 0       # Next index to write to.
        self.count = 0       # Total number of updates received.

        # Dictionary to store extrema for each EMA window.
        # Each entry is a list of dictionaries with keys:
        # 'update_idx', 'timestamp', 'price', 'stock_index', 'type'
        # plus raw price details: 'raw_extreme_value' and 'raw_extreme_timestamp'.
        self.extrema = {window: [] for window in ema_windows}

    def update(self, timestamp, price_vector):
        """
        Update the history with a new timestamp and stock prices,
        compute the EMA values, and detect local extrema.
        
        Parameters:
          timestamp: The timestamp for this update.
          price_vector (array-like): A vector of stock prices (length must equal num_stocks).
        """
        price_vector = np.array(price_vector)
        if price_vector.shape[0] != self.num_stocks:
            raise ValueError("price_vector length does not match num_stocks")

        # Insert the new data into the circular buffer.
        self.timestamps[self.index] = timestamp
        self.prices[self.index, :] = price_vector
        self.update_indices[self.index] = self.count

        # Compute EMA for each specified window.
        for window in self.ema_windows:
            a = self.alpha[window]
            if self.adjust:
                # Compute the adjusted weighted average over the entire rolling history.
                hist_prices = self._get_rolling_array(self.prices)
                n = hist_prices.shape[0]
                weights = (1 - a) ** np.arange(n - 1, -1, -1)
                new_ema = np.dot(weights, hist_prices) / weights.sum()
                self.ema[window][self.index, :] = new_ema
            else:
                if self.count == 0:
                    # For the first update, initialize EMA to the current price.
                    self.ema[window][self.index, :] = price_vector
                else:
                    prev_index = (self.index - 1) % self.max_history
                    prev_ema = self.ema[window][prev_index, :]
                    self.ema[window][self.index, :] = a * price_vector + (1 - a) * prev_ema

        # After updating EMA, try detecting local extrema for each EMA window.
        for window in self.ema_windows:
            ema_hist = self._get_rolling_array(self.ema[window])
            ts_hist = self._get_rolling_array(self.timestamps)
            idx_hist = self._get_rolling_array(self.update_indices)
            n = ema_hist.shape[0]
            if n >= 5:
                # Use candidate point at position n - 3.
                cand_pos = n - 1
                for stock in range(self.num_stocks):
                    cand_value = ema_hist[cand_pos, stock]
                    prev_vals = ema_hist[cand_pos-1:cand_pos, stock]
                    next_vals = ema_hist[cand_pos+1:cand_pos+2, stock]

                    is_min = np.all(cand_value < prev_vals) and np.all(cand_value < next_vals)
                    is_max = np.all(cand_value > prev_vals) and np.all(cand_value > next_vals)

                    # Check to avoid double detections.
                    last_event = None
                    for event in reversed(self.extrema[window]):
                        if event['stock_index'] == stock:
                            last_event = event
                            break
                    cand_abs_idx = idx_hist[cand_pos]
                    if last_event is not None:
                        if abs(cand_abs_idx - last_event['update_idx']) < 2:
                            continue
                    if is_min or is_max:
                        event = {
                            'update_idx': int(cand_abs_idx),
                            'timestamp': ts_hist[cand_pos],
                            'price': float(cand_value),
                            'stock_index': stock,
                            'type': 'min' if is_min else 'max'
                        }
                        # --- Compute corresponding raw price extreme ---
                        raw_hist = self._get_rolling_array(self.prices)
                        raw_ts = self._get_rolling_array(self.timestamps)
                        raw_len = raw_hist.shape[0]
                        raw_start_idx = self.count - raw_len
                        # Map candidate's absolute update index to raw history index.
                        raw_index = cand_abs_idx - raw_start_idx
                        # Use a window of length (window/4) (at least 1 period).
                        period = max(int(window / 4), 1)
                        raw_win_left = max(0, raw_index - period)
                        raw_win_right = raw_index + 1  # include event point
                        raw_segment = raw_hist[raw_win_left:raw_win_right, stock]
                        raw_ts_segment = raw_ts[raw_win_left:raw_win_right]
                        if is_min:
                            raw_extreme_value = float(raw_segment.min())
                            local_idx = int(np.argmin(raw_segment))
                        else:
                            raw_extreme_value = float(raw_segment.max())
                            local_idx = int(np.argmax(raw_segment))
                        raw_extreme_timestamp = raw_ts_segment[local_idx]
                        # Add raw details to event.
                        event['raw_extreme_value'] = raw_extreme_value
                        event['raw_extreme_timestamp'] = raw_extreme_timestamp
                        # ---------------------------------------------------

                        self.extrema[window].append(event)

        self._drop_old_extrema()

        # Advance circular buffer pointer.
        self.index = (self.index + 1) % self.max_history
        self.count += 1

    def _get_rolling_array(self, arr):
        """Retrieve the rolling array in chronological order."""
        if self.count < self.max_history:
            return arr[:self.count]
        return np.concatenate((arr[self.index:], arr[:self.index]), axis=0)

    def _drop_old_extrema(self):
        """Remove extrema events that fall outside the current rolling history."""
        threshold = self.count - min(self.count, self.max_history)
        for window in self.ema_windows:
            self.extrema[window] = [e for e in self.extrema[window] if e['update_idx'] >= threshold]

    def get_price_history(self):
        """Returns the raw price history in chronological order."""
        return self._get_rolling_array(self.prices)

    def get_timestamp_history(self):
        """Returns the timestamp history in chronological order."""
        return self._get_rolling_array(self.timestamps)

    def get_ema_history(self, window):
        """
        Retrieve the rolling history of EMA values for a specified window.
        
        Parameters:
          window (int): The EMA window size.
        
        Returns:
          A NumPy array of shape (num_entries, num_stocks) with the EMA values.
        """
        if window not in self.ema_windows:
            raise ValueError(f"Window {window} was not specified at initialization.")
        return self._get_rolling_array(self.ema[window])

    def get_extrema(self):
        """
        Retrieve all known extrema (minima and maxima) for all EMA windows that are still within the rolling history.
        
        Returns:
          A list of dictionaries, each containing:
            - stock_index: int (0 ... num_stocks - 1)
            - type: 'min' or 'max'
            - price: float (the EMA value)
            - timestamp: the timestamp associated with the event
            - update_idx: the absolute update index
            - raw_extreme_value: float (the corresponding raw price extreme)
            - raw_extreme_timestamp: the timestamp for the raw price extreme
        """
        all_events = []
        for window in self.ema_windows:
            all_events.extend(self.extrema[window])
        all_events.sort(key=lambda e: e['update_idx'])
        return all_events
