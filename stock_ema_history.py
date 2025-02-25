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
        
        # Preallocate arrays for timestamps, update indices, and prices.
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
        # Each entry is a list of dictionaries with keys: update_idx, timestamp, price, stock_index, type ('min' or 'max').
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
                # Weights: earliest price gets weight (1 - a)^(n-1), newest gets weight 1.
                weights = (1 - a) ** np.arange(n - 1, -1, -1)
                new_ema = np.dot(weights, hist_prices) / weights.sum()
                self.ema[window][self.index, :] = new_ema
            else:
                if self.count == 0:
                    # For the first update, initialize EMA to the current price.
                    self.ema[window][self.index, :] = price_vector
                else:
                    # Use the recursive update from the previous EMA value.
                    prev_index = (self.index - 1) % self.max_history
                    prev_ema = self.ema[window][prev_index, :]
                    self.ema[window][self.index, :] = a * price_vector + (1 - a) * prev_ema

        # After updating EMA, try detecting local extrema for each EMA window.
        for window in self.ema_windows:
            # Retrieve the EMA history, timestamps, and update indices for this window.
            ema_hist = self._get_rolling_array(self.ema[window])
            ts_hist = self._get_rolling_array(self.timestamps)
            idx_hist = self._get_rolling_array(self.update_indices)
            n = ema_hist.shape[0]
            # Require at least 5 points to have two neighbors on each side.
            if n >= 5:
                # We'll check the candidate point at position n - 3 (ensuring it has two later points).
                cand_pos = n - 3
                for stock in range(self.num_stocks):
                    cand_value = ema_hist[cand_pos, stock]
                    # Compare with two previous and two next points.
                    prev_vals = ema_hist[cand_pos-2:cand_pos, stock]
                    next_vals = ema_hist[cand_pos+1:cand_pos+3, stock]
                    
                    is_min = np.all(cand_value < prev_vals) and np.all(cand_value < next_vals)
                    is_max = np.all(cand_value > prev_vals) and np.all(cand_value > next_vals)
                    
                    # Heuristic: avoid double detections by checking the last recorded event for this stock/window.
                    last_event = None
                    for event in reversed(self.extrema[window]):
                        if event['stock_index'] == stock:
                            last_event = event
                            break
                    # Use absolute update index from candidate.
                    cand_abs_idx = idx_hist[cand_pos]
                    if last_event is not None:
                        if abs(cand_abs_idx - last_event['update_idx']) < 2:
                            # Too close in time to the last event; skip detection.
                            continue
                    if is_min or is_max:
                        event = {
                            'update_idx': int(cand_abs_idx),
                            'timestamp': ts_hist[cand_pos],
                            'price': float(cand_value),
                            'stock_index': stock,
                            'type': 'min' if is_min else 'max'
                        }
                        self.extrema[window].append(event)
        
        # Drop any extrema events that are outside the current rolling history.
        self._drop_old_extrema()
        
        # Advance the circular buffer pointer.
        self.index = (self.index + 1) % self.max_history
        self.count += 1

    def _get_rolling_array(self, arr):
        """
        Retrieve the rolling array in chronological order.
        """
        if self.count < self.max_history:
            return arr[:self.count]
        # When the buffer is full, rearrange so that the oldest element comes first.
        return np.concatenate((arr[self.index:], arr[:self.index]), axis=0)

    def _drop_old_extrema(self):
        """
        Remove extrema events that fall outside the current rolling history.
        An event is considered old if its update index is less than the threshold.
        """
        threshold = self.count - min(self.count, self.max_history)
        for window in self.ema_windows:
            self.extrema[window] = [e for e in self.extrema[window] if e['update_idx'] >= threshold]

    def get_price_history(self):
        """
        Returns:
          A NumPy array of shape (num_entries, num_stocks) containing the rolling history of prices.
        """
        return self._get_rolling_array(self.prices)
    
    def get_timestamp_history(self):
        """
        Returns:
          A NumPy array of timestamps in chronological order.
        """
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
            - update_idx: the absolute update index (for internal reference)
        """
        # Combine events from all EMA windows.
        all_events = []
        for window in self.ema_windows:
            all_events.extend(self.extrema[window])
        # Optionally, sort events by update_idx.
        all_events.sort(key=lambda e: e['update_idx'])
        return all_events

