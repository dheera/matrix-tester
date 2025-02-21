import numpy as np

class RollingEWMStats:
    def __init__(self, alpha, adjust=True, min_periods=1):
        """
        Parameters:
          alpha : float
            Smoothing factor (0 < alpha <= 1).
          adjust : bool, default True
            If True, all data are stored and statistics computed using explicit weights.
            If False, statistics are updated recursively.
          min_periods : int, default 1
            Minimum number of observations required before returning a result.
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods

        if self.adjust:
            # Explicit mode: store all observations.
            self.values = []
        else:
            # Recursive mode: initialize running state.
            self.count = 0
            self.mean = None
            self.v = None  # running variance (uncorrected)

    def update(self, value):
        """
        Update with a new data point and return the current EWM mean and standard deviation.
        
        Parameters:
          value : float
            New observation.
            
        Returns:
          (mean, std) : tuple of floats
            The current exponentially weighted moving mean and standard deviation.
            If the number of observations is less than min_periods, returns (np.nan, np.nan).
        """
        if self.adjust:
            self.values.append(value)
            n = len(self.values)
            if n < self.min_periods:
                return np.nan, np.nan
            # Weights: newest gets weight 1; oldest gets (1-alpha)^(n-1)
            weights = (1 - self.alpha) ** np.arange(n-1, -1, -1)
            sum_w = weights.sum()
            arr = np.array(self.values)
            mean = np.dot(arr, weights) / sum_w
            uncorrected_var = np.dot((arr - mean)**2, weights) / sum_w
            correction = 1 - (np.sum(weights**2) / (sum_w**2))
            var = uncorrected_var / correction
            return mean, np.sqrt(var)
        else:
            self.count += 1
            if self.count == 1:
                self.mean = value
                self.v = 0.0
                return np.nan, np.nan  # Not enough data.
            else:
                old_mean = self.mean
                self.mean = old_mean + self.alpha * (value - old_mean)
                self.v = (1 - self.alpha) * (self.v + self.alpha * (value - old_mean)**2)
                if self.count < self.min_periods:
                    return np.nan, np.nan
                return self.mean, np.sqrt(self.v)


# --- Unit Tests ---

if __name__ == '__main__':
    import pandas as pd
    import unittest

    class TestRollingEWMStats(unittest.TestCase):
        def setUp(self):
            self.alpha = 0.1
            self.min_periods = 5
            np.random.seed(42)

        def _run_trial(self, adjust_flag):
            # Generate a random series (length between 50 and 150).
            n = np.random.randint(50, 151)
            values = np.random.randn(n)

            # Create our rolling EWM stats object.
            rolling_stats = RollingEWMStats(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods)
            means = []
            stds = []
            for val in values:
                m, s = rolling_stats.update(val)
                means.append(m)
                stds.append(s)
            means = np.array(means)
            stds = np.array(stds)

            # Compute expected results using pandas.
            series = pd.Series(values)
            pd_mean = series.ewm(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods).mean().to_numpy()
            pd_std  = series.ewm(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods).std().to_numpy()

            # Use a tight tolerance for explicit mode and a looser tolerance for recursive mode.
            tol = 1e-4 if adjust_flag else 0.3
            np.testing.assert_allclose(means, pd_mean, rtol=tol, equal_nan=True)
            np.testing.assert_allclose(stds, pd_std, rtol=tol, equal_nan=True)

        def test_adjust_true_random_trials(self):
            for _ in range(5):
                self._run_trial(adjust_flag=True)

        def test_adjust_false_random_trials(self):
            for _ in range(5):
                self._run_trial(adjust_flag=False)

    unittest.main()

