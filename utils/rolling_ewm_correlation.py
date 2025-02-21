import numpy as np
import unittest

class RollingEWMCorr:
    def __init__(self, alpha, adjust=True, min_periods=1):
        """
        Parameters:
          alpha : float
            Smoothing factor (0 < alpha <= 1).
          adjust : bool, default True
            When True, stores all data points and computes the correlation using explicit exponential weights.
            When False, uses recursive updates to maintain the weighted means, variances, and covariance.
          min_periods : int, default 1
            Minimum number of data points required for a valid correlation output.
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods

        if self.adjust:
            # Store all data points.
            self.xs = []
            self.ys = []
        else:
            # Recursive mode: initialize state variables.
            self.count = 0
            self.mean_x = None
            self.mean_y = None
            self.var_x = None
            self.var_y = None
            self.cov = None

    def update(self, x, y):
        """
        Update the correlation with a new (x, y) data point and return the latest correlation.

        Returns:
          corr : float
            Latest correlation estimate. Returns np.nan if there are fewer than min_periods points,
            or if the variances are zero.
        """
        if self.adjust:
            # Append new data.
            self.xs.append(x)
            self.ys.append(y)
            n = len(self.xs)
            if n < self.min_periods:
                return np.nan

            # Compute weights: newest gets weight 1; oldest gets (1-alpha)^(n-1)
            weights = (1 - self.alpha) ** np.arange(n-1, -1, -1)
            x_arr = np.array(self.xs)
            y_arr = np.array(self.ys)
            sum_w = weights.sum()

            # Weighted means.
            mean_x = np.sum(x_arr * weights) / sum_w
            mean_y = np.sum(y_arr * weights) / sum_w

            # Weighted covariance and variances.
            cov = np.sum(weights * (x_arr - mean_x) * (y_arr - mean_y)) / sum_w
            var_x = np.sum(weights * (x_arr - mean_x)**2) / sum_w
            var_y = np.sum(weights * (y_arr - mean_y)**2) / sum_w

            if var_x > 0 and var_y > 0:
                return cov / np.sqrt(var_x * var_y)
            else:
                return np.nan

        else:
            # Recursive update mode.
            self.count += 1
            if self.count == 1:
                self.mean_x = x
                self.mean_y = y
                self.var_x = 0.0
                self.var_y = 0.0
                self.cov = 0.0
                return np.nan
            else:
                delta_x = x - self.mean_x
                delta_y = y - self.mean_y

                # Update exponentially weighted means.
                self.mean_x += self.alpha * delta_x
                self.mean_y += self.alpha * delta_y

                # Update covariance and variances.
                self.cov = (1 - self.alpha) * (self.cov + self.alpha * delta_x * delta_y)
                self.var_x = (1 - self.alpha) * (self.var_x + self.alpha * delta_x**2)
                self.var_y = (1 - self.alpha) * (self.var_y + self.alpha * delta_y**2)

                if self.count < self.min_periods or self.var_x <= 0 or self.var_y <= 0:
                    return np.nan
                else:
                    return self.cov / np.sqrt(self.var_x * self.var_y)

# Unit test for RollingEWMCorr
class TestRollingEWMCorr(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.1
        self.min_periods = 5
        np.random.seed(42)  # for reproducibility

    def _run_trial(self, adjust_flag):
        import pandas as pd
        # Create a random data series of length between 50 and 150.
        n = np.random.randint(50, 151)
        x_vals = np.random.randn(n)
        # Create a correlated y: y = 0.5*x + noise.
        y_vals = 0.5 * x_vals + np.random.randn(n) * 0.5

        # Initialize our rolling EWM correlation.
        rolling_corr = RollingEWMCorr(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods)

        # Collect rolling correlation results.
        rolling_results = []
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            corr = rolling_corr.update(x, y)
            rolling_results.append(corr)

        rolling_results = np.array(rolling_results)

        # Now compute the expected results using pandas on each growing window.
        expected_results = []
        for i in range(n):
            if i + 1 < self.min_periods:
                expected_results.append(np.nan)
            else:
                # For each window, generate the appropriate weights.
                if adjust_flag:
                    # Explicit weighting:
                    window = pd.DataFrame({'x': x_vals[:i+1], 'y': y_vals[:i+1]})
                    # pandas uses the same weighting as we do (newest weight=1, oldest=(1-alpha)^(n-1))
                    expected_corr = window['x'].ewm(alpha=self.alpha, adjust=True,
                                                      min_periods=self.min_periods).corr(window['y']).iloc[-1]
                    expected_results.append(expected_corr)
                else:
                    # For recursive mode, pandas supports adjust=False.
                    window = pd.DataFrame({'x': x_vals[:i+1], 'y': y_vals[:i+1]})
                    expected_corr = window['x'].ewm(alpha=self.alpha, adjust=False,
                                                      min_periods=self.min_periods).corr(window['y']).iloc[-1]
                    expected_results.append(expected_corr)

        expected_results = np.array(expected_results)

        # Compare the rolling results with the expected results from pandas.
        np.testing.assert_allclose(rolling_results, expected_results, rtol=1e-6, equal_nan=True)

    def test_adjust_true_random_trials(self):
        # Run 5 random trials with adjust=True.
        for _ in range(5):
            self._run_trial(adjust_flag=True)

    def test_adjust_false_random_trials(self):
        # Run 5 random trials with adjust=False.
        for _ in range(5):
            self._run_trial(adjust_flag=False)

if __name__ == '__main__':
    unittest.main()

