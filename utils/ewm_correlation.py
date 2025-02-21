import numpy as np
import unittest

class EWMCorrelation:
    def __init__(self, alpha, adjust=True, min_periods=1):
        """
        Parameters:
          alpha : float
            Smoothing factor, with 0 < alpha <= 1.
          adjust : bool, default True
            When True, weights are computed using (1-alpha)**(n-1-i) for the i-th observation.
            When False, recursive updates are used.
          min_periods : int, default 1
            Minimum number of observations in window required to have a valid result.
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods

    def corr(self, x, y):
        """
        Compute the exponentially weighted moving correlation between two arrays.

        Parameters:
          x, y : array_like (1D)
            The two data series to correlate. Must have the same length.

        Returns:
          result : numpy.ndarray
            Array of correlation values; indices with fewer than min_periods observations are np.nan.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Only 1D arrays are supported.")
        if len(x) != len(y):
            raise ValueError("x and y must be the same length.")

        n = len(x)
        result = np.full(n, np.nan)

        if self.adjust:
            # Compute using explicit weights:
            for t in range(n):
                if t + 1 < self.min_periods:
                    continue  # Not enough observations yet
                # weights for observations 0 ... t: latest gets weight 1, oldest gets (1-alpha)**t
                weights = (1 - self.alpha) ** np.arange(t, -1, -1)
                sum_w = weights.sum()
                mean_x = np.sum(x[:t+1] * weights) / sum_w
                mean_y = np.sum(y[:t+1] * weights) / sum_w
                cov = np.sum(weights * (x[:t+1] - mean_x) * (y[:t+1] - mean_y)) / sum_w
                var_x = np.sum(weights * (x[:t+1] - mean_x)**2) / sum_w
                var_y = np.sum(weights * (y[:t+1] - mean_y)**2) / sum_w

                if var_x > 0 and var_y > 0:
                    result[t] = cov / np.sqrt(var_x * var_y)
                else:
                    result[t] = np.nan
        else:
            # Compute recursively:
            mean_x = np.empty(n)
            mean_y = np.empty(n)
            var_x = np.empty(n)
            var_y = np.empty(n)
            cov = np.empty(n)

            # Initialization with first observation:
            mean_x[0] = x[0]
            mean_y[0] = y[0]
            var_x[0] = 0.0
            var_y[0] = 0.0
            cov[0] = 0.0
            result[0] = np.nan  # not defined for a single observation

            for t in range(1, n):
                delta_x = x[t] - mean_x[t-1]
                delta_y = y[t] - mean_y[t-1]
                mean_x[t] = mean_x[t-1] + self.alpha * delta_x
                mean_y[t] = mean_y[t-1] + self.alpha * delta_y
                cov[t] = (1 - self.alpha) * (cov[t-1] + self.alpha * delta_x * delta_y)
                var_x[t] = (1 - self.alpha) * (var_x[t-1] + self.alpha * delta_x**2)
                var_y[t] = (1 - self.alpha) * (var_y[t-1] + self.alpha * delta_y**2)
                if t + 1 >= self.min_periods and var_x[t] > 0 and var_y[t] > 0:
                    result[t] = cov[t] / np.sqrt(var_x[t] * var_y[t])
                else:
                    result[t] = np.nan

        return result

class TestEWMCorrelation(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.5
        self.min_periods = 2
        # We'll use a fixed random seed for reproducibility
        np.random.seed(42)

    def _run_random_trial(self, adjust_flag):
        import pandas as pd
        # Generate a random length between 50 and 150
        n = np.random.randint(50, 151)
        # Create two random arrays
        x = np.random.randn(n)
        y = np.random.randn(n)
        custom_ewm = EWMCorrelation(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods)
        custom_result = custom_ewm.corr(x, y)
        pd_result = pd.Series(x).ewm(alpha=self.alpha, adjust=adjust_flag, min_periods=self.min_periods)\
                              .corr(pd.Series(y)).to_numpy()
        print(custom_result, pd_result)
        np.testing.assert_allclose(custom_result, pd_result, rtol=1e-6, equal_nan=True)

    def test_adjust_true_random_trials(self):
        # Run 5 random trials with adjust=True
        for _ in range(5):
            self._run_random_trial(adjust_flag=True)

    def test_adjust_false_random_trials(self):
        # Run 5 random trials with adjust=False
        for _ in range(5):
            self._run_random_trial(adjust_flag=False)

if __name__ == '__main__':
    unittest.main()

