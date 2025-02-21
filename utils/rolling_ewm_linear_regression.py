import numpy as np

class RollingEWMLinearRegression:
    def __init__(self, alpha, adjust=True, min_periods=1,
                 outlier_method=None, ransac_iterations=100, ransac_threshold=2.0):
        """
        Parameters:
          alpha : float
            Smoothing factor (0 < alpha <= 1).
          adjust : bool, default True
            When True, all observed data are stored and explicit weights computed.
            When False, a recursive update is used (robust outlier rejection is not available in this mode).
          min_periods : int, default 1
            Minimum number of data points required for a valid regression output.
          outlier_method : str or None, default None
            Specify 'ransac' to perform robust regression via RANSAC when using adjust=True.
          ransac_iterations : int, default 100
            Number of iterations for the RANSAC procedure.
          ransac_threshold : float, default 2.0
            Scaling factor for the inlier threshold in RANSAC.
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.adjust = adjust
        self.min_periods = min_periods
        self.outlier_method = outlier_method
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold

        if self.adjust:
            # For explicit mode, store all data points.
            self.xs = []
            self.ys = []
        else:
            # For recursive mode, initialize weighted sums.
            self.count = 0
            self.S0 = 0.0   # weighted sum of weights
            self.S1 = 0.0   # weighted sum of x
            self.S2 = 0.0   # weighted sum of x^2
            self.Sy = 0.0   # weighted sum of y
            self.Sxy = 0.0  # weighted sum of x*y

    def _weighted_linear_regression(self, x, y, weights):
        """Compute weighted least squares estimates for slope and intercept."""
        sum_w = weights.sum()
        mean_x = np.sum(x * weights) / sum_w
        mean_y = np.sum(y * weights) / sum_w
        cov = np.sum(weights * (x - mean_x) * (y - mean_y))
        var_x = np.sum(weights * (x - mean_x)**2)
        if var_x == 0:
            return np.nan, np.nan
        slope = cov / var_x
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def _ransac_regression(self, x, y, weights):
        """
        Perform a simple RANSAC-based robust regression.
        """
        n = len(x)
        if n < 2:
            return np.nan, np.nan
        best_error = np.inf
        best_inlier_mask = None

        for _ in range(self.ransac_iterations):
            # Randomly sample 2 distinct points.
            idx = np.random.choice(n, 2, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]
            if x_sample[1] == x_sample[0]:
                continue
            candidate_slope = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
            candidate_intercept = y_sample[0] - candidate_slope * x_sample[0]
            y_pred = candidate_slope * x + candidate_intercept
            residuals = np.abs(y - y_pred)
            median_resid = np.median(residuals)
            thresh = self.ransac_threshold * (median_resid if median_resid > 0 else 1.0)
            inlier_mask = residuals < thresh
            if inlier_mask.sum() < max(2, self.min_periods):
                continue
            inlier_error = np.sum(weights[inlier_mask] * residuals[inlier_mask])
            if inlier_error < best_error:
                best_error = inlier_error
                best_inlier_mask = inlier_mask

        if best_inlier_mask is not None and best_inlier_mask.sum() >= max(2, self.min_periods):
            return self._weighted_linear_regression(x[best_inlier_mask],
                                                    y[best_inlier_mask],
                                                    weights[best_inlier_mask])
        else:
            return np.nan, np.nan

    def update(self, x, y):
        """
        Update the regression with a new (x, y) data point and return the latest slope and intercept.

        Returns:
          slope, intercept : tuple of floats
            Latest regression parameters. If there are fewer than min_periods data points,
            both slope and intercept are returned as np.nan.
        """
        if self.adjust:
            # Store the new data point.
            self.xs.append(x)
            self.ys.append(y)
            n = len(self.xs)
            if n < self.min_periods:
                return np.nan, np.nan
            # Compute weights for all points (most recent has weight 1).
            weights = (1 - self.alpha) ** np.arange(n-1, -1, -1)
            x_arr = np.array(self.xs)
            y_arr = np.array(self.ys)
            if self.outlier_method == 'ransac':
                slope, intercept = self._ransac_regression(x_arr, y_arr, weights)
            else:
                slope, intercept = self._weighted_linear_regression(x_arr, y_arr, weights)
            return slope, intercept
        else:
            # Recursive update mode.
            self.count += 1
            if self.count == 1:
                self.S0 = 1.0
                self.S1 = x
                self.S2 = x**2
                self.Sy = y
                self.Sxy = x*y
                return np.nan, np.nan
            else:
                # Decay the previous sums and add the new observation.
                self.S0 = (1 - self.alpha) * self.S0 + 1.0
                self.S1 = (1 - self.alpha) * self.S1 + x
                self.S2 = (1 - self.alpha) * self.S2 + x**2
                self.Sy = (1 - self.alpha) * self.Sy + y
                self.Sxy = (1 - self.alpha) * self.Sxy + x*y

                if self.count < self.min_periods:
                    return np.nan, np.nan

                mean_x = self.S1 / self.S0
                mean_y = self.Sy / self.S0
                var_x = self.S2 / self.S0 - mean_x**2
                cov = self.Sxy / self.S0 - mean_x * mean_y
                if var_x > 0:
                    slope = cov / var_x
                    intercept = mean_y - slope * mean_x
                else:
                    slope, intercept = np.nan, np.nan
                return slope, intercept

# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a rolling EWM regression object with explicit weights and robust RANSAC.
    ewmlr = RollingEWMLinearRegression(alpha=0.05, adjust=True, min_periods=20,
                                       outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)

    # Generate synthetic data with a linear trend plus noise.
    np.random.seed(0)
    n_points = 200
    xs = np.linspace(0, 10, n_points)
    true_slope = 2.0
    true_intercept = 1.0
    noise = np.random.randn(n_points)
    ys = true_slope * xs + true_intercept + noise

    # Introduce some outliers.
    outlier_indices = np.random.choice(n_points, size=10, replace=False)
    ys[outlier_indices] += np.random.randn(10) * 20

    slopes = []
    intercepts = []

    for xi, yi in zip(xs, ys):
        slope, intercept = ewmlr.update(xi, yi)
        slopes.append(slope)
        intercepts.append(intercept)

    # Plot the evolving slope estimates.
    plt.figure(figsize=(12, 5))
    plt.plot(xs, slopes, label='Rolling EWM Slope (RANSAC)', color='red')
    plt.axhline(true_slope, color='black', linestyle='--', label='True Slope')
    plt.xlabel('x')
    plt.ylabel('Slope')
    plt.legend()
    plt.title('Rolling EWM Linear Regression: Slope Estimates')
    plt.show()

    # Plot the final regression line.
    final_slope = slopes[-1]
    final_intercept = intercepts[-1]
    plt.figure(figsize=(12, 5))
    plt.scatter(xs, ys, label='Data', s=10, alpha=0.6)
    plt.plot(xs, final_slope * xs + final_intercept, label='Final Regression', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final Regression Line from Rolling EWM Linear Regression')
    plt.show()

