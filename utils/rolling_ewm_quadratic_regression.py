import numpy as np

class RollingEWMQuadraticRegression:
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
            Specify 'ransac' to perform robust regression via RANSAC (only available in explicit mode).
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
            # Explicit mode: store all data points.
            self.xs = []
            self.ys = []
        else:
            # Recursive mode: initialize weighted sums for quadratic regression.
            self.count = 0
            # Sums for powers of x:
            self.S0 = 0.0    # sum of weights
            self.S1 = 0.0    # sum of x
            self.S2 = 0.0    # sum of x^2
            self.S3 = 0.0    # sum of x^3
            self.S4 = 0.0    # sum of x^4
            # Sums for y:
            self.Sy = 0.0    # sum of y
            self.Sxy = 0.0   # sum of x*y
            self.Sx2y = 0.0  # sum of x^2*y

    def _weighted_quadratic_regression(self, x, y, weights):
        """
        Compute weighted least squares estimates for quadratic regression.
        The model is y = a x^2 + b x + c.
        """
        # Construct the design matrix with columns: x^2, x, 1.
        X = np.vstack((x**2, x, np.ones_like(x))).T
        # Multiply each row by sqrt(weight) to get weighted least squares.
        W = np.sqrt(weights)
        Xw = X * W[:, None]
        yw = y * W
        # Solve the least squares problem.
        try:
            beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        a, b, c = beta
        return a, b, c

    def _ransac_regression(self, x, y, weights):
        """
        Perform a simple RANSAC-based robust regression for the quadratic model.
        Randomly selects 3 distinct points to compute candidate parameters.
        """
        n = len(x)
        if n < 3:
            return np.nan, np.nan, np.nan
        best_error = np.inf
        best_inlier_mask = None

        for _ in range(self.ransac_iterations):
            # Randomly sample 3 distinct indices.
            idx = np.random.choice(n, 3, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]
            # Form design matrix and solve for quadratic coefficients.
            X_sample = np.vstack((x_sample**2, x_sample, np.ones_like(x_sample))).T
            try:
                beta = np.linalg.solve(X_sample, y_sample)
            except np.linalg.LinAlgError:
                continue
            a, b, c = beta
            y_pred = a * x**2 + b * x + c
            residuals = np.abs(y - y_pred)
            median_resid = np.median(residuals)
            thresh = self.ransac_threshold * (median_resid if median_resid > 0 else 1.0)
            inlier_mask = residuals < thresh
            if inlier_mask.sum() < max(3, self.min_periods):
                continue
            inlier_error = np.sum(weights[inlier_mask] * residuals[inlier_mask])
            if inlier_error < best_error:
                best_error = inlier_error
                best_inlier_mask = inlier_mask

        if best_inlier_mask is not None and best_inlier_mask.sum() >= max(3, self.min_periods):
            return self._weighted_quadratic_regression(x[best_inlier_mask],
                                                       y[best_inlier_mask],
                                                       weights[best_inlier_mask])
        else:
            return np.nan, np.nan, np.nan

    def update(self, x, y):
        """
        Update the regression with a new (x, y) data point and return the latest quadratic coefficients.
        
        Returns:
          a, b, c : tuple of floats
            Coefficients of the quadratic model y = a*x^2 + b*x + c.
            If there are fewer than min_periods data points, returns (np.nan, np.nan, np.nan).
        """
        if self.adjust:
            # Explicit mode: store the new data point.
            self.xs.append(x)
            self.ys.append(y)
            n = len(self.xs)
            if n < self.min_periods:
                return np.nan, np.nan, np.nan
            weights = (1 - self.alpha) ** np.arange(n-1, -1, -1)
            x_arr = np.array(self.xs)
            y_arr = np.array(self.ys)
            if self.outlier_method == 'ransac':
                a, b, c = self._ransac_regression(x_arr, y_arr, weights)
            else:
                a, b, c = self._weighted_quadratic_regression(x_arr, y_arr, weights)
            return a, b, c
        else:
            # Recursive mode: update weighted sums.
            self.count += 1
            if self.count == 1:
                # Initialize with the first observation.
                self.S0 = 1.0
                self.S1 = x
                self.S2 = x**2
                self.S3 = x**3
                self.S4 = x**4
                self.Sy = y
                self.Sxy = x * y
                self.Sx2y = (x**2) * y
                return np.nan, np.nan, np.nan
            else:
                # Update sums with decay.
                self.S0 = (1 - self.alpha) * self.S0 + 1.0
                self.S1 = (1 - self.alpha) * self.S1 + x
                self.S2 = (1 - self.alpha) * self.S2 + x**2
                self.S3 = (1 - self.alpha) * self.S3 + x**3
                self.S4 = (1 - self.alpha) * self.S4 + x**4
                self.Sy = (1 - self.alpha) * self.Sy + y
                self.Sxy = (1 - self.alpha) * self.Sxy + x * y
                self.Sx2y = (1 - self.alpha) * self.Sx2y + (x**2) * y

                if self.count < self.min_periods:
                    return np.nan, np.nan, np.nan

                # Construct the normal equations matrix M and vector B.
                M = np.array([[self.S4, self.S3, self.S2],
                              [self.S3, self.S2, self.S1],
                              [self.S2, self.S1, self.S0]])
                B = np.array([self.Sx2y, self.Sxy, self.Sy])
                try:
                    beta = np.linalg.solve(M, B)
                    a, b, c = beta
                except np.linalg.LinAlgError:
                    a, b, c = np.nan, np.nan, np.nan
                return a, b, c

# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a rolling EWM quadratic regression object.
    # For explicit mode with RANSAC robust regression.
    ewmlr = RollingEWMQuadraticRegression(alpha=0.05, adjust=True, min_periods=20,
                                            outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)

    # Generate synthetic data with a quadratic trend plus noise.
    np.random.seed(0)
    n_points = 200
    xs = np.linspace(-10, 10, n_points)
    # True quadratic: y = 0.5*x^2 - 2*x + 3.
    true_a = 0.5
    true_b = -2.0
    true_c = 3.0
    noise = np.random.randn(n_points) * 5.0
    ys = true_a * xs**2 + true_b * xs + true_c + noise

    # Introduce some outliers.
    outlier_indices = np.random.choice(n_points, size=10, replace=False)
    ys[outlier_indices] += np.random.randn(10) * 30

    a_list = []
    b_list = []
    c_list = []

    for xi, yi in zip(xs, ys):
        a, b, c = ewmlr.update(xi, yi)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    # Plot the evolving quadratic coefficient estimates.
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(xs, a_list, label='Estimate a', color='red')
    plt.axhline(true_a, color='black', linestyle='--', label='True a')
    plt.ylabel('Coefficient a')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(xs, b_list, label='Estimate b', color='blue')
    plt.axhline(true_b, color='black', linestyle='--', label='True b')
    plt.ylabel('Coefficient b')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(xs, c_list, label='Estimate c', color='green')
    plt.axhline(true_c, color='black', linestyle='--', label='True c')
    plt.xlabel('x')
    plt.ylabel('Coefficient c')
    plt.legend()
    plt.suptitle('Rolling EWM Quadratic Regression Coefficient Estimates')
    plt.show()

    # Plot the final regression curve.
    final_a = a_list[-1]
    final_b = b_list[-1]
    final_c = c_list[-1]
    plt.figure(figsize=(12, 5))
    plt.scatter(xs, ys, label='Data', s=10, alpha=0.6)
    xs_fit = np.linspace(min(xs), max(xs), 300)
    ys_fit = final_a * xs_fit**2 + final_b * xs_fit + final_c
    plt.plot(xs_fit, ys_fit, label='Final Regression', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final Quadratic Regression Curve')
    plt.show()

