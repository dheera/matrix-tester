import numpy as np

class EWMQuadraticRegression:
    def __init__(self, alpha, adjust=True, min_periods=1,
                 outlier_method=None, ransac_iterations=100, ransac_threshold=2.0):
        """
        Parameters:
          alpha : float
            Smoothing factor (0 < alpha <= 1).
          adjust : bool, default True
            When True, explicit weights are computed for all observations (like pandas).
            When False, a recursive update is used (robust outlier rejection is ignored in this mode).
          min_periods : int, default 1
            Minimum number of observations required for a valid result.
          outlier_method : str or None, default None
            Specify 'ransac' for robust outlier rejection using a simple RANSAC procedure.
          ransac_iterations : int, default 100
            Number of iterations to run the RANSAC procedure.
          ransac_threshold : float, default 2.0
            Scaling factor applied to the median absolute residual error to determine the inlier threshold.
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
            # Recursive mode: initialize weighted sums.
            self.count = 0
            # Sums for powers of x:
            self.S0 = 0.0   # sum of weights
            self.S1 = 0.0   # sum of x
            self.S2 = 0.0   # sum of x^2
            self.S3 = 0.0   # sum of x^3
            self.S4 = 0.0   # sum of x^4
            # Sums for y:
            self.Sy = 0.0   # sum of y
            self.Sxy = 0.0  # sum of x*y
            self.Sx2y = 0.0 # sum of x^2*y

    def _weighted_quadratic_regression(self, x, y, weights):
        """
        Compute weighted least squares estimates for quadratic regression.
        The model is y = a*x^2 + b*x + c.
        """
        # Construct the design matrix: columns: x^2, x, 1.
        X = np.vstack((x**2, x, np.ones_like(x))).T
        # Solve weighted least squares using weights (applied as sqrt(weight)).
        W = np.sqrt(weights)
        Xw = X * W[:, None]
        yw = y * W
        try:
            beta, residuals, rank, s = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        a, b, c = beta
        return a, b, c

    def _ransac_regression(self, x, y, weights):
        """
        Perform a simple RANSAC-based robust regression for the quadratic model.
        Randomly selects 3 distinct points, computes the candidate quadratic model,
        then evaluates all points for inliers. The best candidate is refit using weighted LS.
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
            # Construct design matrix and attempt to solve.
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

    def regress(self, x, y):
        """
        Compute the EWM quadratic regression estimates (a, b, c) for each time step.

        Parameters:
          x, y : array_like (1D)
            The independent and dependent variable series (must have the same length).

        Returns:
          a_coeffs, b_coeffs, c_coeffs : tuple of np.ndarray
            Arrays containing the quadratic coefficients for each time point.
            For indices with fewer than min_periods observations, np.nan is returned.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Only 1D arrays are supported.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")

        n = len(x)
        a_coeffs = np.full(n, np.nan)
        b_coeffs = np.full(n, np.nan)
        c_coeffs = np.full(n, np.nan)

        if self.adjust:
            # Explicit mode: compute regression on [0...t] at each time step.
            for t in range(n):
                if t + 1 < self.min_periods:
                    continue
                weights = (1 - self.alpha) ** np.arange(t, -1, -1)
                x_arr = x[:t+1]
                y_arr = y[:t+1]
                if self.outlier_method == 'ransac':
                    a, b, c = self._ransac_regression(x_arr, y_arr, weights)
                else:
                    a, b, c = self._weighted_quadratic_regression(x_arr, y_arr, weights)
                a_coeffs[t] = a
                b_coeffs[t] = b
                c_coeffs[t] = c
        else:
            # Recursive mode: update weighted sums for quadratic regression.
            # Initialize with the first observation.
            self.count = 0
            for t in range(n):
                self.count += 1
                if self.count == 1:
                    self.S0 = 1.0
                    self.S1 = x[0]
                    self.S2 = x[0]**2
                    self.S3 = x[0]**3
                    self.S4 = x[0]**4
                    self.Sy = y[0]
                    self.Sxy = x[0]*y[0]
                    self.Sx2y = (x[0]**2)*y[0]
                    a_coeffs[0] = np.nan
                    b_coeffs[0] = np.nan
                    c_coeffs[0] = np.nan
                    continue
                # Decay previous sums and add new observation.
                self.S0 = (1 - self.alpha) * self.S0 + 1.0
                self.S1 = (1 - self.alpha) * self.S1 + x[t]
                self.S2 = (1 - self.alpha) * self.S2 + x[t]**2
                self.S3 = (1 - self.alpha) * self.S3 + x[t]**3
                self.S4 = (1 - self.alpha) * self.S4 + x[t]**4
                self.Sy = (1 - self.alpha) * self.Sy + y[t]
                self.Sxy = (1 - self.alpha) * self.Sxy + x[t]*y[t]
                self.Sx2y = (1 - self.alpha) * self.Sx2y + (x[t]**2)*y[t]

                if self.count < self.min_periods:
                    continue

                # Construct the normal equations:
                # [ S4  S3  S2 ] [a]   = [ Sx2y ]
                # [ S3  S2  S1 ] [b]   = [ Sxy  ]
                # [ S2  S1  S0 ] [c]   = [ Sy   ]
                M = np.array([[self.S4, self.S3, self.S2],
                              [self.S3, self.S2, self.S1],
                              [self.S2, self.S1, self.S0]])
                B = np.array([self.Sx2y, self.Sxy, self.Sy])
                try:
                    beta = np.linalg.solve(M, B)
                    a, b, c = beta
                except np.linalg.LinAlgError:
                    a, b, c = np.nan, np.nan, np.nan
                a_coeffs[t] = a
                b_coeffs[t] = b
                c_coeffs[t] = c

        return a_coeffs, b_coeffs, c_coeffs

# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate synthetic data with a quadratic trend plus noise.
    np.random.seed(0)
    n_points = 200
    xs = np.linspace(-10, 10, n_points)
    true_a = 0.5
    true_b = -2.0
    true_c = 3.0
    noise = np.random.randn(n_points) * 5.0
    ys = true_a * xs**2 + true_b * xs + true_c + noise

    # Introduce some outliers.
    outlier_indices = np.random.choice(n_points, size=10, replace=False)
    ys[outlier_indices] += np.random.randn(10) * 30

    # Create quadratic regression objects.
    # 1. Standard weighted regression (explicit mode, no outlier rejection)
    ewm_quad_standard = EWMQuadraticRegression(alpha=0.05, adjust=True, min_periods=20)
    a_std, b_std, c_std = ewm_quad_standard.regress(xs, ys)

    # 2. Robust regression using RANSAC (explicit mode)
    ewm_quad_ransac = EWMQuadraticRegression(alpha=0.05, adjust=True, min_periods=20,
                                              outlier_method='ransac', ransac_iterations=200, ransac_threshold=2.5)
    a_ransac, b_ransac, c_ransac = ewm_quad_ransac.regress(xs, ys)

    # Plot evolving coefficient estimates.
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(xs, a_std, label='Standard a', color='blue')
    plt.plot(xs, a_ransac, label='Robust a (RANSAC)', color='red')
    plt.axhline(true_a, color='black', linestyle='--', label='True a')
    plt.ylabel('Coefficient a')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(xs, b_std, label='Standard b', color='blue')
    plt.plot(xs, b_ransac, label='Robust b (RANSAC)', color='red')
    plt.axhline(true_b, color='black', linestyle='--', label='True b')
    plt.ylabel('Coefficient b')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(xs, c_std, label='Standard c', color='blue')
    plt.plot(xs, c_ransac, label='Robust c (RANSAC)', color='red')
    plt.axhline(true_c, color='black', linestyle='--', label='True c')
    plt.xlabel('x')
    plt.ylabel('Coefficient c')
    plt.legend()
    plt.suptitle('EWM Quadratic Regression Coefficient Estimates')
    plt.show()

    # Plot the final regression curves.
    final_a_std = a_std[-1]
    final_b_std = b_std[-1]
    final_c_std = c_std[-1]
    final_a_ransac = a_ransac[-1]
    final_b_ransac = b_ransac[-1]
    final_c_ransac = c_ransac[-1]

    xs_fit = np.linspace(min(xs), max(xs), 300)
    ys_fit_std = final_a_std * xs_fit**2 + final_b_std * xs_fit + final_c_std
    ys_fit_ransac = final_a_ransac * xs_fit**2 + final_b_ransac * xs_fit + final_c_ransac

    plt.figure(figsize=(12, 5))
    plt.scatter(xs, ys, label='Data', s=10, alpha=0.6)
    plt.plot(xs_fit, ys_fit_std, label='Standard Regression', color='blue', linewidth=2)
    plt.plot(xs_fit, ys_fit_ransac, label='Robust Regression (RANSAC)', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final Quadratic Regression Curves')
    plt.show()

