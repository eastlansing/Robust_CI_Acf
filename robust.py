import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import copy
import cmath

class HAC_robust_conf_int:

    """
    Implements the Heteroskedasticity and Autocorrelation Consistent (HAC) robust confidence intervals for autocorrelation functions.

    Parameters
    ----------
    data : array-like
        The input data series for which the HAC robust confidence intervals are to be calculated.
    
    lag : int
        The maximum lag to be considered for autocorrelation.
    
    null_imp : bool, default=True
        Indicates if the null hypothesis of no importance should be considered.
    
    method : str, default='fixedb'
        The method used for estimation.
    
    bandwidth : str, default="SPJ"
        Specifies the bandwidth method used in the estimation.
    
    time_trend : bool, default=False
        If True, a time trend is considered in the model.
    
    diff : bool, default=False
        If True, calculates the difference series of the data.
    
    alpha : float, default=0.05
        Significance level for the confidence interval.

    Note
    ----
    Ensure that the provided parameters, especially 'method' and 'bandwidth', align with the domain-specific options available. The provided options are for demonstration purposes and may need further refinement.

    """

    def __init__(self, data, lag, null_imp=True, method='fixedb', bandwidth="SPJ", time_trend=False, diff=False, alpha=0.05):
        self.data = np.copy(np.asarray(data).reshape(-1, 1))
        self.lag = lag
        self.null_imp = null_imp
        self.method = method
        self.bandwidth = bandwidth
        self.time_trend = time_trend
        self.diff = diff
        self.alpha = alpha
        self.rho_est_values = []
        self.cv_values = []
        self.var_unres_values = []
        self._calculate_estimations()

    def _calculate_estimations(self):
        if self.diff:
            self.data = np.diff(self.data, axis=0)
        k_values = range(1, self.lag+1)  # 1 to lag k

        self.rho_est_values = []
        
        if self.null_imp:
            self.lower_values = []
            self.upper_values = []
            self.index_values = []
            self.cv_values = []
            self.var_res_under_zero_values = []

            for k in k_values:
                # calculate rho_est, M_n_band
                if self.time_trend:
                    rho_est, w_t, res_y_t_k = regression_residuals(data=self.data, k=k)
                    vhat_opt3 = np.copy(res_y_t_k*w_t)
                    vhat_opt3 = vhat_opt3 - vhat_opt3.mean(axis=0)
                else:
                    y, x, nmp, X, coeffs = reg_estimating_equation_v1(data=self.data, lag=k)  # assumed function name and arguments
                    uhat_opt3 = y - np.dot(X, coeffs)  # null is not imposed
                    vhat_opt3 = np.copy(X * uhat_opt3)
                    vhat_opt3 = vhat_opt3 - vhat_opt3.mean(axis=0)
                    rho_est = coeffs[1]

                # method and bandwidth processing
                if self.method == "OS":
                    pass
                else:
                    if self.bandwidth == "SPJ":  # assuming bandwidth is an attribute
                        spj_function = create_spj_function(w=10)  # assumed function name and arguments
                        M_n_spj = spj_function(vhat_opt3)
                        M_n_band = M_n_spj
                    elif self.bandwidth == "AD":
                        M_n_AD = AD_band(vhat_opt3)  # assumed function name and arguments
                        M_n_band = M_n_AD  # assumed variable name

                # Calculate Q2
                Q_2 = compute_Q2(self.data, k)
                Tnum1 = len(self.data) - k

                # Calculate Omegas
                k_function_vectorized = parzen_k_function_vectorized
                omegas_res_dm = compute_omegas_vectorized_demean(self.data, M_n_band, k_function_vectorized, k)

                # fixed-b
                fixedb_coeffs = [0.43754, 0.11912, 0.08640, 0.49629, -0.57879, 0.43266, 0.02543, -0.02379, -0.02376]
                fixed_cv_res = calc_fixed_cv_org_alpha(fixed_b=M_n_band, Tnum=Tnum1, fixedb_coeffs=fixedb_coeffs, alpha=self.alpha)

                # Choose CV by condition
                if self.method == "fixedb":
                    cv_real = fixed_cv_res
                elif self.method == "normal":
                    cv_real = stats.norm.ppf(1 - self.alpha / 2)

                # calc roots
                c_coef_res_dm = compute_c_coefficients(omegas=omegas_res_dm, cv=cv_real, y=self.data, k_val=k, rho_k_til=rho_est, Q_2=Q_2)
                roots_res_dm = quadratic_roots(c_coef_res_dm[0], c_coef_res_dm[1], c_coef_res_dm[2])

                roots = roots_res_dm
                c_coef = c_coef_res_dm

                is_first_root_complex = roots[0].imag != 0
                is_second_root_complex = roots[1].imag != 0

                if is_first_root_complex or is_second_root_complex:
                    index = 2 if c_coef[0] < 0 else 3
                    if index == 3:
                        raise ValueError("Open upward, vertex is above 0, where there is no such collection of null value rejecting null hypothesis. Impossible.")
                    upper = 1
                    lower = -1
                else:
                    index = 0 if c_coef[0] > 0 else 1
                    lower = min(roots[0].real, roots[1].real)
                    upper = max(roots[0].real, roots[1].real)
                    
                    #Truncate if greater than 1
                    #if np.float64(lower) < -1:
                    #    lower = -1
                    #if np.float64(upper) > 1:
                    #    upper = 1

                # Appending results for this k value to their respective lists
                self.rho_est_values.append(np.float64(rho_est))
                self.lower_values.append(np.float64(lower))
                self.upper_values.append(np.float64(upper))
                self.index_values.append(index)
                
                var_res_under_zero11 = self._calculate_var_res_under_zero(k)
                self.var_res_under_zero_values.append(np.float64(var_res_under_zero11))
                self.cv_values.append(np.float64(cv_real))

        else:
            self.cv_values = []
            self.var_unres_values = []

            for k in k_values:
                if self.time_trend:
                    rho_est, w_t, res_y_t_k = regression_residuals(data=self.data, k=k)
                    vhat_opt1 = np.copy(res_y_t_k * w_t)
                    X = np.copy(res_y_t_k)
                else:
                    y, x, nmp, X, coeffs = reg_estimating_equation_v1(data=self.data, lag=k)  # assumed function name and arguments
                    uhat_opt1 = y - np.dot(X, coeffs)  # null is not imposed
                    vhat_opt1 = np.copy(X * uhat_opt1)
                    rho_est = coeffs[1]

                if self.method == "OS":
                    pass
                else:
                    if self.bandwidth == "SPJ":  # assuming bandwidth is an attribute
                        spj_function = create_spj_function(w=10)  # assumed function name and arguments
                        M_n_spj = spj_function(vhat_opt1)
                        M_n_band = M_n_spj
                    elif self.bandwidth == "AD":
                        M_n_AD = AD_band(vhat_opt1)  # assumed function name and arguments
                        M_n_band = M_n_AD  # assumed variable name

                    fixedb_coeffs = [0.43754, 0.11912, 0.08640, 0.49629, -0.57879, 0.43266, 0.02543, -0.02379, -0.02376]
                    fixed_cv, var_unres = process_var_fixedbcv_alpha(vhat=vhat_opt1, M_n=M_n_band, X=X, nmp=len(X), fixedb_coeffs=fixedb_coeffs, alpha=self.alpha)  # assumed function name and arguments
                    if self.time_trend:
                        pass
                    else:
                        var_unres = np.copy(var_unres[1,1])
                    
                    
                    if self.method == "fixedb":
                        cv_real = fixed_cv
                    elif self.method == "normal":
                        cv_real = stats.norm.ppf(1 - self.alpha / 2)

                self.rho_est_values.append(np.float64(rho_est))
                self.cv_values.append(np.float64(cv_real))  # updated cv to cv_real
                self.var_unres_values.append(np.float64(var_unres))
                
    
    def _calculate_var_res_under_zero(self, k):
        """
        Calculates the variance estimator under null imposed for a given lag.
        
        Parameters:
        - k (int): The lag order for the regression estimating equation.
        
        Returns:
        - var_res_under_zero11 (float): The element at index (1,1) of the variance estimates under null imposed.

        """
        
        y_z, x_z, nmp_z, X_z, coeffs_z = reg_estimating_equation_v1(data=self.data, lag=k)
        null_val = 0
        uhat_z = (y_z - np.mean(y_z)) - null_val*(x_z - np.mean(x_z))
        vhat_z = np.copy(X_z*uhat_z)
        vhat_z = vhat_z - vhat_z.mean(axis=0)  # Stock Watson demeaning

        if self.bandwidth == "SPJ":
            spj_function = create_spj_function(w=10)
            M_n_z = spj_function(vhat_z)
        elif self.bandwidth == "AD":
            M_n_z = AD_band(vhat_z)
        
        fixedb_coeffs = [0.43754, 0.11912, 0.08640, 0.49629, -0.57879, 0.43266, 0.02543, -0.02379, -0.02376]
        _, var_res_under_zero = process_var_fixedbcv_alpha(vhat_z, M_n_z, X_z, len(X_z), fixedb_coeffs, self.alpha)
        var_res_under_zero11 = var_res_under_zero[1,1]

        return var_res_under_zero11

    def plot_acf(self, CI_HAC=True, CB_HAC=False, CB_stata=False, CB_bart=False, title="Autocorrelogram with HAC robust CI", save_as_pdf=False, filename="autocorrelogram.pdf"):
        """
        Plot the autocorrelation function (ACF).

        Parameters
        ----------
        CI_HAC : bool, default=True
            Whether to plot HAC robust confidence intervals.
        CB_HAC : bool, default=False
            Whether to plot HAC robust confidence bands.
        CB_stata : bool, default=False
            Whether to plot the Stata confidence bands.
        CB_bart : bool, default=False
            Whether to plot the Bartlett confidence bands.
        title : str, default="Autocorrelogram with HAC robust CI"
            Title for the plot.
        save_as_pdf : bool, default=False
            If True, save the plot as a PDF, otherwise display it.
        filename : str, default="autocorrelogram.pdf"
            Filename for saving the plot if save_as_pdf is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object containing the ACF plot.

        Notes
        -----
        This method selects the appropriate sub-routine for plotting based on the null_imp attribute.
        """

        if self.null_imp:
            self._plot_acf_null_imp(CI_HAC, CB_HAC, CB_stata, CB_bart, title, save_as_pdf, filename)
        else:
            self._plot_acf_null_not_imp(CI_HAC, CB_HAC, CB_stata, CB_bart, title, save_as_pdf, filename)
            
    
    def _stacked_autocorrelation(self, k):
        data = np.copy(self.data)
        
        if self.time_trend:
            t = np.arange(1, len(data) + 1).reshape(-1, 1)
            residuals_data = partial_regression(np.copy(data), t, coef=False, cons=True)
            data = np.copy(residuals_data)
            
            
        data = data.flatten()  # Reshape the data to 1D if it's 2D    
        n = len(data)
        mean_val = np.mean(data)
        acf_values = []

        for lag in range(1, k + 1):  # Lag 0 to k
            numerator = np.sum((data[lag:] - mean_val) * (data[:-lag] - mean_val))
            denominator = np.sum((data - mean_val) ** 2)
            acf = numerator / denominator
            acf_values.append(acf)

        return acf_values

    def _calculate_var_stata(self):
        var_stata = []
        n = len(self.data)
        for k in range(1, len(self.rho_est_values) + 1):
            if k == 1:
                var_stata.append(1 / n)
            else:
                var_stata.append((1 + 2 * sum([rho**2 for rho in self.rho_est_values[:k-1]])) / n)
        return var_stata
    
    def _calculate_var_stata_ori(self):
        var_stata = []
        n = len(self.data)
        stacked_acf = self._stacked_autocorrelation(len(self.rho_est_values))
        for k in range(1, len(self.rho_est_values) + 1):
            if k == 1:
                var_stata.append(1 / n)
            else:
                var_stata.append((1 + 2 * sum([rho**2 for rho in stacked_acf[:k-1]])) / n)
        return var_stata

    def _plot_acf_null_imp(self, CI_HAC, CB_HAC, CB_stata, CB_bart, title, save_as_pdf, filename):
        lags = range(1, len(self.rho_est_values) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.vlines(lags, [0], self.rho_est_values, colors='blue', lw=2)
        ax.plot(lags, self.rho_est_values, 'o', color='blue', label = 'Estimated Autocorrelation')

        ax.axhline(0, color='black', lw=0.5)


        if CI_HAC and self.lower_values and self.upper_values and self.index_values:
            for lag, v, lb, ub, index in zip(lags, self.rho_est_values, self.lower_values, self.upper_values, self.index_values):
                lb = min(max(lb, -1),1)
                ub = max(min(ub, 1),-1)
                
                bracket_length = 0.2
                if index == 0 or index == 2:
                    ax.plot([lag, lag], [ub, lb], color='gray', lw=0.8)
                    ax.plot([lag-bracket_length, lag+bracket_length], [ub, ub], color='black', lw=1.1)
                    ax.plot([lag-bracket_length, lag+bracket_length], [lb, lb], color='black', lw=1.1, label='Confidence Interval - HAC robust' if lag == lags[0] else "")

                elif index == 1:
                    ax.plot([lag, lag], [-1, lb], color='gray', lw=0.8)
                    ax.plot([lag, lag], [ub, 1], color='gray', lw=0.8)
                    ax.plot([lag-bracket_length, lag+bracket_length], [-1, -1], color='black', lw=1.1)
                    ax.plot([lag-bracket_length, lag+bracket_length], [1, 1], color='black', lw=1.1)
                    ax.plot([lag-bracket_length, lag+bracket_length], [lb, lb], color='black', lw=1.1)
                    ax.plot([lag-bracket_length, lag+bracket_length], [ub, ub], color='black', lw=1.1, label='Confidence Interval - HAC robust' if lag == lags[0] else "")

                elif index == 3:
                    raise ValueError("Open upward, vertex is above 0. No valid collection of null values rejecting the null hypothesis. Impossible.")
    
        if CB_HAC and self.cv_values and self.var_res_under_zero_values:
            upper_bounds = [min(1.00, 0 + fc * np.sqrt(vu)) for fc, vu in zip(self.cv_values, self.var_res_under_zero_values)]
            lower_bounds = [max(-1.00, 0 - fc * np.sqrt(vu)) for fc, vu in zip(self.cv_values, self.var_res_under_zero_values)]

            ax.plot(lags, upper_bounds, color='green', linestyle='dashed', lw=0.99)
            ax.plot(lags, lower_bounds, color='green', linestyle='dashed', lw=0.99, label='Confidence Band - HAC robust')
        
        if CB_stata:
            ax.plot(lags, self._stacked_autocorrelation(len(self.rho_est_values)), 'o', color='red', markerfacecolor="None", label='Sample Autocorrelation')
            
            var_stata = self._calculate_var_stata_ori()
            upper_bound = [0 + stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
            lower_bound = [0 - stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
            ax.fill_between(lags, lower_bound, upper_bound, color='grey', alpha=0.2, label='Confidence Band - Stata Formula')
            
        if CB_bart and len(self.data):
            if not CB_stata:
                ax.plot(lags, self._stacked_autocorrelation(len(self.rho_est_values)), 'o', color='red', markerfacecolor="None", label='Sample Autocorrelation')
            
            conf_value = 1.96 / np.sqrt(len(self.data))
            ax.axhline(conf_value, color='red', linestyle='dashed', lw=0.8)
            ax.axhline(-conf_value, color='red', linestyle='dashed', lw=0.8, label='Confidence Band - Bartlett Formula')
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Estimated Autocorrelation')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.25)
        ax.set_xlim([0, len(self.rho_est_values) + 1])
        ax.set_ylim([-1.1, 1.1])

        if save_as_pdf:
            plt.savefig(filename, format='pdf')
        else:
            plt.show()

    def _plot_acf_null_not_imp(self, CI_HAC, CB_HAC, CB_stata, CB_bart, title, save_as_pdf, filename):
        lags = range(1, len(self.rho_est_values) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.vlines(lags, [0], self.rho_est_values, colors='blue', lw=2)
        ax.plot(lags, self.rho_est_values, 'o', color='blue', label = 'Estimated Autocorrelation')
        ax.axhline(0, color='black', lw=0.5)

        if CI_HAC and self.cv_values and self.var_unres_values:
            for lag, v, fc, vu in zip(lags, self.rho_est_values, self.cv_values, self.var_unres_values):
                upper_bound = min(1.00, v + fc * np.sqrt(vu))
                lower_bound = max(-1.00, v - fc * np.sqrt(vu))
                
                
                # Plotting the brackets
                ax.plot([lag, lag], [upper_bound, v], color='gray', lw=0.8)
                ax.plot([lag, lag], [lower_bound, v], color='gray', lw=0.8)
                
                # Plotting the horizontal bar at the tip of brackets
                bracket_length = 0.2
                ax.plot([lag-bracket_length, lag+bracket_length], [upper_bound, upper_bound], color='black', lw=1.1)
                ax.plot([lag-bracket_length, lag+bracket_length], [lower_bound, lower_bound], color='black', lw=1.1, label='Confidence Interval - HAC robust' if lag == lags[0] else "")
                
        #if CB_HAC and self.cv_values and self.var_unres_values:
        #    upper_bounds = [min(1.00, 0 + fc * np.sqrt(vu)) for fc, vu in zip(self.cv_values, self.var_unres_values)]
        #    lower_bounds = [max(-1.00, 0 - fc * np.sqrt(vu)) for fc, vu in zip(self.cv_values, self.var_unres_values)]
            
        #    ax.fill_between(lags, lower_bounds, upper_bounds, color='gray', alpha=0.2)
        
        if CB_HAC and self.cv_values and self.var_unres_values:
            upper_bounds = [min(1.00, 0 + fc * np.sqrt(vu)) for fc, vu in zip(self.cv_values, self.var_unres_values)]
            lower_bounds = [max(-1.00, 0 - fc * np.sqrt(vu)) for fc, vu in zip(self.cv_values, self.var_unres_values)]

            ax.plot(lags, upper_bounds, color='green', linestyle='dashed', lw=0.99)
            ax.plot(lags, lower_bounds, color='green', linestyle='dashed', lw=0.99, label='Confidence Band - HAC robust')

            
        if CB_stata:
            ax.plot(lags, self._stacked_autocorrelation(len(self.rho_est_values)), 'o', color='red', markerfacecolor="None", label='Sample Autocorrelation')
            
            var_stata = self._calculate_var_stata_ori()
            upper_bound = [0 + stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
            lower_bound = [0 - stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
            ax.fill_between(lags, lower_bound, upper_bound, color='grey', alpha=0.2, label='Confidence Band - Stata Formula')
            
        if CB_bart and len(self.data):
            if not CB_stata:
                ax.plot(lags, self._stacked_autocorrelation(len(self.rho_est_values)), 'o', color='red', markerfacecolor="None", label='Sample Autocorrelation')
            
            conf_value = 1.96 / np.sqrt(len(self.data))
            ax.axhline(conf_value, color='red', linestyle='dashed', lw=0.8)
            ax.axhline(-conf_value, color='red', linestyle='dashed', lw=0.8, label='Confidence Band - Bartlett Formula')

        ax.set_xlabel('Lag')
        ax.set_ylabel('Estimated Autocorrelation')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.25)
        ax.set_xlim([0, len(self.rho_est_values) + 1])
        ax.set_ylim([-1.1, 1.1])

        if save_as_pdf:
            plt.savefig(filename, format='pdf')
        else:
            plt.show()
            
    def get_estimated_acf(self):
        """
        Retrieve the estimated autocorrelation function values.

        Returns
        -------
        list of float
            List of estimated ACF values.
        """
        return self.rho_est_values
    
    
    def get_confidence_interval(self):
        """
        Compute the confidence intervals for the ACF values.

        Returns
        -------
        tuple of list of float
            Lower and upper confidence interval bounds for each ACF value.
        """
        if self.null_imp:
            return self.lower_values, self.upper_values
        else:
            lower_bounds = [rho - cv * np.sqrt(var) for rho, cv, var in zip(self.rho_est_values, self.cv_values, self.var_unres_values)]
            upper_bounds = [rho + cv * np.sqrt(var) for rho, cv, var in zip(self.rho_est_values, self.cv_values, self.var_unres_values)]
            return lower_bounds, upper_bounds
        
    def get_confidence_band(self):
        """
        Compute the confidence bands around zero for the ACF values.

        Returns
        -------
        tuple of list of float
            Lower and upper confidence band bounds for each ACF value.
        """
        if self.null_imp:
            lower_bounds = [0 - cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_res_under_zero_values)]
            upper_bounds = [0 + cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_res_under_zero_values)]
            return lower_bounds, upper_bounds
        else:
            lower_bounds = [0 - cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_unres_values)]
            upper_bounds = [0 + cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_unres_values)]
            return lower_bounds, upper_bounds
        
    def get_null_imp_index(self):
        """
        Retrieve the index values if the null is imposed.

        Returns
        -------
        list of int or None
            List of index values if null is imposed; otherwise, print a message and return None.
        """
        if self.null_imp:
            return self.index_values
        else:
            print ("Only null imposed has index")
    
    def get_cb_stata(self):
        """
        Compute the Stata-based confidence bands for the ACF values.

        Returns
        -------
        tuple of list of float
            Lower and upper Stata-based confidence band bounds for each ACF value.
        """
        var_stata = self._calculate_var_stata_ori()
        upper_bounds = [0 + stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
        lower_bounds = [0 - stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
        return lower_bounds, upper_bounds
    
    def get_cv(self):
        """
        Retrieve the critical values used for confidence interval calculations.

        Returns
        -------
        list of float
            List of critical values.
        """
        return self.cv_values
    
    def get_sample_autocorrelation(self):
        """
        Compute the sample autocorrelation for a given number of lags.

        Returns
        -------
        list of float
            List of sample autocorrelation values.
        """
        return self._stacked_autocorrelation(len(self.rho_est_values))

def calc_fixed_cv_org_alpha(fixed_b, Tnum, fixedb_coeffs, alpha):
    """
    Compute fixed-b critical values for t-test

    Args:
    - fixed_b: value of bandwidth (i.e. M)
    - Tnum (int): length of time series, in this autocorrelation testing case, T-k
    - fixedb_coeffs: List of fixed-b cv coefficients [c1, c2, ..., c9]

    Returns:
    - Scalar, fixed-b critical value
    """
    c = copy(fixedb_coeffs)  # shallow copy for list prevent mistake
    n_cv = stats.norm.ppf(1 - alpha / 2)  # eg) stats.norm.ppf(0.975)

    return n_cv + c[0] * (fixed_b / Tnum * n_cv) + c[1] * ((fixed_b / Tnum) * (n_cv ** 2)) + \
           c[2] * ((fixed_b / Tnum) * (n_cv ** 3)) + \
           c[3] * (((fixed_b / Tnum) ** 2) * n_cv) + c[4] * (((fixed_b / Tnum) ** 2) * (n_cv ** 2)) + \
           c[5] * (((fixed_b / Tnum) ** 2) * (n_cv ** 3)) + \
           c[6] * (((fixed_b / Tnum) ** 3) * (n_cv ** 1)) + c[7] * (((fixed_b / Tnum) ** 3) * (n_cv ** 2)) + \
           c[8] * (((fixed_b / Tnum) ** 3) * (n_cv ** 3))

def process_var_fixedbcv_alpha(vhat, M_n, X, nmp, fixedb_coeffs,alpha):
    vhat = np.copy(vhat)
    X =  np.copy(X)
    fixedb_coeffs = copy(fixedb_coeffs)
    
    Q_inv = np.linalg.inv(np.dot(X.T, X))
    LRV = newLRV_vec(vhat, M_n)
    
    var = np.linalg.multi_dot([Q_inv, LRV, Q_inv]) * nmp
    fixed_cv = calc_fixed_cv_org_alpha(M_n, nmp, fixedb_coeffs,alpha)
    #t_stat = (coeffs1 - true_acf1) / np.sqrt(np.float64(var[1, 1]))

    return fixed_cv, var

def regression_residuals(data, k):
    """
    Compute the beta_2 coefficient and residuals w_t after regressing 
    the residuals from partialling out the effects of t and intercept
    on y_t and y_t-k.
    
    Parameters:
    - data: ndarray
        Column vector containing the data series
    - k: int
        Lag value
    
    Returns:
    - beta_2: float
        Coefficient from regressing the residuals of y_t on y_t-k
    - w_t: ndarray
        Residuals from the regression
    """
    data = np.copy(data)
    T_data = len(data)
    
    y_t = np.copy(data[k:T_data, :])
    y_t_minus_k = np.copy(data[0:T_data - k, :])
    t = np.arange(k + 1, T_data + 1).reshape(-1, 1)
    
    # Part 2: Partial out the effect of t and intercept on y_t and y_t-k
    residuals_y_t = partial_regression(y_t, t, coef=False, cons=True)
    residuals_y_t_minus_k = partial_regression(y_t_minus_k, t, coef=False, cons=True)

    # Part 3: Regress the residuals on each other without intercept
    beta_2, _ = partial_regression(residuals_y_t, residuals_y_t_minus_k, coef=True, cons=False)
    
    # w_t residuals from this regression
    w_t = residuals_y_t - beta_2 * residuals_y_t_minus_k

    return beta_2, w_t, residuals_y_t_minus_k

def partial_regression(y, x, coef=False, cons=True):
    """
    Perform a regression of y on x and optionally return the coefficients.
    
    Parameters:
    - y: ndarray
        Dependent variable
    - x: ndarray
        Independent variable
    - coef: bool (default: False)
        If True, return coefficients
    - cons: bool (default: True)
        If True, include intercept in regression
    
    Returns:
    - residuals: ndarray
        Residuals from the regression
    - gamma_1: float (optional)
        Coefficient from the regression (if coef=True)
    """
    y = np.copy(y)
    x = np.copy(x)
    
    y = np.array(y).reshape(-1, 1)
    x = np.array(x).reshape(-1, 1)
    
    if cons:
        cons_array = np.ones((len(x), 1))
        X = np.hstack((cons_array, x))
    else:
        X = x

    coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    y_pred = X.dot(coeffs)
    residuals = y - y_pred

    if coef:
        return coeffs.flatten(), residuals
    else:
        return residuals
    
def AD_band(vhat):
    vhat = np.copy(vhat)
    #vhat = np.copy(xuhat)
    if vhat.shape[1] == 1:
        pass
    elif vhat.shape[1] == 2:
        vhat = np.copy(np.reshape(vhat[:,1],(len(vhat[:,1]),1))) 
    
    T = len(vhat)

    vhat_t = np.copy(vhat[1:])
    vhat_tm1 = np.copy(vhat[:-1])
    y2 = np.copy(vhat_t)
    X2 = np.copy(vhat_tm1)
    rho_hat = np.linalg.inv(X2.transpose().dot(X2)).dot(X2.transpose()).dot(y2) #without constant
    #print (rho_hat)
    #if (np.abs(rho_hat-1) < 0.001) & (rho_hat>=1):
    #    rho_hat = 1.001
    #elif (np.abs(rho_hat-1) < 0.001) & (rho_hat<1):
    #    rho_hat = 0.999
        
    #if np.abs(rho_hat) >= 0.97:
    #    rho_hat = 0.97*np.sign(rho_hat)

    alpha_hat_2 = (4*(rho_hat)**2)/((1-rho_hat)**4) #univariate version CLP
    ST = 2.6614*(T*alpha_hat_2)**(0.2) #bandwidth
    ST = np.float64(ST)
    
    if ST > T:
        ST = np.copy(T)
    
    return ST

def noncen_chisq(k,j,x):
    #k is df
    #j is non-centrality parameter
    pdf_v = 0.5*np.exp(-0.5*(x+j))*((x/j)**(k/4-0.5))*mpmath.besseli(0.5*k-1, np.sqrt(j*x), derivative=0)
    pdf_v2 = np.float64(pdf_v)
    return pdf_v2

def chisq_dfone(x):
    pdf_v = (np.exp(-x/2))/(np.sqrt(2*math.pi*x))
    pdf_v2 = np.float64(pdf_v)
    return pdf_v2

def SPJ_band(vhat,w,z_a=1.96,delta=2,q=2,g=6,c=0.539):
    #for Parzen
    #z_a = 1.96 # w is tuning parameter
    vhat = np.copy(vhat)

    if vhat.shape[1] == 1:
        pass
    elif vhat.shape[1] == 2:
        vhat = np.copy(np.reshape(vhat[:,1],(len(vhat[:,1]),1))) 
    
    T = len(vhat)

    vhat_t = np.copy(vhat[1:])
    vhat_tm1 = np.copy(vhat[:-1])
    y2 = np.copy(vhat_t)
    X2 = np.copy(vhat_tm1)
    rho_hat = np.linalg.inv(X2.transpose().dot(X2)).dot(X2.transpose()).dot(y2) #without constant
    
    #if np.abs(rho_hat) >= 0.97:
    #    rho_hat = 0.97*np.sign(rho_hat)
    
    d_hat = (2*rho_hat)/(1-rho_hat)**2
    d_hat = np.float64(d_hat)
    x_val = z_a**2
    
    ### calculation of b_hat
    G_0 = chisq_dfone(x=x_val)
    G_d = noncen_chisq(k=1,j=delta**2,x=x_val)
    k_d = ((delta**2)/(2*x_val))*noncen_chisq(k=3,j=delta**2,x=x_val)
    
    if d_hat*(w*G_0 - G_d) > 0:
        b_hat = (((q*g*d_hat*(w*G_0 - G_d))/(c*x_val*k_d))**(1/3))*(T**(-2/3))
        #print (b_hat)
    elif d_hat*(w*G_0 - G_d) <= 0:
        b_hat = np.log(T)/T
    else:
        raise Exception("Error")
    spj_band = b_hat * T
    spj_band = np.float64(spj_band)
    
    if spj_band > T:
        spj_band = np.copy(T)    
    
    return spj_band

def create_spj_function(w, z_a=1.96, delta=2, q=2, g=6, c=0.539):
    def spj_function(vhat):
        return SPJ_band(vhat, w=w, z_a=z_a, delta=delta, q=q, g=g, c=c)
    return spj_function

def compute_Q2(y, k):
    """
    Compute Q2 value.
    
    Args:
    - y (np.array): Input data array (time-series)
    - k (int): lag k; i.e. rho_k
    
    Returns:
    - float: Q2 value
    """
    
    T = len(y)

    # Compute the means
    ybar_1_T_minus_k = np.mean(y[:T-k])

    # Demeaning y values
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k

    # Compute Q2
    Q2 = np.mean(y_tilda_minus_k**2)
    
    return Q2

def parzen_k_function_vectorized(x):
    """Vectorized Kernel Function."""
    conditions = [
        (abs(x) <= 0.5),
        (abs(x) > 0.5) & (abs(x) <= 1)
    ]
    
    outputs = [
        1 - 6 * x**2 + 6 * abs(x)**3,
        2 * (1 - abs(x))**3
    ]
    
    return np.select(conditions, outputs, default=0)

def compute_omegas_vectorized_demean(y, M, k_function, k):
    y = y.reshape(-1, 1)
    T = y.shape[0]

    # Compute the means
    ybar_1_T_minus_k = np.mean(y[:T-k])
    ybar_k_plus_1_T = np.mean(y[k:T])

    # Demeaning y values and computing v1 and v2
    y_tilda = y[k:] - ybar_k_plus_1_T
    y_tilda_minus_k = y[:T-k] - ybar_1_T_minus_k

    v1 = (y_tilda * y_tilda_minus_k).squeeze()
    v2 = (y_tilda_minus_k ** 2).squeeze()

    # Calculate the means of v1 and v2 
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)

    # Demean v1 and v2
    v1_demeaned = v1 - mean_v1
    v2_demeaned = v2 - mean_v2

    # Create the kernel matrix
    indices_matrix = np.abs(np.arange(k+1, T+1)[:, None] - np.arange(k+1, T+1))
    K = k_function(indices_matrix / M)

    # Compute Omega values using demeaned v1 and v2
    omega_11 = np.sum(v1_demeaned[:, None] * K * v1_demeaned[None, :]) / (T-k)
    omega_12 = np.sum(v1_demeaned[:, None] * K * v2_demeaned[None, :]) / (T-k)
    omega_22 = np.sum(v2_demeaned[:, None] * K * v2_demeaned[None, :]) / (T-k)

    return omega_11, omega_12, omega_22

def compute_c_coefficients(omegas, cv, y, k_val, rho_k_til, Q_2):
    """
    Compute the coefficients c_0_nd, c_1_nd, and c_2_nd.
    
    Args:
    - omegas (tuple): Omega values (omegas_nd11, omegas_nd12, omegas_nd22)
    - cv (float): Value of cv
    - y (np.array): Input data array
    - k_val (int): Parameter k value
    - rho_k_til (float): Value of rho_k_til
    - Q_2 (float): Value of Q_2
    
    Returns:
    - tuple: Coefficients (c_0_nd, c_1_nd, c_2_nd)
    """
    
    omegas_nd11, omegas_nd12, omegas_nd22 = omegas

    T_minus_k = len(y) - k_val
    Q_2_inv_squared = Q_2 ** (-2)
    common_factor = (1 / T_minus_k) * Q_2_inv_squared * (cv ** 2)

    c_2_nd = 1 - common_factor * omegas_nd22
    c_1_nd = common_factor * omegas_nd12 - rho_k_til
    c_0_nd = (rho_k_til ** 2) - common_factor * omegas_nd11

    return c_2_nd, c_1_nd, c_0_nd

def quadratic_roots(c2, c1, c0):
    """
    Calculate the roots of the equation: c2*a^2 + 2*c1*a + c0 = 0

    Args:
    - c2 (float): Coefficient of a^2
    - 2*c1 (float): Coefficient of a
    - c0 (float): Constant term

    Returns:
    - tuple: Roots of the equation (root1, root2)
    """
    
    # Calculating the discriminant
    D = cmath.sqrt((2*c1)**2 - 4*c2*c0)

    # Calculating the roots
    root1 = (-2*c1 + D) / (2*c2)
    root2 = (-2*c1 - D) / (2*c2)

    return root1, root2

def reg_estimating_equation_v1(data, lag):
    """
    Estimating equtation for autocorrelation, the first option

    Parameters:
    - data (np.ndarray): The input data array. y_t from t=1 to T
    - p (lag) (int): The lag order estimating equation

    Returns:
    - y (np.ndarray): y_t from t=p+1 to T
    - x (np.ndarray): y_t-p
    - nmp (int): The length of x (T - p).
    - X (np.ndarray): T x 2 matrix with intercept and x
    - coeffs (np.ndarray): The estimated coefficients from the estimating equation
    """
    data = np.copy(data)
    p = np.copy(lag)
    
    T_data = len(data)  # T

    # Create y and x based on the lag order p
    y = np.copy(data[p:T_data, :])
    x = np.copy(data[0:T_data - p, :])

    # Calculate nmp (T - p)
    nmp = len(x)

    # Create a constant term for the regression
    cons = np.ones((nmp, 1))

    # Create the design matrix X
    X = np.hstack((cons, x))

    # Calculate the coefficients based on the ordinary least squares (OLS) formula
    coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    return y, x, nmp, X, coeffs

def newLRV_vec(vhat, M_n):
    vhat = np.copy(vhat)
    
    n, k = vhat.shape
    LRV = autocov_est(vhat, 0)
    
    j_values = np.arange(1, n)
    parzen_values = Parzen_vec(j_values / M_n)
    
    autocov_values = np.array([autocov_est(vhat, j) for j in j_values])
    
    for j, p_val in zip(j_values, parzen_values):
        LRV += p_val * (autocov_values[j-1] + autocov_values[j-1].T)

    return LRV

def autocov_est(vhat,j):
    vhat = np.copy(vhat)
    if j >= 0:
        T = len(vhat)
        v_t = vhat[j:T]
        v_tm1 = vhat[0:T-j]
        gamma_hat_j = np.dot(v_t.T,v_tm1)/T
        return gamma_hat_j
    else:
        raise ValueError("j cannot be negative number")
        
def Parzen_vec(x):
    x = np.copy(x)
    
    kx = np.zeros_like(x)
    
    mask1 = (0 <= np.abs(x)) & (np.abs(x) <= 0.5)
    mask2 = (0.5 < np.abs(x)) & (np.abs(x) <= 1)
    
    kx[mask1] = 1 - 6 * (x[mask1] ** 2) + 6 * (np.abs(x[mask1]) ** 3)
    kx[mask2] = 2 * ((1 - np.abs(x[mask2])) ** 3)
    
    return kx