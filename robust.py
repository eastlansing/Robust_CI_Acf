class HAC_robust_conf_int:
    
    """
    This is test

    Parameters
    ----------

    n_factors : int, default=1
        abcd

    intercept : boolean, default=False
        abd

    max_iter : int, default=10000
        abcd
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
        - k (int): The lag order
        
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
        return self.rho_est_values
    
    
    def get_confidence_interval(self):
        if self.null_imp:
            return self.lower_values, self.upper_values
        else:
            lower_bounds = [rho - cv * np.sqrt(var) for rho, cv, var in zip(self.rho_est_values, self.cv_values, self.var_unres_values)]
            upper_bounds = [rho + cv * np.sqrt(var) for rho, cv, var in zip(self.rho_est_values, self.cv_values, self.var_unres_values)]
            return lower_bounds, upper_bounds
        
    def get_confidence_band(self):
        if self.null_imp:
            lower_bounds = [0 - cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_res_under_zero_values)]
            upper_bounds = [0 + cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_res_under_zero_values)]
            return lower_bounds, upper_bounds
        else:
            lower_bounds = [0 - cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_unres_values)]
            upper_bounds = [0 + cv * np.sqrt(var) for cv, var in zip(self.cv_values, self.var_unres_values)]
            return lower_bounds, upper_bounds
        
    def get_null_imp_index(self):
        if self.null_imp:
            return self.index_values
        else:
            print ("Only null imposed has index")
    
    def get_cb_stata(self):
        var_stata = self._calculate_var_stata_ori()
        upper_bounds = [0 + stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
        lower_bounds = [0 - stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(var) for var in var_stata]
        return lower_bounds, upper_bounds
    
    def get_cv(self):
        return self.cv_values
    
    def get_sample_autocorrelation(self):
        return self._stacked_autocorrelation(len(self.rho_est_values))
