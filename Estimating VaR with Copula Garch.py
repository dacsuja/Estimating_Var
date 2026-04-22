"""
"Estimating value at risk of portfolio by conditional copula-GARCH method"
Huang, Lee, Liang, Lin (2009), Insurance: Mathematics and Economics 45, 315-324."""


import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model


def download_data():
    import yfinance as yf

    start_date = "2000-07-01"
    end_date = "2007-05-20"

    print("Downloading Nikkei 225 (^N225) daily prices...")
    nikkei = yf.download("^N225", start=start_date, end=end_date, auto_adjust=True)

    print("Downloading CAC 40 (^FCHI) daily prices...")
    cac40 = yf.download("^FCHI", start=start_date, end=end_date, auto_adjust=True)

    nikkei_close = nikkei["Close"].squeeze().dropna()
    cac40_close = cac40["Close"].squeeze().dropna()

    return nikkei_close, cac40_close


def align_and_compute_returns(nikkei_close, cac40_close):
    common_dates = nikkei_close.index.intersection(cac40_close.index)
    nikkei_aligned = nikkei_close.loc[common_dates].sort_index()
    cac40_aligned = cac40_close.loc[common_dates].sort_index()

    print(f"\nCommon trading days (both markets open): {len(common_dates)}")

    nikkei_returns = 100.0 * np.log(nikkei_aligned / nikkei_aligned.shift(1)).dropna()
    cac40_returns = 100.0 * np.log(cac40_aligned / cac40_aligned.shift(1)).dropna()

    common_ret_dates = nikkei_returns.index.intersection(cac40_returns.index)
    nikkei_returns = nikkei_returns.loc[common_ret_dates]
    cac40_returns = cac40_returns.loc[common_ret_dates]

    print(f"Return observations: {len(nikkei_returns)}")
    print(f"Date range: {common_ret_dates[0].strftime('%Y-%m-%d')} to "
          f"{common_ret_dates[-1].strftime('%Y-%m-%d')}")

    return cac40_returns, nikkei_returns


def compute_descriptive_statistics(series):
    n = len(series)
    mean_val = np.mean(series)
    std_val = np.std(series, ddof=1)
    skew_val = stats.skew(series, bias=True)
    excess_kurt_val = stats.kurtosis(series, bias=True)

    return {
        'Sample number': n,
        'Mean': mean_val,
        'Standard deviation': std_val,
        'Skewness': skew_val,
        'Excess of Kurtosis': excess_kurt_val
    }


def engle_lm_test(series, nlags):
    residuals = series - np.mean(series)

    lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=nlags)

    return {'LM-statistic': lm_stat, 'P-value': lm_pval}


def print_table1(cac40_stats, nikkei_stats, cac40_arch, nikkei_arch, lags):
    print("\n" + "=" * 75)
    print("TABLE 1")
    print("Descriptive statistic and Engle tests.")
    print("=" * 75)

    print(f"\n  {'Statistics':<25} {'TWIEX':<30} {'nikkei':<30}")
    print(f"  {'-' * 85}")

    for key in ['Sample number', 'Mean', 'Standard deviation', 'Skewness',
                'Excess of Kurtosis']:
        t_val = cac40_stats[key]
        n_val = nikkei_stats[key]
        if key == 'Sample number':
            print(f"  {key:<25} {t_val:<30} {n_val:<30}")
        else:
            print(f"  {key:<25} {t_val:<30.4f} {n_val:<30.4f}")

    print(f"  {'Engle-test':<25} {'Q-statistic':<15} {'P-value':<15} "
          f"{'Q-statistic':<15} {'P-value':<15}")

    for i, lag in enumerate(lags):
        t_q = cac40_arch[i]['LM-statistic']
        t_p = cac40_arch[i]['P-value']
        n_q = nikkei_arch[i]['LM-statistic']
        n_p = nikkei_arch[i]['P-value']
        print(f"  LM({lag:<2})                 {t_q:<15.4f} ({t_p:.4f})       "
              f"{n_q:<15.4f} ({n_p:.4f})")

def estimate_garch_model(returns, dist='normal'):
    am = arch_model(
        returns,
        mean='Constant',
        vol='GARCH',
        p=1,
        q=1,
        dist='normal' if dist == 'normal' else 'studentst'
    )

    res = am.fit(disp='off')

    mu = res.params['mu']
    alpha0 = res.params['omega']
    alpha1 = res.params['alpha[1]']
    beta = res.params['beta[1]']

    mu_std = res.std_err['mu']
    alpha0_std = res.std_err['omega']
    alpha1_std = res.std_err['alpha[1]']
    beta_std = res.std_err['beta[1]']

    d = None
    d_std = None
    if dist == 't':
        d = res.params['nu']
        d_std = res.std_err['nu']

    llf = res.loglikelihood

    nparams = res.num_params
    nobs = res.nobs

    aic = -2 * llf + 2 * nparams
    bic = -2 * llf + nparams * np.log(nobs)

    std_resid = res.std_resid

    cond_vol = res.conditional_volatility

    return {
        'mu': mu, 'mu_std': mu_std,
        'alpha0': alpha0, 'alpha0_std': alpha0_std,
        'alpha1': alpha1, 'alpha1_std': alpha1_std,
        'beta': beta, 'beta_std': beta_std,
        'd': d, 'd_std': d_std,
        'LLF': llf, 'AIC': aic, 'BIC': bic,
        'std_resid': std_resid,
        'cond_vol': cond_vol,
        'nobs': nobs,
        'nparams': nparams,
        'result': res
    }


def diagnostic_tests_garch(std_resid):
    resid = np.asarray(std_resid).flatten()

    ljung_box_lags = [1, 3, 5, 7]
    lb_results = []

    for lag in ljung_box_lags:
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        q_stat = lb['lb_stat'].values[0]
        p_val = lb['lb_pvalue'].values[0]
        lb_results.append({
            'lag': lag,
            'Q-statistic': q_stat,
            'P-value': p_val
        })

    engle_lags = [4, 6, 8, 10]
    engle_results = []

    for lag in engle_lags:
        lm_stat, lm_pval, f_stat, f_pval = het_arch(resid, nlags=lag)
        engle_results.append({
            'lag': lag,
            'Q-statistic': lm_stat,
            'P-value': lm_pval
        })

    return {
        'ljung_box': lb_results,
        'engle': engle_results
    }


def print_table2(twiex_gn, twiex_gt, nikkei_gn, nikkei_gt):
    print("\n" + "=" * 120)
    print("TABLE 2")
    print("Parameter estimates of GARCH model and statistic test.")
    print("=" * 120)

    print(f"\n{'':>15} {'GARCH-n':^50} {'GARCH-t':^50}")
    print(f"{'':>15} {'TWIEX':^25} {'nikkei':^25} {'TWIEX':^25} {'nikkei':^25}")
    print(f"{'Parameter':>15} {'Value':>10} {'Std':>10}   {'Value':>10} {'Std':>10}   "
          f"{'Value':>10} {'Std':>10}   {'Value':>10} {'Std':>10}")
    print("-" * 120)

    models = [twiex_gn, nikkei_gn, twiex_gt, nikkei_gt]

    for pname, key, key_std in [
        ('μ', 'mu', 'mu_std'),
        ('α₀', 'alpha0', 'alpha0_std'),
        ('α₁', 'alpha1', 'alpha1_std'),
        ('β', 'beta', 'beta_std')
    ]:
        vals = [m[key] for m in models]
        stds = [m[key_std] for m in models]
        print(f"{pname:>15} {vals[0]:>10.4f} {stds[0]:>10.4f}   "
              f"{vals[1]:>10.4f} {stds[1]:>10.4f}   "
              f"{vals[2]:>10.4f} {stds[2]:>10.4f}   "
              f"{vals[3]:>10.4f} {stds[3]:>10.4f}")

    print(f"{'d':>15} {'':>10} {'':>10}   {'':>10} {'':>10}   "
          f"{twiex_gt['d']:>10.4f} {twiex_gt['d_std']:>10.4f}   "
          f"{nikkei_gt['d']:>10.4f} {nikkei_gt['d_std']:>10.4f}")

    print(f"{'LLF':>15} {twiex_gn['LLF']:>10.2f} {'':>10}   "
          f"{nikkei_gn['LLF']:>10.2f} {'':>10}   "
          f"{twiex_gt['LLF']:>10.2f} {'':>10}   "
          f"{nikkei_gt['LLF']:>10.2f} {'':>10}")
    print(f"{'AIC':>15} {twiex_gn['AIC']:>10.2f} {'':>10}   "
          f"{nikkei_gn['AIC']:>10.2f} {'':>10}   "
          f"{twiex_gt['AIC']:>10.2f} {'':>10}   "
          f"{nikkei_gt['AIC']:>10.2f} {'':>10}")
    print(f"{'BIC':>15} {twiex_gn['BIC']:>10.2f} {'':>10}   "
          f"{nikkei_gn['BIC']:>10.2f} {'':>10}   "
          f"{twiex_gt['BIC']:>10.2f} {'':>10}   "
          f"{nikkei_gt['BIC']:>10.2f} {'':>10}")

    twiex_gn_diag = diagnostic_tests_garch(twiex_gn['std_resid'])
    nikkei_gn_diag = diagnostic_tests_garch(nikkei_gn['std_resid'])
    twiex_gt_diag = diagnostic_tests_garch(twiex_gt['std_resid'])
    nikkei_gt_diag = diagnostic_tests_garch(nikkei_gt['std_resid'])

    print(f"\n{'':>15} {'P-value':>10} {'Q-stat':>10}   {'P-value':>10} {'Q-stat':>10}   "
          f"{'P-value':>10} {'Q-stat':>10}   {'P-value':>10} {'Q-stat':>10}")
    print("-" * 120)
    print("Ljung-Box test")

    for i, lag in enumerate([1, 3, 5, 7]):
        gn_tw = twiex_gn_diag['ljung_box'][i]
        gn_nq = nikkei_gn_diag['ljung_box'][i]
        gt_tw = twiex_gt_diag['ljung_box'][i]
        gt_nq = nikkei_gt_diag['ljung_box'][i]
        print(f"{'QW(' + str(lag) + ')':>15} {gn_tw['P-value']:>10.4f} {gn_tw['Q-statistic']:>10.4f}   "
              f"{gn_nq['P-value']:>10.4f} {gn_nq['Q-statistic']:>10.4f}   "
              f"{gt_tw['P-value']:>10.4f} {gt_tw['Q-statistic']:>10.4f}   "
              f"{gt_nq['P-value']:>10.4f} {gt_nq['Q-statistic']:>10.4f}")

    print("\nEngles test")
    for i, lag in enumerate([4, 6, 8, 10]):
        gn_tw = twiex_gn_diag['engle'][i]
        gn_nq = nikkei_gn_diag['engle'][i]
        gt_tw = twiex_gt_diag['engle'][i]
        gt_nq = nikkei_gt_diag['engle'][i]
        print(f"{'LM(' + str(lag) + ')':>15} {gn_tw['P-value']:>10.4f} {gn_tw['Q-statistic']:>10.4f}   "
              f"{gn_nq['P-value']:>10.4f} {gn_nq['Q-statistic']:>10.4f}   "
              f"{gt_tw['P-value']:>10.4f} {gt_tw['Q-statistic']:>10.4f}   "
              f"{gt_nq['P-value']:>10.4f} {gt_nq['Q-statistic']:>10.4f}")

def estimate_gjr_model(returns, dist='normal'):
    am = arch_model(
        returns,
        mean='Constant',
        vol='GARCH',
        p=1,
        o=1,
        q=1,
        dist='normal' if dist == 'normal' else 'studentst'
    )

    res = am.fit(disp='off')

    mu = res.params['mu']
    alpha0 = res.params['omega']
    alpha1 = res.params['alpha[1]']
    beta = res.params['beta[1]']
    gamma = res.params['gamma[1]']

    mu_std = res.std_err['mu']
    alpha0_std = res.std_err['omega']
    alpha1_std = res.std_err['alpha[1]']
    beta_std = res.std_err['beta[1]']
    gamma_std = res.std_err['gamma[1]']

    d = None
    d_std = None
    if dist == 't':
        d = res.params['nu']
        d_std = res.std_err['nu']

    llf = res.loglikelihood

    nparams = res.num_params
    nobs = res.nobs

    aic = -2 * llf + 2 * nparams
    bic = -2 * llf + nparams * np.log(nobs)

    std_resid = res.std_resid

    cond_vol = res.conditional_volatility

    return {
        'mu': mu, 'mu_std': mu_std,
        'alpha0': alpha0, 'alpha0_std': alpha0_std,
        'alpha1': alpha1, 'alpha1_std': alpha1_std,
        'beta': beta, 'beta_std': beta_std,
        'gamma': gamma, 'gamma_std': gamma_std,
        'd': d, 'd_std': d_std,
        'LLF': llf, 'AIC': aic, 'BIC': bic,
        'std_resid': std_resid,
        'cond_vol': cond_vol,
        'nobs': nobs,
        'nparams': nparams,
        'result': res
    }


def print_table3(twiex_gn, twiex_gt, nikkei_gn, nikkei_gt):
    print("\n" + "=" * 120)
    print("TABLE 3")
    print("Parameter estimates of GJR model and statistic test.")
    print("=" * 120)

    print(f"\n{'':>15} {'GJR-n':^50} {'GJR-t':^50}")
    print(f"{'':>15} {'TWIEX':^25} {'nikkei':^25} {'TWIEX':^25} {'nikkei':^25}")
    print(f"{'Parameter':>15} {'Value':>10} {'Std':>10}   {'Value':>10} {'Std':>10}   "
          f"{'Value':>10} {'Std':>10}   {'Value':>10} {'Std':>10}")
    print("-" * 120)

    models = [twiex_gn, nikkei_gn, twiex_gt, nikkei_gt]

    for pname, key, key_std in [
        ('μ', 'mu', 'mu_std'),
        ('α₀', 'alpha0', 'alpha0_std'),
        ('α₁', 'alpha1', 'alpha1_std'),
        ('β', 'beta', 'beta_std'),
        ('γ', 'gamma', 'gamma_std')
    ]:
        vals = [m[key] for m in models]
        stds = [m[key_std] for m in models]
        print(f"{pname:>15} {vals[0]:>10.4f} {stds[0]:>10.4f}   "
              f"{vals[1]:>10.4f} {stds[1]:>10.4f}   "
              f"{vals[2]:>10.4f} {stds[2]:>10.4f}   "
              f"{vals[3]:>10.4f} {stds[3]:>10.4f}")

    print(f"{'d':>15} {'':>10} {'':>10}   "
          f"{'':>10} {'':>10}   "
          f"{twiex_gt['d']:>10.4f} {twiex_gt['d_std']:>10.4f}   "
          f"{nikkei_gt['d']:>10.4f} {nikkei_gt['d_std']:>10.4f}")

    print(f"{'LLF':>15} {twiex_gn['LLF']:>10.2f} {'':>10}   "
          f"{nikkei_gn['LLF']:>10.2f} {'':>10}   "
          f"{twiex_gt['LLF']:>10.2f} {'':>10}   "
          f"{nikkei_gt['LLF']:>10.2f} {'':>10}")
    print(f"{'AIC':>15} {twiex_gn['AIC']:>10.2f} {'':>10}   "
          f"{nikkei_gn['AIC']:>10.2f} {'':>10}   "
          f"{twiex_gt['AIC']:>10.2f} {'':>10}   "
          f"{nikkei_gt['AIC']:>10.2f} {'':>10}")
    print(f"{'BIC':>15} {twiex_gn['BIC']:>10.2f} {'':>10}   "
          f"{nikkei_gn['BIC']:>10.2f} {'':>10}   "
          f"{twiex_gt['BIC']:>10.2f} {'':>10}   "
          f"{nikkei_gt['BIC']:>10.2f} {'':>10}")

    twiex_gn_diag = diagnostic_tests_garch(twiex_gn['std_resid'])
    nikkei_gn_diag = diagnostic_tests_garch(nikkei_gn['std_resid'])
    twiex_gt_diag = diagnostic_tests_garch(twiex_gt['std_resid'])
    nikkei_gt_diag = diagnostic_tests_garch(nikkei_gt['std_resid'])

    print(f"\n{'':>15} {'P-value':>10} {'Q-stat':>10}   {'P-value':>10} {'Q-stat':>10}   "
          f"{'P-value':>10} {'Q-stat':>10}   {'P-value':>10} {'Q-stat':>10}")
    print("-" * 120)
    print("Ljung-Box test")

    for i, lag in enumerate([1, 3, 5, 7]):
        gn_tw = twiex_gn_diag['ljung_box'][i]
        gn_nq = nikkei_gn_diag['ljung_box'][i]
        gt_tw = twiex_gt_diag['ljung_box'][i]
        gt_nq = nikkei_gt_diag['ljung_box'][i]
        print(f"{'QW(' + str(lag) + ')':>15} {gn_tw['P-value']:>10.4f} {gn_tw['Q-statistic']:>10.4f}   "
              f"{gn_nq['P-value']:>10.4f} {gn_nq['Q-statistic']:>10.4f}   "
              f"{gt_tw['P-value']:>10.4f} {gt_tw['Q-statistic']:>10.4f}   "
              f"{gt_nq['P-value']:>10.4f} {gt_nq['Q-statistic']:>10.4f}")

    print("\nEngles test")
    for i, lag in enumerate([4, 6, 8, 10]):
        gn_tw = twiex_gn_diag['engle'][i]
        gn_nq = nikkei_gn_diag['engle'][i]
        gt_tw = twiex_gt_diag['engle'][i]
        gt_nq = nikkei_gt_diag['engle'][i]
        print(f"{'LM(' + str(lag) + ')':>15} {gn_tw['P-value']:>10.4f} {gn_tw['Q-statistic']:>10.4f}   "
              f"{gn_nq['P-value']:>10.4f} {gn_nq['Q-statistic']:>10.4f}   "
              f"{gt_tw['P-value']:>10.4f} {gt_tw['Q-statistic']:>10.4f}   "
              f"{gt_nq['P-value']:>10.4f} {gt_nq['Q-statistic']:>10.4f}")

# TABLE 4: Copula parameter estimation
# After estimating marginal GARCH/GJR models, we transform the standardized
# residuals to uniform [0,1] via the probability integral transform (PIT):
#   u_i = F_i(epsilon_i)
# where F_i is the CDF of the assumed innovation distribution (N(0,1) or t_d).
#
# Then we estimate copula parameters by:
# - Gaussian copula: Kendall's tau inversion (Equation 32)
# - All others: MLE or IFM (Equations 16-22)

from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import norm, t as student_t, kendalltau


def probability_integral_transform(std_resid, dist='normal', d=None):
    """
    Apply the probability integral transform to standardized residuals.

    Section 3.1 (Sklar's theorem) and Section 3.3 (Estimation method):
    The IFM method requires transforming residuals to uniform [0,1]:
        u_i = F_i(epsilon_i)

    For GARCH-n / GJR-n: F_i = Phi (standard normal CDF)
    For GARCH-t / GJR-t: F_i = t_d (standardized Student-t CDF)

    Parameters
    ----------
    std_resid : array-like
        Standardized residuals epsilon_t from GARCH/GJR model
    dist : str
        'normal' or 't'
    d : float or None
        Degrees of freedom for Student-t (required if dist='t')

    Returns
    -------
    u : array of uniform [0,1] values
    """
    resid = np.asarray(std_resid).flatten()

    if dist == 'normal':
        u = norm.cdf(resid)
    elif dist == 't':
        # The arch package's standardized Student-t has unit variance,
        # meaning it's scaled by sqrt((d-2)/d). The CDF is:
        # F(x) = t_d(x * sqrt(d/(d-2)))
        # However, the arch package's std_resid already accounts for this
        # scaling, so we need to use the standard t_d CDF directly on
        # the standardized residuals.
        # Actually, arch's studentst distribution is already standardized
        # to have unit variance, so std_resid ~ t_d * sqrt((d-2)/d).
        # To get the correct CDF, we scale back:
        u = student_t.cdf(resid * np.sqrt(d / (d - 2)), d)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    u = np.clip(u, 1e-10, 1 - 1e-10)

    return u


# --- Copula log-likelihood functions ---
# Each function computes the log-likelihood of the copula density c(u1, u2)
# evaluated at the transformed residuals.
# Section 3.3, Equation (16): l(theta) = sum_t ln c(F1(x1t), F2(x2t))
# (the marginal part is already handled in the IFM first stage)


def gaussian_copula_loglik(rho, u1, u2):
    """
    Gaussian copula log-likelihood (Section 3.2, Equation 8).

    C_Gaussian(u1, u2; rho) = Phi_rho(Phi^{-1}(u1), Phi^{-1}(u2))

    The density is:
    c(u1, u2) = (1/sqrt(1-rho^2)) * exp(-(rho^2*(x1^2+x2^2) - 2*rho*x1*x2) / (2*(1-rho^2)))
    where x1 = Phi^{-1}(u1), x2 = Phi^{-1}(u2)
    """
    x1 = norm.ppf(u1)
    x2 = norm.ppf(u2)

    rho2 = rho ** 2

    loglik = -0.5 * np.log(1 - rho2) - (rho2 * (x1 ** 2 + x2 ** 2) - 2 * rho * x1 * x2) / (2 * (1 - rho2))

    return np.sum(loglik)


def student_t_copula_loglik(params, u1, u2):
    """
    Student-t copula log-likelihood (Section 3.2, Equation 9).

    C_T(u1, u2; rho, d) = t_{d,rho}(t_d^{-1}(u1), t_d^{-1}(u2))

    The bivariate t copula density:
    c(u1,u2) = f_{2,d,rho}(t_d^{-1}(u1), t_d^{-1}(u2)) / (f_{1,d}(t_d^{-1}(u1)) * f_{1,d}(t_d^{-1}(u2)))

    where f_{2,d,rho} is the bivariate t density with correlation rho and d dof,
    and f_{1,d} is the univariate t density with d dof.
    """
    rho, d = params

    if d <= 2 or abs(rho) >= 1:
        return -1e10

    x1 = student_t.ppf(u1, d)
    x2 = student_t.ppf(u2, d)

    # Bivariate t density
    # f_2(x1, x2; rho, d) = (1/(2*pi)) * (1/sqrt(1-rho^2)) *
    #   Gamma((d+2)/2) / Gamma(d/2) * (1/d) *
    #   (1 + (x1^2 + x2^2 - 2*rho*x1*x2) / (d*(1-rho^2)))^(-(d+2)/2)

    rho2 = rho ** 2
    Q = (x1 ** 2 + x2 ** 2 - 2 * rho * x1 * x2) / (d * (1 - rho2))

    # Log of bivariate t density
    log_f2 = (gammaln((d + 2) / 2) - gammaln(d / 2)
              - np.log(d * np.pi) - 0.5 * np.log(1 - rho2)
              - ((d + 2) / 2) * np.log(1 + Q))

    # Log of univariate t densities (marginals)
    log_f1_x1 = (gammaln((d + 1) / 2) - gammaln(d / 2)
                 - 0.5 * np.log(d * np.pi)
                 - ((d + 1) / 2) * np.log(1 + x1 ** 2 / d))
    log_f1_x2 = (gammaln((d + 1) / 2) - gammaln(d / 2)
                 - 0.5 * np.log(d * np.pi)
                 - ((d + 1) / 2) * np.log(1 + x2 ** 2 / d))

    # Copula density = bivariate density / product of marginal densities
    loglik = np.sum(log_f2 - log_f1_x1 - log_f1_x2)

    return loglik


def clayton_copula_loglik(omega, u1, u2):
    """
    Clayton copula log-likelihood (Section 3.2, Equation 10).

    C_Clayton(u1, u2; omega) = (u1^{-omega} + u2^{-omega} - 1)^{-1/omega}

    Density:
    c(u1, u2) = (1+omega) * (u1*u2)^{-(1+omega)} * (u1^{-omega} + u2^{-omega} - 1)^{-(2+1/omega)}
    """
    if omega <= -1 or omega == 0:
        return -1e10

    A = u1 ** (-omega) + u2 ** (-omega) - 1

    # Check for valid values
    if np.any(A <= 0):
        return -1e10

    loglik = (np.log(1 + omega)
              - (1 + omega) * (np.log(u1) + np.log(u2))
              - (2 + 1 / omega) * np.log(A))

    return np.sum(loglik)


def rotated_clayton_copula_loglik(omega, u1, u2):
    """
    Rotated-Clayton (180-degree rotation) log-likelihood (Section 3.2, Equation 11).

    C_Rotated-Clayton(u1, u2; omega) = u1 + u2 - 1 + C_Clayton(1-u1, 1-u2; omega)

    The density of the rotated copula is:
    c_rotated(u1, u2) = c_Clayton(1-u1, 1-u2)
    """
    return clayton_copula_loglik(omega, 1 - u1, 1 - u2)


def plackett_copula_loglik(eta, u1, u2):
    """
    Plackett copula log-likelihood (Section 3.2, Equation 12).

    C_Plackett(u1, u2; eta) = [1 + (eta-1)(u1+u2) - sqrt((1+(eta-1)(u1+u2))^2 - 4*eta*(eta-1)*u1*u2)] / (2*(eta-1))

    Density:
    c(u1, u2) = eta * [1 + (eta-1)*(u1+u2-2*u1*u2)] / [(1+(eta-1)*(u1+u2))^2 - 4*eta*(eta-1)*u1*u2]^{3/2}
    """
    if eta <= 0:
        return -1e10

    S = u1 + u2
    P = u1 * u2

    D = (1 + (eta - 1) * S) ** 2 - 4 * eta * (eta - 1) * P

    if np.any(D <= 0):
        return -1e10

    loglik = (np.log(eta)
              + np.log(1 + (eta - 1) * (S - 2 * P))
              - 1.5 * np.log(D))

    return np.sum(loglik)


def frank_copula_loglik(lam, u1, u2):
    """
    Frank copula log-likelihood (Section 3.2, Equation 13).

    C_Frank(u1, u2; lambda) = -(1/lambda) * log[(1-e^{-lambda}) - (1-e^{-lambda*u1})*(1-e^{-lambda*u2})) / (1-e^{-lambda})]

    Density:
    c(u1, u2) = -lambda*(1-e^{-lambda})*e^{-lambda*(u1+u2)} / [(1-e^{-lambda}) - (1-e^{-lambda*u1})*(1-e^{-lambda*u2})]^2
    """
    if abs(lam) < 1e-10:
        return -1e10

    e_lam = np.exp(-lam)
    e_lam_u1 = np.exp(-lam * u1)
    e_lam_u2 = np.exp(-lam * u2)

    denom = (1 - e_lam) - (1 - e_lam_u1) * (1 - e_lam_u2)

    if np.any(denom <= 0):
        return -1e10

    loglik = (np.log(abs(lam)) + np.log(np.abs(1 - e_lam)) - lam * (u1 + u2)
              - 2 * np.log(np.abs(denom)))

    return np.sum(loglik)


def gumbel_copula_loglik(delta, u1, u2):
    """
    Gumbel copula log-likelihood (Section 3.2, Equation 14).

    C_Gumbel(u1, u2; delta) = exp(-[(-log u1)^delta + (-log u2)^delta]^{1/delta})

    Density (derived from the CDF):
    c(u1, u2) = C * (1/(u1*u2)) * ((-log u1)*(-log u2))^{delta-1} *
                (A^{1/delta} + delta - 1) * A^{1/delta - 2}
    where A = (-log u1)^delta + (-log u2)^delta
    """
    if delta < 1:
        return -1e10

    t1 = -np.log(u1)
    t2 = -np.log(u2)

    A = t1 ** delta + t2 ** delta
    A_inv_d = A ** (1 / delta)

    # Log of Gumbel copula density
    logC = -A_inv_d  # log of C(u1, u2)

    loglik = (logC
              - np.log(u1) - np.log(u2)
              + (delta - 1) * np.log(t1) + (delta - 1) * np.log(t2)
              + np.log(A_inv_d + delta - 1)
              + (1 / delta - 2) * np.log(A))

    return np.sum(loglik)


def rotated_gumbel_copula_loglik(delta, u1, u2):
    """
    Rotated-Gumbel (180-degree rotation) log-likelihood (Section 3.2, Equation 15).

    C_Rotated-Gumbel(u1, u2; delta) = u1 + u2 - 1 + C_Gumbel(1-u1, 1-u2; delta)

    The density of the rotated copula is:
    c_rotated(u1, u2) = c_Gumbel(1-u1, 1-u2)
    """
    return gumbel_copula_loglik(delta, 1 - u1, 1 - u2)


def estimate_copulas(u1, u2):
    """
    Estimate all 8 copula families as described in Section 3.2-3.3 and Section 4.3.


    Estimation methods:
    - Gaussian: Kendall's tau transform (Equation 32)
      rho_tau = (2/pi) * arcsin(rho)  =>  rho = sin(pi/2 * rho_tau)

    - Student-t: MLE over (rho, d)
    - Clayton, Rotated-Clayton, Plackett, Frank, Gumbel, Rotated-Gumbel: MLE

    Model selection: AIC and BIC (Section 4.3)
    AIC = -2 * LLF + 2 * k
    BIC = -2 * LLF + k * ln(T)

    Parameters
    ----------
    u1, u2 : array-like
        Uniform [0,1] transformed residuals from the marginal models

    Returns
    -------
    dict of copula estimation results
    """
    n = len(u1)
    results = {}

    # ---- 1. Gaussian copula (Equation 8, estimated via Equation 32) ----
    # Kendall's tau transform: rho_tau = (2/pi) * arcsin(rho)
    # Inversion: rho = sin(pi/2 * rho_tau)
    tau, _ = kendalltau(u1, u2)
    rho_gaussian = np.sin(np.pi / 2 * tau)
    llf_gaussian = gaussian_copula_loglik(rho_gaussian, u1, u2)

    results['Gaussian'] = {
        'params': {'rho': rho_gaussian},
        'LLF': llf_gaussian,
        'AIC': -2 * llf_gaussian + 2 * 1,  # 1 parameter
        'BIC': -2 * llf_gaussian + 1 * np.log(n),
        'nparams': 1
    }

    # ---- 2. Student-t copula (Equation 9, estimated by MLE) ----
    # Optimize over (rho, d) jointly
    def neg_t_copula_ll(params):
        return -student_t_copula_loglik(params, u1, u2)

    # Initial values: use Gaussian rho and d=5
    best_ll = -np.inf
    best_params_t = None
    for rho_init in [rho_gaussian, 0.3, 0.5]:
        for d_init in [2.5, 5, 10, 20]:
            try:
                res = minimize(neg_t_copula_ll, [rho_init, d_init],
                               method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
                if -res.fun > best_ll and res.x[1] > 2 and abs(res.x[0]) < 1:
                    best_ll = -res.fun
                    best_params_t = res.x
            except:
                pass

    if best_params_t is not None:
        rho_t, d_t = best_params_t
        llf_t = best_ll
    else:
        rho_t, d_t, llf_t = rho_gaussian, 10.0, -1e10

    results['Student-t'] = {
        'params': {'rho': rho_t, 'd': d_t},
        'LLF': llf_t,
        'AIC': -2 * llf_t + 2 * 2,  # 2 parameters
        'BIC': -2 * llf_t + 2 * np.log(n),
        'nparams': 2
    }

    # ---- 3. Clayton copula (Equation 10, estimated by MLE) ----
    def neg_clayton_ll(omega):
        return -clayton_copula_loglik(omega, u1, u2)

    best_ll_c = -np.inf
    best_omega_c = None
    for omega_init in [0.1, 0.5, 1.0, 2.0]:
        try:
            res = minimize_scalar(neg_clayton_ll, bounds=(-1 + 1e-6, 50), method='bounded')
            if -res.fun > best_ll_c:
                best_ll_c = -res.fun
                best_omega_c = res.x
        except:
            pass

    results['Clayton'] = {
        'params': {'omega': best_omega_c if best_omega_c else 0.1},
        'LLF': best_ll_c if best_ll_c > -1e9 else 0,
        'AIC': -2 * best_ll_c + 2 * 1,
        'BIC': -2 * best_ll_c + 1 * np.log(n),
        'nparams': 1
    }

    # ---- 4. Rotated-Clayton copula (Equation 11, estimated by MLE) ----
    def neg_rot_clayton_ll(omega):
        return -rotated_clayton_copula_loglik(omega, u1, u2)

    best_ll_rc = -np.inf
    best_omega_rc = None
    for omega_init in [0.1, 0.5, 1.0, 2.0]:
        try:
            res = minimize_scalar(neg_rot_clayton_ll, bounds=(-1 + 1e-6, 50), method='bounded')
            if -res.fun > best_ll_rc:
                best_ll_rc = -res.fun
                best_omega_rc = res.x
        except:
            pass

    results['Rotated-Clayton'] = {
        'params': {'omega': best_omega_rc if best_omega_rc else 0.1},
        'LLF': best_ll_rc if best_ll_rc > -1e9 else 0,
        'AIC': -2 * best_ll_rc + 2 * 1,
        'BIC': -2 * best_ll_rc + 1 * np.log(n),
        'nparams': 1
    }

    # ---- 5. Plackett copula (Equation 12, estimated by MLE) ----
    def neg_plackett_ll(eta):
        return -plackett_copula_loglik(eta, u1, u2)

    res = minimize_scalar(neg_plackett_ll, bounds=(1e-6, 100), method='bounded')
    eta_p = res.x
    llf_p = -res.fun

    results['Plackett'] = {
        'params': {'eta': eta_p},
        'LLF': llf_p,
        'AIC': -2 * llf_p + 2 * 1,
        'BIC': -2 * llf_p + 1 * np.log(n),
        'nparams': 1
    }

    # ---- 6. Frank copula (Equation 13, estimated by MLE) ----
    def neg_frank_ll(lam):
        return -frank_copula_loglik(lam, u1, u2)

    best_ll_f = -np.inf
    best_lam_f = None
    # Frank allows both positive and negative lambda
    for lam_init in [0.5, 1.0, 2.0, 5.0, -1.0]:
        try:
            res = minimize(neg_frank_ll, [lam_init], method='Nelder-Mead',
                           options={'maxiter': 5000})
            if -res.fun > best_ll_f:
                best_ll_f = -res.fun
                best_lam_f = res.x[0]
        except:
            pass

    results['Frank'] = {
        'params': {'lambda': best_lam_f if best_lam_f else 1.0},
        'LLF': best_ll_f if best_ll_f > -1e9 else 0,
        'AIC': -2 * best_ll_f + 2 * 1,
        'BIC': -2 * best_ll_f + 1 * np.log(n),
        'nparams': 1
    }

    # ---- 7. Gumbel copula (Equation 14, estimated by MLE) ----
    def neg_gumbel_ll(delta):
        return -gumbel_copula_loglik(delta, u1, u2)

    res = minimize_scalar(neg_gumbel_ll, bounds=(1.001, 50), method='bounded')
    delta_g = res.x
    llf_g = -res.fun

    results['Gumbel'] = {
        'params': {'delta': delta_g},
        'LLF': llf_g,
        'AIC': -2 * llf_g + 2 * 1,
        'BIC': -2 * llf_g + 1 * np.log(n),
        'nparams': 1
    }

    # ---- 8. Rotated-Gumbel copula (Equation 15, estimated by MLE) ----
    def neg_rot_gumbel_ll(delta):
        return -rotated_gumbel_copula_loglik(delta, u1, u2)

    res = minimize_scalar(neg_rot_gumbel_ll, bounds=(1.001, 50), method='bounded')
    delta_rg = res.x
    llf_rg = -res.fun

    results['Rotated-Gumbel'] = {
        'params': {'delta': delta_rg},
        'LLF': llf_rg,
        'AIC': -2 * llf_rg + 2 * 1,
        'BIC': -2 * llf_rg + 1 * np.log(n),
        'nparams': 1
    }

    return results


def estimate_table4(twiex_models, nikkei_models):
    """
    Estimate Table 4: copula parameters for all 8 copulas across 4 marginal specs.

    The IFM method (Section 3.3, Equations 19-21):
    Stage 1: Estimate marginal parameters (already done in Tables 2-3)
    Stage 2: Transform residuals to uniform via PIT, then estimate copula

    For each of the 4 marginal specifications (GARCH-n, GARCH-t, GJR-n, GJR-t):
    - Extract standardized residuals from the marginal model
    - Apply PIT using the appropriate distribution (normal or t_d)
    - Estimate all 8 copulas on the resulting uniform pairs
    """
    marginal_specs = {
        'GARCH-n': ('normal', None),
        'GARCH-t': ('t', 'use_model_d'),
        'GJR-n': ('normal', None),
        'GJR-t': ('t', 'use_model_d'),
    }

    all_results = {}

    for marg_name, (dist, _) in marginal_specs.items():
        print(f"\n  --- Copula estimation with {marg_name} marginals ---")

        tw_model = twiex_models[marg_name]
        nq_model = nikkei_models[marg_name]

        # Get degrees of freedom for Student-t models
        tw_d = tw_model.get('d', None)
        nq_d = nq_model.get('d', None)

        # Apply PIT to get uniform residuals
        u1 = probability_integral_transform(tw_model['std_resid'], dist, tw_d)
        u2 = probability_integral_transform(nq_model['std_resid'], dist, nq_d)

        # Estimate all 8 copulas
        copula_results = estimate_copulas(u1, u2)
        all_results[marg_name] = copula_results

        # Print brief summary
        for cop_name, cop_res in copula_results.items():
            param_str = ', '.join(f"{k}={v:.4f}" for k, v in cop_res['params'].items())
            print(f"    {cop_name:<20} {param_str:<30} LLF={cop_res['LLF']:.4f}  "
                  f"AIC={cop_res['AIC']:.4f}")

    return all_results


def print_table4(all_results):
    print("\n" + "=" * 100)
    print("TABLE 4")
    print("Parameter estimates for families of copula and model selection statistic.")
    print("=" * 100)

    print(f"\n  {'Copula':<20} {'Parameter':<12} {'GARCH-n':>12} {'GARCH-t':>12} "
          f"{'GJR-n':>12} {'GJR-t':>12}")
    print("  " + "-" * 90)

    copula_param_names = {
        'Gaussian': [('rho', 'ρ')],
        'Student-t': [('rho', 'ρ'), ('d', 'd')],
        'Clayton': [('omega', 'ω')],
        'Rotated-Clayton': [('omega', 'ω')],
        'Plackett': [('eta', 'η')],
        'Frank': [('lambda', 'λ')],
        'Gumbel': [('delta', 'δ')],
        'Rotated-Gumbel': [('delta', 'δ')]
    }

    marginals = ['GARCH-n', 'GARCH-t', 'GJR-n', 'GJR-t']

    for cop_name in ['Gaussian', 'Student-t', 'Clayton', 'Rotated-Clayton',
                     'Plackett', 'Frank', 'Gumbel', 'Rotated-Gumbel']:
        # Print parameter(s)
        for param_key, param_label in copula_param_names[cop_name]:
            vals = []
            for m in marginals:
                v = all_results[m][cop_name]['params'].get(param_key, None)
                vals.append(f"{v:.3f}" if v is not None else "")

            label = cop_name if param_key == copula_param_names[cop_name][0][0] else ""
            print(f"  {label:<20} {param_label:<12} {vals[0]:>12} {vals[1]:>12} "
                  f"{vals[2]:>12} {vals[3]:>12}")

        # Print LLF, AIC, BIC
        for stat in ['LLF', 'AIC', 'BIC']:
            vals = [all_results[m][cop_name][stat] for m in marginals]
            print(f"  {'':<20} {stat:<12} {vals[0]:>12.4f} {vals[1]:>12.4f} "
                  f"{vals[2]:>12.4f} {vals[3]:>12.4f}")

        print()

def plot_figure1(cac40_returns, nikkei_returns, save_path="figure1.png"):
    cac40_abs = np.abs(cac40_returns)
    nikkei_abs = np.abs(nikkei_returns)

    dates = cac40_returns.index

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(dates, cac40_returns.values, color='blue', linewidth=0.5)
    ax.set_title('Return-cac40', fontsize=12, fontweight='bold')
    ax.set_ylabel('Returns (%)', fontsize=10)
    ax.set_ylim(-15, 15)
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.grid(False)

    ax = axes[0, 1]
    ax.plot(dates, nikkei_returns.values, color='blue', linewidth=0.5)
    ax.set_title('Return-nikkei', fontsize=12, fontweight='bold')
    ax.set_ylabel('Returns (%)', fontsize=10)
    ax.set_ylim(-15, 15)
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.grid(False)

    ax = axes[1, 0]
    ax.plot(dates, cac40_abs.values, color='red', linewidth=0.5)
    ax.set_title('Absolute Returns -cac40', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Returns (%)', fontsize=10)
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.grid(False)

    ax = axes[1, 1]
    ax.plot(dates, nikkei_abs.values, color='red', linewidth=0.5)
    ax.set_title('Absolute Returns -nikkei', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Returns (%)', fontsize=10)
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.grid(False)

    import datetime
    tick_dates = [
        datetime.datetime(2000, 7, 1),
        datetime.datetime(2002, 10, 1),
        datetime.datetime(2005, 1, 1),
        datetime.datetime(2007, 5, 1),
    ]
    tick_labels = ['July 00', 'Oct 02', 'Jan 05', 'May 07']

    for ax in axes.flat:
        ax.set_xticks(tick_dates)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlim(dates[0], dates[-1])

    fig.text(0.5, 0.01,
             'Fig. 1.  Daily returns and absolute returns of TWIEX and nikkei.',
             ha='center', fontsize=11, fontstyle='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Figure 1 saved to: {save_path}")


# FIGURE 2 / SECTION 4.4: VaR Estimation by Conditional Copula-GARCH
# Section 3.4, Equations 23-27: VaR estimation
# Section 4.4: Rolling window out-of-sample VaR backtesting
#
# Key methodology:
# - Sample-in: first 1000 observations for parameter estimation
# - Sample-out: remaining 639 observations for VaR testing
# - Rolling window: at each step, use 1000 most recent observations
# - Re-estimate all parameters at each step
# - Compute one-day-ahead VaR via Monte Carlo simulation from the
#   copula-GARCH model
#
# For Figure 2: Student-t copula with GARCH-n marginal model
# Portfolio: equal weights w = 1/2 (Equation 26)
# Confidence levels: alpha = 0.05 and alpha = 0.01


def estimate_rolling_var_t_copula_garch_n(cac40_returns, nikkei_returns,
                                          window=1000, n_sim=10000,
                                          alpha_levels=[0.05, 0.01]):
    """
    Estimate rolling VaR using Student-t copula with GARCH-n marginals.

    This implements the exact procedure from Sections 3.4 and 4.4:

    Section 4.4: "This paper initially uses the sample-in data, which contains
    1000 return observations, to estimate VaR_{1001} at a time t = 1001, and at
    each new observation we re-estimate VaR, because of the conditional level and
    the VaR estimation formula. It means that we estimate VaR_{1002} by using
    observations t = 2 to t = 1001 and estimate VaR_{1003} by using observations
    t = 3 to t = 1002 until the sample-out observations we have updated are used up."

    VaR computation (Equations 25-27):
    - Portfolio return: X_{p,t} = (1/2)*X_{1,t} + (1/2)*X_{2,t}  (Eq. 24, w=1/2)
    - P(X_{p,t} <= VaR_t | Omega_{t-1}) = alpha  (Eq. 26)
    - Using the copula-GARCH joint distribution (Eq. 27)

    We use Monte Carlo simulation to evaluate the double integral in Eq. 27:
    1. Estimate GARCH-n parameters for each asset on the rolling window
    2. Forecast one-step-ahead conditional variance: sigma^2_{t+1|t}
    3. Estimate Student-t copula on PIT-transformed residuals
    4. Simulate from the Student-t copula
    5. Transform simulated uniforms back to returns using conditional distributions
    6. Compute portfolio returns and extract the alpha-quantile as VaR
    """
    np.random.seed(42)

    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window

    print(f"    Total observations: {T}")
    print(f"    Rolling window: {window}")
    print(f"    Out-of-sample: {n_out}")

    var_estimates = {alpha: np.full(n_out, np.nan) for alpha in alpha_levels}
    portfolio_returns = np.full(n_out, np.nan)

    for i in range(n_out):
        # Rolling window: observations [i, i+window)
        tw_window = cac40[i:i + window]
        nq_window = nikkei[i:i + window]

        # The actual out-of-sample return at time t = i + window
        # Portfolio return (Eq. 24): X_{p,t} = (1/2)*X_{1,t} + (1/2)*X_{2,t}
        portfolio_returns[i] = 0.5 * cac40[i + window] + 0.5 * nikkei[i + window]

        try:
            # ---- Step 1: Estimate GARCH-n for each asset ----
            # Section 2.1, Equation (1): GARCH(1,1) with normal innovations
            am_tw = arch_model(tw_window, mean='Constant', vol='GARCH',
                               p=1, q=1, dist='normal')
            res_tw = am_tw.fit(disp='off', show_warning=False)

            am_nq = arch_model(nq_window, mean='Constant', vol='GARCH',
                               p=1, q=1, dist='normal')
            res_nq = am_nq.fit(disp='off', show_warning=False)

            # ---- Step 2: One-step-ahead variance forecast ----
            # sigma^2_{t+1|t} = alpha_0 + alpha_1 * a^2_t + beta * sigma^2_t
            # From the GARCH model, the forecast uses the last observation
            forecast_tw = res_tw.forecast(horizon=1)
            forecast_nq = res_nq.forecast(horizon=1)

            sigma_tw = np.sqrt(forecast_tw.variance.values[-1, 0])
            sigma_nq = np.sqrt(forecast_nq.variance.values[-1, 0])
            mu_tw = res_tw.params['mu']
            mu_nq = res_nq.params['mu']

            # ---- Step 3: Estimate Student-t copula ----
            # Transform standardized residuals to uniform via PIT (normal CDF)
            std_resid_tw = res_tw.std_resid
            std_resid_nq = res_nq.std_resid

            u1 = norm.cdf(std_resid_tw)
            u2 = norm.cdf(std_resid_nq)
            u1 = np.clip(u1, 1e-10, 1 - 1e-10)
            u2 = np.clip(u2, 1e-10, 1 - 1e-10)

            # Estimate Student-t copula parameters (rho, d) by MLE
            def neg_ll(params):
                return -student_t_copula_loglik(params, u1, u2)

            # Use Kendall's tau for initial rho
            tau, _ = kendalltau(u1, u2)
            rho_init = np.sin(np.pi / 2 * tau)

            best_ll = -np.inf
            best_params = [rho_init, 5.0]
            for d_init in [2.5, 5, 10]:
                try:
                    res_cop = minimize(neg_ll, [rho_init, d_init],
                                       method='Nelder-Mead',
                                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
                    if -res_cop.fun > best_ll and res_cop.x[1] > 2 and abs(res_cop.x[0]) < 1:
                        best_ll = -res_cop.fun
                        best_params = res_cop.x
                except:
                    pass

            rho_cop, d_cop = best_params

            # ---- Step 4: Simulate from Student-t copula ----
            # To simulate from bivariate Student-t copula with parameters (rho, d):
            # 1. Generate bivariate t with correlation rho and d degrees of freedom
            # 2. Apply t_d CDF to get uniform samples
            # 3. These uniforms have the Student-t copula dependence structure

            # Generate from bivariate t distribution
            # Method: X = Z / sqrt(W/d) where Z ~ N(0, Sigma) and W ~ chi2(d)
            mean_vec = np.array([0, 0])
            cov_mat = np.array([[1, rho_cop], [rho_cop, 1]])

            z = np.random.multivariate_normal(mean_vec, cov_mat, size=n_sim)
            w = np.random.chisquare(d_cop, size=n_sim)
            t_samples = z / np.sqrt(w[:, np.newaxis] / d_cop)

            # Transform to uniform via t_d CDF
            u_sim1 = student_t.cdf(t_samples[:, 0], d_cop)
            u_sim2 = student_t.cdf(t_samples[:, 1], d_cop)

            # ---- Step 5: Transform back to returns ----
            # The conditional distribution of X_{i,t+1} given Omega_t is:
            # X_{i,t+1} = mu_i + sigma_{i,t+1|t} * epsilon_{i,t+1}
            # where epsilon ~ N(0,1) for GARCH-n
            #
            # From the copula simulation, u_sim are uniform variables.
            # Transform to standard normal quantiles (since GARCH-n uses normal):
            # epsilon = Phi^{-1}(u)
            # Then: X = mu + sigma * epsilon

            eps_tw = norm.ppf(u_sim1)
            eps_nq = norm.ppf(u_sim2)

            x_tw_sim = mu_tw + sigma_tw * eps_tw
            x_nq_sim = mu_nq + sigma_nq * eps_nq

            # ---- Step 6: Portfolio return and VaR ----
            # X_{p,t} = (1/2)*X_{1,t} + (1/2)*X_{2,t}  (Eq. 24 with w=1/2)
            portfolio_sim = 0.5 * x_tw_sim + 0.5 * x_nq_sim

            # VaR_t(alpha) = inf{s : F_t(s) >= alpha}  (Eq. 23)
            # This is the alpha-quantile of the simulated portfolio distribution
            for alpha in alpha_levels:
                var_estimates[alpha][i] = np.percentile(portfolio_sim, alpha * 100)

        except Exception as e:
            # If estimation fails, use previous VaR or a conservative estimate
            if i > 0:
                for alpha in alpha_levels:
                    var_estimates[alpha][i] = var_estimates[alpha][i - 1]

        # Progress reporting
        if (i + 1) % 100 == 0 or i == 0:
            print(f"    Completed {i + 1}/{n_out} out-of-sample VaR estimates...")

    # Count violations (Section 4.4)
    violations = {}
    for alpha in alpha_levels:
        violations[alpha] = np.sum(portfolio_returns < var_estimates[alpha])

    return {
        'var_estimates': var_estimates,
        'portfolio_returns': portfolio_returns,
        'violations': violations,
        'n_out': n_out,
        'expected_violations': {alpha: int(np.round(n_out * alpha)) for alpha in alpha_levels}
    }


def plot_figure2(var_result, save_path="figure2.png"):
    """
    Replicate Figure 2 from the paper:
    "Estimated VaR using Student-t copula with GARCH-n model."

    The figure shows:
    - Blue dots: portfolio returns (X_{p,t} = 0.5*X_{1,t} + 0.5*X_{2,t})
    - Magenta/pink line: VaR at alpha = 5%
    - Brown/dark orange line: VaR at alpha = 1%

    Axis specifications (matched to paper's Figure 2):
    - Title: "Estimated VaR with T Copula GARCHN method"
    - Y-axis: "Portfolio return / VaR (%)", range approximately [-4, 3]
    - X-axis: "Observation", range [0, ~639]
    - Legend: "Portfolio Return", "VaR at a=5%", "VaR ar a=1%"
    - Text annotations: "VaR at α=1%" and "VaR at α=5%" on the plot

    Section 4.4: "Fig. 2 shows the VaR plot we estimate using the Student-t
    copula with a marginal distribution, the GARCH-n model at α = 0.05 and
    α = 0.01. VaR is an estimate of investment loss in the worst case scenario
    with a relatively high level of confidence. In this Figure the VaR of
    portfolio is located almost below the portfolio returns, and describes
    the expectation of investment loss well. The portfolio return of VaR with
    a 99% confidence is surely lower than that with a 95% confidence."
    """
    portfolio_returns = var_result['portfolio_returns']
    var_05 = var_result['var_estimates'][0.05]
    var_01 = var_result['var_estimates'][0.01]
    n_out = var_result['n_out']

    observations = np.arange(n_out)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Portfolio returns as blue dots (scatter plot, matching paper)
    ax.scatter(observations, portfolio_returns, color='blue', s=8,
               label='Portfolio Return', zorder=3)

    # VaR at alpha = 5% (magenta/pink line, matching paper)
    ax.plot(observations, var_05, color='magenta', linewidth=1.2,
            label='VaR at a=5%', zorder=2)

    # VaR at alpha = 1% (brown/dark orange line, matching paper)
    ax.plot(observations, var_01, color='saddlebrown', linewidth=1.2,
            label='VaR ar a=1%', zorder=2)

    # Title matching paper exactly
    ax.set_title('Estimated VaR with T Copula GARCHN method', fontsize=13, fontweight='bold')
    ax.set_xlabel('Observation', fontsize=11)
    ax.set_ylabel('Portfolio return / VaR (%)', fontsize=11)

    # Y-axis range matching paper
    ax.set_ylim(-4, 3)
    ax.set_xlim(0, n_out)

    # Legend matching paper position (upper right)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Text annotations matching paper's Figure 2
    # "VaR at α=1%" annotation near observation ~150, below the brown line
    ax.annotate('VaR at  α=1%', xy=(150, -2.8), fontsize=10)
    # "VaR at α=5%" annotation near observation ~300
    ax.annotate('VaR at  α =5%', xy=(300, -1.7), fontsize=10)

    ax.grid(False)

    # Caption
    fig.text(0.5, -0.02,
             'Fig. 2.  Estimated VaR using Student-t copula with GARCH-n model.',
             ha='center', fontsize=11, fontstyle='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Figure 2 saved to: {save_path}")


# FIGURE 3 / TABLE 6: Comparison of VaR using various methods
# Section 3.4 and Section 4.4
#
# Traditional VaR methods to compare against copula-GARCH:
# 1. Historical Simulation (HS)
# 2. Variance-Covariance (VC) - Equation 28
# 3. Exponential Weighted Moving Average (EWMA) - Equation 29
# 4. Univariate GARCH-VaR (GARCH-n) - Equation 30
# 5. T-copula-GARCH-n (already computed in Figure 2)


def estimate_var_historical_simulation(portfolio_returns_full, window=1000, alpha=0.01):
    T = len(portfolio_returns_full)
    n_out = T - window
    var_hs = np.full(n_out, np.nan)

    for i in range(n_out):
        # Use the past 'window' portfolio returns
        past_returns = portfolio_returns_full[i:i + window]
        # VaR = alpha-quantile of the empirical distribution
        var_hs[i] = np.percentile(past_returns, alpha * 100)

    return var_hs


def estimate_var_variance_covariance(cac40_returns, nikkei_returns, window=1000, alpha=0.01):
    """
    Variance-Covariance VaR (Equation 28).

    Section 3.4, Equation 28:
        sigma^2_{p,t} = [w1, w2] * [[sigma^2_{1,t}, sigma_{12,t}],
                                     [sigma_{21,t}, sigma^2_{2,t}]] * [w1, w2]'
                       = w * Sigma_t * w'

        VaR_{p,t}(alpha) = sigma_{p,t} * Z_alpha + mu_{p,t}

    where mu_{p,t} and sigma^2_{p,t} are the return and variance of the portfolio
    in time t, w_i is the portfolio weight, and Z_alpha is the standardized
    normal inverse with alpha probability.
    """
    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window
    var_vc = np.full(n_out, np.nan)
    w = np.array([0.5, 0.5])  # Equal weights (Section 3.4)
    z_alpha = norm.ppf(alpha)  # Z_alpha: standardized normal inverse

    for i in range(n_out):
        tw_win = cac40[i:i + window]
        nq_win = nikkei[i:i + window]

        # Sample covariance matrix Sigma_t
        cov_matrix = np.cov(tw_win, nq_win)  # 2x2 covariance matrix

        # Portfolio variance: sigma^2_p = w' * Sigma * w
        sigma2_p = w @ cov_matrix @ w
        sigma_p = np.sqrt(sigma2_p)

        # Portfolio mean return
        mu_p = 0.5 * np.mean(tw_win) + 0.5 * np.mean(nq_win)

        # VaR = sigma_p * Z_alpha + mu_p (Equation 28)
        var_vc[i] = sigma_p * z_alpha + mu_p

    return var_vc


def estimate_var_ewma(cac40_returns, nikkei_returns, window=1000, alpha=0.01):
    """
    EWMA (Exponential Weighted Moving Average) VaR (Equation 29).

    Section 3.4: "The EWMA method is generally used in Riskmetrics methodology.
    Assuming a normality of portfolio return distribution."

    Equation 29:
        sigma^2_{p,t|t-1} = (1-lambda) * x^2_{p,t-1} + lambda * sigma^2_{p,t-1|t-2}

    """
    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window
    var_ewma = np.full(n_out, np.nan)
    z_alpha = norm.ppf(alpha)

    for i in range(n_out):
        tw_win = cac40[i:i + window]
        nq_win = nikkei[i:i + window]
        # Portfolio returns for the window
        xp = 0.5 * tw_win + 0.5 * nq_win

        # Optimize lambda by minimizing sum(sigma^2_{(i+1)|i} - x^2_{i+1})^2
        # as described in the paper (citing Palaro and Hotta, 2006)
        def ewma_loss(lam):
            if lam <= 0 or lam >= 1:
                return 1e10
            n = len(xp)
            # Initialize variance with sample variance
            sigma2 = np.var(xp)
            total_loss = 0.0
            for j in range(1, n):
                sigma2_forecast = (1 - lam) * xp[j - 1] ** 2 + lam * sigma2
                total_loss += (sigma2_forecast - xp[j] ** 2) ** 2
                sigma2 = sigma2_forecast
            return total_loss

        # Optimize lambda
        res = minimize_scalar(ewma_loss, bounds=(0.8, 0.9999), method='bounded')
        lam_opt = res.x

        # Compute the final one-step-ahead variance forecast
        sigma2 = np.var(xp)
        for j in range(1, len(xp)):
            sigma2 = (1 - lam_opt) * xp[j - 1] ** 2 + lam_opt * sigma2
        # sigma2 now holds sigma^2_{t|t-1} using the last observation in window
        sigma2_forecast = (1 - lam_opt) * xp[-1] ** 2 + lam_opt * sigma2

        sigma_p = np.sqrt(sigma2_forecast)
        mu_p = np.mean(xp)

        var_ewma[i] = sigma_p * z_alpha + mu_p

    return var_ewma


def estimate_var_univariate_garch(cac40_returns, nikkei_returns, window=1000, alpha=0.01):
    """
    Univariate GARCH-VaR (Equation 30).

    Section 3.4: "For the univariate GARCH-VaR method, we fit the portfolio
    return series X_{p,t} directly. The model is the same as what we introduced
    in Section 2, including GARCH and GJR with normal and Student-t innovation."

    Equation 30:
        VaR_{p,t}(alpha)|Omega_{t-1} = (sigma_{p,t} * Z_alpha + mu_t)|Omega_{t-1}

    "It is not difficult to estimate sigma^2_t|Omega_{t-1}, the conditional
    variance of the portfolio return, by the GARCH or GJR model."

    We use GARCH-n (GARCH with normal innovation) on the portfolio return,
    matching the comparison in Table 6 ("VaR GARCHN" in Figure 3).
    """
    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window
    var_garch = np.full(n_out, np.nan)
    z_alpha = norm.ppf(alpha)

    for i in range(n_out):
        tw_win = cac40[i:i + window]
        nq_win = nikkei[i:i + window]
        # Portfolio returns for the window
        xp = 0.5 * tw_win + 0.5 * nq_win

        try:
            # Fit GARCH(1,1) with normal innovations to portfolio returns
            am = arch_model(xp, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
            res = am.fit(disp='off', show_warning=False)

            # One-step-ahead forecast
            forecast = res.forecast(horizon=1)
            sigma_p = np.sqrt(forecast.variance.values[-1, 0])
            mu_p = res.params['mu']

            # VaR = sigma_p * Z_alpha + mu (Equation 30)
            var_garch[i] = sigma_p * z_alpha + mu_p

        except:
            if i > 0:
                var_garch[i] = var_garch[i - 1]

    return var_garch


def estimate_all_traditional_var(cac40_returns, nikkei_returns, window=1000, alpha=0.01):
    """
    Estimate VaR by all traditional methods for comparison with copula-GARCH.

    Section 4.4 and Table 6: Compare t-copula-GARCH-n against:
    - Historical Simulation (HS)
    - Variance-Covariance (VC)
    - EWMA
    - Univariate GARCH-n (VaR GARCHN)
    """
    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()

    # Portfolio returns for the full series
    portfolio_full = 0.5 * cac40 + 0.5 * nikkei

    print(f"    Estimating Historical Simulation VaR...")
    var_hs = estimate_var_historical_simulation(portfolio_full, window, alpha)

    print(f"    Estimating Variance-Covariance VaR...")
    var_vc = estimate_var_variance_covariance(cac40, nikkei, window, alpha)

    print(f"    Estimating EWMA VaR...")
    var_ewma = estimate_var_ewma(cac40, nikkei, window, alpha)

    print(f"    Estimating Univariate GARCH-n VaR...")
    var_garch = estimate_var_univariate_garch(cac40, nikkei, window, alpha)

    return {
        'HS': var_hs,
        'VC': var_vc,
        'EWMA': var_ewma,
        'VaR GARCHN': var_garch
    }


def plot_figure3(var_result_copula, traditional_vars, save_path="figure3.png"):
    portfolio_returns = var_result_copula['portfolio_returns']
    var_copula_01 = var_result_copula['var_estimates'][0.01]
    n_out = var_result_copula['n_out']

    observations = np.arange(n_out)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Portfolio returns as blue dots
    ax.scatter(observations, portfolio_returns, color='blue', s=8,
               label='Portfolio Return', zorder=5)

    # T Copula GARCHN - thick black solid line (matching paper)
    ax.plot(observations, var_copula_01, color='black', linewidth=2.0,
            linestyle='-', label='T Copula GARCHN', zorder=4)

    # HS - gray dotted line (matching paper)
    ax.plot(observations, traditional_vars['HS'], color='gray', linewidth=1.2,
            linestyle=':', label='HS', zorder=3)

    # VC - black dashed line (matching paper)
    ax.plot(observations, traditional_vars['VC'], color='black', linewidth=1.0,
            linestyle='--', label='VC', zorder=3)

    # EWMA - light gray dash-dot line (matching paper)
    ax.plot(observations, traditional_vars['EWMA'], color='silver', linewidth=1.5,
            linestyle='-.', label='EWMA', zorder=3)

    # VaR GARCHN - dark gray thick dotted line (matching paper)
    ax.plot(observations, traditional_vars['VaR GARCHN'], color='dimgray', linewidth=1.5,
            linestyle=(0, (2, 1)), label='VaR GARCHN', zorder=3)

    # Title matching paper exactly
    ax.set_title('Compare VaR using various method', fontsize=13, fontweight='bold')
    ax.set_xlabel('Observation', fontsize=11)
    ax.set_ylabel('Portfolio return  VaR (%)', fontsize=11)

    # Y-axis range matching paper
    ax.set_ylim(-7, 3)
    ax.set_xlim(0, n_out)

    # X-axis ticks matching paper
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])

    # Legend matching paper position (upper right)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    ax.grid(False)

    # Caption
    fig.text(0.5, -0.02,
             r'Fig. 3.  Comparison of VaR using various methods at $\alpha = 0.01$.',
             ha='center', fontsize=11, fontstyle='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Figure 3 saved to: {save_path}")


# TABLE 5: Number of violations of the VaR estimation
# Section 4.4: Rolling VaR for all 8 copulas × 4 marginal specifications
# at alpha = 0.05 and alpha = 0.01
#
# "The number of violations of the VaR estimation are calculated using
# various copula functions are presented in Table 5."
#
# "The mean error shows for each copula function, the average absolute
# discrepancy per marginal model between the observed and expected number
# of violations."


def simulate_from_copula(copula_name, copula_params, n_sim):
    """
    Simulate n_sim pairs (u1, u2) from the specified copula.

    Uses the copula parameters estimated on the rolling window.
    Returns uniform [0,1] pairs with the copula dependence structure.
    """
    if copula_name == 'Gaussian':
        rho = copula_params['rho']
        z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_sim)
        u1 = norm.cdf(z[:, 0])
        u2 = norm.cdf(z[:, 1])

    elif copula_name == 'Student-t':
        rho = copula_params['rho']
        d = copula_params['d']
        cov_mat = np.array([[1, rho], [rho, 1]])
        z = np.random.multivariate_normal([0, 0], cov_mat, size=n_sim)
        w = np.random.chisquare(d, size=n_sim)
        t_samples = z / np.sqrt(w[:, np.newaxis] / d)
        u1 = student_t.cdf(t_samples[:, 0], d)
        u2 = student_t.cdf(t_samples[:, 1], d)

    elif copula_name == 'Clayton':
        omega = copula_params['omega']
        # Conditional sampling method for Clayton copula
        # u1 ~ Uniform(0,1)
        # u2 = C^{-1}_{2|1}(v | u1) where v ~ Uniform(0,1)
        u1 = np.random.uniform(0, 1, n_sim)
        v = np.random.uniform(0, 1, n_sim)
        # Conditional inverse: u2 = ((u1^{-omega})*(v^{-omega/(1+omega)} - 1) + 1)^{-1/omega}
        u2 = (u1 ** (-omega) * (v ** (-omega / (1 + omega)) - 1) + 1) ** (-1 / omega)

    elif copula_name == 'Rotated-Clayton':
        omega = copula_params['omega']
        u1_raw = np.random.uniform(0, 1, n_sim)
        v = np.random.uniform(0, 1, n_sim)
        u2_raw = (u1_raw ** (-omega) * (v ** (-omega / (1 + omega)) - 1) + 1) ** (-1 / omega)
        # 180-degree rotation
        u1 = 1 - u1_raw
        u2 = 1 - u2_raw

    elif copula_name == 'Frank':
        lam = copula_params['lambda']
        # Conditional sampling for Frank copula
        u1 = np.random.uniform(0, 1, n_sim)
        v = np.random.uniform(0, 1, n_sim)
        # u2 = -(1/lam)*log(1 - (1-e^{-lam}) / ((v^{-1} - 1)*e^{-lam*u1} + 1))
        el = np.exp(-lam)
        u2 = -(1 / lam) * np.log(1 - (1 - el) / ((1 / v - 1) * np.exp(-lam * u1) + 1))

    elif copula_name == 'Plackett':
        eta = copula_params['eta']
        # Conditional sampling for Plackett copula
        u1 = np.random.uniform(0, 1, n_sim)
        v = np.random.uniform(0, 1, n_sim)
        # Solve for u2 from conditional CDF C_{2|1}(u2|u1) = v
        # Using the closed-form inverse
        a = v * (1 - v)
        b = eta + a * (eta - 1) ** 2
        c = 2 * a * (u1 * eta ** 2 + 1 - u1) + eta * (1 - 2 * a)
        d_val = np.sqrt(eta * (eta + 4 * a * u1 * (1 - u1) * (1 - eta) ** 2))
        u2 = (c - (1 - 2 * v) * d_val) / (2 * b)
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)

    elif copula_name == 'Gumbel':
        delta = copula_params['delta']
        # Marshall-Olkin method for Gumbel copula
        # Generate stable random variable S with Laplace transform exp(-t^{1/delta})
        # Then u1 = exp(-(E1/S)^{1/delta}), u2 = exp(-(E2/S)^{1/delta})
        # where E1, E2 are independent Exp(1)
        # For the stable variable with alpha=1/delta, beta=1, use the Chambers-Mallows-Stuck method
        theta = 1.0 / delta
        # Generate stable(theta, 1, cos(pi*theta/2)^(1/theta), 0) using CMS method
        V = np.random.uniform(-np.pi / 2, np.pi / 2, n_sim)
        W = np.random.exponential(1, n_sim)
        if abs(theta - 1.0) < 1e-10:
            S = np.ones(n_sim)
        else:
            S = (np.sin(theta * (V + np.pi / 2)) / np.cos(V) ** (1 / theta)) * \
                (np.cos(V - theta * (V + np.pi / 2)) / W) ** ((1 - theta) / theta)

        E1 = np.random.exponential(1, n_sim)
        E2 = np.random.exponential(1, n_sim)
        u1 = np.exp(-(E1 / S) ** (1.0 / delta))
        u2 = np.exp(-(E2 / S) ** (1.0 / delta))

    elif copula_name == 'Rotated-Gumbel':
        delta = copula_params['delta']
        theta = 1.0 / delta
        V = np.random.uniform(-np.pi / 2, np.pi / 2, n_sim)
        W = np.random.exponential(1, n_sim)
        if abs(theta - 1.0) < 1e-10:
            S = np.ones(n_sim)
        else:
            S = (np.sin(theta * (V + np.pi / 2)) / np.cos(V) ** (1 / theta)) * \
                (np.cos(V - theta * (V + np.pi / 2)) / W) ** ((1 - theta) / theta)
        E1 = np.random.exponential(1, n_sim)
        E2 = np.random.exponential(1, n_sim)
        u1_raw = np.exp(-(E1 / S) ** (1.0 / delta))
        u2_raw = np.exp(-(E2 / S) ** (1.0 / delta))
        # 180-degree rotation
        u1 = 1 - u1_raw
        u2 = 1 - u2_raw
    else:
        raise ValueError(f"Unknown copula: {copula_name}")

    u1 = np.clip(u1, 1e-10, 1 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1 - 1e-10)
    return u1, u2


def estimate_rolling_var_copula(cac40_returns, nikkei_returns,
                                copula_name, marginal_name,
                                window=1000, n_sim=10000,
                                alpha_levels=[0.05, 0.01]):
    np.random.seed(42)

    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window

    # Parse marginal specification
    if marginal_name == 'GARCH-n':
        vol_type, vol_o, dist_type = 'GARCH', 0, 'normal'
    elif marginal_name == 'GARCH-t':
        vol_type, vol_o, dist_type = 'GARCH', 0, 'studentst'
    elif marginal_name == 'GJR-n':
        vol_type, vol_o, dist_type = 'GARCH', 1, 'normal'
    elif marginal_name == 'GJR-t':
        vol_type, vol_o, dist_type = 'GARCH', 1, 'studentst'
    else:
        raise ValueError(f"Unknown marginal: {marginal_name}")

    violations = {alpha: 0 for alpha in alpha_levels}

    for i in range(n_out):
        tw_window = cac40[i:i + window]
        nq_window = nikkei[i:i + window]
        actual_port_ret = 0.5 * cac40[i + window] + 0.5 * nikkei[i + window]

        try:
            # Step 1: Estimate marginal models
            if vol_o == 0:
                am_tw = arch_model(tw_window, mean='Constant', vol='GARCH',
                                   p=1, q=1, dist=dist_type)
                am_nq = arch_model(nq_window, mean='Constant', vol='GARCH',
                                   p=1, q=1, dist=dist_type)
            else:
                am_tw = arch_model(tw_window, mean='Constant', vol='GARCH',
                                   p=1, o=1, q=1, dist=dist_type)
                am_nq = arch_model(nq_window, mean='Constant', vol='GARCH',
                                   p=1, o=1, q=1, dist=dist_type)

            res_tw = am_tw.fit(disp='off', show_warning=False)
            res_nq = am_nq.fit(disp='off', show_warning=False)

            # Step 2: Forecast one-step-ahead variance
            forecast_tw = res_tw.forecast(horizon=1)
            forecast_nq = res_nq.forecast(horizon=1)
            sigma_tw = np.sqrt(forecast_tw.variance.values[-1, 0])
            sigma_nq = np.sqrt(forecast_nq.variance.values[-1, 0])
            mu_tw = res_tw.params['mu']
            mu_nq = res_nq.params['mu']

            # Step 3: PIT on standardized residuals
            std_resid_tw = res_tw.std_resid
            std_resid_nq = res_nq.std_resid

            if dist_type == 'normal':
                u1_data = norm.cdf(std_resid_tw)
                u2_data = norm.cdf(std_resid_nq)
            else:
                d_tw = res_tw.params['nu']
                d_nq = res_nq.params['nu']
                u1_data = student_t.cdf(std_resid_tw * np.sqrt(d_tw / (d_tw - 2)), d_tw)
                u2_data = student_t.cdf(std_resid_nq * np.sqrt(d_nq / (d_nq - 2)), d_nq)

            u1_data = np.clip(u1_data, 1e-10, 1 - 1e-10)
            u2_data = np.clip(u2_data, 1e-10, 1 - 1e-10)

            # Step 4: Estimate copula parameters
            if copula_name == 'Gaussian':
                tau, _ = kendalltau(u1_data, u2_data)
                cop_params = {'rho': np.sin(np.pi / 2 * tau)}
            elif copula_name == 'Student-t':
                tau, _ = kendalltau(u1_data, u2_data)
                rho_init = np.sin(np.pi / 2 * tau)
                best_ll, best_p = -np.inf, [rho_init, 5]
                for d_init in [2.5, 5, 10]:
                    try:
                        r = minimize(lambda p: -student_t_copula_loglik(p, u1_data, u2_data),
                                     [rho_init, d_init], method='Nelder-Mead',
                                     options={'maxiter': 3000})
                        if -r.fun > best_ll and r.x[1] > 2 and abs(r.x[0]) < 1:
                            best_ll, best_p = -r.fun, r.x
                    except:
                        pass
                cop_params = {'rho': best_p[0], 'd': best_p[1]}
            elif copula_name == 'Clayton':
                r = minimize_scalar(lambda w: -clayton_copula_loglik(w, u1_data, u2_data),
                                    bounds=(-1 + 1e-6, 50), method='bounded')
                cop_params = {'omega': r.x}
            elif copula_name == 'Rotated-Clayton':
                r = minimize_scalar(lambda w: -rotated_clayton_copula_loglik(w, u1_data, u2_data),
                                    bounds=(-1 + 1e-6, 50), method='bounded')
                cop_params = {'omega': r.x}
            elif copula_name == 'Plackett':
                r = minimize_scalar(lambda e: -plackett_copula_loglik(e, u1_data, u2_data),
                                    bounds=(1e-6, 100), method='bounded')
                cop_params = {'eta': r.x}
            elif copula_name == 'Frank':
                best_ll, best_l = -np.inf, 1.0
                for l_init in [0.5, 1, 2, 5]:
                    try:
                        r = minimize(lambda l: -frank_copula_loglik(l[0], u1_data, u2_data),
                                     [l_init], method='Nelder-Mead')
                        if -r.fun > best_ll:
                            best_ll, best_l = -r.fun, r.x[0]
                    except:
                        pass
                cop_params = {'lambda': best_l}
            elif copula_name == 'Gumbel':
                r = minimize_scalar(lambda d: -gumbel_copula_loglik(d, u1_data, u2_data),
                                    bounds=(1.001, 50), method='bounded')
                cop_params = {'delta': r.x}
            elif copula_name == 'Rotated-Gumbel':
                r = minimize_scalar(lambda d: -rotated_gumbel_copula_loglik(d, u1_data, u2_data),
                                    bounds=(1.001, 50), method='bounded')
                cop_params = {'delta': r.x}

            # Step 5: Simulate from copula
            u_sim1, u_sim2 = simulate_from_copula(copula_name, cop_params, n_sim)

            # Step 6: Transform back to returns
            if dist_type == 'normal':
                eps_tw = norm.ppf(u_sim1)
                eps_nq = norm.ppf(u_sim2)
            else:
                eps_tw = student_t.ppf(u_sim1, d_tw) * np.sqrt((d_tw - 2) / d_tw)
                eps_nq = student_t.ppf(u_sim2, d_nq) * np.sqrt((d_nq - 2) / d_nq)

            x_tw_sim = mu_tw + sigma_tw * eps_tw
            x_nq_sim = mu_nq + sigma_nq * eps_nq

            # Portfolio VaR
            portfolio_sim = 0.5 * x_tw_sim + 0.5 * x_nq_sim

            for alpha in alpha_levels:
                var_val = np.percentile(portfolio_sim, alpha * 100)
                if actual_port_ret < var_val:
                    violations[alpha] += 1

        except Exception as e:
            pass  # Failed estimation doesn't count as violation

        if (i + 1) % 200 == 0:
            print(f"      {copula_name}/{marginal_name}: {i + 1}/{n_out}...")

    return violations


def estimate_table5(cac40_returns, nikkei_returns, window=1000, n_sim=10000):
    copula_names = ['Gaussian', 'Student-t', 'Clayton', 'Rotated-Clayton',
                    'Plackett', 'Frank', 'Gumbel', 'Rotated-Gumbel']
    marginal_names = ['GARCH-n', 'GARCH-t', 'GJR-n', 'GJR-t']
    alpha_levels = [0.05, 0.01]

    results = {}

    for cop_name in copula_names:
        results[cop_name] = {}
        for marg_name in marginal_names:
            print(f"    Estimating: {cop_name} / {marg_name}...")
            violations = estimate_rolling_var_copula(
                cac40_returns, nikkei_returns,
                cop_name, marg_name,
                window=window, n_sim=n_sim,
                alpha_levels=alpha_levels
            )
            results[cop_name][marg_name] = violations

    return results


def print_table5(table5_results, n_out):
    copula_names = ['Gaussian', 'Student-t', 'Clayton', 'Rotated-Clayton',
                    'Plackett', 'Frank', 'Gumbel', 'Rotated-Gumbel']
    marginal_names = ['GARCH-n', 'GARCH-t', 'GJR-n', 'GJR-t']

    for alpha in [0.05, 0.01]:
        expected = int(np.round(n_out * alpha))

        print(f"\n  At α = {alpha}")
        print(f"  Trading days    {n_out}    Expected no. of violations    {expected}")
        print(f"  {'Copula':<20} {'GARCH-n':>10} {'GARCH-t':>10} {'GJR-n':>10} {'GJR-t':>10} {'Mean error':>12}")
        print(f"  {'-' * 75}")

        for cop_name in copula_names:
            vals = []
            for marg_name in marginal_names:
                v = table5_results[cop_name][marg_name][alpha]
                vals.append(v)

            # Mean error = (1/4) * sum|violations_i - expected|
            mean_error = np.mean([abs(v - expected) for v in vals])

            print(f"  {cop_name:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {mean_error:>12.2f}")

def estimate_var_univariate_general(cac40_returns, nikkei_returns,
                                    model_type='GARCH', dist='normal',
                                    window=1000, alpha_levels=[0.05, 0.01]):

    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window

    violations = {alpha: 0 for alpha in alpha_levels}

    dist_arch = 'normal' if dist == 'normal' else 'studentst'

    for i in range(n_out):
        xp = 0.5 * cac40[i:i + window] + 0.5 * nikkei[i:i + window]
        actual_port_ret = 0.5 * cac40[i + window] + 0.5 * nikkei[i + window]

        try:
            if model_type == 'GARCH':
                am = arch_model(xp, mean='Constant', vol='GARCH',
                                p=1, q=1, dist=dist_arch)
            else:  # GJR
                am = arch_model(xp, mean='Constant', vol='GARCH',
                                p=1, o=1, q=1, dist=dist_arch)

            res = am.fit(disp='off', show_warning=False)

            forecast = res.forecast(horizon=1)
            sigma_p = np.sqrt(forecast.variance.values[-1, 0])
            mu_p = res.params['mu']

            for alpha in alpha_levels:
                if dist == 'normal':
                    # Z_alpha from standard normal
                    q_alpha = norm.ppf(alpha)
                else:
                    # For Student-t: use the standardized t quantile
                    # The arch package's studentst is standardized to unit variance
                    # so the quantile needs scaling by sqrt((d-2)/d)
                    d_est = res.params['nu']
                    q_alpha = student_t.ppf(alpha, d_est) * np.sqrt((d_est - 2) / d_est)

                var_val = sigma_p * q_alpha + mu_p
                if actual_port_ret < var_val:
                    violations[alpha] += 1

        except:
            pass

        if (i + 1) % 200 == 0:
            print(f"      {model_type}-{dist}: {i + 1}/{n_out}...")

    return violations


def estimate_table6(cac40_returns, nikkei_returns,
                    table5_results=None,
                    var_result_copula_garch_n=None,
                    traditional_vars_01=None,
                    window=1000):

    cac40 = np.asarray(cac40_returns).flatten()
    nikkei = np.asarray(nikkei_returns).flatten()
    T = len(cac40)
    n_out = T - window
    portfolio_full = 0.5 * cac40 + 0.5 * nikkei
    alpha_levels = [0.05, 0.01]

    results = {}

    # ---- Copula methods (from Table 5 if available, otherwise re-estimate) ----
    if table5_results is not None:
        results['t-copula-GARCH-n'] = table5_results['Student-t']['GARCH-n']
        results['t-copula-GJR-n'] = table5_results['Student-t']['GJR-n']
    else:
        # Re-estimate the two copula-GARCH methods needed
        print("    Estimating t-copula-GARCH-n...")
        results['t-copula-GARCH-n'] = estimate_rolling_var_copula(
            cac40, nikkei, 'Student-t', 'GARCH-n', window=window, n_sim=10000,
            alpha_levels=alpha_levels)
        print("    Estimating t-copula-GJR-n...")
        results['t-copula-GJR-n'] = estimate_rolling_var_copula(
            cac40, nikkei, 'Student-t', 'GJR-n', window=window, n_sim=10000,
            alpha_levels=alpha_levels)

    # ---- Traditional methods: HS, VC, EWMA ----
    # Need violations at BOTH alpha = 0.05 and alpha = 0.01
    for method_name, est_func in [
        ('HS', lambda a: estimate_var_historical_simulation(portfolio_full, window, a)),
        ('VC', lambda a: estimate_var_variance_covariance(cac40, nikkei, window, a)),
        ('EWMA', lambda a: estimate_var_ewma(cac40, nikkei, window, a)),
    ]:
        print(f"    Estimating {method_name} at alpha=0.05 and 0.01...")
        violations = {}
        for alpha in alpha_levels:
            var_series = est_func(alpha)
            port_oos = np.array([0.5 * cac40[j + window] + 0.5 * nikkei[j + window]
                                 for j in range(n_out)])
            violations[alpha] = int(np.sum(port_oos < var_series))
        results[method_name] = violations

    # ---- Univariate GARCH/GJR methods ----
    for model_name, model_type, dist in [
        ('GARCH-n', 'GARCH', 'normal'),
        ('GARCH-t', 'GARCH', 't'),
        ('GJR-n', 'GJR', 'normal'),
        ('GJR-t', 'GJR', 't'),
    ]:
        print(f"    Estimating univariate {model_name}...")
        results[model_name] = estimate_var_univariate_general(
            cac40, nikkei, model_type, dist, window, alpha_levels)

    return results, n_out


def print_table6(table6_results, n_out):
    print("\n" + "=" * 75)
    print("TABLE 6")
    print("Number of violations of VaR estimation.")
    print("=" * 75)

    expected_05 = int(np.round(n_out * 0.05))
    expected_01 = int(np.round(n_out * 0.01))

    print(f"\n  Trading days            {n_out}")
    print(f"  α                       5%            1%")
    print(f"  Expected no. of violations  {expected_05}            {expected_01}          Mean error")
    print(f"  {'-' * 65}")

    method_order = ['t-copula-GARCH-n', 't-copula-GJR-n', 'HS', 'VC', 'EWMA',
                    'GARCH-n', 'GARCH-t', 'GJR-n', 'GJR-t']

    for method in method_order:
        v05 = table6_results[method][0.05]
        v01 = table6_results[method][0.01]
        # Mean error: average absolute deviation across both alpha levels
        mean_error = abs(v05 - expected_05) + abs(v01 - expected_01)
        print(f"  {method:<25} {v05:>6}        {v01:>6}        {mean_error:>8.0f}")


def main():
    print("=" * 75)
    print("REPLICATION OF TABLE 1 AND FIGURE 1")
    print("Huang, Lee, Liang, Lin (2009)")
    print("'Estimating value at risk of portfolio by conditional")
    print(" copula-GARCH method'")
    print("Insurance: Mathematics and Economics 45, 315-324")
    print("=" * 75)

    print("\n[Step 1] Downloading data from Yahoo Finance...")
    nikkei_close, cac40_close = download_data()

    print(f"  nikkei raw trading days: {len(nikkei_close)}")
    print(f"  cac40 raw trading days:  {len(cac40_close)}")

    print("\n[Step 2] Aligning dates and computing log returns (Eq. 31)...")
    cac40_returns, nikkei_returns = align_and_compute_returns(
        nikkei_close, cac40_close
    )

    print("\n[Step 3] Computing descriptive statistics...")
    cac40_stats = compute_descriptive_statistics(cac40_returns.values)
    nikkei_stats = compute_descriptive_statistics(nikkei_returns.values)

    print("\n[Step 4] Computing Engle's ARCH-LM tests...")
    lags = [4, 6, 8, 10]

    cac40_arch = []
    nikkei_arch = []
    for lag in lags:
        cac40_arch.append(engle_lm_test(cac40_returns.values, lag))
        nikkei_arch.append(engle_lm_test(nikkei_returns.values, lag))

    print_table1(cac40_stats, nikkei_stats, cac40_arch, nikkei_arch, lags)

    print("\n[Step 5] Generating Figure 1...")
    print("  Figure 1: Daily returns and absolute returns of TWIEX and nikkei")
    print("  Layout: 2x2 panel")
    print("    Top-left:     Return-cac40 (blue, daily log returns)")
    print("    Top-right:    Return-nikkei (blue, daily log returns)")
    print("    Bottom-left:  Absolute Returns-cac40 (red, |log returns|)")
    print("    Bottom-right: Absolute Returns-nikkei (red, |log returns|)")
    plot_figure1(cac40_returns, nikkei_returns, save_path="figure1.png")

    # For Tables 2-4: use sample-in data (first 1000 observations)
    cac40_sample_in = cac40_returns.values[:1000]
    nikkei_sample_in = nikkei_returns.values[:1000]

    print("\n" + "=" * 75)
    print("[Step 6] Estimating GARCH models (Table 2)...")
    print("=" * 75)
    print("  Section 2.1, Equation (1):")
    print("    x_t = mu + a_t")
    print("    a_t = sigma_t * epsilon_t")
    print("    sigma^2_t = alpha_0 + alpha_1 * a^2_{t-1} + beta * sigma^2_{t-1}")
    print("  GARCH-n: epsilon_t ~ N(0,1)")
    print("  GARCH-t: epsilon_t ~ t_d (standardized Student-t)")
    print("  Estimation: Maximum Likelihood (MLE)")

    print("\n  Estimating GARCH-n for TWIEX...")
    twiex_garch_n = estimate_garch_model(cac40_sample_in, dist='normal')

    print("  Estimating GARCH-n for nikkei...")
    nikkei_garch_n = estimate_garch_model(nikkei_sample_in, dist='normal')

    print("  Estimating GARCH-t for TWIEX...")
    twiex_garch_t = estimate_garch_model(cac40_sample_in, dist='t')

    print("  Estimating GARCH-t for nikkei...")
    nikkei_garch_t = estimate_garch_model(nikkei_sample_in, dist='t')

    print_table2(twiex_garch_n, twiex_garch_t, nikkei_garch_n, nikkei_garch_t)

    print("\n" + "=" * 75)
    print("[Step 7] Estimating GJR models (Table 3)...")
    print("=" * 75)
    print("  Section 2.2, Equation (4):")
    print("    x_t = mu + a_t")
    print("    a_t = sigma_t * epsilon_t")
    print("    sigma^2_t = alpha_0 + alpha_1*a^2_{t-1} + beta*sigma^2_{t-1}")
    print("                + gamma*s_{t-1}*a^2_{t-1}")
    print("    where s_{t-1} = 1 if a_{t-1} < 0, else 0")
    print("  GJR-n: epsilon_t ~ N(0,1)")
    print("  GJR-t: epsilon_t ~ t_d (standardized Student-t)")
    print("  Estimation: Maximum Likelihood (MLE)")

    print("\n  Estimating GJR-n for TWIEX...")
    twiex_gjr_n = estimate_gjr_model(cac40_sample_in, dist='normal')

    print("  Estimating GJR-n for nikkei...")
    nikkei_gjr_n = estimate_gjr_model(nikkei_sample_in, dist='normal')

    print("  Estimating GJR-t for TWIEX...")
    twiex_gjr_t = estimate_gjr_model(cac40_sample_in, dist='t')

    print("  Estimating GJR-t for nikkei...")
    nikkei_gjr_t = estimate_gjr_model(nikkei_sample_in, dist='t')

    print_table3(twiex_gjr_n, twiex_gjr_t, nikkei_gjr_n, nikkei_gjr_t)

    print("\n" + "=" * 75)
    print("[Step 8] Estimating Copula models (Table 4)...")
    print("=" * 75)
    print("  Section 3.2: Eight copula families")
    print("    Gaussian, Student-t, Clayton, Rotated-Clayton,")
    print("    Plackett, Frank, Gumbel, Rotated-Gumbel")
    print("  Section 3.3: Estimation by IFM/MLE")
    print("    Gaussian: Kendall's tau inversion (Eq. 32)")
    print("    Others: MLE on copula log-likelihood (Eq. 16)")
    print("  Applied to PIT-transformed residuals from each marginal model")

    twiex_models = {
        'GARCH-n': twiex_garch_n,
        'GARCH-t': twiex_garch_t,
        'GJR-n': twiex_gjr_n,
        'GJR-t': twiex_gjr_t
    }
    nikkei_models = {
        'GARCH-n': nikkei_garch_n,
        'GARCH-t': nikkei_garch_t,
        'GJR-n': nikkei_gjr_n,
        'GJR-t': nikkei_gjr_t
    }

    table4_results = estimate_table4(twiex_models, nikkei_models)

    print_table4(table4_results)

    print("\n" + "=" * 75)
    print("[Step 9] Estimating rolling VaR for Figure 2...")
    print("=" * 75)
    print("  Method: Student-t copula with GARCH-n marginals")
    print("  Section 4.4, Equations 23-27")
    print("  Portfolio: X_{p,t} = 0.5*X_{1,t} + 0.5*X_{2,t} (Eq. 24)")
    print("  Rolling window: 1000 observations")
    print("  Confidence levels: alpha = 0.05 and alpha = 0.01")
    print("  Monte Carlo simulation from copula-GARCH model")

    var_result = estimate_rolling_var_t_copula_garch_n(
        cac40_returns.values, nikkei_returns.values,
        window=1000, n_sim=10000, alpha_levels=[0.05, 0.01]
    )

    print(f"\n  --- VaR Violation Summary ---")
    print(f"  Out-of-sample observations: {var_result['n_out']}")
    for alpha in [0.05, 0.01]:
        expected = var_result['expected_violations'][alpha]
        actual = var_result['violations'][alpha]
        print(f"  alpha={alpha}: expected violations={expected}, "
              f"actual violations={actual}")

    print("\n  Generating Figure 2...")
    plot_figure2(var_result, save_path="figure2.png")

    print("\n" + "=" * 75)
    print("[Step 10] Estimating traditional VaR methods for Figure 3...")
    print("=" * 75)
    print("  Comparing at alpha = 0.01 (99% confidence)")
    print("  Traditional methods:")
    print("    1. Historical Simulation (HS) - empirical quantile")
    print("    2. Variance-Covariance (VC) - Equation 28")
    print("    3. EWMA - Equation 29, optimized lambda")
    print("    4. Univariate GARCH-n VaR - Equation 30")

    traditional_vars = estimate_all_traditional_var(
        cac40_returns.values, nikkei_returns.values,
        window=1000, alpha=0.01
    )

    n_out = var_result['n_out']
    portfolio_rets = var_result['portfolio_returns']

    print(f"\n  --- Violation Comparison at alpha = 0.01 ---")
    print(f"  Trading days: {n_out}")
    print(f"  Expected violations: {int(np.round(n_out * 0.01))}")
    print(f"  {'Method':<25} {'Violations':>12}")
    print(f"  {'-' * 40}")
    print(f"  {'t-copula-GARCH-n':<25} {var_result['violations'][0.01]:>12}")
    for method_name, var_series in traditional_vars.items():
        violations = np.sum(portfolio_rets < var_series)
        print(f"  {method_name:<25} {violations:>12}")

    print("\n  Generating Figure 3...")
    plot_figure3(var_result, traditional_vars, save_path="figure3.png")

    print("\n" + "=" * 75)
    print("[Step 11] Estimating Table 5: VaR violations for all copula-marginal pairs...")
    print("=" * 75)
    print("  8 copulas × 4 marginals × 639 rolling windows")
    print("  Alpha levels: 0.05 and 0.01")
    print("  WARNING: This step is very computationally intensive.")
    print("  Estimated runtime: several hours.")

    table5_results = estimate_table5(
        cac40_returns.values, nikkei_returns.values,
        window=1000, n_sim=10000
    )

    # Display Table 5
    n_out_t5 = len(cac40_returns) - 1000
    print_table5(table5_results, n_out_t5)

    print("\n" + "=" * 75)
    print("[Step 12] Estimating Table 6: Copula vs traditional VaR comparison...")
    print("=" * 75)
    print("  Methods: t-copula-GARCH-n, t-copula-GJR-n, HS, VC, EWMA,")
    print("           GARCH-n, GARCH-t, GJR-n, GJR-t")
    print("  Alpha levels: 0.05 and 0.01")

    table6_results, n_out_t6 = estimate_table6(
        cac40_returns.values, nikkei_returns.values,
        table5_results=table5_results,
        window=1000
    )

    print_table6(table6_results, n_out_t6)


if __name__ == "__main__":
    main()