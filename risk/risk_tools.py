from tracemalloc import Statistic
from turtle import color
from unittest import result
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize
import ipywidgets as widgets
from IPython.display import display


def drawdown(return_series: pd.Series, amount: float = 1000):
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    """
    wealth_index = amount * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdowns': drawdowns
    })


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv',
                      header=0, index_col=0, parse_dates=True)

    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')

    return hfi


def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Wighted Monthly Returns
    """
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header=0,
                      index_col=0, parse_dates=True) / 100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()

    return ind


def get_total_market_index_returns():
    """
    """
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis='columns')
    ind_capweight = ind_mktcap.divide(total_mktcap, axis='rows')
    total_market_return = (ind_capweight * ind_return).sum(axis='columns')
    return total_market_return


def get_ind_nfirms():
    """
    """
    ind = pd.read_csv('data/ind30_m_nfirms.csv', header=0,
                      index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()

    return ind


def get_ind_size():
    """
    """
    ind = pd.read_csv('data/ind30_m_size.csv', header=0,
                      index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()

    return ind


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a Dataframe
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame Returns
    a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()

    return exp/sigma_r ** 3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame Returns
    a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()

    return exp/sigma_r ** 4


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall bellow that number, and the level (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1) * s/6 +
             (z**3 - 3*z) * (k-3)/24 -
             (2*z**3 - 5*z) * (s**2)/36
             )

    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')


def annualized_return(r):
    n_months = r.shape[0]
    r = r / 100
    annualized_return = (r + 1).prod() ** (12/n_months) - 1

    return annualized_return


def annualized_volatility(r):
    r = r / 100
    return r.std() * np.sqrt(12)


def annualized_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    """
    compunded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compunded_growth ** (periods_per_year/n_periods) - 1


def annualized_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    We should infer the periods per year
    """

    return r.std() * (periods_per_year ** 0.5)


def sharp_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_rets(excess_ret, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)

    return ann_ex_ret / ann_vol


def portfolio_return(weights, returns):
    """
    Calculate portfolio returns
    Weights -> Returns
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Calculate portfolio volatility
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights) ** 0.5


def plot_ef2(n_points, er, cov, style='.-'):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError('plot_ef2 can only plot 2-asset frontiers')

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })

    return ef.plot.line(x='Volatility', y='Returns', style=style)


def optimal_weights(n_points, er, cov):
    """
    -> list of weights ro run the optimizer on to minimize the volatility
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov)
               for target_return in target_rs]

    return weights


def gmv(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the weights of the Global Minimum Volatility portfolio
    by given covariance matrix
    """
    n = cov.shape[0]
    return max_sharp_ratio(0, np.repeat(1, n), cov)


def plot_ef(n_points: int, er: pd.DataFrame, cov: pd.DataFrame, show_cml=False, style='.-', riskfree_rate=0, show_ew=False, show_gmv=False) -> pd.DataFrame:
    """
    Plots the N-asset efficient frontier
    show_ew = equally wighted
    """

    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })

    ax = ef.plot.line(x='Volatility', y='Returns', style=style)

    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display GMV
        ax.plot([vol_gmv], [r_gmv], color='midnightblue',
                marker='o', markersize=10)
    if show_cml:
        ax.set_xlim(left=0)
        weights_msr = max_sharp_ratio(riskfree_rate, er, cov)
        returns_msr = portfolio_return(weights_msr, er)
        volatility_msr = portfolio_vol(weights_msr, cov)

        # Add Capital Market Line
        cml_x = [0, volatility_msr]
        cml_y = [riskfree_rate, returns_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o',
                linestyle='dashed', markersize=12, linewidth=2)

    return ax


def minimize_vol(target_return, er, cov):
    """
    target_ret -> w
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = minimize(
        portfolio_vol,
        init_guess, args=(cov,),
        method='SLSQP',
        options={'disp': False},
        constraints=(return_is_target, weights_sum_to_1), bounds=bounds)

    return results.x


def negative_sharp_ratio(weights, riskfree_rate, er, cov):
    """
        Returns the negative of the sharp ratio, given wights
        """
    r = portfolio_return(weights, er)
    vol = portfolio_vol(weights, cov)

    return -(r - riskfree_rate) / vol


def max_sharp_ratio(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharp ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = minimize(
        negative_sharp_ratio,
        init_guess,
        args=(riskfree_rate, er, cov,),
        method='SLSQP',
        options={'disp': False},
        constraints=(weights_sum_to_1), bounds=bounds)

    return results.x


def run_cppi(risky_return, safe_return=None, multiplier=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # 1. Cushion - (Asset Value - Floor Value)
    # 2. Compute allocation to the safe and risky assets -> m * risk_budget
    # 3. Recompute the asset value based on the returns

    # set up the CPPI parameters
    dates = risky_return.index
    number_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_return, pd.Series):
        risky_return = pd.DataFrame(risky_return, columns=['R'])

    if safe_return is None:
        safe_return = pd.DataFrame().reindex_like(risky_return)
        # fast way to set all values to a number
        safe_return.values[:] = riskfree_rate / 12

    account_history = pd.DataFrame().reindex_like(risky_return)
    cushion_history = pd.DataFrame().reindex_like(risky_return)
    risky_weight_history = pd.DataFrame().reindex_like(risky_return)

    for step in range(number_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value  # risk budget
        risky_weight = multiplier * cushion

        # don't go above 100% and bellow 0%
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)

        safe_weight = 1 - risky_weight

        risky_allocation = account_value * risky_weight
        safe_allocation = account_value * safe_weight

        # update the account value for this account step
        account_value = (risky_allocation * (1 + risky_return.iloc[step])) + (
            safe_allocation * (1 + safe_return.iloc[step]))

        # save the value to be able to plot it
        cushion_history.iloc[step] = cushion
        risky_weight_history.iloc[step] = risky_weight
        account_history.iloc[step] = account_value

    risky_wealth = start * (1 + risky_return).cumprod()

    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'Risk Budget': cushion_history,
        'Risky Allocation': risky_weight_history,
        'multiplier': multiplier,
        'start': start,
        'floor': floor,
        'risky_return': risky_return,
        'safe_return': safe_return
    }

    return backtest_result


def summary_stats(r: pd.DataFrame, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualized_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualized_vol, periods_per_year=12)
    ann_sr = r.aggregate(
        sharp_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)

    return pd.DataFrame({
        'Annualized Return': ann_r,
        'Annualized Vol': ann_vol,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish-Fisher VaR (5%)': cf_var5,
        'Historic CVaR (5%)': hist_cvar5,
        'Sharp Ratio': ann_sr,
        'Max Drawdown': dd
    })


def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(
        loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0 * \
        pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


def show_gbm(n_scenarios, mu, sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    s_0 = 100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color='indianred',
                     alpha=0.5, linewidth=2, figsize=(12, 5))
    ax.axhline(y=s_0, ls=':', color='black')
    ax.set_ylim(top=400)

    # draw a dot at the origin
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)


def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100):
    """
    Plots the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma,
                   prices=False, steps_per_year=12)
    risky_r = pd.DataFrame(sim_rets)

    # run the back-test
    btr = run_cppi(risky_return=pd.DataFrame(risky_r),
                   riskfree_rate=riskfree_rate, multiplier=m, start=start, floor=floor)
    wealth = btr['Wealth']
    y_max = wealth.values.max() * y_max / 100
    ax = wealth.plot(legend=False, alpha=0.3,
                     color='indianred', figsize=(12, 6))
    ax.axhline(y=start, ls=':', color='black')
    ax.axhline(y=start * floor, ls='--', color='red')
    ax.set_ylim(top=y_max)
