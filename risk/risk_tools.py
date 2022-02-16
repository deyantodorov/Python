from tracemalloc import Statistic
from unittest import result
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize


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


def annualized_rets(r, peridos_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    """
    compunded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compunded_growth ** (peridos_per_year/n_periods) - 1


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
