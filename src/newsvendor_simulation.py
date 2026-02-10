import numpy as np
import pandas as pd
import scipy.stats as stats

def simulate_one_rep(weekday_stats, price, cost, salvage, days_per_year=365):
    """
    Simulate one year and return daily metrics arrays for that replication.
    """
    # Set random seed for reproducibility.
    rng = np.random.default_rng(1234)

    # Precompute an array of weekdays for the year (length = days_per_year).
    # Use cyclical calendar starting on Monday for simplicity.
    # Simulate by cycling weekdays in order (randomize start day if desired).
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd_list = np.tile(weekday_order, int(np.ceil(days_per_year/7)))[:days_per_year]
    
    # Initialize daily metrics arrays
    daily_profit = np.zeros(days_per_year)
    daily_stockout = np.zeros(days_per_year, dtype=int)
    daily_leftover = np.zeros(days_per_year)
    daily_demand = np.zeros(days_per_year)
    daily_sales = np.zeros(days_per_year)
    
    # Simulate day by day
    for i, wd in enumerate(wd_list):
        row = weekday_stats.loc[weekday_stats['weekday'] == wd].iloc[0]
        mu = float(row['mu'])
        sigma = float(row['sigma'])
        Q = int(row['Q_star'])
        
        # Draw demand from Normal truncated at 0 (and round to int)
        D = rng.normal(loc=mu, scale=sigma)
        D = max(int(round(D)), 0)

        sales = min(Q, D)
        leftover = max(Q - D, 0)
        stockout = 1 if D > Q else 0

        profit = price * sales - cost * Q + salvage * leftover  # salvage may be 0
        
        daily_demand[i] = D
        daily_sales[i] = sales
        daily_leftover[i] = leftover
        daily_stockout[i] = stockout
        daily_profit[i] = profit
    
    # Return a dictionary of daily metrics arrays
    return {
        'profit': daily_profit,
        'stockout': daily_stockout,
        'leftover': daily_leftover,
        'demand': daily_demand,
        'sales': daily_sales
    }

def simulate_one_rep_with_offsets(weekday_stats, price, cost, salvage,
                                  offsets=[0], days_per_year=365, rng=None):
    """
    Simulate one full year of daily pizza demand and outcomes for different Q* offsets.
    
    Parameters
    ----------
    weekday_stats : DataFrame with columns ['weekday','mu','sigma','Q_star']
    price, cost, salvage : economic values
    offsets : list of integers to add/subtract from Q_star
    days_per_year : number of simulated days (default: 365)
    rng : optional numpy random generator (for CRN)
    
    Returns
    -------
    results : dict
        results[offset][metric] is a daily array (length = days_per_year)
    """
    
    # Random generator setup
    if rng is None:
        rng = np.random.default_rng(1234)
    
    # Cycle weekday names for the year (start Monday)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd_list = np.tile(weekday_order, int(np.ceil(days_per_year/7)))[:days_per_year]

    results = {}  # nested dictionary output structure
    
    for offset in offsets:
        # Storage for daily outcomes for this policy
        daily_profit   = np.zeros(days_per_year)
        daily_stockout = np.zeros(days_per_year, dtype=int)
        daily_leftover = np.zeros(days_per_year)
        daily_demand   = np.zeros(days_per_year)
        daily_sales    = np.zeros(days_per_year)

        for i, wd in enumerate(wd_list):
            row = weekday_stats.loc[weekday_stats['weekday'] == wd].iloc[0]
            mu = float(row['mu'])
            sigma = float(row['sigma'])
            Q = int(row['Q_star'] + offset)

            # Draw demand from Normal, truncate at zero, round to int
            D = rng.normal(mu, sigma)
            D = max(int(round(D)), 0)

            sales = min(Q, D)
            leftover = max(Q - D, 0)
            stockout = int(D > Q)
            
            profit = price * sales - cost * Q + salvage * leftover

            daily_profit[i]   = profit
            daily_stockout[i] = stockout
            daily_leftover[i] = leftover
            daily_demand[i]   = D
            daily_sales[i]    = sales

        # Store results for this offset
        results[offset] = {
            'profit': daily_profit,
            'stockout': daily_stockout,
            'leftover': daily_leftover,
            'demand': daily_demand,
            'sales': daily_sales
        }

    return results

def simulate_many_reps_with_offsets(weekday_stats, price, cost, salvage, offsets, days_per_year, replications, rng):
    """
    Perform Monte Carlo with multiple inventory offsets.
    Returns:
        summary_tables: dict of DataFrames summarizing metrics per offset
        raw_samples: dict of raw replication result arrays per offset
    """

    # Storage for raw simulation outcomes for each offset
    raw_samples = {
        offset: {
            'profit': np.zeros(replications),
            'stockout_rate': np.zeros(replications),
            'leftover_per_day': np.zeros(replications),
            'fill_rate': np.zeros(replications)
        }
        for offset in offsets
    }

    # Monte Carlo Loop
    for r in range(replications):
        rep_results = simulate_one_rep_with_offsets(
            weekday_stats,
            price,
            cost,
            salvage,
            offsets=offsets,
            days_per_year=days_per_year,
            rng=rng
        )

        for offset in offsets:
            daily = rep_results[offset]

            raw_samples[offset]['profit'][r] = daily['profit'].sum()
            raw_samples[offset]['stockout_rate'][r] = daily['stockout'].mean()
            raw_samples[offset]['leftover_per_day'][r] = daily['leftover'].mean()
            raw_samples[offset]['fill_rate'][r] = (
                daily['sales'].sum() / daily['demand'].sum()
                if daily['demand'].sum() > 0 else np.nan
            )

    # Build summary tables from raw results
    summary_tables = {}
    z = stats.t.ppf(0.975, df=replications - 1)

    for offset in offsets:
        out = raw_samples[offset]

        def ci(arr):
            se = arr.std(ddof=1) / np.sqrt(replications)
            return arr.mean() - z * se, arr.mean() + z * se

        summary_tables[offset] = pd.DataFrame([
            ["Annual Profit", out['profit'].mean(), *ci(out['profit'])],
            ["Stockout Rate", out['stockout_rate'].mean(), *ci(out['stockout_rate'])],
            ["Leftover per Day", out['leftover_per_day'].mean(), *ci(out['leftover_per_day'])],
            ["Fill Rate", out['fill_rate'].mean(), *ci(out['fill_rate'])]
        ], columns=["Metric", "Mean", "CI Lower (95%)", "CI Upper (95%)"])

    return summary_tables, raw_samples

def mean_ci(x, confidence=0.95):
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin

def compute_metric_ci(name, data):
    mean, lower, upper = mean_ci(data)
    return {
        f'{name}_mean': mean,
        f'{name}_ci_lower': lower,
        f'{name}_ci_upper': upper
    }

def summarize_results(annual_profit, annual_stockout_rate, annual_avg_leftover, annual_fill_rate, replications):
    return pd.DataFrame({
        "mean_annual_profit": [annual_profit.mean()],
        "std_annual_profit": [annual_profit.std(ddof=1)],
        "ci_profit_lower": [annual_profit.mean() - stats.t.ppf(0.975, replications-1) * annual_profit.std(ddof=1) / np.sqrt(replications)],
        "ci_profit_upper": [annual_profit.mean() + stats.t.ppf(0.975, replications-1) * annual_profit.std(ddof=1) / np.sqrt(replications)],

        "mean_stockout_rate": [annual_stockout_rate.mean()],
        "std_stockout_rate": [annual_stockout_rate.std(ddof=1)],

        "mean_leftover_per_day": [annual_avg_leftover.mean()],
        "std_leftover_per_day": [annual_avg_leftover.std(ddof=1)],

        "mean_fill_rate": [annual_fill_rate.mean()],
        "std_fill_rate": [annual_fill_rate.std(ddof=1)]
    })