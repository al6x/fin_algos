import pandas as pd
import numpy as np
from scipy.optimize import minimize
from helpers import save_asset, report, cached
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import inspect
import math
from scipy.stats import boxcox
from scipy.special import inv_boxcox

np.random.seed(0)
show=False

doc_before = r"""Exploring Put Premiums from Historical Data

Premiums calculated as:

  P_{eu}(K|Q_{vol}) = E[(K/S_T-S_T/S_0)^+|Q_{vol}]
"""

doc_after = """
# Data

  - period - period, days
  - vol - current volatility as std (in scale unit, not variance).
  - vol_dc - volatility decile 1..10
  - k - strike
  - kq - quantile or strike
  - p_exp - realised put premium using price at expiration (lower bound, european option)
  - p_min - realised put premium, using min price during option lifetime (upper bound, max possible
    for american option).
"""

def make_normalise_vol(df):
  report("""
    # Strike normalisation

    Making mad(m) = 1 for all periods.

      m = log(k)/(vol*vol_p(period | P))
      vol_p(period | P) = exp(p1 + p2*log(period) + p3*log(period)^2)
      P ~ min L2 mean_abs_dev(m) - 1
  """)
  def normalise(vol, period, params):
    p1, p2, p3 = params
    lp = np.log(period)
    return df['vol']*np.exp(p1 + p2*lp + p3*lp**2)

  def fit(df):
    def loss(params):
      m = np.log(df['k']) / (normalise(df['vol'], df['period'], params))
      mad = np.mean(np.abs(m - np.median(m)))
      return (mad - 1)**2
    res = minimize(loss, x0=[0.0, 1.0, 0.0], method='Nelder-Mead')
    return tuple(res.x)

  params = cached('normalise_vol_params', lambda: fit(df))
  report(f"Found params: {', '.join(f'{x:.4f}' for x in params[:3])}")

  return lambda vol, period: normalise(vol, period, params)

def make_normalise_premium_pow_t(df):
  report("""
    # Premium normalisation

    Making mean premium for k=1 same as 30d period mean.

      np_exp = p_exp / (period/30)**pow
      pow ~ min mean(p_exp|k=1,period) - mean(p_exp|k=1,period=30d)
  """, False)

  v = df.loc[df.k==1, ['period','p_exp']]

  def normalise(prem, period, a):
    return prem / (period/30)**a

  def fit(df):
    periods = v.period.unique()
    def loss(x):
      pow = float(x[0])
      s = normalise(v.p_exp, v.period, pow)
      m = s.groupby(v.period).mean()
      # sum squared diff vs base‐period mean
      return sum((m.loc[p] - m.loc[30])**2 for p in periods)
    res = minimize(loss, x0=[0.5], method='Nelder-Mead')
    return float(res.x[0])

  a = cached('normalise_premium_pow_t', lambda: fit(df))
  report(f"Found pow: {a:.4f}")
  return lambda p, per: p / (np.array(per)/30)**a

def load():
  df = pd.read_csv('data/put_premiums.csv')

  # Normalising strikes
  normalise_vol = make_normalise_vol(df)
  nvol = normalise_vol(df['vol'], df['period'])
  df['m'] = np.log(df['k']) / nvol

  # Normalising premiums, so that mean and mad = 1 for each period and k=1
  # normalise_premium = make_normalise_premium_log_z_score(df)
  normalise_premium = make_normalise_premium_pow_t(df)
  df['np_exp'] = normalise_premium(df['p_exp'], df['period'])
  df['np_min'] = normalise_premium(df['p_min'], df['period'])

  return df

def plot_premium(title, df, x, y, x_min=-5, x_max=1, y_min=0.001, y_max = 0.2):
  vols = sorted(df['vol'].unique())
  n_vols = len(vols)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

  for vi, (dc, grp) in enumerate(df.groupby('vol_dc')):
    ks = grp[x].values
    prem = grp[y].values
    color = colors[vi]
    vol = grp['vol'].iloc[0]    # actual volatility for this decile

    # Plot on linear axis
    ax1.plot(ks, prem, '--', color=color, alpha=0.7)
    ax1.scatter(ks, prem, color=color, s=5, alpha=0.9, label=f"vol={vol:.4f}")

    # Plot on log axis
    ax2.plot(ks, prem, '--', color=color, alpha=0.7)
    ax2.scatter(ks, prem, color=color, s=5, alpha=0.9)

  # Legends
  ax1.legend()

  # Y scales
  ax1.set_yscale('linear')
  ax2.set_yscale('log')

  for ax in (ax1, ax2):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which='both', ls=':')

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()

  save_asset(fig, title)

def plot_premium_all_(title, df, x, y, y_max=0.2, y_min=0.001, x_min=-5, x_max=1.0, scale='linear'):
  periods = sorted(df['period'].unique())
  ncols = 3
  nrows = int(np.ceil(len(periods) / ncols))

  vols = sorted(df['vol'].unique())
  n_vols = len(vols)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 5, nrows * 4),
    sharex=True, sharey=True
  )

  for pi, period in enumerate(periods):
    row = pi // ncols
    col = pi % ncols
    ax  = axes[row, col]

    sub = df[df['period'] == period]
    for vi, vol in enumerate(vols):
      grp   = sub[sub['vol']==vol]
      ks    = grp[x].values
      prem  = grp[y].values
      color = colors[vi]                # direct by index

      ax.plot(  ks, prem, '--', color=color, alpha=0.7)
      ax.scatter(ks, prem,       color=color, s=5, alpha=0.9, label=f"vol={vol:.4f}")

    ax.set_title(f"{period}d")
    # if pi == 0:
    #   ax.legend(loc='upper left', fontsize='small')
    ax.set_yscale(scale)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, which='both', ls=':')

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()

  save_asset(fig, title)

def plot_premium_all(title, df, x, y, y_max=0.2, y_min=0.001, x_min=-5, x_max=1.0):
  plot_premium_all_(f'{title} (lin)', df, x, y, x_min=x_min, x_max=x_max, y_max=y_max, y_min=y_min, scale='linear')
  plot_premium_all_(f'{title} (log)', df, x, y, x_min=x_min, x_max=x_max, y_max=y_max, y_min=y_min, scale='log')

def plot_ratio(title, df, x):
  vols = sorted(df['vol'].unique())
  n_vols = len(vols)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  fig, ax = plt.subplots(figsize=(8, 6))

  for vi, (dc, grp) in enumerate(df.groupby('vol_dc')):
    ks = grp[x].values
    ratio = (grp['p_min'] / grp['p_exp']).values
    color = colors[vi]
    vol = grp['vol'].iloc[0]

    # Plot on linear axis
    ax.plot(ks, ratio, '--', color=color, alpha=0.7)
    ax.scatter(ks, ratio, color=color, s=5, alpha=0.9, label=f"vol={vol:.4f}")

  # Legend
  ax.legend()

  # Y scale and limits
  # ax.set_xscale('log')
  ax.set_ylim(1, 2.5)
  ax.grid(True, which='both', ls=':')

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()

  save_asset(fig, title)

def plot_ratio_all(title, df, x, y_min=1.0, y_max=5):
  periods = sorted(df['period'].unique())
  ncols = 3
  nrows = int(np.ceil(len(periods) / ncols))

  vols   = sorted(df['vol'].unique())
  n_vols = len(vols)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 5, nrows * 4),
    sharex=True, sharey=True
  )

  for pi, period in enumerate(periods):
    row = pi // ncols
    col = pi % ncols
    ax  = axes[row, col]

    sub = df[df['period'] == period]
    for vi, vol in enumerate(vols):
      grp = sub[sub['vol'] == vol]
      if grp.empty:
        continue

      ks    = grp[x].values
      ratio = (grp['p_min'] / grp['p_exp']).values
      color = colors[vi]                # direct by index

      ax.plot(  ks, ratio, '--', color=color, alpha=0.7)
      ax.scatter(ks, ratio,       color=color, s=5, alpha=0.9, label=f"vol={vol:.4f}")

    ax.set_title(f"{period}d")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which='both', ls=':')

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()

  save_asset(fig, title)

def run_60d(df):
  df = df[df['period'] == 60]
  report("# Puts 60d")

  plot_premium(
    "Premiums 60d, Raw Strikes k=K/S",
    df, x='k', y='p_exp', x_min=0.5, x_max=1, y_min=0.001, y_max=0.1
  )
  plot_premium(
    "Premiums 60d, Normalised Strikes m=ln(K/S)/nvol",
    df, x='m', y='p_exp', x_min=-5, x_max=0, y_min=0.001, y_max=0.1
  )
  plot_premium(
    "Premiums 60d, Strike Quantiles kq=CDF(k|vol)",
    df, x='kq', y='p_exp', x_min=0, x_max=0.5, y_max=0.1
  )
  report("#note quantiles produce linear curves")
  plot_premium(
    "Premiums 60d, Norma Strikes and Norm Premium",
    df, x='m', y='np_exp', x_min=-5, x_max=0, y_min=0.001, y_max=0.1
  )

  plot_ratio("Ratio of Premium Min / Exp, 60d", df, x='m')
  report("#note bounds for american put: eu < am < 2eu")

def run_all(df):
  df = df[df['period'] != 1095]
  report("# Puts all periods")

  plot_premium_all(
    "Premiums, Raw Strikes k=K/S",
    df, x='k', y='p_exp', x_min=0.5, x_max=1, y_min=0.001, y_max=0.1
  )
  plot_premium_all(
    "Premiums, Norm Strikes m=ln(K/S)/nvol",
    df, x='m', y='p_exp', x_min=-5, x_max=0, y_min=0.001, y_max=0.1
  )
  plot_premium_all(
    "Premiums, Strike Quantiles kq=CDF(k|vol)",
    df, x='kq', y='p_exp', x_min=0, x_max=0.5, y_max=0.1
  )
  plot_premium_all(
    "Premiums, Norm Strikes and Norm Premiums",
    df, x='m', y='np_exp', x_min=-5, x_max=0, y_min=0.001, y_max=0.1
  )

  plot_ratio_all("Ratio of Premium Min / Exp", df, x='m')
  report("#note bounds for american put: eu < am < 2eu")

def run():
  report(doc_before, False)

  df = load()

  run_60d(df)
  run_all(df)

  report(doc_after, False)

if __name__ == "__main__":
  run()