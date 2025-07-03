import pandas as pd
import numpy as np
from scipy.optimize import minimize
from helpers import save_asset, report
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import inspect
import math

np.random.seed(0)

report("""Fitting Premiums for OTM Put from Historical Data.

Data, csv file:

  - period - period, days
  - vol - current volatility as std (in scale unit, not variance).
  - vol_dc - volatility decile 1..10
  - k - strike
  - kq - quantile or strike
  - p_exp - realised put premium using price at expiration (lower bound, european option)
  - p_min - realised put premium, using min price during option lifetime (upper bound, max possible
    for american option).

**Strike normalisation**

Strikes normalised so that mad(m) = 1 for all periods.

  m = log(k)/vol_p(vol, period | P)
  vol_p(vol, period | P) = exp(p1 + p2*log(vol) + p3*log(vol)^2)
  P ~ min L2 mean_abs_dev(m) - 1

**Report**
""", False)

def nvol(vol, period, params):
  p1, p2, p3 = params
  lv = np.log(vol)
  return np.exp(p1 + p2 * lv + p3 * lv**2)

def fit_nvol(df):
  lv = np.log(df['vol'].values)
  def loss(params):
    p1, p2, p3 = params
    m = np.log(df['k'].values) / np.exp(p1 + p2 * lv + p3 * lv**2)
    mad = np.mean(np.abs(m - np.median(m)))
    return (mad - 1)**2
  res = minimize(loss, x0=[0.0, 1.0, 0.0], method='Nelder-Mead')
  return tuple(res.x)

def load():
  df = pd.read_csv('data/put_premiums.csv')

  # Normalising strikes
  df['m'] = np.log(df['k']) / nvol(df['vol'], df['period'], fit_nvol(df))

  return df

def plot_premium(df, x, y, y_max = 0.2, y_min=0.001, scale='linear', show=True, model=None):
  vols = sorted(df['vol'].unique())
  n_vols = len(vols)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

  for vi, (vol, grp) in enumerate(df.groupby('vol')):
    ks = grp[x].values
    prem = grp[y].values
    color = colors[vi]

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

  # Titles
  ax1.set_title("Linear Scale")
  ax2.set_title("Log Scale")

  for ax in (ax1, ax2):
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which='both', ls=':')

  fig.suptitle("Put Premiums")
  plt.tight_layout()



  # fig, ax = plt.subplots(figsize=(8, 6))
  # cmap = plt.get_cmap('coolwarm')

  # vols = sorted(df['vol'].unique())
  # n_vols = len(vols)
  # colors = cmap(np.linspace(0, 1, n_vols))

  # for vi, (vol, grp) in enumerate(df.groupby('vol')):
  #   ks = grp[x].values
  #   prem = grp[y].values
  #   color = colors[vi]
  #   ax.plot(ks, prem, '--', color=color, alpha=0.7)
  #   ax.scatter(ks, prem, color=color, s=5, alpha=0.9, label=f"vol={vol:.4f}")

  #   # if model is not None:
  #   #   model_vals = np.array([model(vol, k) for k in ks])
  #   #   ax.plot(ks, model_vals, color=color, label=f"vol={vol:.4f}")

  # ax.legend()
  # ax.set_yscale(scale)
  # ax.set_ylim(y_min)
  # # ax.set_xlim(-2, 0)
  # # ax.set_ylim(y_min, y_max)
  # ax.grid(True, which='both', ls=':')

  # # if model is not None:
  # #   ax.legend()

  # fig.suptitle("Put Premiums")
  # plt.tight_layout()

  if show:
    plt.show()

  return fig

def run():
  df = load()
  save_asset(plot_premium(df[df['period'] == 60], x='m', y='p_exp'), 'premiums_exp_60.png')
  save_asset(plot_premium(df[df['period'] == 60], x='m', y='p_min'), 'premiums_min_60.png')

if __name__ == "__main__":
  run()