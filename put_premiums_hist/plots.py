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
from misc import prune_params, fit_multi_init
from matplotlib.lines import Line2D

show=False

def plot_lmean(title, df, model=None):
  df = df.drop_duplicates(subset=['period','vol_dc']).copy()

  vols   = sorted(df.vol_dc.unique())
  cmap   = plt.get_cmap('coolwarm')
  colors = cmap(np.linspace(0, 1, len(vols)))

  df['mmean_true'] = np.exp(df.lmean_t2 + 0.5*df.scale_t2**2)
  if model is not None:
    df['mmean_model'] = df.apply(lambda r: model(r.period, r.vol), axis=1)

  def _plot_raw(df, ax):
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Mean')
    ax.set_title('Mean, E[S_T/S_0] at expiration')
    # ax.set_yscale('log')
    for i, v in enumerate(vols):
      sub = df[df.vol_dc == v].sort_values('period')
      x, y = sub.period, sub.mmean_true
      ax.scatter(x, y, color=colors[i], s=20, alpha=0.7, edgecolor='k', lw=0.2, label=f'{v}')
      ax.plot(x, y, linestyle='--', color=colors[i], alpha=0.5)
      if model is not None:
        ax.plot(x, sub.mmean_model, color=colors[i], lw=1.5)
    ax.grid(True, which='both', ls=':')
    ax.legend(title='vol decile', fontsize='small', loc='upper right')

  def _plot_trans(df, ax):
    df = df.copy()
    df = df[df['vol_dc'] != 10]

    def trans_mmean(mmean, period, vol):
      # return mmean * 1500*period*vol**(period**0.1)
      # return mmean * 1500*period**0.05*vol**(period**0.05)
      return np.log(mmean)/period

    df['mmean_true_norm'] = trans_mmean(df.mmean_true, df.period, df.vol)
    if model is not None:
      df['mmean_model_norm'] = df.apply(lambda r: trans_mmean(r.mmean_model, r.period, r.vol), axis=1)

    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Trans Mean')
    ax.set_title('Trans Mean, log(E[S_T/S_0])/T (vol 10 not shown)')
    ax.set_yscale('log')
    for i, v in enumerate(vols):
      sub = df[df.vol_dc == v].sort_values('period')
      x, y = sub.period, sub.mmean_true_norm
      ax.scatter(x, y, color=colors[i], s=20, alpha=0.7, edgecolor='k', lw=0.2, label=f'{v}')
      ax.plot(x, y, linestyle='--', color=colors[i], alpha=0.5)
      if model is not None:
        ax.plot(x, sub.mmean_model_norm, color=colors[i], lw=1.5)
    ax.grid(True, which='both', ls=':')
    ax.legend(title='vol decile', fontsize='small', loc='upper right')

  def _plot_errors(df, ax):
    df = df.copy()
    df = df[df['vol_dc'] != 10]

    df['rel_error'] = df['mmean_model'] / df['mmean_true']
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Rel Error')
    ax.set_title('Relative Error (vol 10 not shown)')
    # ax.set_yscale('log')
    for i, v in enumerate(vols):
      sub = df[df.vol_dc == v].sort_values('period')
      x, y = sub.period, sub.rel_error
      ax.scatter(x, y, color=colors[i], s=20, alpha=0.7, edgecolor='k', lw=0.2)
      ax.plot(x, y, color=colors[i], alpha=0.5)
    ax.axhline(1.0, color='gray', linestyle=':')
    ax.grid(True, ls=':')

  # Draw subplots
  fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # 2 rows, 2 cols
  _plot_raw(df, axes[0, 0])
  axes[0, 1].axis('off')
  if model is not None:
    _plot_errors(df, axes[1, 1])
  _plot_trans(df, axes[1, 0])  # row=2, col=1
  fig.suptitle(title)
  plt.tight_layout()
  if show:
    plt.show()
  save_asset(fig, title)

def plot_lmean_heatmap(title, vol_range, period_range, mmean_, df):
  # Create grid
  vol_vals = np.linspace(vol_range[0], vol_range[1], 100)
  period_vals = np.linspace(period_range[0], period_range[1], 100)
  period_mesh, vol_mesh = np.meshgrid(period_vals, vol_vals)

  # Compute model surface
  Z = mmean_(period_mesh, vol_mesh)

  # Prepare transformed axes for plotting
  X = np.sqrt(period_mesh)     # x-axis: sqrt(period)
  Y = np.log(vol_mesh)         # y-axis: log(vol)

  # Plot heatmap + contours
  fig, ax = plt.subplots(figsize=(8, 6))
  cf = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
  # cs = ax.contour( X, Y, Z, levels=10, colors='white', linewidths=0.8)
  # levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 10)
  levels = [1.01, 1.025, 1.05, 1.1, 1.2, 1.3]
  cs = ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=0.8)
  ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

  # Custom x-ticks: map back to original period values
  tick_periods = [30, 90, 180, 365, 730]
  ax.set_xticks([np.sqrt(t) for t in tick_periods])
  ax.set_xticklabels([str(t) for t in tick_periods])

  # Custom y-ticks: decile labels at median log-vol
  deciles = sorted(df['vol_dc'].unique())
  yticks = [np.log(df.loc[df.vol_dc==d, 'vol'].median()) for d in deciles]
  ylabels = [f"{d}" for d in deciles]
  ax.set_yticks(yticks)
  ax.set_yticklabels(ylabels)

  # Labels, title, colorbar
  ax.set_xlabel('Period (days, sqrt scaled)')
  ax.set_ylabel('Vol Decile (log scaled)')
  ax.set_title(title)
  fig.colorbar(cf, ax=ax, label='E[S_T/S_0]')

  plt.tight_layout()
  if show:
    plt.show()
  save_asset(fig, title)

def plot_estimated_scale(title, df, model = None):
  vdcats = sorted(df['vol_dc'].unique())
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(vdcats)))
  fig = plt.figure(figsize=(8, 6))
  for i, vdc in enumerate(vdcats):
    sub = df[df['vol_dc']==vdc]
    periods = sub['period'].values
    x, y, vols = periods, sub['scale_t2'].values, sub['vol'].values
    plt.scatter(x, y, color=colors[i], s=20, alpha=0.7, edgecolor='k', lw=0.2, label=f'{vdc}')
    idx = np.argsort(x)
    x_ord = x[idx]
    if model is not None:
      y_pred = np.array([model(v, p) for v, p in zip(periods, vols)])[idx]
      plt.plot(x_ord, y_pred, color=colors[i], linewidth=1.5)
  plt.xlabel('Period')
  # plt.xticks(np.sqrt(periods), [str(p) for p in periods])
  plt.ylabel('Scale')
  plt.xscale('log')
  plt.yscale('log')
  plt.title(title)
  plt.grid(True, which='both', ls=':')
  plt.legend(title='vol_dc', fontsize='small')
  plt.tight_layout()
  if show: plt.show()
  save_asset(fig, title)

def plot_premium(title, df, x, y, x_min=-4, x_max=1, y_min=0.001, y_max = 0.2):
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
    ax1.scatter(ks, prem, color=color, s=5, alpha=0.9, label=f"{vol:.4f}")

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

def plot_strikes_vs_strike_quantiles_by_period(title, df, x, y, min, max):
  # identify unique periods and layout
  periods = sorted(df['period'].unique())
  n = len(periods)
  cols = np.minimum(3, n)
  rows = math.ceil(n / cols)

  fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
  # flatten axes array
  axes = axes.flatten() if hasattr(axes, "__len__") else [axes]

  for i, period in enumerate(periods):
    ax = axes[i]
    sub = df[df['period'] == period]

    vol_dcs = sorted(sub['vol_dc'].unique())
    colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(vol_dcs)))

    for vi, (dc, grp) in enumerate(sub.groupby('vol_dc')):
      ks = grp[x].values
      qs = grp[y].values
      # if scale == 'log':
      #   qs = np.where(qs > 0.5, 1 - qs, qs)
      c = colors[vi]
      # ax.plot(ks, qs, color=c, alpha=0.7)
      ax.plot(ks, qs, color=c, alpha=0.7)
      # ax.scatter(ks, qs, color=c, s=5, alpha=0.9, label=f"{dc}" if i == 0 else None)

    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.grid(True, which='both', ls=':')
    ax.set_title(f"{period}")
    ax.set_xlabel('NStrike')
    ax.set_ylabel('True NStrike')

  # shared legend on first subplot
  # axes[0].legend(title='Vol deciles', loc='upper left')
  handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(vol_dcs))]
  axes[0].legend(handles, vol_dcs, title='Vol Decile')

  # hide any unused subplots
  for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()
  save_asset(fig, title)

def plot_premium_by_period(title, df, x, x_title, p, c, y_max, y_min, x_min, x_max, yscale='linear', xscale='linear'):
  periods = sorted(df['period'].unique())
  ncols = 3
  nrows = int(np.ceil(len(periods) / ncols))

  vol_dcs = sorted(df['vol_dc'].unique())
  n_vols = len(vol_dcs)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 5, nrows * 4),
    sharex=True, sharey=True
  )
  if isinstance(axes, plt.Axes):     # Single plot
    axes = np.array([axes])
  else:
    axes = axes.flatten()
  for ax in axes[len(periods):]:
    ax.set_visible(False)

  for pi, period in enumerate(periods):
    row = pi // ncols
    col = pi % ncols
    # ax  = axes[row, col]
    ax = axes[pi]

    sub = df[df['period'] == period]
    for vi, vol_dc in enumerate(vol_dcs):
      grp   = sub[sub['vol_dc']==vol_dc]
      ks    = grp[x].values
      color = colors[vi]                # direct by index

      cs = grp[c].values
      # ax.plot(ks[ks < center], cs[ks < center], color=color, alpha=0.5, linewidth=0.7)
      # ax.plot(ks[ks > center], cs[ks > center], color=color, label=f"{vol_dc}")
      ax.plot(ks, cs, color=color, label=f"{vol_dc}")

      ps = grp[p].values
      # ax.plot(ks[ks < center], ps[ks < center], color=color)
      # ax.plot(ks[ks > center], ps[ks > center], color=color, alpha=0.5, linewidth=0.7)
      ax.plot(ks, ps, color=color)

    ax.set_title(f"{period}d")
    if pi == 0:
      ax.legend(title='Vol Deciles', loc='upper left', fontsize='small')
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, which='both', ls=':')
    ax.set_xlabel(x_title)
    ax.set_ylabel('Premium')

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()

  save_asset(fig, title)

def plot_ratio_by_period(title, df, y_min=1.0, y_max=3.5):
  periods = sorted(df['period'].unique())
  ncols = 3
  nrows = int(np.ceil(len(periods) / ncols))

  vols   = sorted(df['vol'].unique())
  n_vols = len(vols)
  colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, n_vols))

  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True)
  if isinstance(axes, plt.Axes):     # Single plot
    axes = np.array([axes])
  else:
    axes = axes.flatten()
  for ax in axes[len(periods):]:
    ax.set_visible(False)

  for pi, period in enumerate(periods):
    row = pi // ncols
    col = pi % ncols
    # ax  = axes[row, col]
    ax = axes[pi]

    sub = df[df['period'] == period]
    for vi, vol in enumerate(vols):
      grp = sub[sub['vol'] == vol]
      if grp.empty:
        continue

      kp    = grp['kq'].values
      c_ratio = (grp['c_max'] / grp['c_exp']).values
      p_ratio = (grp['p_max'] / grp['p_exp']).values
      color = colors[vi]                # direct by index

      # ax.plot(kp[kp < 0.5], p_ratio[kp < 0.5], color=color)
      # ax.plot(kp[kp > 0.5], c_ratio[kp > 0.5], color=color)

      ax.plot(kp, p_ratio, color=color, linestyle='--')
      ax.plot(kp, c_ratio, color=color)


    ax.set_title(f"{period}d")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which='both', ls=':')

  fig.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()

  save_asset(fig, title)

def plot_vols_by_periods(title, df):
  vols = sorted(df['vol_dc'].unique())
  cmap = plt.get_cmap('coolwarm')
  colors = cmap(np.linspace(0, 1, len(vols)))

  fig, ax = plt.subplots(figsize=(8, 6))
  for i, v in enumerate(vols):
    sub = df[df['vol_dc'] == v].sort_values('period')
    ax.scatter(sub['period'], sub['vol'], color=colors[i], s=20, alpha=0.7, edgecolor='k', lw=0.2, label=f'{v}'
    )
    ax.plot(sub['period'], sub['vol'], color=colors[i], alpha=0.9 )

  ax.set_xlabel('Period (days)')
  ax.set_ylabel('Volatility')
  ax.set_title(title)
  ax.grid(True, ls=':')
  ax.legend(title='Vol Decile', fontsize='small', loc='upper left')

  plt.tight_layout()
  if show:
    plt.show()
  save_asset(fig, title)

def plot_scatter(title, x, y):
  fig = plt.figure(figsize=(6, 6))
  plt.scatter(x, y, alpha=0.7)

  # Add 45-degree line for reference
  plt.plot([x.min(), x.max()], [y.min(), y.max()], 'r--', label='y = x', linewidth=0.8, color='gray')
  plt.title(title)
  plt.grid(True)
  if show:
    plt.show()
  save_asset(fig, title)

def plot_line(title, x, y, xdomain=None, ydomain=None):
  fig = plt.figure(figsize=(6, 6))
  plt.plot(x, y, alpha=0.7)  # line plot instead of scatter
  plt.title(title)
  plt.grid(True)

  if xdomain:
    plt.xlim(xdomain)
  if ydomain:
    plt.ylim(ydomain)

  if show:
    plt.show()
  save_asset(fig, title)

def plot_mmean_ratio(title, ratio):
  fig = plt.figure(figsize=(8, 4))
  plt.plot(ratio.values, 'o-')
  plt.axhline(1, color='red', linestyle='--', label='y=1')
  plt.title(title)
  plt.grid(True)
  if show:
    plt.show()
  save_asset(fig, title)
