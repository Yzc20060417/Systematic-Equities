import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Optional, Dict, Any, Callable
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cvxpy as cp

# Backtest Uitilities

## 1. Print Performance Metrics of a Strategy
def performance_summary(pnl: pd.Series, periods_per_year: int = 252):
    pnl = pnl.dropna()
    non_zero = pnl != 0
    if non_zero.any():
        first_idx = pnl[non_zero].index[0]
        pnl = pnl.loc[first_idx:]
    n = len(pnl)
    # Cumulative equity
    equity = (1 + pnl).cumprod()
    # Total return
    total_return = equity.iloc[-1] - 1.0
    # CAGR
    years = n / periods_per_year
    if years > 0:
        cagr = equity.iloc[-1] ** (1 / years) - 1
    else:
        cagr = np.nan
    # Annualized volatility
    ann_vol = pnl.std(ddof=1) * np.sqrt(periods_per_year)
    # Sharpe
    sharpe = cagr / ann_vol if pd.notna(ann_vol) and ann_vol != 0 else np.nan
    # Drawdowns
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()
    # Calmar
    calmar = cagr / abs(max_dd) if pd.notna(max_dd) and max_dd < 0 else np.nan
    # Hit rate
    hit_rate = (pnl > 0).mean()
    print("Performance Summary")
    print(f"Periods             : {n}")
    print(f"Total Return        : {total_return:8.2%}")
    print(f"CAGR                : {cagr:8.2%}")
    print(f"Ann. Volatility     : {ann_vol:8.2%}")
    print(f"Sharpe Ratio        : {sharpe:8.2f}")
    print(f"Max Drawdown        : {max_dd:8.2%}")
    print(f"Calmar Ratio        : {calmar:8.2f}")
    print(f"Hit Rate (p>0)      : {hit_rate:8.2%}")

## 2. Plot Equity Backtest Curve
def plot_equity_curve(pnl: pd.Series, title: str = "Equity Curve"):
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    non_zero = pnl != 0
    if non_zero.any():
        first_idx = pnl[non_zero].index[0]
        pnl = pnl.loc[first_idx:]
        
    equity = (1 + pnl).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

## 3. Equal-weight top-k / bottom-k long-short. gross=1 => long +0.5 gross, short -0.5 gross.
def build_long_short_weights(S: pd.DataFrame, tradable: pd.DataFrame,
                             k: int = 2, gross: float = 1.0,
                             min_names=30):
    w_list = []
    for dt in S.index:
        x = S.loc[dt].where(tradable.loc[dt]).dropna()
        if len(x) < min_names:
            w_list.append(pd.Series(0.0, index=S.columns, name=dt))
            continue

        top = x.nlargest(k).index
        bot = x.nsmallest(k).index

        w = pd.Series(0.0, index=S.columns, name=dt)
        w_long = +0.5 * gross / k
        w_short = -0.5 * gross / k
        w.loc[top] = w_long
        w.loc[bot] = w_short
        w_list.append(w)

    return pd.DataFrame(w_list).sort_index()

## 4. Weekly rebalance: choose top/bottom k on rebalance dates, hold weights constant until next rebalance. rebalance: pandas offset alias e.g. 'W-FRI' for Friday close rebalance
def build_long_short_weights_rebal(
    S: pd.DataFrame,
    tradable: pd.DataFrame,
    k: int = 10,
    gross: float = 1.0,
    rebalance: str = "W-FRI",
    min_names: int = 30,
):
    rebal_dates = S.resample(rebalance).last().index
    rebal_dates = rebal_dates.intersection(S.index)

    # Compute weights only on rebalance dates
    w_rebal = {}
    for dt in rebal_dates:
        x = S.loc[dt].where(tradable.loc[dt]).dropna()
        if len(x) < min_names:
            w_rebal[dt] = pd.Series(0.0, index=S.columns)
            continue
        top = x.nlargest(k).index
        bot = x.nsmallest(k).index

        w = pd.Series(0.0, index=S.columns)
        w_long = +0.5 * gross / k
        w_short = -0.5 * gross / k
        w.loc[top] = w_long
        w.loc[bot] = w_short
        w_rebal[dt] = w

    w_rebal = pd.DataFrame(w_rebal).T.sort_index()
    w_rebal.index.name = "Date"
    # Forward-fill weights to all trading days (hold between rebalances)
    w_daily = w_rebal.reindex(S.index).ffill().fillna(0.0)
    W = []
    for dt in w_daily.index:
        w = w_daily.loc[dt].where(tradable.loc[dt]).fillna(0.0)
        g = w.abs().sum()
        if g > 0:
            w = w * (gross / g)
        W.append(w.rename(dt))
    return pd.DataFrame(W)

## 5. Equal-weight top-k long-only. gross=1 => total long exposure of +1.
def build_long_only_weights(
    S: pd.DataFrame,
    tradable: pd.DataFrame,
    k: int = 10,
    gross: float = 1.0,
    min_names: int = 30,
):
    w_list = []
    for dt in S.index:
        x = S.loc[dt].where(tradable.loc[dt]).dropna()
        if len(x) < min_names:
            w_list.append(pd.Series(0.0, index=S.columns, name=dt))
            continue

        top = x.nlargest(k).index
        k_eff = len(top)

        w = pd.Series(0.0, index=S.columns, name=dt)
        if k_eff > 0:
            w_long = gross / k_eff
            w.loc[top] = w_long
        w_list.append(w)

    return pd.DataFrame(w_list).sort_index()

## 6. Rebalancing top-k long-only: pick on rebalance dates, hold until next rebalance.
def build_long_only_weights_rebal(
    S: pd.DataFrame,
    tradable: pd.DataFrame,
    k: int = 10,
    gross: float = 1.0,
    rebalance: str = "W-FRI",
    min_names: int = 30,
):
    rebal_dates = S.resample(rebalance).last().index
    rebal_dates = rebal_dates.intersection(S.index)

    w_rebal = {}
    for dt in rebal_dates:
        x = S.loc[dt].where(tradable.loc[dt]).dropna()
        if len(x) < min_names:
            w_rebal[dt] = pd.Series(0.0, index=S.columns)
            continue

        top = x.nlargest(k).index
        k_eff = len(top)

        w = pd.Series(0.0, index=S.columns)
        if k_eff > 0:
            w_long = gross / k_eff
            w.loc[top] = w_long
        w_rebal[dt] = w

    w_rebal = pd.DataFrame(w_rebal).T.sort_index()
    w_rebal.index.name = "Date"
    w_daily = w_rebal.reindex(S.index).ffill().fillna(0.0)

    W = []
    for dt in w_daily.index:
        w = w_daily.loc[dt].where(tradable.loc[dt]).fillna(0.0)
        g = w.sum()
        if g > 0:
            w = w * (gross / g)
        W.append(w.rename(dt))
    return pd.DataFrame(W)

## 7. Cost model 1
def cost_model_1(
    w: pd.DataFrame,
    R_fwd: pd.DataFrame,
    vol: pd.DataFrame,
    adv_dollar: pd.DataFrame,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0.,
    impact_eta: float = 0.10,
    impact_alpha: float = 0.50,
    aum: float = 1.0,
    adv_floor: float = 1e4,
    vol_floor: float = 0.0,
    fillna_weight: float = 0.0,
):
    # Align columns across all panels
    cols = w.columns.intersection(R_fwd.columns).intersection(vol.columns).intersection(adv_dollar.columns)
    w0 = w[cols].copy()

    R0 = R_fwd.reindex(w0.index)[cols]
    port_ret_gross = (w0 * R0).sum(axis=1, min_count=1).dropna()

    idx = port_ret_gross.index
    w1 = w0.reindex(idx).fillna(fillna_weight)
    dw = w1.diff().abs().fillna(0.0)

    # Align vol/adv on the same dates
    sig = vol.reindex(idx)[cols].fillna(0.0).clip(lower=vol_floor)
    adv = adv_dollar.reindex(idx)[cols].astype(float)
    adv = adv.replace([np.inf, -np.inf], np.nan).fillna(adv_floor).clip(lower=adv_floor)

    # Linear cost (bps per unit turnover)
    linear_rate = (fee_bps + slippage_bps) * 1e-4
    linear_cost = linear_rate * dw.sum(axis=1)

    # Impact cost (vol + participation vs ADV)
    # cost_i,t (fraction of NAV) = eta * sigma_i,t * |Δw_i,t| * ( (|Δw_i,t|*AUM)/ADV_i,t )^alpha
    part = (dw * float(aum)) / adv
    impact_cost = (impact_eta * sig * dw * (part ** impact_alpha)).sum(axis=1).fillna(0.0)

    cost = (linear_cost + impact_cost).astype(float)
    port_ret_net = (port_ret_gross - cost).astype(float)

    return port_ret_net, cost