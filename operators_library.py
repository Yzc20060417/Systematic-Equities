#!/usr/bin/env python
# coding: utf-8

# # **Factor Pool**

# ## Packages

# In[37]:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ## Common Functions

# In[38]:


# Rolling time series mean
def ts_mean(df,window):
    return df.rolling(window,min_periods=window).mean()

# x-ts_mean
def ts_av_diff(df, d):
    rolling_mean = df.rolling(d, min_periods=1).mean()
    return df - rolling_mean

# Rolling time series std
def ts_std_dev(df,window):
    return df.rolling(window,min_periods=window).std()

# Sum of days
def ts_sum(df,window):
    return df.rolling(window,min_periods=window).sum()

# Rolling z-score
def ts_zscore(df,window):
    df = df.astype(float)
    m = df.rolling(window).mean()
    s = df.rolling(window).std(ddof=1)
    return (df-m)/s

# Rolling minimum
def ts_min(df,window):
    return df.rolling(window,min_periods=window).min()

# Compute cross-section z-score
def zscore(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0,np.nan)
    z = df.sub(mean,axis=0).div(std,axis=0)
    return z.fillna(0.0)   

# Rank cross-sectionally according to values
def rank(df):
    return df.rank(axis=1,pct=True)

# Time series rank of the last observation in each window
def ts_rank(df,window):
    def rank_last(df):
        s = pd.Series(df)
        r = s.rank()
        return r.iloc[-1]/len(s)
    return df.rolling(window,min_periods=window).apply(rank_last,raw=False)

# Signed Power
def signed_power(x, y):
    return np.sign(x) * (np.abs(x) ** y)

# Group cross-sectionally according to rank
def bucket(df,start,end,step):
    range = np.arange(start, end+1e-9, step)
    b = pd.DataFrame(np.digitize(df.values,range,right=True),index=df.index,columns=df.columns).astype('float')
    return b

# Forward gaps of rank bucket e.g. for five units with bucket(1,1,3,3,5) densify to (1,1,2,2,3)
def densify(df):
    out = df.copy()
    for time in df.index:
        vals = df.loc[time]
        uniq = pd.unique(vals.dropna())
        mapping = {g: k + 1 for k, g in enumerate(sorted(uniq))}
        out.loc[time] = vals.map(mapping)
    return out

# Shift (Delay) d days
def ts_delay(df,d):
    return df.shift(d)

# Assign group to dataframe
def grouped_panel(df,group):
    gm = group.reindex(df.columns)
    arr = np.tile(gm.values,(len(df.index),1))
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

# Subtract groupwise mean for each date
def group_neutralize(df,group):
    out = df.copy()
    for time in df.index:
        row = df.loc[time]
        g = group.loc[time]
    for grp in pd.unique(g.dropna()):
        mask = (g == grp)
        vals = row[mask]
        if len(vals) == 0:
            continue
        out.loc[time,mask] = vals - vals.mean()
    return out

# Winsorize within each group per date using mean ± n_std * std
def group_winsorize_std(df,group_panel,n_std):
    df, group_panel = df.align(group_panel, join="inner", axis=0)
    df, group_panel = df.align(group_panel, join="inner", axis=1)

    out = df.astype(float).copy()

    for t in out.index:
        x = out.loc[t]
        g = group_panel.loc[t]

        for grp in pd.unique(g.dropna()):
            mask = (g == grp)
            vals = x[mask].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() < 3:
                continue

            v = vals[finite]
            mu = v.mean()
            sd = v.std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                continue

            lo, hi = mu - n_std * sd, mu + n_std * sd
            vals2 = vals.copy()
            vals2[finite] = np.clip(vals[finite], lo, hi)

            out.loc[t, mask] = vals2

    return out

# Scale x for each date to sum of x == 1
def scale(df):
    out = df.copy()
    denom = out.abs().sum(axis=1)
    denom[denom==0] = np.nan
    out = out.div(denom,axis=0)
    out = out.fillna(0.0)
    return out

# Trade when signal reaches entry/exit
def trade_when(entry,signal,exit):
    entry = entry.astype(bool)
    exit = exit.astype(bool)
    sig = signal.copy()

    pos = pd.DataFrame(0.0,index=signal.index,columns=signal.columns)
    prev = np.zeros(signal.shape[1],dtype=float)
    for i,time in enumerate(signal.index):
        e = entry.loc[time].values
        x = exit.loc[time].values
        s = sig.loc[time].values

        cur = prev.copy()
        cur[x] = 0.0
        mask_enter = e & (~x)
        cur[mask_enter] = s[mask_enter]

        pos.iloc[i,:] = cur
        prev = cur
    return pos

def trade_when_hold(cond_df, val_df):
    """
    Stateful trade_when for this alpha:
    - if cond is True  -> alpha_t = val_df
    - if cond is False -> alpha_t = alpha_{t-1}  (hold previous)
    - if no previous value (first day) and cond is False -> NaN
    All inputs are DataFrames with same shape.
    """
    cond = cond_df.reindex(val_df.index).fillna(False).astype(bool)
    vals = val_df.astype(float)

    out = pd.DataFrame(index=vals.index, columns=vals.columns, dtype=float)

    prev = np.full(vals.shape[1], np.nan, dtype=float)

    for i, date in enumerate(vals.index):
        c_row = cond.loc[date].values
        v_row = vals.loc[date].values.astype(float)

        cur = prev.copy()
        # where cond is true today, overwrite with today's values
        mask = c_row
        cur[mask] = v_row[mask]

        out.iloc[i, :] = cur
        prev = cur

    return out

# Hump smooth along time for each stock
def hump(df, hump):
    df = df.copy()
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # we’ll build it row by row, but handle "first valid" per column
    prev = np.full(len(df.columns), np.nan, dtype=float)

    for i, date in enumerate(df.index):
        raw = df.iloc[i, :].values.astype(float)

        # for names with no previous value but now a finite raw value,
        # treat this as their starting point (no hump limit on first obs)
        start_mask = np.isnan(prev) & np.isfinite(raw)
        prev[start_mask] = raw[start_mask]

        # if still NaN (never had a finite value), keep it NaN
        still_nan = np.isnan(prev) & ~np.isfinite(raw)
        raw[still_nan] = np.nan

        # for the rest, apply hump logic
        # use previous cross-section to set the per-day limit
        limit = hump * np.nansum(np.abs(prev))
        if not np.isfinite(limit) or limit == 0:
            cur = prev
        else:
            # any new NaNs in raw -> hold prev
            raw = np.where(np.isfinite(raw), raw, prev)
            delta = raw - prev
            step = np.where(np.abs(delta) <= limit, 0.0,
                            np.sign(delta) * limit)
            cur = prev + step

        out.iloc[i, :] = cur
        prev = cur

    return out

# Orthogonalize the alpha from b each day
def vector_neut(a,b):
    a = a.copy()
    b = b.copy()
    out = pd.DataFrame(index=a.index,columns=a.columns,dtype=float)
    for time in a.index:
        av = a.loc[time].values.astype(float)
        bv = b.loc[time].values.astype(float)
        mask = np.isinfinite(av) & np.isinfinite(bv) 
        if mask.sum() < 2 or np.all(bv[mask] == 0):
            out.loc[time] = av
            continue
        cov_ab = np.dot(av[mask],bv[mask])
        var_b = np.dot(bv[mask],bv[mask])
        beta = cov_ab / var_b
        adj = av - beta * bv
        out.loc[time] = adj
    return out

# For each date and stock, locate its industry and compute the weighted mean of the industry
def group_mean(df,weights,industry):
    out = df.copy()
    inds = industry.dropna()
    for time in df.index:
        row = df.loc[time]
        w = weights.loc[time]
        for ind in inds.unique():
            mask = (inds == ind)
            vals = row[mask]
            ws = w[mask]
            if len(vals) == 0:
                continue
            w_abs = np.abs(ws)
            w_sum = w_abs.sum()
            if w_sum == 0:
                mean_val = vals.mean()
            else:
                mean_val = (vals * w_abs/w_sum).sum()
            out.loc[time,mask] = mean_val
    return out

# Cross-sectional Cauchy quantile transform
def quantile_cauchy(df,eps=1e-6):
    u = df.rank(axis=1,pct=True)
    u = u.clip(eps,1-eps)
    return np.tan(np.pi*(u-0.5))

# Time-series Cauchy quantile of today's value vs last `window` days.
def ts_quantile_cauchy(df,window,eps=1e-6):
    df = df.astype(float)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        x = df[col].values
        n = len(x)
        res = np.full(n, np.nan, dtype=float)

        for i in range(window - 1, n):
            w = x[i - window + 1 : i + 1]

            # need finite last value and at least 1 finite in window
            if not np.isfinite(w[-1]):
                continue
            mask = np.isfinite(w)
            if mask.sum() == 0:
                continue

            w_valid = w[mask]
            last = w[-1]

            # percentile of today's value within window
            rank = (w_valid <= last).sum()
            u = rank / (mask.sum() + 1.0)   # (0,1) approx

            # clip to avoid infinities in tan()
            u = min(max(u, eps), 1 - eps)
            res[i] = np.tan(np.pi * (u - 0.5))

        out[col] = res

    return out

# Standard OLS on the last d points
# y, x: DataFrames [dates × tickers]
# Rettype: 0:Residual 3:Predicted Value 6:R^2
def ts_regression(y,x,window,lag,rettype):
    x_lag = x.shift(lag)
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    for col in y.columns:
        ys = y[col].values.astype(float)
        xs = x_lag[col].values.astype(float)
        n = len(ys)
        res = np.full(n, np.nan)
        for i in range(window - 1, n):
            # window indices [i-window+1, ..., i]
            y_win = ys[i - window + 1 : i + 1]
            x_win = xs[i - window + 1 : i + 1]
            mask = np.isfinite(y_win) & np.isfinite(x_win)
            if mask.sum() < 2:
                continue
            yw = y_win[mask]
            xw = x_win[mask]
            x_mean = xw.mean()
            y_mean = yw.mean()
            cov = ((xw - x_mean) * (yw - y_mean)).mean()
            var = ((xw - x_mean) ** 2).mean()
            if var == 0:
                beta = 0.0
            else:
                beta = cov / var
            alpha = y_mean - beta * x_mean
            y_hat = alpha + beta * xw
            if rettype == 0:
                res[i] = (yw-y_hat)[-1]
            elif rettype == 2:
                res[i] = beta
            elif rettype == 3:
                res[i] = y_hat[-1]
            elif rettype == 6:
                sse = ((yw - y_hat) ** 2).sum()
                sst = ((yw - y_mean) ** 2).sum()
                r2 = 1.0 - sse / sst if sst > 0 else 0.0
                res[i] = r2
            else:
                raise ValueError("rettype not implemented in ts_regression_core")
        out[col] = res
    return out

# Days Counter
def ts_step(df):
    n = len(df.index)
    step_vals = np.arange(1,n+1,dtype=float)
    step = pd.DataFrame(np.tile(step_vals[:,None],(1,df.shape[1])),index=df.index,columns=df.columns)
    return step

# Cross-Sectionally rank within the group on a given t
def group_rank(df,group):
    df, group = df.align(group, join="inner", axis=0) 
    df, group = df.align(group, join="inner", axis=1)
    out = df.copy().astype(float)
    for t in df.index:
        vals = df.loc[t]
        g = group.loc[t]
        for grp in pd.unique(g.dropna()):
            mask = (g == grp)
            sub = vals[mask]
            if len(sub) == 0:
                continue
            out.loc[t, mask] = sub.rank(method="average", pct=True)
    return out

# Cross-Sectionally Clip the values
def winsorize(df,std):
    out = df.copy().astype(float)
    for t in out.index:
        row = out.loc[t].to_numpy(dtype=float)
        mask = np.isfinite(row)
        if mask.sum() == 0:
            continue
        vals = row[mask]
        mu = vals.mean()
        sigma = vals.std(ddof=1)
        if sigma == 0 or not np.isfinite(sigma):
            continue
        lo = mu - std * sigma
        hi = mu + std * sigma
        row2 = row.copy()
        row2[mask] = np.clip(row[mask], lo, hi)
        row2[~mask] = np.nan
        out.loc[t] = row2
    return out

# Backward fill window of missing values
def ts_backfill(df,window):
    return df.astype(float).ffill(limit=window)

# Check whether there's min value in past d days
def ts_arg_min(df,window):
    df = df.astype(float)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        vals = df[col].values
        n = len(vals)
        res = np.full(n, np.nan, dtype=float)
        for i in range(window - 1, n):
            window_vals = vals[i - window + 1 : i + 1]
            mask = np.isfinite(window_vals)
            if mask.sum() == 0:
                continue
            valid_vals = window_vals[mask]
            idxs = np.arange(window_vals.shape[0])[mask]
            # position (0..window-1) where min occurs
            pos_min = idxs[np.argmin(valid_vals)]
            # convert to 'days ago'
            res[i] = (window - 1) - pos_min
        out[col] = res
    return out

# Weighted Average across time series
def ts_decay_linear(df,window):
    def decay(x):
        x = np.array(x,float)
        x = np.nan_to_num(x,nan=0.0)
        n = len(x)
        w = np.arange(1,n+1)
        return np.dot(x,w)/w.sum()
    return df.rolling(window=window,min_periods=1).apply(decay,raw=True)

# If else Logic
def if_else(cond, a, b):
    """
    Vectorised if-else:
        cond : DataFrame or Series of booleans
        a, b : scalars, Series or DataFrames

    Returns something with the same shape/index/columns as `cond`.
    """
    # Normalize cond to a DataFrame for easier broadcasting
    if isinstance(cond, pd.Series):
        cond_df = cond.to_frame()
    else:
        cond_df = cond
    if np.isscalar(a) and np.isscalar(b):
        return pd.DataFrame(
            np.where(cond_df.values, a, b),
            index=cond_df.index,
            columns=cond_df.columns,
        )
    def _as_like(x):
        if isinstance(x, pd.DataFrame):
            return x.reindex_like(cond_df)
        elif isinstance(x, pd.Series):
            # broadcast series across columns
            return pd.DataFrame(
                np.repeat(x.values[:, None], cond_df.shape[1], axis=1),
                index=cond_df.index,
                columns=cond_df.columns,
            )
        else:  # scalar
            return pd.DataFrame(
                np.full(cond_df.shape, x),
                index=cond_df.index,
                columns=cond_df.columns,
            )
    A = _as_like(a)
    B = _as_like(b)
    out = A.where(cond_df, other=B)
    if isinstance(cond, pd.Series):
        return out.iloc[:, 0]
    return out

# Scale within a certain group to (0,1)
def group_scale(x, group):
    group = group.reindex(x.columns)
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)

    for g in group.dropna().unique():
        cols = group.index[group == g]
        block = x[cols]
        row_min = block.min(axis=1)
        row_max = block.max(axis=1)
        denom = (row_max - row_min).replace(0, np.nan)
        out[cols] = block.sub(row_min, axis=0).div(denom, axis=0)

    return out

# Compute the days passed from the last change of current value
def days_from_last_change(series):
    changes = series != series.shift(1)
    last_change_index = changes[::-1].idxmax()

    days_since_change = (series.index[-1] - last_change_index).days
    return days_since_change

# Non-explode Divide
def safe_div(a, b, eps=1e-12):
    return a / (b.replace(0, np.nan) + eps)

# NLTSMOM (MLP)
def mlp1d_apply(s_df, w1, w2):
    """
    Apply 1D MLP: f(s)= sum_j w2[j] * tanh(w1[j] * s)
    s_df: DataFrame (T x N) or Series
    w1, w2: 1D arrays length H
    """
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    if isinstance(s_df, pd.Series):
        x = s_df.to_numpy(dtype=float)
        h = np.tanh(x[:, None] * w1[None, :])
        out = (h * w2[None, :]).sum(axis=1)
        return pd.Series(out, index=s_df.index, name=s_df.name)

    x = s_df.to_numpy(dtype=float)               # (T, N)
    h = np.tanh(x[:, :, None] * w1[None, None, :])  # (T, N, H)
    out = (h * w2[None, None, :]).sum(axis=2)    # (T, N)
    return pd.DataFrame(out, index=s_df.index, columns=s_df.columns)

# 1D MLP fit (Sharpe objective)
def fit_mlp1d_sharpe(
    s_df, y_df,
    train_frac=0.6,
    val_frac=0.2,
    n_hidden=16,
    epochs=40,
    batch_size=200000,
    lr=1e-3,
    weight_decay=1e-3,
    seed=42,
    max_samples=2_000_000,   # cap pooled samples for speed
    device=None,
    scale_mode="unit_var",   # "unit_var" or "match_s" or None
):
    """
    Learns w1,w2 for f(s)=sum w2*tanh(w1*s) by maximizing Sharpe of r_strat = f(s)*y.
    Pools across all assets (stacked) within each time slice.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- chronological split by time index (rows)
    T = len(s_df)
    t_train = int(train_frac * T)
    t_val = int((train_frac + val_frac) * T)

    s_tr = s_df.iloc[:t_train].to_numpy(dtype=np.float32).ravel()
    y_tr = y_df.iloc[:t_train].to_numpy(dtype=np.float32).ravel()

    s_va = s_df.iloc[t_train:t_val].to_numpy(dtype=np.float32).ravel()
    y_va = y_df.iloc[t_train:t_val].to_numpy(dtype=np.float32).ravel()

    # --- drop NaNs / inf
    def _clean(s, y):
        m = np.isfinite(s) & np.isfinite(y)
        return s[m], y[m]

    s_tr, y_tr = _clean(s_tr, y_tr)
    s_va, y_va = _clean(s_va, y_va)

    # --- optional subsample for speed
    rng = np.random.default_rng(seed)
    if len(s_tr) > max_samples:
        idx = rng.choice(len(s_tr), size=max_samples, replace=False)
        s_tr, y_tr = s_tr[idx], y_tr[idx]
    if len(s_va) > max_samples // 4:
        idx = rng.choice(len(s_va), size=max_samples // 4, replace=False)
        s_va, y_va = s_va[idx], y_va[idx]

    # --- odd symmetry augmentation: (s,y) plus (-s,-y)
    s_tr = np.concatenate([s_tr, -s_tr], axis=0)
    y_tr = np.concatenate([y_tr, -y_tr], axis=0)

    # --- torch tensors
    torch.manual_seed(seed)
    s_tr_t = torch.from_numpy(s_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    s_va_t = torch.from_numpy(s_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    # --- parameters (no bias)
    w1 = torch.nn.Parameter(0.1 * torch.randn(n_hidden, device=device))
    w2 = torch.nn.Parameter(0.1 * torch.randn(n_hidden, device=device))

    opt = torch.optim.AdamW([w1, w2], lr=lr, weight_decay=weight_decay)

    def f_map(s):
        # s: (B,)
        h = torch.tanh(s[:, None] * w1[None, :])          # (B,H)
        return (h * w2[None, :]).sum(dim=1)              # (B,)

    def neg_sharpe(rs, eps=1e-8):
        num = rs.sum()
        den = torch.sqrt((rs * rs).sum() + eps)
        return -(num / den)

    best = {"score": -1e18, "w1": None, "w2": None}

    n = s_tr_t.shape[0]
    for ep in range(epochs):
        # minibatch shuffle
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            j = idx[start:start + batch_size]
            sb = s_tr_t[j]
            yb = y_tr_t[j]

            rs = f_map(sb) * yb
            loss = neg_sharpe(rs)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation score
        with torch.no_grad():
            rs_va = f_map(s_va_t) * y_va_t
            score = (-neg_sharpe(rs_va)).item()
            if score > best["score"]:
                best["score"] = score
                best["w1"] = w1.detach().cpu().numpy().copy()
                best["w2"] = w2.detach().cpu().numpy().copy()

    # --- post scaling (train-only) to stabilize magnitude
    w1b, w2b = best["w1"], best["w2"]
    f_tr = np.tanh(s_tr[:len(s_tr)//2][:, None] * w1b[None, :]) @ w2b  # de-aug first half approx ok
    if scale_mode == "unit_var":
        scale = 1.0 / (np.nanstd(f_tr) + 1e-12)
    elif scale_mode == "match_s":
        scale = (np.nanstd(s_tr[:len(s_tr)//2]) / (np.nanstd(f_tr) + 1e-12))
    else:
        scale = 1.0

    return {
        "w1": w1b,
        "w2": w2b,
        "scale": float(scale),
        "val_score": float(best["score"]),
        "device": device,
    }