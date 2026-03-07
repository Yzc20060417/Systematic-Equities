import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# 1. Basic Helpers

## 1.1 X_row is a Series indexed by MultiIndex (alpha, ticker). Return DataFrame index=ticker, cols=alpha.
def _row_to_matrix(X_row,alpha_panel) -> pd.DataFrame:
    return X_row.unstack(alpha_panel).sort_index(axis=1)

## 1.2 For exposures, apply same sign so "positive score" always means expected positive return
def apply_sign(X_panel, alpha_panel, sign):
    X = X_panel.copy()
    for a, s in sign.items():
        if a in X.columns.get_level_values(alpha_panel):  # Alpha panel name
            X.loc[:, (a, slice(None))] = X.loc[:, (a, slice(None))] * float(s)
    return X

## 1.3 cs winsorization
def _winsorize_by_row(df, limits=(0.01, 0.99)):
    lo, hi = limits
    q_lo = df.quantile(lo, axis=1)
    q_hi = df.quantile(hi, axis=1)
    return df.clip(lower=q_lo, upper=q_hi, axis=0)

## 1.4 cs zscore
def _zscore_by_row(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)

# 1.5 Robust quantile binning per date using percentile ranks (handles ties + avoids qcut failures). Returns integers 1..q, NaN where factor is NaN
def _factor_quantiles_by_row(factor, quantiles= 5):
    q = quantiles
    pct = factor.rank(axis=1, method="first", pct=True)
    bins = np.ceil(pct * q).clip(1, q)
    bins = bins.where(factor.notna())
    return bins.astype("float")

## 1.6 Normalizing Stock Code: 000001.SZ -> 000001 ; 1 -> 000001 ; '000001' -> 000001
def _norm_code(x) -> str:
    s = str(x).strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"\D", "", s)
    if len(s) == 0:
        return s
    return s.zfill(6)

## 1.7 Ensure datetime index (if date_col given or index looks like date) and 6-digit string code columns
def normalize_panel_codes(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if date_col is not None and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col)

    # if index is not datetime but looks like date strings, convert
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass

    # normalize columns
    new_cols = []
    for c in out.columns:
        # keep non-code columns
        if isinstance(c, str) and c.lower() in {"date"}:
            new_cols.append(c)
        else:
            new_cols.append(_norm_code(c))
    out.columns = new_cols
    return out

## 1.8 Forward-fill monthly values to daily frequency on the daily_index
def expand_monthly_to_daily(monthly_df: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    return monthly_df.reindex(daily_index, method="ffill")

## 1.9 Reindex to daily_index x codes and mask out non-members -> NaN.
def align_and_mask_daily_panel(df: pd.DataFrame, daily_index: pd.DatetimeIndex,
                               codes: list[str], comp_daily: pd.DataFrame) -> pd.DataFrame:
    out = df.reindex(index=daily_index, columns=codes)
    mask = comp_daily.reindex(index=daily_index, columns=codes).astype("float")
    return out.where(mask == 1)

## 1.10 If columns are MultiIndex with constant metadata levels, drop those levels
def simplify_columns(df):
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        keep_levels = []
        for i in range(out.columns.nlevels):
            if out.columns.get_level_values(i).nunique() > 1:
                keep_levels.append(i)
        if len(keep_levels) == 1:
            out.columns = out.columns.get_level_values(keep_levels[0])
        else:
            out.columns = pd.Index(["|".join(map(str, tup)) for tup in out.columns.to_list()])
    out.columns = pd.Index([str(c).strip() for c in out.columns])
    return out

## 1.11 Enusre for a df the datetime columns
def ensure_datetime_index(df):
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out

## 1.12 Ensure Numeric
def to_numeric(df):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

## 1.13 Give Statistical Summarization of a Series
def summarize_series(s: pd.Series, name="metric") -> pd.Series:
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    return pd.Series({
        f"{name}_mean": mu,
        f"{name}_std": sd,
        f"{name}_ir": (mu / sd) if sd > 0 else np.nan,
        f"{name}_n": int(s.notna().sum()),
    })

# 2. Collinearity Check Utilities

## 2.1 Row-wise Pearson corr between X and Y
def _rowwise_corr_nan(X, Y, min_n: int = 5):
    valid = np.isfinite(X) & np.isfinite(Y)
    n = valid.sum(axis=1).astype(float)

    out = np.full(X.shape[0], np.nan, dtype=float)
    ok = n >= max(min_n, 2)
    if not ok.any():
        return out
    # sums
    Xv = np.where(valid, X, 0.0)
    Yv = np.where(valid, Y, 0.0)

    mx = Xv.sum(axis=1) / np.where(n == 0, np.nan, n)
    my = Yv.sum(axis=1) / np.where(n == 0, np.nan, n)

    X0 = np.where(valid, X - mx[:, None], 0.0)
    Y0 = np.where(valid, Y - my[:, None], 0.0)

    # sample covariance / variance
    denom = np.where(n > 1, (n - 1.0), np.nan)
    cov = (X0 * Y0).sum(axis=1) / denom
    vx = (X0 * X0).sum(axis=1) / denom
    vy = (Y0 * Y0).sum(axis=1) / denom

    corr = cov / np.sqrt(vx * vy)
    out[ok] = corr[ok]
    return out

## 2.2 Row-wise Spearman corr across tickers between xi[t,:] and xj[t,:]
def _cs_spearman_corr(xi, xj, min_n=30):
    # align
    xi, xj = xi.align(xj, join="inner", axis=0)
    corrs = []
    dates = xi.index
    for t in dates:
        a = xi.loc[t]
        b = xj.loc[t]
        m = a.notna() & b.notna()
        if m.sum() < min_n:
            corrs.append(np.nan)
        else:
            corrs.append(a[m].corr(b[m], method="spearman"))
    return pd.Series(corrs, index=dates)

## 2.3 Build Correlation Matrix for a factor pool
def factor_corr_matrix(alpha_rank, univ, alphas, date_mask, viable=None, min_n=30):
    alphas = list(alphas)
    C = pd.DataFrame(np.nan, index=alphas, columns=alphas)
    N = pd.DataFrame(0, index=alphas, columns=alphas, dtype=int)
    # pre-mask
    X = {a: alpha_rank[a].loc[date_mask].where(univ.loc[date_mask]) for a in alphas}

    for a in alphas:
        C.loc[a, a] = 1.0
        N.loc[a, a] = int(date_mask.sum())
    for a, b in combinations(alphas, 2):
        xi, xj = X[a], X[b]

        # apply viability intersection if provided
        if viable is not None:
            v = viable[a].loc[date_mask] & viable[b].loc[date_mask]
            xi = xi.loc[v]
            xj = xj.loc[v]

        s = _cs_spearman_corr(xi, xj, min_n=min_n).dropna()
        C.loc[a, b] = C.loc[b, a] = float(s.mean()) if len(s) else np.nan
        N.loc[a, b] = N.loc[b, a] = int(len(s))

    return C, N

## 2.4 Return the top correlation pairs above the threshold
def top_corr_pairs(C, N, min_abs_corr=0.3, min_days=756, topk=30):
    alphas = list(C.index)
    out = []
    for i in range(len(alphas)):
        for j in range(i+1, len(alphas)):
            a, b = alphas[i], alphas[j]
            cij = C.loc[a, b]
            nij = N.loc[a, b]
            if pd.isna(cij): 
                continue
            if abs(cij) >= min_abs_corr and nij >= min_days:
                out.append((a, b, float(cij), int(nij)))
    out.sort(key=lambda x: abs(x[2]), reverse=True)
    return pd.DataFrame(out, columns=["alpha_i","alpha_j","mean_spearman","N_days"]).head(topk)

## 2.5 Plot Eigenvalues of a Correlation matrix C, use a eps to ensure division
def plot_eigenspectrum(C,eps = 1e-12):
    good = C.notna().all(axis=1)
    C0 = C.loc[good, good].copy()
    # If any tiny NaNs remain, fill with 0 as a conservative default
    C0 = C0.fillna(0.0)
    # Ensure symmetry
    C0 = (C0 + C0.T) / 2
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(C0.values)
    eigvals_sorted = np.sort(eigvals)[::-1]

    cond = float(eigvals_sorted[0] / max(eigvals_sorted[-1], eps))

    # Effective rank (entropy-based)
    p = np.clip(eigvals_sorted, 0, None)
    p = p / (p.sum() + eps)
    eff_rank = float(np.exp(-(p * np.log(p + eps)).sum()))

    print(f"#factors used: {C0.shape[0]}")
    print(f"min eig: {eigvals_sorted[-1]:.6g}  max eig: {eigvals_sorted[0]:.6g}")
    print(f"condition number: {cond:.2f}")
    print(f"effective rank: {eff_rank:.2f}")

    plt.figure()
    plt.plot(eigvals_sorted, marker="o")
    plt.title("Eigenvalues of factor correlation matrix")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.show()

    return {
        "factors_used": C0.shape[0],
        "Min Eigenvalue": eigvals_sorted[-1],
        "Max Eigenvalue": eigvals_sorted[0],
        "Condition Number": cond,
        "Effective Rank": eff_rank
    }

## 2.6 Estimate daily factor returns via cross-sectional OLS per day.
## X_panel: DataFrame indexed by date, columns MultiIndex (alpha,ticker)
## R_fwd:   DataFrame indexed by date, columns ticker
## sample_mask: boolean DataFrame indexed by date, columns ticker
def fama_macbeth_factor_returns(X_panel: pd.DataFrame,
                               R_fwd: pd.DataFrame,
                               sample_mask: pd.DataFrame,
                               alpha_panel,
                               factors: list[str],
                               add_intercept: bool = True,
                               min_nobs: int = 30,
                               fill_missing_exposure_with_zero: bool = True):
    dates = X_panel.index.intersection(R_fwd.index).intersection(sample_mask.index)
    betas = []
    meta = []

    for t in dates:
        y = R_fwd.loc[t]
        m = sample_mask.loc[t] & y.notna()
        if m.sum() < min_nobs:
            continue

        X_mat = _row_to_matrix(X_panel.loc[t],alpha_panel)
        X_mat = X_mat.reindex(index=y.index, columns=factors)
        X_mat = X_mat.loc[m.index[m]]
        y_t = y.loc[m.index[m]].astype(float)

        # handle missing exposures
        if fill_missing_exposure_with_zero:
            X_mat = X_mat.fillna(0.0)
        else:
            good = X_mat.notna().all(axis=1)
            X_mat = X_mat.loc[good]
            y_t = y_t.loc[good]
        if len(y_t) < min_nobs:
            continue

        # design matrix
        X_np = X_mat.values
        if add_intercept:
            X_np = np.column_stack([np.ones(len(y_t)), X_np])

        # OLS
        beta, *_ = np.linalg.lstsq(X_np, y_t.values, rcond=None)

        if add_intercept:
            alpha0 = beta[0]
            b = beta[1:]
        else:
            alpha0 = np.nan
            b = beta

        # R^2
        y_hat = X_np @ beta
        ss_res = np.sum((y_t.values - y_hat) ** 2)
        ss_tot = np.sum((y_t.values - y_t.values.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        betas.append(pd.Series(b, index=factors, name=t))
        meta.append({"date": t, "nobs": int(len(y_t)), "r2": float(r2), "intercept": float(alpha0)})

    B = pd.DataFrame(betas).sort_index()
    stats = pd.DataFrame(meta).set_index("date").sort_index()
    return B, stats

## 2.7 Plot Cluster-Ordered Heatmap
def plot_heatmap(C):
    # Distance matrix
    C0 = C.copy()
    D = 1.0 - np.abs(C0.values)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    order = leaves_list(Z)
    ordered = C0.iloc[order, order]

    # Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(ordered.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Mean daily Spearman corr")
    plt.xticks(range(len(ordered.columns)), ordered.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(ordered.index)), ordered.index, fontsize=7)
    plt.title("Factor correlation Heatmap")
    plt.tight_layout()
    plt.show()

# 3 Factors Metrics

## 3.1 NW T-stat Test for a returns series, L should roughly equal to H/2H where H is your forward window
def newey_west_tstat(series: pd.Series, L: int = 5):
    x = series.dropna().values
    T = len(x)
    if T < 10:
        return np.nan
    mu = x.mean()
    y = x - mu
    gamma0 = np.dot(y, y) / T
    var = gamma0
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        gam = np.dot(y[l:], y[:-l]) / T
        var += 2 * w * gam
    se = np.sqrt(var / T) if var > 0 else np.nan
    return mu / se if (se is not None and se > 0) else np.nan

## 3.2 Compute daily ic between two df
def daily_ic(score_df: pd.DataFrame, ret_df: pd.DataFrame, method="spearman", min_n=30):
    ics = []
    for t in score_df.index.intersection(ret_df.index):
        s = score_df.loc[t]
        r = ret_df.loc[t]
        m = s.notna() & r.notna()
        if m.sum() >= min_n:
            ics.append((t, s[m].corr(r[m], method=method)))
    return pd.Series(dict(ics)).sort_index()

## 3.3 Compute ICIR from IC
def icir(ic: pd.Series):
    mu = float(ic.mean())
    sd = float(ic.std(ddof=1))
    return mu / sd if sd > 0 else -1e9

## 3.4 Simple top-bottom quantile spread in returns using predictions for ranking.
def daily_ls_spread(pred: pd.Series, y: pd.Series, q=0.2, min_names=30) -> pd.Series:
    df = pd.DataFrame({"pred": pred, "y": y}).dropna()
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) < min_names:
            continue
        g = g.sort_values("pred")
        n = len(g)
        k = int(np.floor(q * n))
        if k < 1:
            continue
        bot = g.iloc[:k]["y"].mean()
        top = g.iloc[-k:]["y"].mean()
        out[d] = float(top - bot)
    return pd.Series(out).sort_index()

## 3.5 Compute score per ticker from one day's (alpha,ticker) exposures and alpha weights.
def score_from_weights(X_row: pd.Series, w: pd.Series, alpha_panel) -> pd.Series:
    Xmat = X_row.unstack(alpha_panel).reindex(columns=w.index)
    Xmat = Xmat.fillna(0.0)
    s = Xmat.values @ w.values
    return pd.Series(s, index=Xmat.index)

## 3.6 Generalized Factor Metrics Analytics
def compute_alpha_metrics_wide(
    alpha_df: pd.DataFrame,
    close_df: pd.DataFrame,
    periods=(1, 5, 10, 20),
    quantiles: int = 5,
    ic_method: str = "spearman",          # "spearman" (rank IC) or "pearson"
    min_assets: int = 50,
    winsorize_alpha: tuple | None = (0.01, 0.99),
    winsorize_fwdret: tuple | None = None,
    zscore_alpha: bool = False,
    annualization: int = 252,             # for ICIR scaling
    drop_extreme_prices: bool = True,     # treat <=0 as missing
):
    """
    Wide-panel alpha analytics akin to core Alphalens outputs.
    Returns dict with:
      - factor: aligned factor panel (post-processing)
      - prices: aligned close prices
      - fwd_returns: {p: DataFrame}
      - ic: DataFrame (dates x periods)
      - ic_summary: DataFrame (periods x stats)
      - quantiles: DataFrame (dates x assets) with 1..Q
      - qret: {p: DataFrame} (dates x quantiles) mean fwd return per quantile
      - qret_mean: {p: Series} time-average quantile returns
      - spread: {p: Series} (top - bottom) daily
      - spread_summary: DataFrame (periods x stats)
      - coverage: Series (#assets used per date)
    """
    common_dates = alpha_df.index.intersection(close_df.index)
    common_assets = alpha_df.columns.intersection(close_df.columns)
    factor = alpha_df.loc[common_dates, common_assets].astype(float)
    prices = close_df.loc[common_dates, common_assets].astype(float)

    if drop_extreme_prices:
        prices = prices.where(prices > 0)
    if winsorize_alpha is not None:
        factor = _winsorize_by_row(factor, winsorize_alpha)
    if zscore_alpha:
        factor = _zscore_by_row(factor)

    # Coverage diagnostics (per date)
    coverage = factor.notna().sum(axis=1)

    # Forward returns
    fwd_returns = {}
    for p in periods:
        fwd = prices.pct_change(p).shift(-p)
        # mask where factor is missing
        fwd = fwd.where(factor.notna())
        if winsorize_fwdret is not None:
            fwd = _winsorize_by_row(fwd, winsorize_fwdret)
        fwd_returns[p] = fwd

    # IC series
    ic = pd.DataFrame(index=factor.index, columns=list(periods), dtype=float)
    X = factor.to_numpy()

    if ic_method.lower() == "spearman":
        X_ic = factor.rank(axis=1).to_numpy()
    elif ic_method.lower() == "pearson":
        X_ic = X
    else:
        raise ValueError("ic_method must be 'spearman' or 'pearson'")

    for p in periods:
        Y = fwd_returns[p].to_numpy()
        if ic_method.lower() == "spearman":
            Y_ic = fwd_returns[p].rank(axis=1).to_numpy()
        else:
            Y_ic = Y

        ic[p] = _rowwise_corr_nan(X_ic, Y_ic, min_n=min_assets)

    # IC summary
    ic_summary = []
    for p in periods:
        s = ic[p].dropna()
        n = len(s)
        mu = s.mean() if n else np.nan
        sd = s.std(ddof=1) if n > 1 else np.nan
        icir = (mu / sd) * np.sqrt(annualization) if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else np.nan
        tstat = (mu / (sd / np.sqrt(n))) if (n > 1 and np.isfinite(sd) and sd > 0) else np.nan
        ic_summary.append({
            "period": p,
            "n": n,
            "mean": mu,
            "std": sd,
            "ICIR": icir,
            "tstat": tstat,
            "skew": s.skew() if n > 2 else np.nan,
            "kurt": s.kurt() if n > 3 else np.nan,
            "p05": s.quantile(0.05) if n else np.nan,
            "p50": s.quantile(0.50) if n else np.nan,
            "p95": s.quantile(0.95) if n else np.nan,
        })
    ic_summary = pd.DataFrame(ic_summary).set_index("period")

    # Factor quantiles + Quantile forward returns
    q_labels = _factor_quantiles_by_row(factor, quantiles=quantiles)

    qret = {}
    qret_mean = {}
    spread = {}
    spread_summary_rows = []

    q_arr = q_labels.to_numpy()

    for p in periods:
        fwd = fwd_returns[p].to_numpy()
        out = np.full((factor.shape[0], quantiles), np.nan, dtype=float)

        for qi in range(1, quantiles + 1):
            m = (q_arr == qi) & np.isfinite(fwd)
            with np.errstate(invalid="ignore"):
                out[:, qi - 1] = np.where(m, fwd, np.nan).mean(axis=1)

        qret_df = pd.DataFrame(out, index=factor.index, columns=[f"Q{qi}" for qi in range(1, quantiles + 1)])
        qret[p] = qret_df
        qret_mean[p] = qret_df.mean(axis=0, skipna=True)

        spr = qret_df[f"Q{quantiles}"] - qret_df["Q1"]
        spread[p] = spr

        spr_s = spr.dropna()
        n = len(spr_s)
        mu = spr_s.mean() if n else np.nan
        sd = spr_s.std(ddof=1) if n > 1 else np.nan
        sharpe = (mu / sd) * np.sqrt(annualization) if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else np.nan
        tstat = (mu / (sd / np.sqrt(n))) if (n > 1 and np.isfinite(sd) and sd > 0) else np.nan

        spread_summary_rows.append({
            "period": p,
            "n": n,
            "mean_spread": mu,
            "std_spread": sd,
            "sharpe_spread": sharpe,
            "tstat_spread": tstat,
        })

    spread_summary = pd.DataFrame(spread_summary_rows).set_index("period")

    return {
        "factor": factor,
        "prices": prices,
        "fwd_returns": fwd_returns,
        "ic": ic,
        "ic_summary": ic_summary,
        "quantiles": q_labels,
        "qret": qret,
        "qret_mean": qret_mean,
        "spread": spread,
        "spread_summary": spread_summary,
        "coverage": coverage,
    }