import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Callable
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna

# 1. Machine Learning Related Utilities

## 1.1 Returns predictions indexed by (date,ticker) for all test dates where training is possible given a certain model
def walkforward_predict(
    data: pd.DataFrame,
    date_column,
    model_builder: Callable[[], object],
    lookback: int = 252,
    min_train_days: int = 60,
    min_n =30
) -> pd.Series:
    all_dates = data.index.get_level_values(date_column).unique().sort_values()
    preds = []

    # pre-split by date for speed
    by_date = {d: data.xs(d, level=date_column) for d in all_dates}

    for i, d_test in enumerate(all_dates):
        train_dates = all_dates[max(0, i - lookback): i]  # strictly before test date
        if len(train_dates) < min_train_days:
            continue

        # stack train
        train_df = pd.concat([by_date[d] for d in train_dates], axis=0)
        test_df = by_date[d_test]

        # Ensure enough names in test cross-section
        if len(test_df) < min_n:
            continue

        X_tr = train_df.drop(columns=["y"]).to_numpy(dtype=float)
        y_tr = train_df["y"].to_numpy(dtype=float)
        X_te = test_df.drop(columns=["y"]).to_numpy(dtype=float)

        model = model_builder()
        model.fit(X_tr, y_tr)
        yhat = model.predict(X_te)

        p = pd.Series(yhat, index=pd.MultiIndex.from_product([[d_test], test_df.index], names=["date", "ticker"]))
        preds.append(p)

    if not preds:
        return pd.Series(dtype=float)

    return pd.concat(preds).sort_index()

## 1.2 Fold Dates for train-test split
def make_date_folds(dates, n_folds=4, val_len=120, gap=0):
    dates = pd.Index(pd.to_datetime(dates)).sort_values()
    folds = []
    end_points = np.linspace(val_len + gap + 300, len(dates), n_folds+1, dtype=int)[1:]
    for end in end_points:
        val_end = min(end, len(dates))
        val_start = max(0, val_end - val_len)
        train_end = max(0, val_start - gap)
        train_dates = dates[:train_end]
        val_dates = dates[val_start:val_end]
        if len(train_dates) >= 300 and len(val_dates) >= 40:
            folds.append((train_dates, val_dates))
    return folds

## 1.3 Build long-format dataset
def panel_to_long(X, y, M,date_column,ticker_column):
    X_long = X.stack(level=ticker_column)
    X_long.index.names = [date_column, ticker_column]

    y_long = y.stack()
    y_long.index.names = [date_column, ticker_column]
    y_long.name = "y"

    m_long = M.stack()
    m_long.index.names = [date_column, ticker_column]
    m_long.name = "m"

    df = X_long.join(y_long, how="inner").join(m_long, how="inner")
    df = df[df["m"]].drop(columns=["m"])
    df = df.dropna()  # drop rows with any missing feature or y
    return df

## 1.4 Build X and y and group sizes by date in row order
def build_Xy_groups(df, dates, feat_cols, y_col, date_col, ticker_col):
    sub = df.loc[df.index.get_level_values(date_col).isin(dates)].copy()
    sub = sub.sort_index(level=[date_col, ticker_col])

    X = sub[feat_cols].values
    y = sub[y_col].values

    counts = sub.groupby(level=date_col).size().values.astype(int)
    idx = sub.index
    return X, y, counts, idx

## 1.5 Cache the Data once
def make_date_cache(df, feat_cols, date_col, y_rel_col, y_eval_col):
    dates = df.index.get_level_values(date_col).unique().sort_values()
    X_by_date = {}
    yrel_by_date = {}
    yeval_by_date = {}
    idx_by_date = {}
    n_by_date = {}

    for d in dates:
        g = df.xs(d, level=date_col).copy()
        # g index is ticker
        X = g[feat_cols].values.astype(np.float32)
        yrel = g[y_rel_col].values.astype(np.float32)
        yeval = g[y_eval_col].values.astype(np.float32)

        X_by_date[d] = X
        yrel_by_date[d] = yrel
        yeval_by_date[d] = yeval
        idx_by_date[d] = g.index  # tickers for reconstruction
        n_by_date[d] = len(g)

    return dates, X_by_date, yrel_by_date, yeval_by_date, idx_by_date, n_by_date

## 1.6 Build window from cache
def build_window_from_cache(date_list, X_by_date, y_by_date):
    Xs = [X_by_date[d] for d in date_list]
    ys = [y_by_date[d] for d in date_list]
    groups = np.array([len(y_by_date[d]) for d in date_list], dtype=np.int32)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y, groups

## 1.7 IC Objective Function
def daily_ic_long(pred, y, date_col, min_n=30):
    df = pd.DataFrame({"pred": pred, "y": y}).dropna()
    out = {}
    for d, g in df.groupby(level=date_col):
        if len(g) >= min_n:
            if g["pred"].nunique() <= 1 or g["y"].nunique() <= 1:
                out[d] = 0.0  # treat degenerate as 0 instead of dropping the day
            else:
                out[d] = g["pred"].corr(g["y"], method="spearman")
    return pd.Series(out).sort_index()

# 2. Data Engineerers

## 2.1 for a given folder with equities price-volume data, aggrgate into a long panel
def build_wide_panel(
    data_dir,
    value_col,
    date_col="DateTime",
    pv_col_map,
    keep="last",            # how to resolve duplicate dates in a ticker file
    min_date=None,
    max_date=None,
):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))

    col_en = pv_col_map.get(value_col, value_col)
    chunks = []

    for fp in files:
        ticker = fp.stem.strip().upper()  # e.g. A.csv -> A
        df = pd.read_csv(fp)

        x = df[[date_col, col_en]].copy()
        x = x.rename(columns={date_col: "Date", col_en: "Value"})

        x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
        x["Value"] = pd.to_numeric(x["Value"], errors="coerce")
        x["Ticker"] = ticker

        x = x.dropna(subset=["Date", "Value"])
        x = x.sort_values("Date").drop_duplicates(subset=["Date"], keep=keep)

        if min_date is not None:
            x = x[x["Date"] >= pd.to_datetime(min_date)]
        if max_date is not None:
            x = x[x["Date"] <= pd.to_datetime(max_date)]

        chunks.append(x[["Date", "Ticker", "Value"]])
    long_df = pd.concat(chunks, ignore_index=True)
    wide = (
        long_df
        .pivot(index="Date", columns="Ticker", values="Value")
        .sort_index()
        .sort_index(axis=1)
    )
    wide.index.name = "Date"
    wide.columns.name = "Ticker"
    return wide