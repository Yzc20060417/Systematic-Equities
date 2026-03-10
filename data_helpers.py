# Data Helpers
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM
import statsmodels.api as sm

# 1. Basic Helpers

## 1.1 Rolling z-score using only local history. Weekly data: 104 ~= 2 years, 52 ~= 1 year minimum.
def rolling_zscore(df, window = 104, min_periods = 52, clip: float | None = 4.0):
    mu = df.rolling(window=window, min_periods=min_periods).mean()
    sd = df.rolling(window=window, min_periods=min_periods).std()
    z = (df - mu) / sd.replace(0, np.nan)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z

etf_path_dir = Path.home() / 'Desktop' / 'Data' / 'ETF' / 'ETF'

## 1.2 Read one ETF/stock csv and return a daily price series.
def _load_price_series(file_path):
    df = pd.read_csv(file_path)

    date_col = 'DateTime'
    price_col = 'Close'

    s = (
        df[[date_col, price_col]]
        .rename(columns={date_col: "date", price_col: "price"})
        .dropna(subset=["date", "price"])
        .copy()
    )

    s["date"] = pd.to_datetime(s["date"])
    s["price"] = pd.to_numeric(s["price"], errors="coerce")
    s = s.dropna(subset=["price"]).sort_values("date")
    s = s.drop_duplicates(subset=["date"], keep="last")
    s = s.set_index("date")["price"]

    s.name = file_path.stem.upper()
    return s

## 1.3 Load daily price panel from folder of csv files.
def load_prices(folder, tickers_set):
    series_list = []

    for file_path in sorted(folder.glob("*.csv")):
        ticker = file_path.stem.upper()
        if tickers_set is not None and ticker not in tickers_set:
            continue
        s = _load_price_series(file_path)
        series_list.append(s)

    prices_daily = pd.concat(series_list, axis=1).sort_index()
    return prices_daily

## 1.4 Convert daily price panel to weekly prices. If research_index is provided, final outputs are aligned exactly to it.
def daily_to_weekly(prices_daily, research_index=None, freq="W-FRI"):
    prices_daily = prices_daily.sort_index()
    weekly_prices = prices_daily.resample(freq).last().ffill()

    if research_index is not None:
        research_index = pd.DatetimeIndex(research_index).sort_values()
        weekly_prices = weekly_prices.reindex(research_index).ffill()
    return weekly_prices

## 1.5 Convert a daily/irregular level dataframe to weekly Friday level dataframe.
def _to_weekly_level_panel(df, research_index=None, freq="W-FRI", ffill=True):
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.resample(freq).last()

    if ffill:
        out = out.ffill()
    if research_index is not None:
        idx = pd.DatetimeIndex(research_index).sort_values()
        out = out.reindex(idx)
        if ffill:
            out = out.ffill()
    return out

## 1.6 Trim the raw panel to the first date where all required columns are available.
def trim_to_common_sample(df, required_cols=None):
    out = df.copy()
    cols = list(out.columns) if required_cols is None else list(required_cols)
    tmp = out[cols].dropna(how="any")
    if tmp.empty:
        raise ValueError("No common sample exists across the required columns.")

    start = tmp.index.min()
    out_common = out.loc[start:].copy()
    out_common = out.dropna(subset=cols, how="any").copy()
    return out_common, start

# 2. HMM Oriented

## 2.1 Return time span for certain regime
def run_lengths(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        return pd.DataFrame(columns=["state", "start_idx", "end_idx", "duration"])

    runs = []
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            runs.append({
                "state": labels[start],
                "start_idx": start,
                "end_idx": i - 1,
                "duration": i - start
            })
            start = i

    runs.append({
        "state": labels[start],
        "start_idx": start,
        "end_idx": len(labels) - 1,
        "duration": len(labels) - start
    })
    return pd.DataFrame(runs)

## 2.2 Reorder arbitrary state labels into a more interpretable order.
## Ordering score: macro_score = stress - growth + 0.25 * rates_pressure
## Lower score  -> easier / lower-stress regime
## Higher score -> more stressed / weaker-growth regime
def relabel_states_by_macro_order(X, labels, pressure_coeff = 0.25):
    X = X.copy()
    labels = pd.Series(labels, index=X.index, name="state")

    means = X.groupby(labels).mean()

    macro_score = (
        means["stress"]
        - means["growth"]
        + pressure_coeff * means["rates_pressure"]
    )

    old_order = macro_score.sort_values().index.tolist()
    mapping = {old_label: new_label for new_label, old_label in enumerate(old_order)}

    relabeled = labels.map(mapping).to_numpy()
    means_relabeled = means.rename(index=mapping).sort_index()

    return relabeled, mapping, means_relabeled

## 2.3 Regime summary using relabeled states.
def summarize_regimes(X, labels):
    labels = pd.Series(labels, index=X.index, name="state")
    means = X.groupby(labels).mean()
    stds = X.groupby(labels).std()
    counts = labels.value_counts().sort_index()
    occupancy = counts / len(labels)

    runs = run_lengths(labels.to_numpy())
    duration_stats = runs.groupby("state")["duration"].agg(
        avg_duration="mean",
        median_duration="median",
        max_duration="max",
        n_runs="count"
    )

    summary = means.copy()
    summary.columns = [f"{c}_mean" for c in summary.columns]

    for c in stds.columns:
        summary[f"{c}_std"] = stds[c]

    summary["n_obs"] = counts
    summary["occupancy"] = occupancy

    summary = summary.join(duration_stats, how="left")
    return summary.sort_index()

## 2.4 Approximate number of free parameters in Gaussian HMM.
def hmm_num_params(n_states, n_features, covariance_type="diag"):
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features

    if covariance_type == "diag":
        covars = n_states * n_features
    elif covariance_type == "full":
        covars = n_states * n_features * (n_features + 1) // 2
    else:
        raise ValueError("Only diag/full handled here.")

    return startprob + transmat + means + covars

## 2.5 Convert state sequence into contiguous date spans for plotting.
def contiguous_state_spans(index, labels):
    labels = np.asarray(labels)
    runs = run_lengths(labels)
    spans = []
    for _, row in runs.iterrows():
        spans.append({
            "state": int(row["state"]),
            "start": index[int(row["start_idx"])],
            "end": index[int(row["end_idx"])]
        })
    return spans

## 2.6 Fit Gaussian Mixture Matrix for a certain input of features
def fit_gmm_grid(X, ks=(2, 3, 4), n_init=50, random_state=42):
    Xv = X.values
    results = []
    models = {}
    for k in ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            random_state=random_state
        )
        gmm.fit(Xv)

        raw_labels = gmm.predict(Xv)
        labels, mapping, means = relabel_states_by_macro_order(X, raw_labels)
        probs = gmm.predict_proba(Xv)

        # reorder probability columns to match relabeled state order
        inv_map = {new: old for old, new in mapping.items()}
        probs = probs[:, [inv_map[i] for i in range(k)]]

        summary = summarize_regimes(X, labels)

        sil = np.nan
        if k > 1:
            sil = silhouette_score(Xv, labels)

        results.append({
            "model": "GMM",
            "k": k,
            "loglik": gmm.score(Xv) * len(X),
            "aic": gmm.aic(Xv),
            "bic": gmm.bic(Xv),
            "silhouette": sil
        })
        models[k] = {
            "model": gmm,
            "labels": pd.Series(labels, index=X.index, name=f"GMM_{k}"),
            "probs": pd.DataFrame(
                probs,
                index=X.index,
                columns=[f"state_{i}" for i in range(k)]
            ),
            "summary": summary,
            "means": means
        }
    results_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    return results_df, models

## 2.7 Fit HMM given features X and number of states k
def fit_best_hmm(X, k, covariance_type="diag", seeds=range(20), n_iter=2000, tol=1e-4):
    Xv = X.values

    best_model = None
    best_loglik = -np.inf
    best_seed = None

    for seed in seeds:
        try:
            hmm = GaussianHMM(
                n_components=k,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=seed
            )
            hmm.fit(Xv)
            ll = hmm.score(Xv)

            if np.isfinite(ll) and ll > best_loglik:
                best_loglik = ll
                best_model = hmm
                best_seed = seed

        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"No HMM fit succeeded for k={k}")

    raw_labels = best_model.predict(Xv)
    labels, mapping, means = relabel_states_by_macro_order(X, raw_labels)

    probs = best_model.predict_proba(Xv)
    inv_map = {new: old for old, new in mapping.items()}
    probs = probs[:, [inv_map[i] for i in range(k)]]

    n_params = hmm_num_params(k, X.shape[1], covariance_type=covariance_type)
    bic = -2 * best_loglik + n_params * np.log(len(X))
    aic = -2 * best_loglik + 2 * n_params

    summary = summarize_regimes(X, labels)

    return {
        "model": best_model,
        "best_seed": best_seed,
        "loglik": best_loglik,
        "aic": aic,
        "bic": bic,
        "labels": pd.Series(labels, index=X.index, name=f"HMM_{k}"),
        "probs": pd.DataFrame(
            probs,
            index=X.index,
            columns=[f"state_{i}" for i in range(k)]
        ),
        "summary": summary,
        "means": means,
    }

## 2.8 Fit HMM among a grid of k
def fit_hmm_grid(X, ks=(2, 3, 4), covariance_type="diag", seeds=range(20), n_iter=2000, tol=1e-4):
    results = []
    models = {}
    for k in ks:
        out = fit_best_hmm(
            X=X,
            k=k,
            covariance_type=covariance_type,
            seeds=seeds,
            n_iter=n_iter,
            tol=tol
        )
        results.append({
            "model": "HMM",
            "k": k,
            "best_seed": out["best_seed"],
            "loglik": out["loglik"],
            "aic": out["aic"],
            "bic": out["bic"]
        })
        models[k] = out
    results_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    return results_df, models


## 2.9 Reorder raw HMM transition matrix into the relabeled macro order. hmm_out is one entry from hmm_models[k]
def reorder_transition_matrix(hmm_out):
    T_raw = hmm_out["model"].transmat_.copy()
    mapping = hmm_out["mapping"] if "mapping" in hmm_out else None
    if mapping is None:
        # fallback: assume already ordered
        T = T_raw
        order = list(range(T.shape[0]))
    else:
        # mapping: old_label -> new_label
        inv_map = {new: old for old, new in mapping.items()}
        order = [inv_map[i] for i in range(len(inv_map))]
        T = T_raw[np.ix_(order, order)]
    idx = [f"state_{i}" for i in range(T.shape[0])]
    T = pd.DataFrame(T, index=idx, columns=idx)
    return T, order

## 2.10 For a Markov chain, implied expected duration in state i is 1 / (1 - p_ii).
def implied_state_durations(T):
    pii = np.diag(T.values)
    out = pd.Series(1 / np.maximum(1 - pii, 1e-12), index=T.index, name="implied_duration")
    return out

## 2.11 Build a clean panel with state labels and posterior probabilities.

def make_regime_panel(X, hmm_out):
    panel = X.copy()
    panel["state"] = hmm_out["labels"]

    probs = hmm_out["probs"].copy()
    probs.columns = [f"p_state_{i}" for i in range(probs.shape[1])]
    panel = panel.join(probs, how="left")

    return panel

## 2.12 Fit best HMM on one training sample.
def fit_best_hmm_single_sample(X_train, k=3, covariance_type="diag",
                               seeds=range(10), n_iter=1000, tol=1e-4):
    Xv = X_train.values
    best_model = None
    best_loglik = -np.inf
    best_seed = None

    for seed in seeds:
        try:
            m = GaussianHMM(
                n_components=k,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=seed
            )
            m.fit(Xv)
            ll = m.score(Xv)

            if np.isfinite(ll) and ll > best_loglik:
                best_loglik = ll
                best_model = m
                best_seed = seed
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("No HMM fit succeeded on training sample.")
    return best_model, best_seed, best_loglik

## 2.13 Relabel states using macro order on the training sample.
## Returns mapping and reordered probabilities for the training sample.
def relabel_hmm_outputs(X_train, model):
    raw_labels = model.predict(X_train.values)
    relabeled, mapping, means_relabeled = relabel_states_by_macro_order(X_train, raw_labels)

    probs = model.predict_proba(X_train.values)
    inv_map = {new: old for old, new in mapping.items()}
    probs = probs[:, [inv_map[i] for i in range(len(inv_map))]]

    return relabeled, mapping, means_relabeled, probs

## 2.14 Probability smoothing before hard assignment
def smooth_state_probs(probs_oos, window=3):
    prob_cols = [c for c in probs_oos.columns if c.startswith("p_state_")]
    out = probs_oos.copy()
    out[prob_cols] = out[prob_cols].rolling(window=window, min_periods=1).mean()
    out["state"] = out[prob_cols].values.argmax(axis=1)
    return out

## 2.15 Expanding-window filtered state inference.
## For each date t >= min_train, fit on X[:t] and record P(state_t | X_1...X_t).
def expanding_filtered_hmm(
    X,
    k=3,
    min_train=156,          # 3 years of weekly data
    refit_every=1,          # set to 4 for monthly-ish refits if speed is an issue
    covariance_type="diag",
    seeds=range(10),
    n_iter=1000,
    tol=1e-4,
):
    X = X.dropna().copy()
    dates = X.index

    prob_list = []
    meta = []
    fitted_models = {}

    last_model = None
    last_mapping = None
    last_refit_idx = None

    for t in range(min_train - 1, len(X)):
        need_refit = (last_model is None) or ((t - (last_refit_idx or 0)) >= refit_every)

        X_train = X.iloc[:t + 1]

        if need_refit:
            model, best_seed, best_loglik = fit_best_hmm_single_sample(
                X_train=X_train,
                k=k,
                covariance_type=covariance_type,
                seeds=seeds,
                n_iter=n_iter,
                tol=tol
            )

            _, mapping, means_relabeled, probs_train = relabel_hmm_outputs(X_train, model)

            last_model = model
            last_mapping = mapping
            last_refit_idx = t

            fitted_models[dates[t]] = {
                "model": model,
                "mapping": mapping,
                "means": means_relabeled,
                "best_seed": best_seed,
                "loglik": best_loglik,
            }

        # use current model on training sample up to t
        probs = last_model.predict_proba(X_train.values)
        inv_map = {new: old for old, new in last_mapping.items()}
        probs = probs[:, [inv_map[i] for i in range(len(inv_map))]]

        p_t = probs[-1]
        state_t = int(np.argmax(p_t))

        row = pd.Series(
            p_t,
            index=[f"p_state_{i}" for i in range(k)],
            name=dates[t]
        )
        row["state"] = state_t
        prob_list.append(row)

        meta.append({
            "date": dates[t],
            "refit_date": dates[last_refit_idx],
            "refit_used": need_refit
        })

    probs_oos = pd.DataFrame(prob_list)
    probs_oos.index = pd.to_datetime(probs_oos.index)
    probs_oos["state"] = probs_oos["state"].astype(int)

    meta_df = pd.DataFrame(meta).set_index("date")
    meta_df.index = pd.to_datetime(meta_df.index)

    return probs_oos, meta_df, fitted_models

## 2.16 - full-sample HMM
## - pseudo-OOS expanding-window HMM
## - smoothed pseudo-OOS probabilities
## - OOS state feature means
def run_regime_workflow(
    X_model,
    k,
    refit_every=4,
    smooth_window=3,
    min_train=104,
    covariance_type="diag",
    seeds=range(20),
    n_iter=1500,
    tol=1e-4,
):
    X_model = X_model.dropna().copy()

    # Full-sample HMM
    hmm_results_df, hmm_models_dict = fit_hmm_grid(
        X_model,
        ks=(k,),
        covariance_type=covariance_type,
        seeds=seeds,
        n_iter=n_iter,
        tol=tol,
    )
    hmm_k = hmm_models_dict[k]

    # Transition matrix
    if "mapping" not in hmm_k:
        raw_labels = hmm_k["model"].predict(X_model.values)
        relabeled, mapping, means = relabel_states_by_macro_order(X_model, raw_labels)
        hmm_k["mapping"] = mapping
        hmm_k["means"] = means

    T_k, _ = reorder_transition_matrix(hmm_k)
    dur_k = implied_state_durations(T_k)

    # Pseudo-OOS
    probs_oos, meta_oos, fitted_oos = expanding_filtered_hmm(
        X=X_model,
        k=k,
        min_train=min_train,
        refit_every=refit_every,
        covariance_type=covariance_type,
        seeds=seeds,
        n_iter=n_iter,
        tol=tol,
    )

    probs_oos_s = smooth_state_probs(probs_oos, window=smooth_window)

    # OOS state feature means
    oos_means = oos_state_feature_means(X_model, probs_oos_s["state"])

    # OOS state duration summary
    oos_summary = summarize_state_sequence(probs_oos_s["state"])

    return {
        "X_model": X_model,
        "full_hmm_results": hmm_results_df,
        "full_hmm": hmm_k,
        "transition_matrix": T_k,
        "implied_duration": dur_k,
        "probs_oos_raw": probs_oos,
        "probs_oos_s": probs_oos_s,
        "meta_oos": meta_oos,
        "fitted_oos": fitted_oos,
        "oos_means": oos_means,
        "oos_summary": oos_summary,
    }

# 3. Regime Analyzers

## 3.1 Plot Regime timeline
def plot_regime_timeline(X, labels, title="Regime Timeline"):
    colors = plt.cm.Set2.colors
    spans = contiguous_state_spans(X.index, labels)

    fig, axes = plt.subplots(
        X.shape[1], 1,
        figsize=(14, 2.8 * X.shape[1]),
        sharex=True
    )

    if X.shape[1] == 1:
        axes = [axes]

    for ax, col in zip(axes, X.columns):
        for sp in spans:
            ax.axvspan(
                sp["start"], sp["end"],
                color=colors[sp["state"] % len(colors)],
                alpha=0.18
            )
        ax.plot(X.index, X[col], lw=1.5)
        ax.set_title(col)

    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

## 3.2 Plot the means of labels within the regimes
def plot_state_means(summary_df, title="State Conditional Means"):
    mean_cols = [c for c in summary_df.columns if c.endswith("_mean")]
    M = summary_df[mean_cols].copy()
    M.columns = [c.replace("_mean", "") for c in M.columns]

    fig, ax = plt.subplots(figsize=(8, 1.6 * len(M)))
    im = ax.imshow(M.values, aspect="auto", cmap="coolwarm")

    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels([f"state_{i}" for i in M.index])

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.show()

## 3.3 Plot the posterior probabilities for the hmm
def plot_hmm_posteriors(probs, title="HMM Posterior Probabilities"):
    fig, axes = plt.subplots(probs.shape[1], 1, figsize=(14, 2.2 * probs.shape[1]), sharex=True)

    if probs.shape[1] == 1:
        axes = [axes]

    for i, col in enumerate(probs.columns):
        axes[i].plot(probs.index, probs[col], lw=1.4)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].set_title(col)

    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

## 3.4 Plot the transition matrix
def plot_transition_matrix(T, title="Transition Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T.values, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(T.shape[1]))
    ax.set_xticklabels(T.columns)
    ax.set_yticks(range(T.shape[0]))
    ax.set_yticklabels(T.index)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax.text(j, i, f"{T.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.show()

## 3.5 Report of regimes
def compact_regime_report(summary_df):
    cols = [
        "growth_mean", "rates_pressure_mean", "stress_mean",
        "occupancy", "avg_duration", "median_duration", "n_runs"
    ]
    cols = [c for c in cols if c in summary_df.columns]
    return summary_df[cols].copy()

## 3.6 Plot OOS State Probabilities
def plot_oos_state_probs(probs_oos, title="Pseudo-OOS Filtered State Probabilities"):
    prob_cols = [c for c in probs_oos.columns if c.startswith("p_state_")]

    fig, axes = plt.subplots(len(prob_cols), 1, figsize=(14, 2.2 * len(prob_cols)), sharex=True)
    if len(prob_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, prob_cols):
        ax.plot(probs_oos.index, probs_oos[col], lw=1.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(col)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

## 3.7 Plot OOS State Timeline
def plot_oos_state_timeline(X, probs_oos, title="Pseudo-OOS State Timeline"):
    Xp = X.loc[probs_oos.index].copy()
    labels = probs_oos["state"].values

    plot_regime_timeline(Xp, labels, title=title)

## 3.8 Summarize the state occupancy and duration in OOS
def summarize_state_sequence(state_series):
    runs = run_lengths(state_series.astype(int).values)

    summary = pd.DataFrame({
        "n_obs": state_series.value_counts().sort_index()
    })
    summary["occupancy"] = summary["n_obs"] / len(state_series)

    duration_stats = runs.groupby("state")["duration"].agg(
        avg_duration="mean",
        median_duration="median",
        max_duration="max",
        n_runs="count"
    )

    summary = summary.join(duration_stats, how="left")
    return summary.sort_index()

## 3.9 Date Table of regime switches
def regime_switch_table(state_series):
    s = state_series.astype(int)
    switch = s != s.shift(1)

    out = pd.DataFrame({
        "date": s.index,
        "state": s.values,
        "prev_state": s.shift(1).values,
        "switch": switch.values
    })

    out = out.loc[out["switch"]].copy()
    return out.reset_index(drop=True)

## 3.10 Inspect means of features in regimes
def oos_state_feature_means(X_features, state_series):
    df = X_features.loc[state_series.index].copy()
    df["state"] = state_series
    return df.groupby("state").mean()

# 4. Asset Behavior Analyzers

## 4.1 Compute forward return stats by state for a single asset return series. Horizons are in weeks
def state_forward_return_stats(state_series, ret_series, horizons=(1, 4, 12), name="asset"):
    df = pd.DataFrame({
        "state": state_series,
        "ret": ret_series
    }).dropna()

    out = {}

    for h in horizons:
        fwd = (1 + df["ret"]).rolling(h).apply(np.prod, raw=True).shift(-h + 1) - 1
        tmp = pd.DataFrame({
            "state": df["state"],
            f"fwd_{h}w": fwd
        }).dropna()

        stats = tmp.groupby("state")[f"fwd_{h}w"].agg(
            mean="mean",
            median="median",
            std="std",
            hit_rate=lambda x: (x > 0).mean(),
            n_obs="count"
        )
        out[h] = stats

    return out

## 4.2 Forward compounded return over `horizon` weeks.
def make_forward_return(ret_series, horizon):
    return (1.0 + ret_series).rolling(horizon).apply(np.prod, raw=True).shift(-horizon + 1) - 1.0

## 4.3 Compute Max drawdown for forward returns
def max_drawdown_from_returns(ret_series):
    eq = (1.0 + ret_series.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()

## 4.4 Align states and returns date lable
def align_state_and_returns(state_series, ret_df):
    idx = pd.DatetimeIndex(state_series.index).intersection(pd.DatetimeIndex(ret_df.index))
    state_series = state_series.loc[idx].astype(int)
    ret_df = ret_df.loc[idx].copy()
    return state_series, ret_df

## 4.5 Compute forward return statistics by hard regime state for each asset.
def regime_forward_return_stats(
    state_series,
    asset_returns,
    horizons=(1, 4, 12),
):
    state_series = state_series.copy()
    asset_returns = asset_returns.copy()

    idx = state_series.index.intersection(asset_returns.index)
    state_series = state_series.loc[idx]
    asset_returns = asset_returns.loc[idx]

    results = {}

    for asset in asset_returns.columns:
        asset_res = {}
        for h in horizons:
            fwd = make_forward_return(asset_returns[asset], h)
            tmp = pd.DataFrame({
                "state": state_series,
                "fwd_ret": fwd
            }).dropna()
            stats = tmp.groupby("state")["fwd_ret"].agg(
                mean="mean",
                median="median",
                std="std",
                hit_rate=lambda x: (x > 0).mean(),
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                n_obs="count"
            )
            asset_res[h] = stats
        results[asset] = asset_res
    return results

## 4.6 Probability-weighted forward returns for each state and asset.
## probs_df should contain columns like p_state_0, p_state_1, ...
def probability_weighted_forward_returns(
    probs_df,
    asset_returns,
    horizons=(1, 4, 12),
):
    prob_cols = [c for c in probs_df.columns if c.startswith("p_state_")]
    idx = probs_df.index.intersection(asset_returns.index)
    probs_df = probs_df.loc[idx]
    asset_returns = asset_returns.loc[idx]

    results = {}

    for asset in asset_returns.columns:
        asset_res = {}
        for h in horizons:
            fwd = make_forward_return(asset_returns[asset], h)
            tmp = probs_df.copy()
            tmp["fwd_ret"] = fwd
            tmp = tmp.dropna()
            rows = {}
            for pcol in prob_cols:
                w = tmp[pcol]
                x = tmp["fwd_ret"]
                wsum = w.sum()
                mean = np.nan if wsum == 0 else (w * x).sum() / wsum

                rows[pcol] = {
                    "prob_weighted_mean": mean,
                    "avg_probability": w.mean(),
                    "n_obs": len(tmp),
                }
            asset_res[h] = pd.DataFrame(rows).T
        results[asset] = asset_res
    return results

## 4.7 Build a table: rows = assets, cols = states, values = mean forward returns
def regime_mean_table(results, horizon):
    rows = []
    for asset, asset_res in results.items():
        df = asset_res[horizon]["mean"].rename(asset)
        rows.append(df)

    out = pd.DataFrame(rows)
    out.index.name = "asset"
    return out

## 4.8 Build a table: rows = assets, cols = states, values = hit rates
def regime_hit_rate_table(results, horizon):
    rows = []
    for asset, asset_res in results.items():
        df = asset_res[horizon]["hit_rate"].rename(asset)
        rows.append(df)

    out = pd.DataFrame(rows)
    out.index.name = "asset"
    return out

## 4.9 Plot mean table as a matrix
def plot_mean_table(mean_table, title="Mean Forward Returns by State"):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(mean_table))))
    im = ax.imshow(mean_table.values, aspect="auto", cmap="coolwarm")

    ax.set_xticks(range(mean_table.shape[1]))
    ax.set_xticklabels(mean_table.columns)
    ax.set_yticks(range(mean_table.shape[0]))
    ax.set_yticklabels(mean_table.index)

    for i in range(mean_table.shape[0]):
        for j in range(mean_table.shape[1]):
            val = mean_table.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.show()

## 4.10 Test whether mean forward return in a given state is different from 0, using HAC/Newey-West standard errors.
def hac_mean_test_by_state(state_series, ret_series, state, horizon=4, nw_lags=None):
    if nw_lags is None:
        nw_lags = max(horizon - 1, 1)
    fwd = make_forward_return(ret_series, horizon)
    df = pd.DataFrame({"state": state_series, "fwd": fwd}).dropna()
    df = df.loc[df["state"] == state].copy()

    if len(df) < 10:
        return pd.Series({
            "state": state,
            "n_obs": len(df),
            "mean": np.nan,
            "se_hac": np.nan,
            "t_hac": np.nan,
            "pvalue_hac": np.nan,
        })

    X = np.ones((len(df), 1))
    model = sm.OLS(df["fwd"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    return pd.Series({
        "state": state,
        "n_obs": len(df),
        "mean": model.params[0],
        "se_hac": model.bse[0],
        "t_hac": model.tvalues[0],
        "pvalue_hac": model.pvalues[0],
    })

## 4.11 Test whether mean forward return differs between state_a and state_b, using HAC/Newey-West SE.
def hac_diff_test_between_states(state_series, ret_series, state_a, state_b, horizon=4, nw_lags=None):
    if nw_lags is None:
        nw_lags = max(horizon - 1, 1)

    fwd = make_forward_return(ret_series, horizon)
    df = pd.DataFrame({"state": state_series, "fwd": fwd}).dropna()
    df = df.loc[df["state"].isin([state_a, state_b])].copy()

    if len(df) < 20:
        return pd.Series({
            "state_a": state_a,
            "state_b": state_b,
            "n_obs": len(df),
            "mean_a": np.nan,
            "mean_b": np.nan,
            "diff_b_minus_a": np.nan,
            "se_hac": np.nan,
            "t_hac": np.nan,
            "pvalue_hac": np.nan,
        })

    df["is_b"] = (df["state"] == state_b).astype(int)

    X = sm.add_constant(df["is_b"])
    model = sm.OLS(df["fwd"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    alpha = model.params["const"]
    beta = model.params["is_b"]

    return pd.Series({
        "state_a": state_a,
        "state_b": state_b,
        "n_obs": len(df),
        "mean_a": alpha,
        "mean_b": alpha + beta,
        "diff_b_minus_a": beta,
        "se_hac": model.bse["is_b"],
        "t_hac": model.tvalues["is_b"],
        "pvalue_hac": model.pvalues["is_b"],
    })

## 4.12 Moving-block bootstrap for difference in mean forward returns: mean(state_b) - mean(state_a)
## Uses overlapping forward returns, so block bootstrap is more appropriate than iid bootstrap.
def moving_block_bootstrap_diff(state_series, ret_series, state_a, state_b, horizon=4,
                                block_size=None, n_boot=2000, random_state=42):
    rng = np.random.default_rng(random_state)

    if block_size is None:
        block_size = max(2 * horizon, 8)
    fwd = make_forward_return(ret_series, horizon)
    df = pd.DataFrame({"state": state_series, "fwd": fwd}).dropna().copy()
    n = len(df)
    if n < block_size + 5:
        return pd.Series({
            "state_a": state_a,
            "state_b": state_b,
            "obs_diff": np.nan,
            "boot_mean": np.nan,
            "boot_std": np.nan,
            "ci_5": np.nan,
            "ci_50": np.nan,
            "ci_95": np.nan,
        })

    obs_a = df.loc[df["state"] == state_a, "fwd"].mean()
    obs_b = df.loc[df["state"] == state_b, "fwd"].mean()
    obs_diff = obs_b - obs_a

    diffs = []
    max_start = n - block_size

    for _ in range(n_boot):
        pieces = []
        total = 0

        while total < n:
            start = rng.integers(0, max_start + 1)
            block = df.iloc[start:start + block_size]
            pieces.append(block)
            total += len(block)

        boot = pd.concat(pieces, axis=0).iloc[:n].copy()

        a_mean = boot.loc[boot["state"] == state_a, "fwd"].mean()
        b_mean = boot.loc[boot["state"] == state_b, "fwd"].mean()
        diffs.append(b_mean - a_mean)

    diffs = np.asarray(diffs)

    return pd.Series({
        "state_a": state_a,
        "state_b": state_b,
        "obs_diff": obs_diff,
        "boot_mean": np.nanmean(diffs),
        "boot_std": np.nanstd(diffs, ddof=1),
        "ci_5": np.nanquantile(diffs, 0.05),
        "ci_50": np.nanquantile(diffs, 0.50),
        "ci_95": np.nanquantile(diffs, 0.95),
    })

## 4.13 Run HAC mean tests, HAC pairwise diff tests, and block bootstrap diff tests for all spread columns and horizons.
def run_caution_checks_all_spreads(state_series, spread_ret_df, horizons=(1, 4, 12), state_pairs=None):
    state_series, spread_ret_df = align_state_and_returns(state_series, spread_ret_df)
    if state_pairs is None:
        states = sorted(state_series.dropna().unique().tolist())
        state_pairs = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                state_pairs.append((states[i], states[j]))
    mean_tests = []
    diff_tests = []
    boot_tests = []
    for asset in spread_ret_df.columns:
        r = spread_ret_df[asset]

        for h in horizons:
            for s in sorted(state_series.unique()):
                row = hac_mean_test_by_state(state_series, r, state=s, horizon=h)
                row["asset"] = asset
                row["horizon"] = h
                mean_tests.append(row)

            for a, b in state_pairs:
                row = hac_diff_test_between_states(state_series, r, state_a=a, state_b=b, horizon=h)
                row["asset"] = asset
                row["horizon"] = h
                diff_tests.append(row)

                brow = moving_block_bootstrap_diff(state_series, r, state_a=a, state_b=b, horizon=h)
                brow["asset"] = asset
                brow["horizon"] = h
                boot_tests.append(brow)
    mean_tests = pd.DataFrame(mean_tests)
    diff_tests = pd.DataFrame(diff_tests)
    boot_tests = pd.DataFrame(boot_tests)

    return mean_tests, diff_tests, boot_tests

## 4.14 Weekly return stats on the subset of weeks belonging to each state.
def state_conditional_risk_stats(state_series, ret_df, ann_factor=52):
    state_series, ret_df = align_state_and_returns(state_series, ret_df)
    rows = []
    states = sorted(state_series.dropna().unique().tolist())

    for asset in ret_df.columns:
        for s in states:
            x = ret_df.loc[state_series == s, asset].dropna()

            if len(x) == 0:
                continue

            mean_w = x.mean()
            vol_w = x.std(ddof=1)
            downside = x[x < 0].std(ddof=1)
            hit = (x > 0).mean()
            mdd = max_drawdown_from_returns(x)

            ann_mean = mean_w * ann_factor
            ann_vol = vol_w * np.sqrt(ann_factor)
            ann_down = downside * np.sqrt(ann_factor) if pd.notna(downside) else np.nan

            sharpe = np.nan if (ann_vol == 0 or pd.isna(ann_vol)) else ann_mean / ann_vol
            sortino = np.nan if (ann_down == 0 or pd.isna(ann_down)) else ann_mean / ann_down

            rows.append({
                "asset": asset,
                "state": s,
                "n_obs": len(x),
                "mean_w": mean_w,
                "vol_w": vol_w,
                "hit_rate": hit,
                "ann_mean": ann_mean,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": mdd,
            })

    return pd.DataFrame(rows).sort_values(["asset", "state"]).reset_index(drop=True)

## 4.15 For each state s and asset: strategy return at t = asset return at t if state_{t-lag} == s else 0
## lag=1 is the safer default for implementability.
def timed_state_strategy_stats(state_series, ret_df, lag=1, ann_factor=52):
    state_series, ret_df = align_state_and_returns(state_series, ret_df)
    state_lag = state_series.shift(lag)

    rows = []
    states = sorted(state_series.dropna().unique().tolist())

    for asset in ret_df.columns:
        r = ret_df[asset].fillna(0.0)

        for s in states:
            signal = (state_lag == s).astype(float)
            strat = signal * r

            mean_w = strat.mean()
            vol_w = strat.std(ddof=1)
            downside = strat[strat < 0].std(ddof=1)
            hit = (strat > 0).mean()
            exposure = signal.mean()
            mdd = max_drawdown_from_returns(strat)

            ann_mean = mean_w * ann_factor
            ann_vol = vol_w * np.sqrt(ann_factor)
            ann_down = downside * np.sqrt(ann_factor) if pd.notna(downside) else np.nan

            sharpe = np.nan if (ann_vol == 0 or pd.isna(ann_vol)) else ann_mean / ann_vol
            sortino = np.nan if (ann_down == 0 or pd.isna(ann_down)) else ann_mean / ann_down

            rows.append({
                "asset": asset,
                "state": s,
                "exposure": exposure,
                "mean_w": mean_w,
                "vol_w": vol_w,
                "hit_rate": hit,
                "ann_mean": ann_mean,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": mdd,
            })

    return pd.DataFrame(rows).sort_values(["asset", "state"]).reset_index(drop=True)

## 4.16 Compact pivot tables for risk metrics
def pivot_metric(df, metric):
    out = df.pivot(index="asset", columns="state", values=metric)
    out.index.name = "asset"
    out.columns.name = "state"
    return out