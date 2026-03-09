# Data Helpers
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM

# 1. Basic Helpers

## 1.1 Rolling z-score using only local history. Weekly data: 104 ~= 2 years, 52 ~= 1 year minimum.
def rolling_zscore(df, window = 104, min_periods = 52, clip: float | None = 4.0):
    mu = df.rolling(window=window, min_periods=min_periods).mean()
    sd = df.rolling(window=window, min_periods=min_periods).std()
    z = (df - mu) / sd.replace(0, np.nan)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z

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

## 2.14 Expanding-window filtered state inference.
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