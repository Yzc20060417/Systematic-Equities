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
def relabel_states_by_macro_order(X, labels):
    X = X.copy()
    labels = pd.Series(labels, index=X.index, name="state")

    means = X.groupby(labels).mean()

    macro_score = (
        means["stress"]
        - means["growth"]
        + 0.25 * means["rates_pressure"]
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