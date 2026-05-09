from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# CONFIG
# ============================================================
WINDOW_SIZE = 100
BASE_CSV = "base_normal_reference.csv"
SCENARIO_CSVS = {
    "Sudden": "sudden_drift.csv",
    "Gradual": "gradual_drift.csv",
    "Recurrent": "recurrent_drift.csv",
}

OUTPUT_DIR = Path("drift_measurement_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
REF_HIST_BINS = 20
N_HEATMAP_SPLITS = 8
LATENT_DIM = 12
K_NEIGHBORS = 10
MAD_MULTIPLIER = 3.0
FAMILY_SW_PROJECTIONS = 16
FAMILY_BASE_SAMPLE = 1500
FAMILY_TARGET_SAMPLE = 600


# ============================================================
# HELPERS
# ============================================================
META_COLS = {
    "stream_name",
    "stream_window_pos",
    "source_window_id",
    "source_window_type",
    "source_stage_label",
    "source_macro_role",
    "source_micro_name",
    "stream_row_id",
    "_orig_row_id",
    "window_id",
    "macro_role",
    "stage_label",
    "micro_name",
}


def load_stream(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "stream_window_pos" not in df.columns:
        df["stream_window_pos"] = np.arange(len(df)) // WINDOW_SIZE
    return df.reset_index(drop=True)


base_df = load_stream(BASE_CSV)
scenario_dfs = {name: load_stream(path) for name, path in SCENARIO_CSVS.items()}


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in META_COLS]
    numeric_cols = []
    categorical_cols = []
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return feature_cols, numeric_cols, categorical_cols


FEATURE_COLS, NUMERIC_COLS, CATEGORICAL_COLS = infer_feature_columns(base_df)
if not FEATURE_COLS:
    raise ValueError("No feature columns were found in the CSV files.")


# ------------------------------------------------------------
# Common preprocessing
# ------------------------------------------------------------
def make_row_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


ROW_PREPROCESSOR = make_row_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
ROW_PREPROCESSOR.fit(base_df[FEATURE_COLS])


def transform_rows(df: pd.DataFrame):
    return ROW_PREPROCESSOR.transform(df[FEATURE_COLS])


# ============================================================
# 1) FEATURE-DISTRIBUTION DRIFT (Jeffreys divergence)
# ============================================================
EPS = 1e-10


def _numeric_bins(values: np.ndarray, max_bins: int = REF_HIST_BINS) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([-0.5, 0.5])
    q01, q99 = np.quantile(values, [0.01, 0.99])
    lo, hi = float(q01), float(q99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, max_bins + 1)


REF_DISTS: Dict[str, Dict] = {}
for col in NUMERIC_COLS:
    vals = pd.to_numeric(base_df[col], errors="coerce").to_numpy(dtype=float)
    bins = _numeric_bins(vals)
    clipped = np.clip(vals[np.isfinite(vals)], bins[0], bins[-1])
    hist, _ = np.histogram(clipped, bins=bins)
    p = hist.astype(float) + EPS
    p /= p.sum()
    REF_DISTS[col] = {"type": "numeric", "bins": bins, "p": p}

for col in CATEGORICAL_COLS:
    s = base_df[col].astype(str).fillna("<NA>")
    cats = sorted(s.unique().tolist())
    p = s.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy(dtype=float)
    p = p + EPS
    p /= p.sum()
    REF_DISTS[col] = {"type": "categorical", "cats": cats, "p": p}


def jeffreys_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum((p - q) * np.log(p / q)))



def distribution_vector(df_slice: pd.DataFrame, col: str) -> np.ndarray:
    spec = REF_DISTS[col]
    if spec["type"] == "numeric":
        vals = pd.to_numeric(df_slice[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            q = np.ones_like(spec["p"], dtype=float)
            q /= q.sum()
            return q
        vals = np.clip(vals, spec["bins"][0], spec["bins"][-1])
        hist, _ = np.histogram(vals, bins=spec["bins"])
        q = hist.astype(float) + EPS
        q /= q.sum()
        return q
    else:
        s = df_slice[col].astype(str).fillna("<NA>")
        q = s.value_counts(normalize=True).reindex(spec["cats"], fill_value=0.0).to_numpy(dtype=float)
        q = q + EPS
        q /= q.sum()
        return q



def compute_window_jd(df_stream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, sub in df_stream.groupby("stream_window_pos", sort=True):
        per_feature = {col: jeffreys_divergence(REF_DISTS[col]["p"], distribution_vector(sub, col)) for col in FEATURE_COLS}
        rows.append({
            "window": int(w),
            "jd_mean": float(np.mean(list(per_feature.values()))),
            "jd_median": float(np.median(list(per_feature.values()))),
            **{f"jd__{k}": v for k, v in per_feature.items()},
        })
    return pd.DataFrame(rows)


BASE_JD = compute_window_jd(base_df)
SCENARIO_JD = {name: compute_window_jd(df) for name, df in scenario_dfs.items()}



def split_stream(df_stream: pd.DataFrame, n_splits: int) -> List[pd.DataFrame]:
    win_ids = sorted(df_stream["stream_window_pos"].unique())
    groups = np.array_split(win_ids, n_splits)
    return [df_stream[df_stream["stream_window_pos"].isin(g)].copy() for g in groups if len(g) > 0]



def cross_split_jd(ref_df: pd.DataFrame, target_df: pd.DataFrame, n_splits: int = N_HEATMAP_SPLITS) -> np.ndarray:
    ref_slices = split_stream(ref_df, n_splits)
    tgt_slices = split_stream(target_df, n_splits)
    M = np.zeros((len(ref_slices), len(tgt_slices)))
    for i, a in enumerate(ref_slices):
        for j, b in enumerate(tgt_slices):
            vals = []
            for col in FEATURE_COLS:
                pa = distribution_vector(a, col)
                pb = distribution_vector(b, col)
                vals.append(jeffreys_divergence(pa, pb))
            M[i, j] = float(np.mean(vals))
    return M


# ============================================================
# 2) WINDOW-LEVEL VISUAL DRIFT INSPECTION (t-SNE)
# ============================================================

def window_mean_embedding(df_stream: pd.DataFrame, max_svd_dim: int = 32) -> Tuple[np.ndarray, pd.DataFrame]:
    X = transform_rows(df_stream)
    if hasattr(X, "toarray"):
        n_comp = min(max_svd_dim, max(2, X.shape[1] - 1))
        reducer = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        Z = reducer.fit_transform(X)
    else:
        X = np.asarray(X)
        n_comp = min(max_svd_dim, X.shape[1])
        if n_comp >= 2:
            reducer = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            Z = reducer.fit_transform(X)
        else:
            Z = X

    rows = []
    grouped = df_stream.groupby("stream_window_pos", sort=True).indices
    for w, idx in grouped.items():
        rows.append(np.asarray(Z[idx]).mean(axis=0))
    W = np.vstack(rows)
    meta = pd.DataFrame({"window": sorted(grouped.keys())})
    return W, meta


# ============================================================
# 3) FAMILY-LEVEL DRIFT (stability + OT-style / sliced Wasserstein)
# ============================================================

def family_name(col: str) -> str:
    c = col.lower()
    if c.startswith("lte_"):
        return "LTE"
    if c.startswith("nr_"):
        return "NR"
    if c.startswith("irat_"):
        return "IRAT"
    if c in {"dl_tput", "ul_tput"} or "tput" in c:
        return "Throughput"
    if any(k in c for k in ["lat", "lon", "lng", "speed", "heading", "bearing", "gps", "location"]):
        return "Mobility"
    if c in {"technology", "lte_band", "nr_band"} or "band" in c:
        return "RadioContext"
    return "Other"


FAMILY_TO_COLS: Dict[str, List[str]] = {}
for c in FEATURE_COLS:
    FAMILY_TO_COLS.setdefault(family_name(c), []).append(c)



def make_family_preprocessor(cols: List[str], ref_df: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in cols if c in NUMERIC_COLS]
    cat = [c for c in cols if c in CATEGORICAL_COLS]
    pre = make_row_preprocessor(num, cat)
    pre.fit(ref_df[cols])
    return pre


FAMILY_PREPROCESSORS = {fam: make_family_preprocessor(cols, base_df) for fam, cols in FAMILY_TO_COLS.items()}



def transform_family(df_slice: pd.DataFrame, fam: str) -> np.ndarray:
    cols = FAMILY_TO_COLS[fam]
    X = FAMILY_PREPROCESSORS[fam].transform(df_slice[cols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X



def sample_rows(X: np.ndarray, n: int, seed: int = RANDOM_STATE) -> np.ndarray:
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]



def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = FAMILY_SW_PROJECTIONS, seed: int = RANDOM_STATE) -> float:
    if len(X) == 0 or len(Y) == 0:
        return np.nan
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= (np.linalg.norm(v) + 1e-12)
        px = X @ v
        py = Y @ v
        scores.append(wasserstein_distance(px, py))
    return float(np.mean(scores))


REF_FAMILY_MATS = {
    fam: sample_rows(transform_family(base_df, fam), FAMILY_BASE_SAMPLE, seed=RANDOM_STATE)
    for fam in FAMILY_TO_COLS
}
REF_FAMILY_MEANS = {fam: REF_FAMILY_MATS[fam].mean(axis=0) for fam in FAMILY_TO_COLS}



def family_drift_by_window(df_stream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, sub in df_stream.groupby("stream_window_pos", sort=True):
        row = {"window": int(w)}
        for fam in FAMILY_TO_COLS:
            X_ref = REF_FAMILY_MATS[fam]
            X_tgt = sample_rows(transform_family(sub, fam), FAMILY_TARGET_SAMPLE, seed=RANDOM_STATE + int(w))
            mu_diff = np.mean(np.abs(X_tgt.mean(axis=0) - REF_FAMILY_MEANS[fam]))
            stability = float(np.exp(-mu_diff))
            sw = sliced_wasserstein(X_ref, X_tgt, n_proj=FAMILY_SW_PROJECTIONS, seed=RANDOM_STATE + int(w))
            row[f"stab__{fam}"] = stability
            row[f"ot__{fam}"] = sw
        rows.append(row)
    return pd.DataFrame(rows)


BASE_FAMILY = family_drift_by_window(base_df)
SCENARIO_FAMILY = {name: family_drift_by_window(df) for name, df in scenario_dfs.items()}


# ============================================================
# 4) SAMPLE-LEVEL LATENT DRIFT (CADE-style kNN + MAD)
# ============================================================
X_base = transform_rows(base_df)
if hasattr(X_base, "toarray"):
    latent_dim = min(LATENT_DIM, max(2, X_base.shape[1] - 1))
    LATENT_REDUCER = TruncatedSVD(n_components=latent_dim, random_state=RANDOM_STATE)
    Z_base = LATENT_REDUCER.fit_transform(X_base)
else:
    X_base = np.asarray(X_base)
    latent_dim = min(LATENT_DIM, X_base.shape[1])
    if latent_dim >= 2:
        LATENT_REDUCER = PCA(n_components=latent_dim, random_state=RANDOM_STATE)
        Z_base = LATENT_REDUCER.fit_transform(X_base)
    else:
        LATENT_REDUCER = None
        Z_base = X_base

BASE_NN = NearestNeighbors(n_neighbors=min(K_NEIGHBORS + 1, len(Z_base)), metric="euclidean")
BASE_NN.fit(Z_base)
base_self_dist, _ = BASE_NN.kneighbors(Z_base)
base_scores_rows = base_self_dist[:, 1:].mean(axis=1)
base_median = float(np.median(base_scores_rows))
base_mad = float(np.median(np.abs(base_scores_rows - base_median))) + 1e-12
LATENT_THRESHOLD = base_median + MAD_MULTIPLIER * 1.4826 * base_mad


def latent_transform(df_stream: pd.DataFrame) -> np.ndarray:
    X = transform_rows(df_stream)
    if LATENT_REDUCER is None:
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)
    return LATENT_REDUCER.transform(X)



def latent_scores(df_stream: pd.DataFrame, is_reference: bool = False) -> pd.DataFrame:
    Z = latent_transform(df_stream)
    if is_reference:
        scores = base_scores_rows
    else:
        dists, _ = BASE_NN.kneighbors(Z, n_neighbors=min(K_NEIGHBORS, len(Z_base)))
        scores = dists.mean(axis=1)

    tmp = pd.DataFrame({
        "stream_window_pos": df_stream["stream_window_pos"].to_numpy(),
        "latent_score": scores,
        "flag": scores > LATENT_THRESHOLD,
    })
    out = tmp.groupby("stream_window_pos", sort=True).agg(
        latent_score_mean=("latent_score", "mean"),
        latent_score_median=("latent_score", "median"),
        latent_flag_rate=("flag", "mean"),
    ).reset_index().rename(columns={"stream_window_pos": "window"})
    return out


BASE_LATENT = latent_scores(base_df, is_reference=True)
SCENARIO_LATENT = {name: latent_scores(df) for name, df in scenario_dfs.items()}


# ============================================================
# 5) PLOTS
# ============================================================
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.2


# 5.1 Jeffreys divergence over time
for name, df_jd in SCENARIO_JD.items():
    plt.figure()
    plt.plot(BASE_JD["window"], BASE_JD["jd_mean"], label="Base reference")
    plt.plot(df_jd["window"], df_jd["jd_mean"], label=f"{name} drift")
    plt.xlabel("Window position")
    plt.ylabel("Mean Jeffreys divergence")
    plt.title(f"Feature-distribution drift over time: Base vs {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"jd_line_{name.lower()}.png", dpi=180)
    plt.show()


# 5.2 Jeffreys heatmaps (base splits x scenario splits)
for name, df_stream in scenario_dfs.items():
    M = cross_split_jd(base_df, df_stream, n_splits=N_HEATMAP_SPLITS)
    plt.figure(figsize=(7, 6))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="Mean Jeffreys divergence")
    plt.xlabel(f"{name} split")
    plt.ylabel("Base split")
    plt.title(f"Feature-distribution heatmap: Base vs {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"jd_heatmap_{name.lower()}.png", dpi=180)
    plt.show()


# 5.3 t-SNE window plots
for name, df_stream in scenario_dfs.items():
    pair = pd.concat([
        base_df.assign(_dataset="Base"),
        df_stream.assign(_dataset=name),
    ], ignore_index=True)
    W, meta = window_mean_embedding(pair)
    meta["dataset"] = ["Base" if i < base_df["stream_window_pos"].nunique() else name for i in range(len(meta))]

    perplexity = min(30, max(5, len(W) // 8))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=RANDOM_STATE)
    Y = tsne.fit_transform(W)

    plt.figure(figsize=(8, 7))
    mask_base = meta["dataset"] == "Base"
    plt.scatter(Y[mask_base, 0], Y[mask_base, 1], s=28, alpha=0.7, label="Base")
    plt.scatter(Y[~mask_base, 0], Y[~mask_base, 1], s=28, alpha=0.7, label=name)
    plt.title(f"Window-level t-SNE: Base vs {name}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"tsne_{name.lower()}.png", dpi=180)
    plt.show()


# 5.4 Family stability heatmap summary
families = list(FAMILY_TO_COLS.keys())
stab_summary = pd.DataFrame(index=SCENARIO_CSVS.keys(), columns=families, dtype=float)
ot_summary = pd.DataFrame(index=SCENARIO_CSVS.keys(), columns=families, dtype=float)
for name, fam_df in SCENARIO_FAMILY.items():
    for fam in families:
        stab_summary.loc[name, fam] = fam_df[f"stab__{fam}"].mean()
        ot_summary.loc[name, fam] = fam_df[f"ot__{fam}"].mean()

plt.figure(figsize=(1.4 * len(families) + 4, 4.8))
plt.imshow(stab_summary.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
plt.colorbar(label="Mean stability score")
plt.xticks(np.arange(len(families)), families, rotation=30, ha="right")
plt.yticks(np.arange(len(stab_summary.index)), stab_summary.index)
plt.title("Family-level stability: Base vs drift scenarios")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "family_stability_heatmap.png", dpi=180)
plt.show()

plt.figure(figsize=(1.4 * len(families) + 4, 4.8))
plt.imshow(ot_summary.to_numpy(dtype=float), aspect="auto")
plt.colorbar(label="Mean OT-style sliced Wasserstein")
plt.xticks(np.arange(len(families)), families, rotation=30, ha="right")
plt.yticks(np.arange(len(ot_summary.index)), ot_summary.index)
plt.title("Family-level OT-style drift: Base vs drift scenarios")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "family_ot_heatmap.png", dpi=180)
plt.show()


# 5.5 Family drift over time (top 4 families by overall OT)
top_families = ot_summary.mean(axis=0).sort_values(ascending=False).head(min(4, len(families))).index.tolist()
for name, fam_df in SCENARIO_FAMILY.items():
    plt.figure(figsize=(11, 6))
    for fam in top_families:
        plt.plot(fam_df["window"], fam_df[f"ot__{fam}"], label=fam)
    plt.xlabel("Window position")
    plt.ylabel("OT-style sliced Wasserstein")
    plt.title(f"Top drifting feature families over time: {name}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"family_ot_lines_{name.lower()}.png", dpi=180)
    plt.show()


# 5.6 Latent-space CADE-style scores
for name, df_lat in SCENARIO_LATENT.items():
    plt.figure()
    plt.plot(BASE_LATENT["window"], BASE_LATENT["latent_score_mean"], label="Base reference")
    plt.plot(df_lat["window"], df_lat["latent_score_mean"], label=f"{name} drift")
    plt.axhline(LATENT_THRESHOLD, linestyle="--", label="MAD threshold")
    plt.xlabel("Window position")
    plt.ylabel("Mean latent kNN distance to base")
    plt.title(f"Latent sample-level drift: Base vs {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"latent_score_{name.lower()}.png", dpi=180)
    plt.show()

    plt.figure()
    plt.plot(BASE_LATENT["window"], BASE_LATENT["latent_flag_rate"], label="Base reference")
    plt.plot(df_lat["window"], df_lat["latent_flag_rate"], label=f"{name} drift")
    plt.xlabel("Window position")
    plt.ylabel("Flag rate")
    plt.title(f"Latent drift flag rate per window: Base vs {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"latent_flag_rate_{name.lower()}.png", dpi=180)
    plt.show()


# ============================================================
# 6) SAVE NUMERIC SUMMARIES
# ============================================================
BASE_JD.to_csv(OUTPUT_DIR / "base_jd_by_window.csv", index=False)
BASE_FAMILY.to_csv(OUTPUT_DIR / "base_family_drift_by_window.csv", index=False)
BASE_LATENT.to_csv(OUTPUT_DIR / "base_latent_scores_by_window.csv", index=False)

for name in SCENARIO_CSVS:
    slug = name.lower()
    SCENARIO_JD[name].to_csv(OUTPUT_DIR / f"{slug}_jd_by_window.csv", index=False)
    SCENARIO_FAMILY[name].to_csv(OUTPUT_DIR / f"{slug}_family_drift_by_window.csv", index=False)
    SCENARIO_LATENT[name].to_csv(OUTPUT_DIR / f"{slug}_latent_scores_by_window.csv", index=False)

summary_rows = []
for name in SCENARIO_CSVS:
    jd_mean = float(SCENARIO_JD[name]["jd_mean"].mean())
    latent_mean = float(SCENARIO_LATENT[name]["latent_score_mean"].mean())
    latent_flag = float(SCENARIO_LATENT[name]["latent_flag_rate"].mean())
    best_fam = ot_summary.loc[name].astype(float).sort_values(ascending=False).index[0]
    summary_rows.append({
        "scenario": name,
        "mean_jeffreys": jd_mean,
        "mean_latent_score": latent_mean,
        "mean_latent_flag_rate": latent_flag,
        "top_family_by_ot": best_fam,
        "top_family_ot": float(ot_summary.loc[name, best_fam]),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("mean_jeffreys", ascending=False)
summary_df.to_csv(OUTPUT_DIR / "drift_summary.csv", index=False)
stab_summary.to_csv(OUTPUT_DIR / "family_stability_summary.csv")
ot_summary.to_csv(OUTPUT_DIR / "family_ot_summary.csv")

print("\nSaved outputs to:", OUTPUT_DIR.resolve())
print("\nDrift summary:")
print(summary_df.to_string(index=False))
