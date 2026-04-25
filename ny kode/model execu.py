# =============================================================================
# FILE 2: portfolio_model.py
# Regime-based portfolio optimization — walk-forward backtest.
#
# Pipeline per month t:
#   1. KMeans fitted on pretrain + all data BEFORE t  (no lookahead)
#   2. After each refit, centroids are MATCHED to the original pretrain
#      centroids via Hungarian algorithm → regime labels never drift.
#      "Regime 0 = recession-like" stays Regime 0 for the entire backtest.
#   3. KNN on centroid-distance history → next-regime transition probabilities
#   4. Per-regime Markowitz max-Sharpe using EXCESS returns (correct formula)
#   5. Blend regime weights by transition probabilities
#   6. Record portfolio return at month t
#
# Key fixes vs previous versions:
#   - Markowitz uses excess returns (ret - RF), not raw returns
#   - KMeans re-fitted on expanding window but centroids pinned to pretrain
#     anchor via Hungarian matching → no regime label drift across time
#   - Pretrain return history used to warm up Markowitz from day 1
#   - SP500 benchmark rebased correctly from same start date
#   - KNN off-by-one in neighbor indexing fixed
#   - Turnover damping: weights smoothed toward previous month
#   - Diagnostic plots reuse the SAME KMeans/scaler from the backtest loop
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment   # Hungarian algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Config ────────────────────────────────────────────────────────────────────
K               = 3    # Number of macro regimes
MIN_OBS         = 3    # Min months in a regime before trusting Markowitz
KNN_NEIGHBORS   = 2    # Neighbours for transition probability estimate
REFIT_MONTHS    = 1    # Re-fit KMeans every N months

ASSETS      = ["SP500", "QQQ", "Gold", "RealEstate"]
RET_COLS    = [f"{a}_ret"    for a in ASSETS]
EXCESS_COLS = [f"{a}_excess" for a in ASSETS]

# ── Load data ─────────────────────────────────────────────────────────────────
backtest       = pd.read_csv("backtest_data.csv",  index_col=0, parse_dates=True)
pretrain       = pd.read_csv("pretrain_macro.csv", index_col=0, parse_dates=True)
MACRO_FEATURES = pd.read_csv("macro_features.csv", header=None)[0].tolist()


# ── Hungarian centroid matcher ────────────────────────────────────────────────
def match_centroids(new_centers: np.ndarray,
                    anchor_centers: np.ndarray) -> np.ndarray:
    """
    Given freshly fitted KMeans centroids (new_centers, shape K x F) and the
    original pretrain anchor centroids (anchor_centers, K x F), return a
    permutation array `perm` such that new_centers relabelled by perm best
    matches anchor_centers in L2 distance.

    After refit:
        perm = match_centroids(kmeans.cluster_centers_, anchor_centers)
        canonical_label = perm[raw_kmeans_label]
    """
    # Cost matrix (K x K): cost[i,j] = dist(anchor i, new centroid j)
    cost = np.linalg.norm(
        anchor_centers[:, None, :] - new_centers[None, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = which new centroid matches anchor i
    # perm[col_ind[i]] = i  →  raw label col_ind[i] becomes canonical label i
    perm = np.empty(K, dtype=int)
    perm[col_ind] = row_ind
    return perm


# ── Markowitz max-Sharpe weights ──────────────────────────────────────────────
def markowitz_weights(excess_mu: pd.Series, cov: pd.DataFrame,
                      reg: float = 1e-4) -> pd.Series:
    """
    Max-Sharpe long-only Markowitz:
        w* proportional to Sigma^{-1} mu_excess
    Clips negatives (long-only), normalises to sum = 1.
    Falls back to equal weight if all excess returns are <= 0.
    """
    C = cov.values.copy() + np.eye(len(cov)) * reg
    w = np.linalg.inv(C) @ excess_mu.values
    w = np.maximum(w, 0)
    if w.sum() == 0:
        w = np.ones(len(excess_mu))
    return pd.Series(w / w.sum(), index=excess_mu.index)


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def run_backtest(backtest: pd.DataFrame, pretrain: pd.DataFrame) -> tuple:

    data    = backtest.copy()
    equal_w = pd.Series(1.0 / len(ASSETS), index=RET_COLS)
    prev_w  = equal_w.copy()

    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)

    regime_log     = []
    weights_log    = []
    dist_history   = []
    regime_history = []

    # ── Fit KMeans on pretrain and lock anchor centroids ──────────────────────
    # The pretrain centroids define what each regime MEANS economically.
    # Every subsequent refit is Hungarian-matched back to these anchors,
    # so Regime 0 always refers to the same macro cluster as it did on day 1.
    X_pre_s        = scaler.fit_transform(pretrain[MACRO_FEATURES].dropna())
    kmeans.fit(X_pre_s)
    anchor_centers = kmeans.cluster_centers_.copy()   # frozen reference in pretrain space
    current_perm   = np.arange(K)                     # identity to start
    last_refit     = -REFIT_MONTHS

    for t in range(len(data)):

        # ── Re-fit KMeans on expanding window ─────────────────────────────────
        if t - last_refit >= REFIT_MONTHS:
            X_train = pd.concat([
                pretrain[MACRO_FEATURES],
                data.iloc[:t][MACRO_FEATURES]
            ]).dropna()
            scaler.fit(X_train)
            kmeans.fit(scaler.transform(X_train))

            # Re-express anchor centroids in the new scaler's space so the
            # Hungarian distance comparison is apples-to-apples.
            # We fit a temporary KMeans on just the pretrain in the new scale,
            # giving us the pretrain centroids in the current feature space.
            X_pre_new    = scaler.transform(pretrain[MACRO_FEATURES].dropna())
            km_anchor    = KMeans(n_clusters=K, random_state=42, n_init=10)
            km_anchor.fit(X_pre_new)
            current_perm = match_centroids(kmeans.cluster_centers_,
                                           km_anchor.cluster_centers_)
            last_refit = t

        # ── Predict & relabel current regime with canonical label ──────────────
        X_now_s    = scaler.transform(data.iloc[[t]][MACRO_FEATURES])
        raw_regime = int(kmeans.predict(X_now_s)[0])
        regime_now = int(current_perm[raw_regime])   # stable canonical label
        regime_log.append(regime_now)

        dist_now = np.linalg.norm(
            kmeans.cluster_centers_ - X_now_s, axis=1
        ).flatten()

        # ── KNN transition probabilities ──────────────────────────────────────
        if len(dist_history) >= 2:
            hist   = np.vstack(dist_history)
            deltas = np.linalg.norm(hist - dist_now, axis=1)
            n_nn   = min(KNN_NEIGHBORS, len(deltas) - 1)
            nn_idx = np.argsort(deltas)[:n_nn]
            valid  = nn_idx[nn_idx + 1 < len(regime_history)]
            next_reg = np.array(regime_history)[valid + 1]
            if len(next_reg) > 0:
                counts = np.bincount(next_reg, minlength=K).astype(float)
                probs  = counts / counts.sum()
            else:
                probs = np.ones(K) / K
        else:
            probs = np.ones(K) / K

        dist_history.append(dist_now)
        regime_history.append(regime_now)

        # ── Per-regime Markowitz weights ──────────────────────────────────────
        regime_weights = {}

        if t >= MIN_OBS:
            past       = data.iloc[:t]
            raw_labels = kmeans.predict(scaler.transform(past[MACRO_FEATURES]))
            past_labels = current_perm[raw_labels]   # canonical labels

            for c in range(K):
                idx = np.where(past_labels == c)[0]
                if len(idx) < MIN_OBS:
                    continue
                excess_mu = past.iloc[idx][EXCESS_COLS].mean()
                cov       = past.iloc[idx][EXCESS_COLS].cov()
                excess_mu.index = RET_COLS
                cov.index       = RET_COLS
                cov.columns     = RET_COLS
                reg = 1e-3 if len(idx) < 20 else (5e-4 if len(idx) < 50 else 1e-4)
                regime_weights[c] = markowitz_weights(excess_mu, cov, reg)

        # ── Blend by transition probabilities ─────────────────────────────────
        raw_w = pd.Series(0.0, index=RET_COLS)
        if not regime_weights:
            raw_w = equal_w.copy()
        else:
            for c in range(K):
                raw_w += probs[c] * regime_weights.get(c, equal_w)
            raw_w = raw_w / raw_w.sum()

        final_w = prev_w + raw_w
        final_w = final_w / final_w.sum()

        assert abs(final_w.sum() - 1.0) < 1e-6
        weights_log.append(final_w.copy())
        prev_w = final_w.copy()

        data.loc[data.index[t], "portfolio_ret"] = float(
            data.iloc[t][RET_COLS] @ final_w
        )

    data["regime"]        = regime_log
    data["portfolio_cum"] = (1 + data["portfolio_ret"].fillna(0)).cumprod()
    data["SP500_cum"]     = (1 + data["SP500_ret"]).cumprod()

    weights_df = pd.DataFrame(weights_log, index=data.index, columns=RET_COLS)

    equal_w_static              = np.ones(len(RET_COLS)) / len(RET_COLS)
    data["equal_portfolio_ret"] = data[RET_COLS] @ equal_w_static
    data["equal_portfolio_cum"] = (1 + data["equal_portfolio_ret"].fillna(0)).cumprod()

    return data, weights_df, kmeans, scaler


# ── Performance statistics ────────────────────────────────────────────────────
def perf_stats(label: str, cum: pd.Series, ret: pd.Series):
    n       = len(ret)
    total   = cum.iloc[-1] - 1
    ann_ret = cum.iloc[-1] ** (12 / n) - 1
    ann_vol = ret.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    print(f"\n-- {label} --")
    print(f"  Total Return    : {total:.1%}")
    print(f"  Ann. Return     : {ann_ret:.1%}")
    print(f"  Ann. Volatility : {ann_vol:.1%}")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd:.1%}")
    print(f"  Calmar Ratio    : {calmar:.2f}")


# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================
from scipy import stats as scipy_stats

print("\n================ EDA ================")
print(f"\nBacktest period : {backtest.index[0].date()} -> {backtest.index[-1].date()}  ({len(backtest)} months)")
print(f"Pretrain period : {pretrain.index[0].date()} -> {pretrain.index[-1].date()}  ({len(pretrain)} months)")
print(f"\nAssets          : {ASSETS}")
print(f"Macro features  : {MACRO_FEATURES}")
print(f"\nAsset return summary (backtest):")
print((backtest[RET_COLS] * 100).describe().round(2).to_string())
print(f"\nMacro feature summary (backtest):")
print(backtest[MACRO_FEATURES].describe().round(4).to_string())


# ── [EDA-1] ASSET CUMULATIVE RETURNS OVER TIME ───────────────────────────────
fig, axes = plt.subplots(len(RET_COLS), 1, figsize=(14, 3 * len(RET_COLS)), sharex=True)
for i, (col, asset) in enumerate(zip(RET_COLS, ASSETS)):
    cum = (1 + backtest[col]).cumprod()
    axes[i].plot(cum.index, cum.values, linewidth=1.5, color=f"C{i}")
    axes[i].set_ylabel("Growth of $1", fontsize=9)
    axes[i].set_title(asset, fontsize=10)
    axes[i].grid(alpha=0.3)
fig.suptitle("Asset Cumulative Returns Over Time (Backtest Period)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("eda_asset_cumulative.png", dpi=150)
plt.show()


# ── [EDA-2] RETURN DISTRIBUTIONS — HISTOGRAM + NORMAL OVERLAY + QQ ───────────
fig, axes = plt.subplots(len(RET_COLS), 2, figsize=(12, 3.5 * len(RET_COLS)))

for i, (col, asset) in enumerate(zip(RET_COLS, ASSETS)):
    r = backtest[col].dropna() * 100
    mu, sigma = r.mean(), r.std()
    skew  = r.skew()
    kurt  = r.kurtosis()
    _, pval = scipy_stats.shapiro(r.iloc[:min(len(r), 5000)])

    ax = axes[i, 0]
    ax.hist(r, bins=40, density=True, alpha=0.6, color=f"C{i}", edgecolor="white")
    xgrid = np.linspace(r.min(), r.max(), 300)
    ax.plot(xgrid, scipy_stats.norm.pdf(xgrid, mu, sigma),
            color="black", linewidth=2, label="Normal fit")
    ax.axvline(mu, color="red", linestyle="--", linewidth=1, label=f"Mean={mu:.2f}%")
    ax.set_title(f"{asset}  |  skew={skew:.2f}  kurt={kurt:.2f}  Shapiro p={pval:.3f}", fontsize=9)
    ax.set_xlabel("Monthly Return (%)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes[i, 1]
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, s=12, alpha=0.6, color=f"C{i}")
    ax2.plot(osm, slope * np.array(osm) + intercept,
             color="black", linewidth=1.5, label="Normal reference")
    ax2.set_title(f"{asset} -- QQ Plot", fontsize=9)
    ax2.set_xlabel("Theoretical quantiles"); ax2.set_ylabel("Sample quantiles")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle("Asset Return Distributions: Normality Check\n"
             "(Shapiro p < 0.05 = reject normality; fat tails inflate Markowitz tail risk)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("eda_return_distributions.png", dpi=150)
plt.show()


# ── [EDA-3] ASSET SUMMARY STATS TABLE ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

summary_rows = []
for col, asset in zip(RET_COLS, ASSETS):
    r   = backtest[col].dropna()
    cum = (1 + r).cumprod()
    ann_r  = cum.iloc[-1] ** (12 / len(r)) - 1
    ann_v  = r.std() * np.sqrt(12)
    sharpe = ann_r / ann_v if ann_v > 0 else np.nan
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    summary_rows.append([
        asset, f"{ann_r:.1%}", f"{ann_v:.1%}",
        f"{sharpe:.2f}", f"{r.skew():.2f}", f"{r.kurtosis():.2f}", f"{max_dd:.1%}"
    ])

col_labels = ["Asset", "Ann. Return", "Ann. Vol", "Sharpe", "Skew", "Excess Kurt", "Max DD"]
tbl = ax.table(cellText=summary_rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(summary_rows) + 1):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

ax.set_title("Asset Summary Statistics (Annualised, Backtest Period)", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig("eda_asset_summary_table.png", dpi=150)
plt.show()


# ── [EDA-4] PAIRWISE ASSET SCATTER MATRIX ────────────────────────────────────
n = len(RET_COLS)
fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))

for i, (ci, ai) in enumerate(zip(RET_COLS, ASSETS)):
    for j, (cj, aj) in enumerate(zip(RET_COLS, ASSETS)):
        ax = axes[i, j]
        ri = backtest[ci].dropna() * 100
        rj = backtest[cj].dropna() * 100
        aligned = pd.concat([ri, rj], axis=1).dropna()

        if i == j:
            ax.hist(aligned.iloc[:, 0], bins=30, density=True,
                    alpha=0.7, color=f"C{i}", edgecolor="white")
            ax.set_title(ai, fontsize=9, fontweight="bold")
        else:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            ax.scatter(aligned.iloc[:, 1], aligned.iloc[:, 0],
                       alpha=0.4, s=10, color=f"C{i}")
            m, b = np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)
            xline = np.linspace(aligned.iloc[:, 1].min(), aligned.iloc[:, 1].max(), 100)
            ax.plot(xline, m * xline + b, color="red", linewidth=1.2)
            ax.set_title(f"rho = {corr:.2f}", fontsize=8,
                         color="darkred" if abs(corr) > 0.6 else "black")

        if i == n - 1: ax.set_xlabel(aj, fontsize=8)
        if j == 0:     ax.set_ylabel(ai, fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.2)

fig.suptitle("Pairwise Asset Return Scatter Matrix\n"
             "(Diagonal = return distribution; off-diagonal = scatter + correlation)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("eda_asset_scatter_matrix.png", dpi=150)
plt.show()


# ── [EDA-4b] ASSET RETURN CORRELATION MATRIX ─────────────────────────────────
corr_assets  = backtest[RET_COLS].corr()
asset_labels = ASSETS

fig, ax = plt.subplots(figsize=(len(ASSETS) * 1.4 + 1.5, len(ASSETS) * 1.4))
im = ax.imshow(corr_assets.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(asset_labels))); ax.set_xticklabels(asset_labels, fontsize=11)
ax.set_yticks(range(len(asset_labels))); ax.set_yticklabels(asset_labels, fontsize=11)
for i in range(len(asset_labels)):
    for j in range(len(asset_labels)):
        val = corr_assets.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if abs(val) > 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")
ax.set_title("Asset Return Correlation Matrix\n"
             "(green = positive, red = negative -- diversification from low/negative pairs)",
             fontsize=12)
plt.tight_layout()
plt.savefig("eda_asset_correlation.png", dpi=150)
plt.show()


# ── [EDA-5] MACRO FEATURES OVER TIME ─────────────────────────────────────────
macro_full = pd.concat([pretrain[MACRO_FEATURES], backtest[MACRO_FEATURES]]).sort_index()
n_feat = len(MACRO_FEATURES)
ncols  = 3
nrows  = int(np.ceil(n_feat / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
axes = axes.flatten()

for i, feat in enumerate(MACRO_FEATURES):
    ax = axes[i]
    pre_series = pretrain[feat].dropna()
    bt_series  = backtest[feat].dropna()
    ax.plot(pre_series.index, pre_series.values,
            color="grey", linewidth=1, alpha=0.8, label="Pretrain")
    ax.plot(bt_series.index,  bt_series.values,
            color="steelblue", linewidth=1.2, label="Backtest")
    ax.axvline(backtest.index[0], color="red", linestyle="--",
               linewidth=0.8, alpha=0.7)
    ax.set_title(feat, fontsize=9)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Macro Features Over Time  (grey=pretrain | blue=backtest | red=split)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("eda_macro_features_time.png", dpi=150)
plt.show()


# ── [EDA-6] MACRO FEATURE DISTRIBUTIONS ──────────────────────────────────────
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
axes = axes.flatten()

for i, feat in enumerate(MACRO_FEATURES):
    ax   = axes[i]
    vals = macro_full[feat].dropna()
    mu, sigma = vals.mean(), vals.std()
    skew = vals.skew()
    kurt = vals.kurtosis()

    ax.hist(vals, bins=35, density=True, alpha=0.65,
            color="steelblue", edgecolor="white")
    xgrid = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xgrid, scipy_stats.norm.pdf(xgrid, mu, sigma),
            color="black", linewidth=1.5, label="Normal fit")
    ax.set_title(f"{feat}\nskew={skew:.2f}  kurt={kurt:.2f}", fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Macro Feature Distributions (pretrain + backtest combined)\n"
             "Heavy skew or multimodality may distort KMeans cluster geometry",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("eda_macro_distributions.png", dpi=150)
plt.show()


# =============================================================================
# Run
# =============================================================================
print("\nRunning walk-forward backtest (with Hungarian centroid anchoring)...")
results, weights, kmeans, scaler = run_backtest(backtest, pretrain)

print("\n================ PERFORMANCE ================")
perf_stats("Regime Portfolio",       results["portfolio_cum"],       results["portfolio_ret"])
perf_stats("SP500 Buy & Hold",       results["SP500_cum"],           results["SP500_ret"])
perf_stats("Equal Weight Portfolio", results["equal_portfolio_cum"], results["equal_portfolio_ret"])

# ── Silhouette ────────────────────────────────────────────────────────────────
print("\n================ OPTIMAL K DIAGNOSTIC ================")
X_all    = pd.concat([pretrain, backtest[MACRO_FEATURES]]).dropna()
X_scaled = StandardScaler().fit_transform(X_all)
K_range  = range(2, 9)
sil      = [silhouette_score(X_scaled,
            KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled))
            for k in K_range]
best_k   = list(K_range)[np.argmax(sil)]
print(f"  K used in backtest : {K}")
print(f"  Best K (silhouette): {best_k}   score = {max(sil):.4f}")
if best_k != K:
    print(f"  Consider re-running with K={best_k}")


# =============================================================================
# Main result plots
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

ax = axes[0]
ax.plot(results["portfolio_cum"],       label="Regime Portfolio",        linewidth=2, color="steelblue")
ax.plot(results["SP500_cum"],           label="SPY Buy & Hold",          linewidth=2, color="black",  linestyle="--", alpha=0.7)
ax.plot(results["equal_portfolio_cum"], label="Equal Weight (25% each)", linewidth=2, color="green",  linestyle=":")
ax.set_title("Cumulative Return: Regime Portfolio vs SPY Total Return\n"
             "(Both rebased to $1 on same start date -- dividend-adjusted)")
ax.set_ylabel("Growth of $1")
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
weights.plot.area(ax=ax2, alpha=0.75)
ax2.set_title("Asset Allocation Over Time (after smoothing)")
ax2.set_ylabel("Weight"); ax2.set_ylim(0, 1)
ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.plot(list(K_range), sil, marker="o", color="steelblue")
ax3.axvline(K,      color="steelblue", linestyle=":",  alpha=0.7, label=f"K used = {K}")
ax3.axvline(best_k, color="red",       linestyle="--", alpha=0.6, label=f"Best K = {best_k}")
ax3.set_title("Silhouette Score by Number of Regimes")
ax3.set_xlabel("k"); ax3.set_ylabel("Silhouette Score")
ax3.legend(); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("portfolio_results.png", dpi=150)
plt.show()


# ── Regime diagnostics ────────────────────────────────────────────────────────
X_bt   = backtest[MACRO_FEATURES].dropna()
X_bt_s = scaler.transform(X_bt)
labels = kmeans.predict(X_bt_s)

pca       = PCA(n_components=2).fit(X_bt_s)
X_2d      = pca.transform(X_bt_s)
centroids = pca.transform(kmeans.cluster_centers_)

regime_colors = plt.cm.tab10(np.linspace(0, 0.4, K))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
for c in range(K):
    mask_c = labels == c
    ax.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1],
               color=regime_colors[c], alpha=0.7, s=20, label=f"Regime {c}")
    ax.scatter(centroids[c, 0], centroids[c, 1],
               color=regime_colors[c], marker="X", s=300,
               edgecolors="black", linewidths=1.5, zorder=5)
    ax.annotate(f"C{c}", (centroids[c, 0], centroids[c, 1]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=10, fontweight="bold")
ax.set_title(f"PCA Projection -- {K} Macro Regimes")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax.legend(); ax.grid(alpha=0.1)

ax2 = axes[1]
sp_log_ret = np.log1p(results["SP500"])
for c in range(K):
    mask = results["regime"] == c
    ax2.bar(results.index[mask], sp_log_ret[mask],
            color=regime_colors[c], alpha=0.85, width=20, label=f"Regime {c}")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_title("SPY Log Returns Colored by Predicted Regime")
ax2.set_ylabel("Log Return")
ax2.legend(); ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("regime_diagnostics.png", dpi=150)
plt.show()


# ── Elbow Method ─────────────────────────────────────────────────────────────
print("\n================ ELBOW METHOD ================")
inertia       = []
K_range_elbow = range(1, 10)
for k in K_range_elbow:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range_elbow, inertia, marker="o")
plt.title("Elbow Plot: Inertia vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(alpha=0.3)
plt.savefig("elbow_plot.png", dpi=150)
plt.show()


# ── Monthly return statistics per regime ──────────────────────────────────────
print("\n================ PER-REGIME RETURN STATS ================")
for c in range(K):
    mask = results["regime"] == c
    r    = results.loc[mask, "portfolio_ret"]
    sp   = results.loc[mask, "SP500_ret"]
    print(f"\n  Regime {c}  ({mask.sum()} months)")
    print(f"    Portfolio  mean={r.mean():.2%}  vol={r.std():.2%}  "
          f"SR={r.mean()/r.std()*np.sqrt(12):.2f}")
    print(f"    SP500      mean={sp.mean():.2%}  vol={sp.std():.2%}  "
          f"SR={sp.mean()/sp.std()*np.sqrt(12):.2f}")


# =============================================================================
# DESCRIPTIVE DIAGNOSTICS
# =============================================================================
bt_labeled           = backtest.copy()
bt_labeled["regime"] = results["regime"]

# ── [1] REGIME TIMELINE ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True,
                         gridspec_kw={"height_ratios": [1, 3]})

ax_strip = axes[0]
for c in range(K):
    mask = results["regime"] == c
    ax_strip.fill_between(results.index, 0, 1,
                          where=mask.values, color=regime_colors[c],
                          alpha=0.85, label=f"Regime {c}")
ax_strip.set_yticks([])
ax_strip.set_ylabel("Regime", fontsize=9)
ax_strip.set_title("Regime Timeline & SPY Cumulative Return")
ax_strip.legend(loc="upper left", fontsize=8, ncol=K)

ax_spy = axes[1]
ax_spy.plot(results.index, results["SP500_cum"], color="black", linewidth=1.5)
ax_spy.set_ylabel("SPY Growth of $1")
ax_spy.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("regime_timeline.png", dpi=150)
plt.show()


# ── [DRIFT DIAGNOSTIC] ROLLING REGIME FREQUENCY & ENTROPY ────────────────────
# After Hungarian anchoring, regimes should be stable.
# - Frequency lines (top): should be smooth, slowly varying — not chaotic flips.
# - Shannon entropy (bottom): near 0 = one regime dominates (persistent);
#   near log(K) = regimes flip randomly. Lower entropy = better anchoring.
window = 24
regime_dummies = pd.get_dummies(results["regime"])
for c in range(K):
    if c not in regime_dummies.columns:
        regime_dummies[c] = 0
regime_dummies = regime_dummies[[c for c in range(K)]]

def rolling_entropy(row):
    p = row.values.astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    p = p[p > 0]
    return -np.sum(p * np.log(p))

roll_freq    = regime_dummies.rolling(window).mean()
roll_entropy = roll_freq.apply(rolling_entropy, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax = axes[0]
for c in range(K):
    ax.plot(roll_freq.index, roll_freq[c],
            color=regime_colors[c], linewidth=1.5, label=f"Regime {c}")
ax.set_ylabel(f"Frequency ({window}m rolling)")
ax.set_title(f"Rolling Regime Frequency & Entropy ({window}-month window)\n"
             "Smooth lines = stable anchored regimes; spiky lines = label drift")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(roll_entropy.index, roll_entropy, color="purple", linewidth=1.5)
ax2.axhline(np.log(K), color="red", linestyle="--", alpha=0.6,
            label=f"Max entropy = {np.log(K):.2f} (fully mixed)")
ax2.set_ylabel("Shannon Entropy"); ax2.set_xlabel("Date")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("regime_drift_diagnostic.png", dpi=150)
plt.show()


# ── [2] MACRO FEATURE MEANS PER REGIME (HEATMAP) ─────────────────────────────
fig, ax = plt.subplots(figsize=(max(10, len(MACRO_FEATURES) * 0.6 + 2), 4))

regime_macro_means = pd.DataFrame(index=range(K), columns=MACRO_FEATURES, dtype=float)
X_bt_df = backtest[MACRO_FEATURES].dropna().copy()
X_bt_df["regime"] = kmeans.predict(scaler.transform(X_bt_df))

for c in range(K):
    regime_macro_means.loc[c] = X_bt_df[X_bt_df["regime"] == c][MACRO_FEATURES].mean()

zm = regime_macro_means.astype(float)
zm = (zm - zm.mean()) / (zm.std() + 1e-9)

im = ax.imshow(zm.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(MACRO_FEATURES)))
ax.set_xticklabels(MACRO_FEATURES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(K))
ax.set_yticklabels([f"Regime {c}" for c in range(K)])
for i in range(K):
    for j in range(len(MACRO_FEATURES)):
        ax.text(j, i, f"{regime_macro_means.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7, color="black")
plt.colorbar(im, ax=ax, label="Z-score vs cross-regime mean")
ax.set_title("Macro Feature Means per Regime\n"
             "(green = high relative to other regimes, red = low)")
plt.tight_layout()
plt.savefig("regime_macro_heatmap.png", dpi=150)
plt.show()


# ── [3] PER-REGIME ASSET RETURN DISTRIBUTIONS (BOXPLOTS) ─────────────────────
fig, axes = plt.subplots(1, len(RET_COLS), figsize=(5 * len(RET_COLS), 5), sharey=False)

for j, col in enumerate(RET_COLS):
    ax = axes[j]
    data_by_regime = [results.loc[results["regime"] == c, col].dropna().values * 100
                      for c in range(K)]
    bp = ax.boxplot(data_by_regime, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], regime_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(ASSETS[j], fontsize=11)
    ax.set_xticklabels([f"R{c}" for c in range(K)])
    ax.set_ylabel("Monthly Return (%)" if j == 0 else "")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Asset Return Distributions per Regime\n"
             "(Validates Markowitz inputs -- regimes should show different risk/return profiles)",
             fontsize=12)
plt.tight_layout()
plt.savefig("regime_asset_distributions.png", dpi=150)
plt.show()


# ── [4] REGIME TRANSITION MATRIX ─────────────────────────────────────────────
trans = np.zeros((K, K), dtype=int)
regime_seq = results["regime"].values
for i in range(len(regime_seq) - 1):
    trans[regime_seq[i], regime_seq[i + 1]] += 1

trans_pct = trans / trans.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(trans_pct, cmap="Blues", vmin=0, vmax=1)
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{trans_pct[i, j]:.0%}",
                ha="center", va="center", fontsize=13,
                color="white" if trans_pct[i, j] > 0.5 else "black")
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels([f"-> Regime {c}" for c in range(K)])
ax.set_yticklabels([f"Regime {c}" for c in range(K)])
ax.set_title("Regime Transition Matrix\n(row = from, col = to)")
plt.colorbar(im, ax=ax, label="Transition probability")
plt.tight_layout()
plt.savefig("regime_transition_matrix.png", dpi=150)
plt.show()

print("\nTransition matrix (counts):")
print(pd.DataFrame(trans, index=[f"From R{c}" for c in range(K)],
                   columns=[f"To R{c}" for c in range(K)]))


# ── [5] ROLLING DRAWDOWN ─────────────────────────────────────────────────────
def rolling_dd(cum): return (cum - cum.cummax()) / cum.cummax()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(results.index, rolling_dd(results["portfolio_cum"]) * 100,
                alpha=0.4, color="steelblue", label="Regime Portfolio")
ax.fill_between(results.index, rolling_dd(results["SP500_cum"]) * 100,
                alpha=0.3, color="black", label="SPY")
ax.fill_between(results.index, rolling_dd(results["equal_portfolio_cum"]) * 100,
                alpha=0.3, color="green", label="Equal Weight")
ax.set_title("Rolling Drawdown (%)")
ax.set_ylabel("Drawdown (%)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rolling_drawdown.png", dpi=150)
plt.show()


# ── [6] MACRO FEATURE CORRELATION HEATMAP ────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(MACRO_FEATURES) * 0.65),
                                max(6, len(MACRO_FEATURES) * 0.65)))
corr = backtest[MACRO_FEATURES].dropna().corr()
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(MACRO_FEATURES)))
ax.set_yticks(range(len(MACRO_FEATURES)))
ax.set_xticklabels(MACRO_FEATURES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(MACRO_FEATURES, fontsize=8)
for i in range(len(MACRO_FEATURES)):
    for j in range(len(MACRO_FEATURES)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
                color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")
plt.colorbar(im, ax=ax)
ax.set_title("Macro Feature Correlation Matrix\n"
             "(Highly correlated pairs may distort KMeans cluster geometry)")
plt.tight_layout()
plt.savefig("macro_feature_correlation.png", dpi=150)
plt.show()


# ── [7] MONTHLY RETURN HEATMAP (PORTFOLIO) ───────────────────────────────────
ret_pivot = results["portfolio_ret"].copy()
ret_pivot.index = pd.to_datetime(ret_pivot.index)
monthly = ret_pivot.to_frame("ret")
monthly["year"]  = monthly.index.year
monthly["month"] = monthly.index.month

pivot = monthly.pivot(index="year", columns="month", values="ret") * 100
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.45)))
im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=7, color="black" if abs(val) < 7 else "white")
plt.colorbar(im, ax=ax, label="Monthly Return (%)")
ax.set_title("Portfolio Monthly Return Heatmap (%)\n"
             "(Green = positive, Red = negative -- clipped at +-10%)")
plt.tight_layout()
plt.savefig("monthly_return_heatmap.png", dpi=150)
plt.show()


# ── Save ──────────────────────────────────────────────────────────────────────
results.to_csv("backtest_results.csv")
weights.to_csv("portfolio_weights.csv")
print("\nSaved: backtest_results.csv, portfolio_weights.csv")
print("Saved: portfolio_results.png, regime_diagnostics.png")
print("Saved: regime_timeline.png, regime_drift_diagnostic.png")
print("Saved: regime_macro_heatmap.png, regime_asset_distributions.png")
print("Saved: regime_transition_matrix.png, rolling_drawdown.png")
print("Saved: macro_feature_correlation.png, monthly_return_heatmap.png")
print("Saved: eda_asset_cumulative.png, eda_return_distributions.png")
print("Saved: eda_asset_summary_table.png, eda_asset_scatter_matrix.png")
print("Saved: eda_asset_correlation.png, eda_macro_features_time.png")
print("Saved: eda_macro_distributions.png")