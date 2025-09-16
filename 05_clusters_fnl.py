# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import re
import psycopg2
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date, timedelta
import glob
import torch
from torch_geometric.data import Data
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


df = pd.read_csv("df_for_clustering.csv").drop(columns=["Unnamed: 0"])
df.head()
# %%
df.info()
# %%

df["community_size"] = df.groupby("community")["community"].transform("size")

# %%
print(df["community"].nunique())
print(df["original_node_id"].nunique())

com_size = df[["community", "community_size"]].drop_duplicates()
com_size = com_size.groupby("community_size").count().reset_index()


com_size = com_size.sort_values("community_size").reset_index(drop=True)


from matplotlib.ticker import MultipleLocator, PercentFormatter

def plot_pareto_by_size(com_size, x_max=20, pct_threshold=90):
    # collapse to counts per size and ensure every integer size appears
    s = (com_size.groupby("community_size", as_index=True)["community"]
                 .sum()
                 .sort_index())
    idx = pd.Index(range(1, x_max + 1), name="community_size")  # start at 1; use range(0, ...) if you have zeros
    s = s.reindex(idx, fill_value=0)

    share = 100 * s / s.sum()   # % of communities per size
    cum   = share.cumsum()      # cumulative %

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.bar(share.index, share.values, width=0.9, edgecolor="black", alpha=0.6, label="% per size")
    ax1.set_xlabel("Community Size")
    ax1.set_ylabel("% of Communities (per size)")
    ax1.yaxis.set_major_formatter(PercentFormatter(100))
    ax1.xaxis.set_major_locator(MultipleLocator(1))     # step = 1
    ax1.set_xlim(0.5, x_max + 0.5)                      # center bars on integers
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(cum.index, cum.values, marker="o", linewidth=2, label="Cumulative %")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Cumulative % of Communities")
    ax2.yaxis.set_major_formatter(PercentFormatter(100))

    # threshold marker (e.g., 90%)
    k = int(np.searchsorted(cum.values, pct_threshold) + 1)
    ax2.axhline(pct_threshold, linestyle="--", alpha=0.7)
    ax2.axvline(k, linestyle="--", alpha=0.7)
    ax2.annotate(f"{pct_threshold}% reached by size ≤ {k}",
                 xy=(k, pct_threshold), xytext=(k+0.5, pct_threshold-8),
                 arrowprops=dict(arrowstyle="->", lw=1), ha="left", va="top")

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")

    plt.title("Communities by Size: Pareto")
    plt.tight_layout()
    plt.show()

# usage
plot_pareto_by_size(com_size, x_max=20, pct_threshold=90)

# %%

def plot_pareto_nodes_by_size(com_size, x_max=20, pct_threshold=90):
    # counts of communities per size
    s = (com_size.groupby("community_size", as_index=True)["community"]
                 .sum()
                 .sort_index())

    # ensure every integer size appears on the x-axis
    idx = pd.Index(range(1, x_max + 1), name="community_size")  # use range(0, ...) if you truly have size 0
    s = s.reindex(idx, fill_value=0)

    # total nodes contributed by each community size = count * size
    nodes_by_size = s.index.to_series() * s

    # shares and cumulative shares of NODES
    share_nodes = 100 * nodes_by_size / nodes_by_size.sum()
    cum_nodes   = share_nodes.cumsum()

    # --- plot ---
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.bar(share_nodes.index, share_nodes.values, width=0.9,
            edgecolor="black", alpha=0.6, label="% of nodes per size")
    ax1.set_xlabel("Community Size")
    ax1.set_ylabel("% of Nodes (per size)")
    ax1.yaxis.set_major_formatter(PercentFormatter(100))
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.set_xlim(0.5, x_max + 0.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(cum_nodes.index, cum_nodes.values, marker="o", linewidth=2,
             label="Cumulative % of nodes")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Cumulative % of Nodes")
    ax2.yaxis.set_major_formatter(PercentFormatter(100))

    # threshold marker (e.g., 90% of nodes)
    pos = np.searchsorted(cum_nodes.values, pct_threshold)
    if pos < len(cum_nodes):
        k_size = int(cum_nodes.index[pos])
        ax2.axhline(pct_threshold, linestyle="--", alpha=0.7)
        ax2.axvline(k_size, linestyle="--", alpha=0.7)
        ax2.annotate(f"{pct_threshold}% nodes by size ≤ {k_size}",
                     xy=(k_size, pct_threshold), xytext=(k_size+0.5, pct_threshold-8),
                     arrowprops=dict(arrowstyle="->", lw=1), ha="left", va="top")

    # combined legend
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")

    plt.title("Nodes by Community Size: Pareto")
    plt.tight_layout()
    plt.show()

# use it
plot_pareto_nodes_by_size(com_size, x_max=20, pct_threshold=90)

# %%
numeric_df = df.select_dtypes(include=["number"])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Display with seaborn heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
# %%
# Compute absolute correlation matrix and flatten it
corr_pairs = corr_matrix.abs().unstack()

# Remove self-correlations and duplicates
corr_pairs = corr_pairs[corr_pairs < 1.0]
corr_pairs = corr_pairs.drop_duplicates()

# Filter for high correlations (e.g., above 0.5)
high_corr = corr_pairs[corr_pairs > 0.5]

# Sort descending
high_corr_sorted = high_corr.sort_values(ascending=False)

# Display result
print(high_corr_sorted)

# %%
selected_features = [
    "pagerank",
    "out_degree",
    "total_amount_sent",
    "total_amount_recv",
    "avg_rel_amt_sent",
    "avg_rel_amt_recv",
    "share_weekend_sent",
    "community_size",
    "share_same_person_sent",
    "share_family_sent"
]

import math

num_features = len(selected_features)
cols = 3
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    sns.histplot(df[feature], ax=axes[i], kde=True, bins=50)
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# %%

df_log = df[selected_features].apply(lambda x: np.log1p(x))
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(df_log.columns):
    sns.histplot(df_log[col], bins=40, ax=axes[i], kde=True, color="steelblue")
    axes[i].set_title(f"Log Distribution: {col}")
    axes[i].set_xlabel("log(1 + x)")

# Remove unused subplots
for j in range(len(df_log.columns), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()
# %%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_log), columns=selected_features)

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(df_scaled.columns):
    sns.histplot(df_scaled[col], bins=40, ax=axes[i], kde=True, color="seagreen")
    axes[i].set_title(f"Standardized Log: {col}")
    axes[i].set_xlabel("Scaled log(1 + x)")

# Remove unused subplots
for j in range(len(df_scaled.columns), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()
# %%
df.sample(10)

# %%
from sklearn.decomposition import PCA

# 1. Initialize PCA (keep all components initially)
pca = PCA(n_components=None)  # we’ll analyze all PCs first
pca_result = pca.fit_transform(df_scaled)

# 2. Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 3. Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker="o")
plt.axhline(y=0.95, color="r", linestyle="--", label="95% variance threshold")
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%

# Reduce to 6 dimensions
pca = PCA(n_components=7)
pca_reduced = pca.fit_transform(df_scaled)

# Create DataFrame with PC labels
pca_df = pd.DataFrame(pca_reduced, columns=[f"PC{i+1}" for i in range(7)])
pca_df.head()

# %%

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(7)],
    index=df_scaled.columns
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
plt.title("PCA Loadings (Feature Contribution to Principal Components)")
plt.tight_layout()
plt.show()

# %%

from sklearn.mixture import GaussianMixture

X = pca_df.values  # PCA-reduced data
aic_scores = []
bic_scores = []
n_components_range = range(2, 13)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
    gmm.fit(X)
    aic_scores.append(gmm.aic(X))
    bic_scores.append(gmm.bic(X))

# Plot AIC/BIC
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, aic_scores, label="AIC", marker="o")
plt.plot(n_components_range, bic_scores, label="BIC", marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Information Criterion")
plt.title("GMM Model Selection via AIC and BIC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# Fit GMM with optimal cluster count
from sklearn.mixture import GaussianMixture


optimal_n = 9
gmm = GaussianMixture(n_components=optimal_n, covariance_type="full", random_state=42)
gmm.fit(pca_df)

# Assign cluster labels
pca_df["gmm_cluster"] = gmm.predict(pca_df)
pca_df["gmm_prob_max"] = gmm.predict_proba(pca_df.drop(columns="gmm_cluster")).max(axis=1)

# %%
## Visualize Clusters in PCA Space 
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1", y="PC2",
    hue="gmm_cluster",
    palette="Set2",
    alpha=0.6,
    s=15
)
plt.title("GMM Clustering Result (Visualized in PC1 vs PC2)")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
cluster_counts = pca_df["gmm_cluster"].value_counts().sort_index()
# Percentages
cluster_percents = (cluster_counts / cluster_counts.sum() * 100).round(1)

plt.figure(figsize=(10, 6))
bars = sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values,
    palette="viridis"
)

# Add percentage labels on top
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    label = f"{cluster_percents.iloc[i]}%"
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (0.01 * height),
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold"
    )

plt.title("User Count per GMM Cluster (with Percentage Labels)")
plt.xlabel("Cluster Label")
plt.ylabel("User Count")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# %%
df["gmm_cluster"] = pca_df["gmm_cluster"].values

# Group by cluster
amounts_by_cluster = df.groupby("gmm_cluster")[
    ["total_amount_sent", "total_amount_recv"]
].sum()

# Compute total network volume
network_totals = amounts_by_cluster.sum()

# Calculate percentage share per cluster
amounts_by_cluster_pct = (amounts_by_cluster / network_totals) * 100
amounts_by_cluster_pct = amounts_by_cluster_pct.round(1)

# Combine counts and percent in one frame for clarity
summary = amounts_by_cluster.copy()
summary["sent_pct"] = amounts_by_cluster_pct["total_amount_sent"]
summary["recv_pct"] = amounts_by_cluster_pct["total_amount_recv"]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot total amount sent share
sns.barplot(
    x=summary.index,
    y="sent_pct",
    data=summary,
    palette="Blues",
    ax=ax[0]
)
ax[0].set_title("Share of Total Amount Sent per Cluster")
ax[0].set_xlabel("Cluster")
ax[0].set_ylabel("Sent Share (%)")
ax[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot total amount received share
sns.barplot(
    x=summary.index,
    y="recv_pct",
    data=summary,
    palette="Greens",
    ax=ax[1]
)
ax[1].set_title("Share of Total Amount Received per Cluster")
ax[1].set_xlabel("Cluster")
ax[1].set_ylabel("Received Share (%)")
ax[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
# %%



# %%
df.to_csv("clustered_df2.csv", index=False)

# %%
df.head()

# %%
features_to_plot = [
    "pagerank", "out_degree", "in_degree", "degree",
    "n_transfers_sent", "n_transfers_recv",
    "total_amount_sent", "total_amount_recv",
    "avg_rel_amt_sent", "avg_rel_amt_recv",
    "avg_roundness_sent", "avg_roundness_recv",
    "inter_transfer_days_mean",  "share_morning_sent", "share_night_sent", "share_weekend_sent", 
    "share_family_sent",  "share_family_recv", 
    "share_same_person_sent", "share_same_person_recv", "community_size"
]

# import math

# def plot_features_by_cluster(data, features, cluster_col="gmm_cluster", batch_size=6):
#     n = len(features)
#     batches = math.ceil(n / batch_size)

#     for i in range(batches):
#         batch = features[i * batch_size: (i + 1) * batch_size]
#         n_cols = 2
#         n_rows = math.ceil(len(batch) / n_cols)

#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
#         axes = axes.flatten()

#         for j, feature in enumerate(batch):
#             sns.boxplot(
#                 data=data,
#                 x=cluster_col,
#                 y=feature,
#                 ax=axes[j],
#                 palette="Set2",
#                 showfliers=False  # optional: hide outliers
#             )
#             axes[j].set_title(f"{feature} by Cluster")
#             axes[j].set_xlabel("Cluster")
#             axes[j].set_ylabel(feature)
#             axes[j].grid(True, linestyle="--", alpha=0.5)

#         # Delete extra subplots
#         for k in range(len(batch), len(axes)):
#             fig.delaxes(axes[k])

#             plt.tight_layout()
#             filename = f"boxplots_batch_{i+1}.png"
#             plt.savefig(filename, dpi=300)
#             plt.show()
        
# plot_features_by_cluster(df, features_to_plot, cluster_col="gmm_cluster", batch_size=6)

# %%

import os
import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_features_by_cluster(data, features, cluster_col="gmm_cluster", batch_size=6, save_dir="boxplots"):
    # Create output directory if it doesn"t exist
    os.makedirs(save_dir, exist_ok=True)

    n = len(features)
    batches = math.ceil(n / batch_size)

    for i in range(batches):
        batch = features[i * batch_size: (i + 1) * batch_size]
        n_cols = 2
        n_rows = math.ceil(len(batch) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten()

        for j, feature in enumerate(batch):
            sns.boxplot(
                data=data,
                x=cluster_col,
                y=feature,
                ax=axes[j],
                palette="Set2",
                showfliers=False
            )
            axes[j].set_title(f"{feature} by Cluster")
            axes[j].set_xlabel("Cluster")
            axes[j].set_ylabel(feature)
            axes[j].grid(True, linestyle="--", alpha=0.5)

        # Remove extra axes
        for k in range(len(batch), len(axes)):
            fig.delaxes(axes[k])

        # Save plot
        plt.tight_layout()
        filename = os.path.join(save_dir, f"alt_boxplots_batch_{i+1}.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)  # Release memory

    print(f"✅ Saved {batches} boxplot images to: {os.path.abspath(save_dir)}")

plot_features_by_cluster(df, features_to_plot, cluster_col="gmm_cluster", batch_size=6)

# %%
commm = df[["community", "community_size"]].drop_duplicates()
commm = commm.groupby("community_size").count().reset_index()
commm["nodes"] = commm["community_size"] * commm["community"] 
commm["nodes_prc"] = 100 * commm["nodes"] / commm["nodes"].sum()
commm = commm.sort_values("community_size").reset_index(drop=True)
commm["nodes_cumprc"] = commm["nodes_prc"].cumsum()
commm.head()

# %%


# data for zoomed panel
zoom = commm[commm["community_size"] <= 100].copy()

fig, axes = plt.subplots(1, 2, figsize=(14,6))  # no sharey

# --- full range ---
axes[0].plot(commm["community_size"], commm["nodes_cumprc"], marker=".")
axes[0].set_title("Full Range")
axes[0].set_xlabel("Community Size")
axes[0].set_ylabel("Cumulative % of Nodes")
axes[0].yaxis.set_major_formatter(PercentFormatter(100))
axes[0].set_ylim(0, 100)           # keep full 0–100% here
axes[0].grid(True)

# --- zoomed (independent y) ---
axes[1].plot(zoom["community_size"], zoom["nodes_cumprc"], marker=".")
axes[1].set_title("Zoomed In (≤ 1000)")
axes[1].set_xlabel("Community Size")
axes[1].yaxis.set_major_formatter(PercentFormatter(100))
axes[1].set_xlim(0, 100)
# independent y-limits with a little padding
ymin, ymax = zoom["nodes_cumprc"].min(), zoom["nodes_cumprc"].max()
pad = 0.05 * max(ymax - ymin, 1e-6)
axes[1].set_ylim(max(0, ymin - pad), min(100, ymax + pad))
axes[1].grid(True)

plt.tight_layout()
plt.show()
# %%
