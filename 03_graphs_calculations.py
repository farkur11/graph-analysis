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


# %% check cuda
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

# %%
dfp = pl.scan_parquet("agg_output.parquet").collect()
dfp = dfp.sample(fraction=0.05, seed=42)

# %% 
#dfp.sample(10)
dfp.head()

# %%

selected_attributes = [
    "n_transfers",
    "total_amount",
    "avg_rel_amount"]

# %% 

torch.cuda.is_available()
from torch_geometric.data import Data

unique_nodes = sorted(set(dfp["sender_card"]) | set(dfp["receiver_card"]))
node_map = {card: i for i, card in enumerate(unique_nodes)}
dfp["sender_idx"] = dfp["sender_card"].map(node_map)
dfp["receiver_idx"] = dfp["receiver_card"].map(node_map)

# Building edge_index (2 x num_edges)
edge_index = torch.tensor(
    [dfp["sender_idx"].values, dfp["receiver_idx"].values],
    dtype=torch.long
)

# Adding edge attributes
edge_attr = torch.tensor(
    dfp[selected_attributes].values,
    dtype=torch.float32
)

# Creating PyG graph object (GPU-ready)
data = Data(edge_index=edge_index, edge_attr=edge_attr)
data.num_nodes = len(unique_nodes)
data = data.to("cuda")

# %% Pagerank

def pagerank_pyg(data, alpha=0.85, max_iter=50, tol=1e-6):
    N = data.num_nodes
    pr = torch.full((N,), 1.0 / N, dtype=torch.float32, device="cuda")
    edge_index = data.edge_index

    for _ in range(max_iter):
        prev_pr = pr.clone()
        message = pr[edge_index[0]]
        out_deg = torch.bincount(edge_index[0], minlength=N).float()
        norm_msg = message / out_deg[edge_index[0]].clamp(min=1)

        pr = torch.zeros(N, device="cuda").scatter_add(0, edge_index[1], norm_msg)
        pr = (1 - alpha) / N + alpha * pr
        if torch.norm(pr - prev_pr, p=1) < tol:
            break
    return pr

pagerank_scores = pagerank_pyg(data)

# %%
out_degree = torch.bincount(edge_index[0], minlength=data.num_nodes)
in_degree = torch.bincount(edge_index[1], minlength=data.num_nodes)
total_degree = in_degree + out_degree

centrality_df = pd.DataFrame({
    "node": unique_nodes,
    "pagerank": pagerank_scores.cpu().numpy(),
    "in_degree": in_degree.cpu().numpy(),
    "out_degree": out_degree.cpu().numpy(),
    "degree": total_degree.cpu().numpy(),
})

# Normalize to match NetworkX-style centrality (divide by N-1)
n = data.num_nodes
centrality_df["in_degree_centrality"] = centrality_df["in_degree"] / (n - 1)
centrality_df["out_degree_centrality"] = centrality_df["out_degree"] / (n - 1)
centrality_df["degree_centrality"] = centrality_df["degree"] / (n - 1)

# %% 
import networkx as nx

# Convert edge_index and edge_attr to edge list with weight
edge_list = dfp[["sender_card", "receiver_card", "total_amount"]].copy()
edge_list["weight"] = edge_list["total_amount"]

# Build NetworkX graph (undirected, weighted)
G_undirected = nx.Graph()
G_undirected.add_weighted_edges_from(
    edge_list[["sender_card", "receiver_card", "weight"]].values
)

# %% Community

import community as community_louvain

# Run Louvain algorithm
node_to_community = community_louvain.best_partition(G_undirected, weight="weight")


# Convert to DataFrame
comm_df = pd.DataFrame({
    "node": list(node_to_community.keys()),
    "community": list(node_to_community.values())
})

# Merge with centrality_df
centrality_df = centrality_df.merge(comm_df, on="node", how="left")

centrality_df.to_csv("centrality.csv")


#%%
comm_sizes = centrality_df.groupby("community")["node"].count().rename("community_size")
centrality_df = centrality_df.merge(comm_sizes, on="community", how="left")
centrality_df.head()

# %%
agg_features = dfp.groupby("sender_card").agg(
    n_transfers_sent=("n_transfers", "sum"),
    total_amount_sent=("total_amount", "sum"),
    avg_rel_amt_sent=("avg_rel_amount", "mean"),
    avg_roundness_sent=("avg_roundness_level", "mean"),
    share_morning_sent=("share_morning", "mean"),
    share_night_sent=("share_night", "mean"),
    share_weekend_sent=("share_weekend", "mean"),
    share_family_sent=("share_family", "mean"),
    share_same_person_sent=("share_same_person", "mean"),
    inter_transfer_days_mean=("inter_transfer_days", "mean")
).reset_index().rename(columns={"sender_card": "node"})

agg_features_recv = dfp.groupby("receiver_card").agg(
    n_transfers_recv=("n_transfers", "sum"),
    total_amount_recv=("total_amount", "sum"),
    avg_rel_amt_recv=("avg_rel_amount", "mean"),
    avg_roundness_recv=("avg_roundness_level", "mean"),
    share_family_recv=("share_family", "mean"),
    share_same_person_recv=("share_same_person", "mean")
).reset_index().rename(columns={"receiver_card": "node"})


centrality_df = centrality_df.merge(agg_features, on="node", how="left")
centrality_df = centrality_df.merge(agg_features_recv, on="node", how="left")
centrality_df.fillna(0, inplace=True)
# %%
centrality_df.sample(5)
# %%
