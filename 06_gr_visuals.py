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
import networkx as nx
from pyvis.network import Network



# %%
centrality_df = pd.read_csv("clustered_df2.csv")
dfp = pd.read_parquet("agg_output.parquet")

centrality_df.info()
# %%
dfp.info()
# %%

centrality_df['original_node_id'] = centrality_df['original_node_id'].astype(str)
dfp['sender_card'] = dfp['sender_card'].astype(str)
dfp['receiver_card'] = dfp['receiver_card'].astype(str)
# %%
dfp.head()
# %%

from pyvis.network import Network
import numpy as np, pandas as pd, os, re

# ─────────── INTERPRET REL AMOUNT ───────────
def interpret_rel_amount(x):
    if pd.isna(x): return "unknown"
    elif x <= 0.05: return "negligible"
    elif x <= 0.15: return "low"
    elif x <= 0.35: return "moderate"
    elif x <= 0.6: return "high"
    elif x <= 0.85: return "heavy"
    else: return "aggressive"

# ─────────── EGO VISUALIZER ───────────
def plot_ego_direct(focal_id, dfp, centrality_df, output_dir="visuals"):
    os.makedirs(output_dir, exist_ok=True)

    focal_id = str(focal_id)
    dfp["sender_card"] = dfp["sender_card"].astype(str)
    dfp["receiver_card"] = dfp["receiver_card"].astype(str)

    # only direct edges involving focal
    dfp_sub = dfp[(dfp["sender_card"]==focal_id) | (dfp["receiver_card"]==focal_id)].copy()
    if dfp_sub.empty:
        print(f"⚠️ No direct edges for node {focal_id}")
        return None

    # ---- node stats ----
    send_stats = dfp_sub.groupby("sender_card").agg(
        sender_gender=('sender_gender','first'),
        sender_name=('sender_name','first'),
        sent_amount=('total_amount','sum')
    ).reset_index().rename(columns={"sender_card":"node"})

    recv_stats = dfp_sub.groupby("receiver_card").agg(
        receiver_gender=('receiver_gender','first'),
        receiver_name=('receiver_name','first'),
        received_amount=('total_amount','sum')
    ).reset_index().rename(columns={"receiver_card":"node"})

    node_meta = pd.merge(send_stats, recv_stats, on="node", how="outer")

    cent_sub = centrality_df[centrality_df["original_node_id"].astype(str).isin(node_meta["node"])]
    cent_sub = cent_sub[["original_node_id","gmm_cluster","total_amount_sent","total_amount_recv"]]
    cent_sub = cent_sub.rename(columns={"original_node_id":"node"})
    node_meta = pd.merge(node_meta, cent_sub, on="node", how="left")

    # fallback names/genders
    node_meta["name"] = node_meta["sender_name"].fillna(node_meta["receiver_name"]).fillna("Unknown")
    node_meta["gender"] = node_meta["sender_gender"].fillna(node_meta["receiver_gender"]).fillna("unknown")

    node_meta["turnover"] = node_meta[["sent_amount","received_amount",
                                       "total_amount_sent","total_amount_recv"]].fillna(0).sum(axis=1)

    node_dict = node_meta.set_index("node").to_dict(orient="index")
    all_nodes = node_meta["node"].tolist()

    # scaling node size
    turnovers = [node_dict[n]["turnover"] for n in all_nodes]
    min_t, max_t = min(turnovers), max(turnovers)
    def node_size(turn):
        if max_t == min_t: return 20
        return 10 + ((turn-min_t)/(max_t-min_t))*70

    def amount_to_color(amount, min_amt, max_amt):
        if max_amt == min_amt: return "rgb(100,180,255)"
        norm = (amount-min_amt)/(max_amt-min_amt)
        return f"rgb({int(180-norm*180)},{int(220-norm*100)},255)"

    # ---- build PyVis net ----
    net = Network(height="800px", width="98vw", directed=True, bgcolor="white")
    net.barnes_hut()

    # add nodes
    for n in all_nodes:
        info = node_dict[n]
        g = str(info.get("gender","unknown")).lower()
        color = "blue" if g=="male" else "red" if g=="female" else "gray"
        nm = info.get("name") or str(n)
        t = info.get("turnover",0)
        size = node_size(t)

        if n == focal_id:
            net.add_node(n, label=nm, color="gold", shape="dot", size=size+10,
                         title=f"FOCAL: {nm}, Turnover {t:,.0f}")
        else:
            net.add_node(n, label=nm, color=color, shape="dot", size=size,
                         title=f"Name: {nm}, Gender: {g}, Turnover: {t:,.0f}")

    # add edges (all solid, arrows pointy, color by direction)
    min_amt, max_amt = dfp_sub["total_amount"].min(), dfp_sub["total_amount"].max()
    for _, row in dfp_sub.iterrows():
        u,v = row["sender_card"], row["receiver_card"]
        amt, transfers = row["total_amount"], int(row["n_transfers"])
        rel_amt = row.get("avg_rel_amount",None)
        rel_label = interpret_rel_amount(rel_amt)

        # default amount gradient
        base_color = amount_to_color(amt, min_amt, max_amt)

        # override color if focal is involved
        if u == focal_id:      # outgoing from focal
            final_color = "blue"
        elif v == focal_id:    # incoming to focal
            final_color = "green"
        else:
            final_color = base_color

        rel_amt_str = f"{rel_amt:.2%}" if pd.notna(rel_amt) else "NA"
        hover = f"{transfers} transfers, Total {amt:,.0f}, Rel {rel_amt_str} ({rel_label})"

        net.add_edge(
            u, v,
            value=transfers,
            color=final_color,
            title=hover,
            arrows={"to": {"enabled": True, "scaleFactor": 0.9, "type": "arrow"}}
        )

    # save (only write, don’t auto-open in browser)
    surname = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+",
                         str(node_dict[focal_id].get("name") or focal_id))
    surname = surname[-1] if surname else focal_id[-8:]
    cluster = node_dict[focal_id].get("gmm_cluster",-1)
    path = os.path.join(output_dir,f"cluster-{cluster}_{surname}.html")
    net.write_html(path)   # <---- changed from .show() to .write_html()
    print("✅ Saved", path)
    return path


# %% BATCH SAMPLER

MAX_DIRECT_EDGES = 50000   # set to None to disable

def count_direct_edges(dfp: pd.DataFrame, node_id: str) -> int:
    node_id = str(node_id)
    m = (dfp["sender_card"].astype(str) == node_id) | (dfp["receiver_card"].astype(str) == node_id)
    return int(m.sum())

def plot_samples_all_clusters(dfp, centrality_df, samples_per_cluster=3, seed=42):
    rng = np.random.default_rng(seed)
    clusters = sorted(pd.Series(centrality_df["gmm_cluster"]).dropna().unique())

    for c in clusters:
        cand = (centrality_df.loc[centrality_df["gmm_cluster"] == c, "original_node_id"]
                .astype(str).dropna().unique())
        if len(cand) == 0:
            print(f"[cluster {c}] no candidates")
            continue

        k = min(samples_per_cluster, len(cand))
        sampled = rng.choice(cand, size=k, replace=False)
        print(f"[cluster {c}] sampled {k} nodes: {list(sampled)}")

        for nid in sampled:
            if MAX_DIRECT_EDGES is not None:
                n_edges = count_direct_edges(dfp, nid)
                if n_edges > MAX_DIRECT_EDGES:
                    print(f"  - skip {nid}: {n_edges} direct edges (> {MAX_DIRECT_EDGES})")
                    continue
            plot_ego_direct(nid, dfp, centrality_df)

# ---- run it ----
plot_samples_all_clusters(dfp, centrality_df, samples_per_cluster=5)


# %%

import random

# pick a random node from cluster 7
cluster_id = 2
nodes_in_cluster = centrality_df.loc[centrality_df["gmm_cluster"] == cluster_id, "original_node_id"]

if nodes_in_cluster.empty:
    print(f"⚠️ No nodes found in cluster {cluster_id}")
else:
    focal_id = str(random.choice(nodes_in_cluster.tolist()))
    print("🎯 Random focal node:", focal_id)

    # plot its ego network
    path = plot_ego_direct(focal_id, dfp, centrality_df, output_dir="visuals")
    print("📂 Visualization saved to:", path)
# %%
