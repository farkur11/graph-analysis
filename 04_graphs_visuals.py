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


# %% check cuda
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

# %%
centrality_df = pd.read_csv("centrality.csv")
dfp = pd.read_parquet("agg_output.parquet")
sen_df = pd.read_parquet("agg_sender.parquet")
rec_df = pd.read_parquet("agg_receiver.parquet")


centrality_df["original_node_id"] = centrality_df["original_node_id"].astype(str)
sen_df["node"] = sen_df["node"].astype(str)
rec_df["node"] = rec_df["node"].astype(str)


# %%
unique_cards = pd.concat([dfp["sender_card"], dfp["receiver_card"]]).nunique()
print(unique_cards)

# %%
print(dfp["receiver_card"].nunique())
print(dfp["sender_card"].nunique())

# %%
##centrality_df["node_x"].nunique()

# %%
centrality_df.info()
# %%
rec_df.info()
# %%
sen_df.info()

# %%
print(centrality_df["original_node_id"].nunique())

# %%
sen_df.head()
print(sen_df["node"].nunique())
# %%
rec_df.info()

# %%


centrality_df.merge(sen_df, left_on="original_node_id", right_on="node", how="inner")

# %%
centrality_df.merge(rec_df, left_on="original_node_id", right_on="node", how="inner")


# %%
# --- 3. Merge All Aggregates into centrality_df ---
centrality_df = centrality_df.merge(sen_df, left_on="original_node_id", right_on="node", how="left")
centrality_df = centrality_df.merge(rec_df, left_on="original_node_id", right_on="node", how="left")

# --- 4. Fill NaNs (some users may be only sender or receiver) ---
centrality_df.fillna(0, inplace=True)

# %%
centrality_df["node_x"].nunique()

# %%
centrality_df[centrality_df["n_transfers_sent"] > 0]


# %%
centrality_df.to_csv("df_for_clustering.csv")
# %%
from pyvis.network import Network
import os

# ───────────────────────────────
def interpret_rel_amount(x):
    if pd.isna(x): return "unknown"
    elif x <= 0.05: return "negligible"
    elif x <= 0.15: return "low"
    elif x <= 0.35: return "moderate"
    elif x <= 0.6: return "high"
    elif x <= 0.85: return "heavy"
    else: return "aggressive"

# ───────────────────────────────
def plot_community_pyvis_efficient(community_id, dfp, centrality_df, output_dir="visuals"):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get list of node IDs in the community
    community_nodes = (
        centrality_df[centrality_df["community"] == community_id]["original_node_id"]
        .dropna().astype(str).unique()
    )
    if len(community_nodes) == 0:
        print(f"⚠️ Community {community_id} is empty.")
        return

    # Step 2: Filter dfp for intra-community edges
    dfp["sender_card"] = dfp["sender_card"].astype(str)
    dfp["receiver_card"] = dfp["receiver_card"].astype(str)

    dfp_sub = dfp[
        dfp["sender_card"].isin(community_nodes) &
        dfp["receiver_card"].isin(community_nodes)
    ].copy()

    if dfp_sub.empty:
        print(f"⚠️ No transactions within community {community_id}.")
        return

    # Step 3: Create minimal node metadata
    sender_stats = dfp_sub.groupby("sender_card").agg(
        sender_gender=("sender_gender", "first"),
        sender_name=("sender_name", "first"),
        sent_amount=("total_amount", "sum")
    ).reset_index().rename(columns={"sender_card": "node"})

    receiver_stats = dfp_sub.groupby("receiver_card").agg(
        receiver_gender=("receiver_gender", "first"),
        receiver_name=("receiver_name", "first"),
        received_amount=("total_amount", "sum")
    ).reset_index().rename(columns={"receiver_card": "node"})

    centrality_sub = centrality_df[centrality_df["community"] == community_id]
    centrality_sub["node"] = centrality_sub["original_node_id"].astype(str)

    node_metadata = (
        sender_stats.merge(receiver_stats, on="node", how="outer")
                    .merge(centrality_sub, on="node", how="left")
    )

    node_dict = node_metadata.set_index("node").to_dict(orient="index")
    all_nodes = node_metadata["node"].tolist()

    # Step 4: Compute turnover for size scaling
    turnovers = [
        (node_dict.get(n, {}).get("sent_amount", 0) or 0) +
        (node_dict.get(n, {}).get("received_amount", 0) or 0)
        for n in all_nodes
    ]
    
    
    min_amt_node, max_amt_node = min(turnovers), max(turnovers)

    def node_size(turnover):
        if max_amt_node == min_amt_node:
            return 20  # fallback if all nodes have same turnover
        norm = (turnover - min_amt_node) / (max_amt_node - min_amt_node)
        return 10 + norm * 60  # range: 10–70


    def amount_to_color(amount, min_amt, max_amt):
        if max_amt == min_amt:
            return "rgb(100,180,255)"
        norm = (amount - min_amt) / (max_amt - min_amt)
        return f"rgb({int(180 - norm * 180)},{int(220 - norm * 100)},255)"

    # Step 5: Create PyVis network
    net = Network(height="800px", width="98vw", directed=True, notebook=False, bgcolor="white")
    net.barnes_hut()

    # Add nodes
    for node in all_nodes:
        info = node_dict.get(node, {})
        sg = info.get("sender_gender")
        rg = info.get("receiver_gender")
        gender = str(sg if pd.notna(sg) and sg else rg if pd.notna(rg) else "").lower()
        color = "blue" if gender == "male" else "red" if gender == "female" else "gray"

        name = info.get("sender_name") or info.get("receiver_name") or "Unknown"
        sent = info.get("sent_amount", 0)
        recv = info.get("received_amount", 0)
        turnover = sent + recv
        size = node_size(turnover)

        net.add_node(
            node,
            label=name,
            color=color,
            size=size,
            shape="dot",
            font={"size": 60},
            title=f"Name: {name}, Gender: {gender}, Sent: {sent:,.0f}, Received: {recv:,.0f}, Turnover: {turnover:,.0f}"
        )

    # Add edges
    min_amt, max_amt = dfp_sub["total_amount"].min(), dfp_sub["total_amount"].max()
    for _, row in dfp_sub.iterrows():
        u, v = row["sender_card"], row["receiver_card"]
        amount = row["total_amount"]
        transfers = int(row["n_transfers"])
        rel_amt = row.get("avg_rel_amount", None)
        rel_label = interpret_rel_amount(rel_amt)
        edge_color = "purple" if row["share_same_person"] == 1 else \
                     "orange" if row["share_family"] == 1 else \
                     amount_to_color(amount, min_amt, max_amt)
        dashes = rel_label == "aggressive"
        hover = f"{transfers} transfers, Total: {amount:,.0f}, Rel Amount: {rel_amt:.2%} ({rel_label})"

        net.add_edge(u, v, value=transfers, color=edge_color, title=hover, dashes=dashes)

    # Save
    path = f"{output_dir}/community_{community_id}_pyvis_efficient.html"
    net.show(path)
    print(f"Saved to: {path}")



# %%
com_filter = centrality_df.groupby("community")["node_x"].count().reset_index()
com_filter[(com_filter["node_x"] > 10000) & (com_filter["node_x"] < 12000)].sample(5)

#  %%
plot_community_pyvis_efficient(community_id=15, dfp=dfp, centrality_df=centrality_df)
# %%
