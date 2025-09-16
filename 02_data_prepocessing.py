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
import os

##os.chdir(os.path.dirname(__file__))


df = pl.read_parquet("parquet_output/*.parquet")

df = df.with_columns([
    pl.col("amount").cast(pl.Float64),
    pl.col("balance_before").cast(pl.Float64)
])

print("import and cast finished")

#  %% Working with names and surnames

surname_suffixes = ["ev", "yev", "ov", "in", "eva", "yeva", "ova", "ina",
                    "skiy", "skaya", "ko", "chuk", "ski", "ska", "dze", "shvili", "yan"]
exotic_surnames = ["kim", "lee", "li", "pak", "yun", "soy", "wang",
                   "zhang", "liu", "chen", "yang", "huang", "zhao", "han", "xan"]

def translit_normalize(name):
    name = name.lower()
    name = name.replace("kh", "h")
    name = name.replace("q", "k")
    name = name.replace("x", "h")
    name = name.replace("dj", "j")
    name = name.replace("yev", "ev")
    name = name.replace("huja", "hoja")
    name = name.replace("jura", "jora")
    return name

def extract_surname_and_name(name_raw):
    if name_raw is None or not str(name_raw).strip():
        return None
    name_clean = re.sub(r"[^\w\s]", "", str(name_raw).strip().lower())
    name_clean = re.sub(r"\d+", "", name_clean)
    name_clean = translit_normalize(name_clean)
    parts = name_clean.split()
    if not parts:
        return None
    elif len(parts) == 1:
        return parts[0].capitalize()
    if parts[0] in exotic_surnames:
        surname = parts[0]
        first_name = parts[1] if len(parts) > 1 else ""
        return f"{first_name.capitalize()} {surname.capitalize()}".strip()
    surname = None
    for part in parts:
        for suf in surname_suffixes:
            if suf == "in" and part.endswith("ddin"):
                continue
            if part.endswith(suf):
                surname = part
                break
        if surname:
            break
    remaining = [p for p in parts if p != surname]
    first_name = remaining[0] if remaining else None
    if not surname and len(parts) >= 2:
        surname = parts[-1]
        first_name = parts[0]
    return f"{(first_name or "").capitalize()} {(surname or "").capitalize()}".strip()

def extract_surname_only(fullname):
    if not isinstance(fullname, str):
        return ""
    parts = fullname.split()
    return parts[1] if len(parts) >= 2 else ""

def trailing_zeros_level(amount):
    try:
        amount_str = str(int(amount))
    except:
        return 0
    for i in range(6, 0, -1):
        if amount_str.endswith("0" * i):
            return i
    return 0


df = df.with_columns([
    pl.col("sender_name").map_elements(extract_surname_and_name, return_dtype=pl.Utf8).alias("sender_name_clean"),
    pl.col("receiver_name").map_elements(extract_surname_and_name, return_dtype=pl.Utf8).alias("receiver_name_clean")
])

df = df.with_columns([
    pl.col("sender_name_clean").map_elements(extract_surname_only, return_dtype=pl.Utf8).alias("sender_surname"),
    pl.col("receiver_name_clean").map_elements(extract_surname_only, return_dtype=pl.Utf8).alias("receiver_surname")
])

print("separation of name and surname finished")

# %% Finding relative transfer amount 

df = df.with_columns([
    (pl.col("amount") / pl.col("balance_before").replace(0, None)).alias("rel_amount"),
    (pl.col("balance_before") - pl.col("amount")).alias("post_balance"),
    ((pl.col("balance_before") - pl.col("amount")) / pl.col("balance_before").replace(0, None)).alias("balance_ratio"),
    (pl.col("balance_before") - pl.col("amount") < 0.1 * pl.col("balance_before")).alias("is_near_empty"),
    pl.col("amount").map_elements(trailing_zeros_level, return_dtype=pl.Int64).alias("roundness_level")
])

df = df.with_columns([
    (pl.col("rel_amount") > 0.9).alias("is_high_drain"),
    (pl.col("balance_before") <= 100_000).alias("is_low_balance_sender")
])

# %% Extraction of date and time variables

# Extracting date parts
df = df.with_columns([
    pl.col("ds").dt.hour().alias("hour"),
    pl.col("ds").dt.day().alias("day"),
    pl.col("ds").dt.month().alias("month"),
    pl.col("ds").dt.weekday().alias("dayofweek"),
])

# Adding boolean flags
df = df.with_columns([
    pl.col("dayofweek").is_in([5, 6]).alias("is_weekend"),
    (pl.col("day") <= 3).alias("is_beginning_of_month"),
    (pl.col("day") >= 28).alias("is_end_of_month"),
])

# Defining day period 
def get_day_period(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

# Adding time_of_day column
df = df.with_columns([
    pl.col("hour").map_elements(get_day_period, return_dtype=pl.Utf8).alias("time_of_day")
])

# Adding time-of-day flags
df = df.with_columns([
    (pl.col("time_of_day") == "morning").alias("is_morning"),
    (pl.col("time_of_day") == "afternoon").alias("is_afternoon"),
    (pl.col("time_of_day") == "evening").alias("is_evening"),
    (pl.col("time_of_day") == "night").alias("is_night"),
])

# %% Name similarity estimation

from rapidfuzz import fuzz

# Normalization and similarity function
def advanced_name_similarity(a, b):
    if a is None or b is None:
        return 0.0
    a = translit_normalize(a)
    b = translit_normalize(b)
    return fuzz.token_sort_ratio(a, b) / 100

# Name similarity
df = df.with_columns([
    pl.struct(["sender_name_clean", "receiver_name_clean"]).map_elements(
        lambda row: advanced_name_similarity(row["sender_name_clean"], row["receiver_name_clean"]),
        return_dtype=pl.Float64
    ).alias("name_similarity")
])

# Surname similarity
df = df.with_columns([
    pl.struct(["sender_surname", "receiver_surname"]).map_elements(
        lambda row: advanced_name_similarity(row["sender_surname"], row["receiver_surname"]),
        return_dtype=pl.Float64
    ).alias("surname_similarity")
])

print("name and surname similarity estimation finished")


# %% Gender detection by surname

male_suffixes = ["ev", "yev", "ov", "in", "skiy", "ski"]
female_suffixes = ["eva", "yeva", "ova", "ina", "skaya", "ska"]

# Detection logic
def detect_gender(surname):
    if not surname:
        return "unknown"
    surname = surname.lower()
    for suf in female_suffixes:
        if surname.endswith(suf):
            return "female"
    for suf in male_suffixes:
        if surname.endswith(suf):
            return "male"
    return "unknown"


df = df.with_columns([
    pl.col("sender_surname").map_elements(detect_gender, return_dtype=pl.Utf8).alias("sender_gender"),
    pl.col("receiver_surname").map_elements(detect_gender, return_dtype=pl.Utf8).alias("receiver_gender")
])


df = df.with_columns([

    (pl.col("name_similarity") >= 0.93).alias("likely_same_person"),

    (
        (pl.col("surname_similarity") >= 0.845) &
        ~(pl.col("name_similarity") >= 0.93)
    ).alias("likely_family")
])

print("gender detection finished")


# %% Dropping extra columns

df = df.drop(["payer_id", "payer_phone", "post_balance", "balance_ratio", "sender_name", "receiver_name",
              "day", "month", "dayofweek", "hour", "time_of_day", "surname_similarity", "name_similarity", "is_high_drain"])


# %% Finding daily interval between transfers

df = df.sort(["sender_card", "receiver_card", "ds"])

df = df.with_columns([
    pl.col("ds").cast(pl.Datetime).diff().over(["sender_card", "receiver_card"])
    .dt.total_days()
    .alias("inter_transfer_days")
])

print("adding inter_transfer_days finished")


# %% Data aggregation

agg_df = (
    df.group_by(["sender_card", "receiver_card"])
    .agg([
        pl.len().alias("n_transfers"),
        pl.sum("amount").alias("total_amount"),
        pl.mean("amount").alias("avg_amount"),
        pl.std("amount").alias("std_amount"),
        pl.median("amount").alias("median_amount"),

        pl.mean("balance_before").alias("avg_balance_before"),
        pl.std("balance_before").alias("std_balance_before"),
        pl.mean("rel_amount").alias("avg_rel_amount"),

        pl.mean("is_near_empty").alias("share_near_empty"),
        pl.mean("is_low_balance_sender").alias("share_low_balance_sender"),

        pl.mean("roundness_level").alias("avg_roundness_level"),
        pl.mean("is_morning").alias("share_morning"),
        pl.mean("is_afternoon").alias("share_afternoon"),
        pl.mean("is_evening").alias("share_evening"),
        pl.mean("is_night").alias("share_night"),
        pl.mean("is_weekend").alias("share_weekend"),
        pl.mean("is_beginning_of_month").alias("share_month_start"),
        pl.mean("is_end_of_month").alias("share_month_end"),

        pl.mean("likely_same_person").alias("share_same_person"),
        pl.mean("likely_family").alias("share_family"),

        pl.first("sender_gender").alias("sender_gender"),
        pl.first("receiver_gender").alias("receiver_gender"),
        pl.median("inter_transfer_days").alias("inter_transfer_days"),

        pl.first("sender_name_clean").alias("sender_name"),
        pl.first("receiver_name_clean").alias("receiver_name")
    ])
)



# %%

unique_nodes = (
    pl.concat([
        agg_df.select(pl.col("sender_card")),
        agg_df.select(pl.col("receiver_card").alias("sender_card"))
    ])
    .unique()
    .sort("sender_card")
    .to_series()
    .to_list()
)

# Step 2: Build card → index mapping
node_map = {card: i for i, card in enumerate(unique_nodes)}

# Step 3: Use .replace() to map to indices
agg_df = agg_df.with_columns([
    pl.col("sender_card").replace(node_map).alias("sender_idx"),
    pl.col("receiver_card").replace(node_map).alias("receiver_idx")
])


# %%

agg_df.head()


# %%

agg_df.write_parquet("agg_output.parquet", compression="zstd", use_pyarrow=True)
print("aggregated files were saved")


# %%

# Sender-side aggregation
agg_features = (
    agg_df.group_by("sender_card")
    .agg([
        pl.sum("n_transfers").alias("n_transfers_sent"),
        pl.sum("total_amount").alias("total_amount_sent"),
        pl.mean("avg_rel_amount").alias("avg_rel_amt_sent"),
        pl.mean("avg_roundness_level").alias("avg_roundness_sent"),
        pl.mean("share_morning").alias("share_morning_sent"),
        pl.mean("share_night").alias("share_night_sent"),
        pl.mean("share_weekend").alias("share_weekend_sent"),
        pl.mean("share_family").alias("share_family_sent"),
        pl.mean("share_same_person").alias("share_same_person_sent"),
        pl.mean("inter_transfer_days").alias("inter_transfer_days_mean")
    ])
    .rename({"sender_card": "node"})
)

# Receiver-side aggregation
agg_features_recv = (
    agg_df.group_by("receiver_card")
    .agg([
        pl.sum("n_transfers").alias("n_transfers_recv"),
        pl.sum("total_amount").alias("total_amount_recv"),
        pl.mean("avg_rel_amount").alias("avg_rel_amt_recv"),
        pl.mean("avg_roundness_level").alias("avg_roundness_recv"),
        pl.mean("share_family").alias("share_family_recv"),
        pl.mean("share_same_person").alias("share_same_person_recv")
    ])
    .rename({"receiver_card": "node"})
)

# %%
agg_features.write_parquet("agg_sender.parquet", compression="zstd", use_pyarrow=True)

# %%
agg_features_recv.write_parquet("agg_receiver.parquet", compression="zstd", use_pyarrow=True)
print("aggregated sen/rec files were saved")

# %% Transfer dataframe into pandas
""" 
dfp = agg_df.to_pandas()
dfp = dfp.fillna(0)

## Adding new variables 

dfp["is_male_to_female"] = ((dfp["sender_gender"] == "male") & (dfp["receiver_gender"] == "female")).astype("int8")
dfp["is_female_to_male"] = ((dfp["sender_gender"] == "female") & (dfp["receiver_gender"] == "male")).astype("int8")
dfp["is_male_to_male"] = ((dfp["sender_gender"] == "male") & (dfp["receiver_gender"] == "male")).astype("int8")
dfp["is_female_to_female"] = ((dfp["sender_gender"] == "female") & (dfp["receiver_gender"] == "female")).astype("int8")

# %% Plotting correlation matrix

# Select behavioral columns
behavioral_cols = [
    "avg_amount", "std_amount", "avg_rel_amount", "avg_balance_before", "std_balance_before",
    "share_near_empty", "share_low_balance_sender", "avg_roundness_level",
    "share_morning", "share_afternoon", "share_evening", "share_night",
    "share_weekend", "share_month_start", "share_month_end",
    "share_same_person", "share_family", "inter_transfer_days"
]

# Compute correlation
corr_matrix = dfp[behavioral_cols].corr()

# Plot
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Behavioral Features", fontsize=16)
plt.tight_layout()
plt.show()
 """
