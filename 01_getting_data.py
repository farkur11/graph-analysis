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


""" ### SQL CODE TO CREATE A TABLE IN THE DATABASE
drop table if exists payme_sandbox.transfers_direction_temp;
create table payme_sandbox.transfers_direction_temp as
select payer_id, card_number sender_card, sendercard_phone payer_phone,
       regexp_replace(trim(sendercard_owner), '\\s+', ' ', 'g') sender_name,
       date_trunc('minute', create_time)::timestamp ds, sensitive_data_balance_before_payment/100 balance_before,
       amount/100 amount, account_number receiver_card,
       regexp_replace(trim(account_cardowner), '\\s+', ' ', 'g') receiver_name
from ods__mdbmn__paycom.receipts r
where r."state" = '4'
and r."type" = '5'
and r.payment_service = '56e7ce796b6ef347d846e3eb'
and r.external = false
and r.meta_owner is null
and create_time < current_date
and create_time > current_date - interval '150 days'
 """

# %% Connection to the DB

conn = psycopg2.connect(
    host="172.17.30.1",
    database="dwh",
    user="",
    password=""
)

# %% Saving data as parquet files
start_date = date(2025, 2, 21)
end_date = date(2025, 7, 20)

output_dir = "parquet_output/"

curr_date = start_date

while curr_date <= end_date:
    next_date = curr_date + timedelta(days=1)

    # Regular cursor (full fetch)
    cur = conn.cursor()

    # SQL query using %s
    query = """
        select * from payme_sandbox.transfers_direction_temp
        where ds >= %s and ds < %s
    """
    cur.execute(query, (curr_date, next_date))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()

    # Load to Polars
    df = pl.DataFrame(rows, schema=columns)

    # Save as Parquet
    output_path = os.path.join(output_dir, f"transfers_{curr_date}.parquet")
    df.write_parquet(output_path)

    print(f"Saved {curr_date} ({df.shape[0]} rows) {output_path}")

    curr_date = next_date

conn.close()