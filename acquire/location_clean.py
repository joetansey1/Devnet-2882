import pandas as pd

df = pd.read_csv("clients_raw_12_mos.csv")

df["day"] = pd.to_datetime(df["time"], utc=True).dt.date
df["day"] = pd.to_datetime(df["day"])  # normalize to midnight UTC

df = df.rename(columns={
    "Connected": "connected",
    "Visitors": "visitors",
    "Passersby": "passersby",
})

df = df.sort_values("day").reset_index(drop=True)

print(df.head())

