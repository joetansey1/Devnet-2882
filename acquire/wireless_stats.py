import pandas as pd

IN_EVENTS_CSV = "wireless_events_last180d.csv"
OUT_HOURLY_CSV = "wireless_assoc_disassoc_hourly_180d.csv"

df = pd.read_csv(IN_EVENTS_CSV, parse_dates=["occurredAt"])

# Basic sanity
df = df.dropna(subset=["occurredAt"])
print("RAW rows:", len(df))
print("RAW earliest:", df["occurredAt"].min())
print("RAW latest:  ", df["occurredAt"].max())
print("RAW distinct types:", df["type"].nunique(dropna=True))

# Keep only assoc/disassoc
df = df[df["type"].isin(["association", "disassociation"])].copy()

# Hour bucket (UTC)
df["hour"] = df["occurredAt"].dt.floor("h")

# Hourly pivot
hourly = (
    df.groupby(["hour", "type"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

# Ensure columns exist even if one type is absent in the dataset
if "association" not in hourly.columns:
    hourly["association"] = 0
if "disassociation" not in hourly.columns:
    hourly["disassociation"] = 0

hourly["total"] = hourly["association"] + hourly["disassociation"]

hourly = hourly.sort_values("hour").reset_index(drop=True)
hourly.to_csv(OUT_HOURLY_CSV, index=False)

print("\nHOURLY rows:", len(hourly))
print("HOURLY earliest:", hourly["hour"].min())
print("HOURLY latest:  ", hourly["hour"].max())
print(f"wrote {OUT_HOURLY_CSV}")

def print_stats(name: str, s: pd.Series) -> None:
    s = s.astype(float)
    mu = s.mean()
    med = s.median()
    sd = s.std()

    print(f"\n{name} events per hour")
    print("-" * 40)
    print(f"count:   {len(s)}")
    print(f"mean:    {mu:.2f}")
    print(f"median:  {med:.2f}")
    print(f"std dev: {sd:.2f}")
    print(f"1 SD:    [{mu - sd:.2f}, {mu + sd:.2f}]")
    print(f"2 SD:    [{mu - 2*sd:.2f}, {mu + 2*sd:.2f}]")
    print(f"min/max: {s.min():.0f} / {s.max():.0f}")

print_stats("ASSOCIATION", hourly["association"])
print_stats("DISASSOCIATION", hourly["disassociation"])
print_stats("TOTAL (assoc+disassoc)", hourly["total"])

print("\nTop 10 hours by TOTAL assoc+disassoc:")
print(hourly.sort_values("total", ascending=False).head(10).to_string(index=False))

