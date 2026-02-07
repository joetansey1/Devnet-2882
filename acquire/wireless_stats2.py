import pandas as pd

CSV = "wireless_assoc_disassoc_hourly_30d.csv"  # change to 180d file when you generate it

df = pd.read_csv(CSV, parse_dates=["hour"])

print("RAW rows:", len(df))
print("RAW earliest:", df["hour"].min())
print("RAW latest:  ", df["hour"].max())
print("RAW hours:", df["hour"].nunique())

for col in ["association", "disassociation"]:
    s = df[col].fillna(0)

    mu = s.mean()
    med = s.median()
    sd = s.std()

    print(f"\n{col.upper()} events per hour")
    print("-" * 40)
    print(f"count:   {len(s)}")
    print(f"mean:    {mu:.2f}")
    print(f"median:  {med:.2f}")
    print(f"std dev: {sd:.2f}")
    print(f"1 SD:    [{mu - sd:.2f}, {mu + sd:.2f}]")
    print(f"2 SD:    [{mu - 2*sd:.2f}, {mu + 2*sd:.2f}]")
    print(f"min/max: {s.min():.0f} / {s.max():.0f}")

