import pandas as pd
import matplotlib.pyplot as plt

INFILE = "wireless_assoc_disassoc_hourly_180d_scored.csv"

def main():
    df = pd.read_csv(INFILE, parse_dates=["hour"]).sort_values("hour")

    # Only plot rows where 30d bands exist (otherwise fill_between breaks)
    valid = df["mu_30d"].notna()

    plt.figure(figsize=(16, 6))

    # 30d ±2σ (light)
    plt.fill_between(
        df.loc[valid, "hour"],
        df.loc[valid, "m2_30d"],
        df.loc[valid, "p2_30d"],
        alpha=0.15,
        label="30d ±2σ",
    )

    # 30d ±1σ (darker)
    plt.fill_between(
        df.loc[valid, "hour"],
        df.loc[valid, "m1_30d"],
        df.loc[valid, "p1_30d"],
        alpha=0.30,
        label="30d ±1σ",
    )

    # Raw hourly signal
    plt.plot(
        df["hour"],
        df["total"],
        lw=0.8,
        alpha=0.45,
        label="assoc + disassoc (hourly)",
    )

    # 7d moving average
    plt.plot(
        df["hour"],
        df["ma_7d"],
        lw=2.2,
        label="7d moving avg",
    )

    plt.title("Wireless association + disassociation events (hourly)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Events per hour")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
