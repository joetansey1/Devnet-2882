import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "wireless_assoc_disassoc_hourly_180d.csv"

# How aggressively to thin the "normal" line (keep every Nth point)
DOWNSAMPLE_NORMAL = 6   # 6 => keep 1 point per 6 hours for normal portion
USE_LOG_Y = False       # set True if you want to compress spikes

df = pd.read_csv(IN_CSV, parse_dates=["hour"])
df = df.sort_values("hour").reset_index(drop=True)

y = df["disassociation"].astype(float)

mu = y.mean()
sd = y.std()
thr1_lo, thr1_hi = mu - sd, mu + sd
thr2_lo, thr2_hi = mu - 2 * sd, mu + 2 * sd
thr2 = mu + 2 * sd

out2 = df[y > thr2].copy()

# "Normal" subset for plotting (downsample)
normal = df[y <= thr2].copy()
normal = normal.iloc[::DOWNSAMPLE_NORMAL, :]

print("disassociation mean/std:", mu, sd)
print("threshold +2σ:", thr2)
print("outliers >2σ hours:", len(out2))

plt.figure(figsize=(16, 4.2))

# Shaded baseline bands
plt.axhspan(thr2_lo, thr2_hi, alpha=0.12, label="mean ± 2σ")
plt.axhspan(thr1_lo, thr1_hi, alpha=0.18, label="mean ± 1σ")

# Mean line
plt.axhline(mu, linestyle="--", linewidth=1, label="mean")

# Normal line (downsampled)
plt.plot(
    normal["hour"],
    normal["disassociation"],
    linewidth=1,
    alpha=0.9,
    label="disassociation (downsampled normal)",
)

# Outliers only
plt.scatter(
    out2["hour"],
    out2["disassociation"],
    s=26,
    label="outliers (> +2σ)",
)

plt.title("Wireless disassociation events per hour (180d) — shaded σ bands + >2σ outliers")
plt.xlabel("hour (UTC)")
plt.ylabel("disassociation events/hour")

if USE_LOG_Y:
    plt.yscale("log")

plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

