import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump
import json

IN_CSV = "wireless_assoc_disassoc_hourly_180d.csv"
MODEL_OUT = "iforest_wireless_assoc_disassoc.joblib"
META_OUT = "iforest_wireless_assoc_disassoc_meta.json"

CONTAMINATION = 0.02
RANDOM_STATE = 42


def main():
    df = pd.read_csv(IN_CSV, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)

    # Feature engineering
    df["hod"] = df["hour"].dt.hour

    feature_cols = [
        "association",
        "disassociation",
        "total",
        "hod",
    ]

    X = df[feature_cols].astype(float)

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X)

    # Persist model
    dump(model, MODEL_OUT)

    # Persist metadata
    with open(META_OUT, "w") as f:
        json.dump(
            {
                "features": feature_cols,
                "contamination": CONTAMINATION,
                "rows_trained": len(df),
                "date_range": [
                    str(df["hour"].min()),
                    str(df["hour"].max()),
                ],
            },
            f,
            indent=2,
        )

    print("Saved wireless assoc/disassoc Isolation Forest")
    print(f"Rows trained: {len(df)}")
    print(f"Range: {df['hour'].min()} → {df['hour'].max()}")


if __name__ == "__main__":
    main()

