#!/usr/bin/env python3
"""
alert_eval.py

Option B: live 24h evaluation against your trained IsolationForest models.

Inputs:
  - Trained models (joblib):
      iforest_location.joblib
      iforest_mx_uplink.joblib
      iforest_wireless_assoc_disassoc.joblib
  - Training score distributions (npy) used for p95/p99 thresholds:
      location_scores.npy
      mx_uplink_scores.npy
      wireless_scores.npy
  - Location daily CSV (your cleaned daily file):
      location_cleaned.csv   (must have columns: day, connected, visitors, passersby)

Live data sources:
  - MX uplink usageHistory (hourly) via Dashboard API
  - Wireless events (association/disassociation) via Dashboard API, aggregated hourly

Alert policy:
  - CRITICAL if any signal >= p99
  - ALERT if 2+ signals >= p95
  - else OK

Environment:
  export MERAKI_DASHBOARD_API_KEY="..."
  export NETWORK_ID="L_..."
"""

import os
import sys
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import meraki


# -----------------------
# Files (adjust if needed)
# -----------------------
IF_LOCATION = "iforest_location.joblib"
IF_MX = "iforest_mx_uplink.joblib"
IF_WIRELESS = "iforest_wireless_assoc_disassoc.joblib"

LOC_SCORES = "location_scores.npy"
MX_SCORES = "mx_uplink_scores.npy"
WIRELESS_SCORES = "wireless_scores.npy"

LOCATION_DAILY_CSV = "location_cleaned.csv"  # output of your location_clean.py (daily)


# -----------------------
# Meraki API params
# -----------------------
PER_PAGE = 1000
SLEEP_SEC = 0.2


# -----------------------
# Helpers
# -----------------------
def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def check_exists(*paths: str) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

def numericize_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce a feature frame to float, handling datetimes safely."""
    Xn = X.copy()
    for c in Xn.columns:
        s = Xn[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            # seconds since epoch
            Xn[c] = (s.view("int64") / 1e9).astype(float)
        else:
            Xn[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return Xn


def anomaly_score_from_model(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """
    Returns anomaly score where higher = more anomalous (so it matches your p95/p99 logic).
    sklearn decision_function: higher = more normal
    We'll invert it.
    """
    Xn = numericize_frame(X)
    s = model.decision_function(Xn)   # higher = more normal
    return (-s)                       # higher = more anomalous

def thresholds_from_scores(scores: np.ndarray) -> Tuple[float, float]:
    """
    Use high-tail thresholds: p95, p99 (because our standardized anomaly_score is higher=more anomalous)
    """
    scores = np.asarray(scores, dtype=float)
    p95 = float(np.nanpercentile(scores, 95))
    p99 = float(np.nanpercentile(scores, 99))
    return p95, p99


def coerce_feature_frame(model, df: pd.DataFrame, candidate_cols: List[str]) -> pd.DataFrame:
    """
    Build an X frame matching the model's expected feature count.
    If model has feature_names_in_ (sklearn >= 1.0 and trained on DataFrame), use that.
    Otherwise, use the first n_features_in_ columns from candidate_cols that exist.
    """
    if hasattr(model, "feature_names_in_"):
        cols = [c for c in list(model.feature_names_in_) if c in df.columns]
        if len(cols) != len(model.feature_names_in_):
            missing = set(model.feature_names_in_) - set(cols)
            die(f"Model expects features {list(model.feature_names_in_)} but missing {sorted(missing)} in live df.")
        return df[cols].copy()

    n = getattr(model, "n_features_in_", None)
    if n is None:
        die("Could not determine model feature count (no feature_names_in_ and no n_features_in_)")

    cols = [c for c in candidate_cols if c in df.columns]
    if len(cols) < n:
        die(f"Need {n} features but only found {len(cols)} in df. Have: {cols}. Missing from candidates: {candidate_cols}")
    return df[cols[:n]].copy()


@dataclass
class SignalResult:
    name: str
    window_start: datetime
    window_end: datetime
    worst_ts: datetime
    worst_score: float
    p95: float
    p99: float
    severity: str  # OK / P95 / P99
    details: Dict[str, Any]


def severity_for(score: float, p95: float, p99: float) -> str:
    if score >= p99:
        return "P99"
    if score >= p95:
        return "P95"
    return "OK"


# -----------------------
# Location (daily, from CSV)
# -----------------------
def eval_location_daily(
    model,
    train_scores: np.ndarray,
    csv_path: str,
    lookback_days: int = 30,
) -> SignalResult:
    df = pd.read_csv(csv_path, parse_dates=["day"])
    df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce")
    df = df.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    if df.empty:
        die(f"{csv_path} had no usable rows")

    # last N days window (for context)
    t1 = df["day"].max()
    t0 = t1 - pd.Timedelta(days=lookback_days)
    w = df[df["day"] >= t0].copy()
    if w.empty:
        w = df.tail(min(len(df), lookback_days)).copy()

    w["dow"] = w["day"].dt.dayofweek.astype(float)

    # candidate features
    candidate = ["connected", "visitors", "passersby", "dow"]
    X = coerce_feature_frame(model, w, candidate)

    a_score = anomaly_score_from_model(model, X)
    w = w.assign(anom_score=a_score)

    worst = w.sort_values("anom_score", ascending=False).iloc[0]
    p95, p99 = thresholds_from_scores(train_scores)
    sev = severity_for(float(worst["anom_score"]), p95, p99)

    return SignalResult(
        name="location_daily",
        window_start=w["day"].min().to_pydatetime(),
        window_end=w["day"].max().to_pydatetime(),
        worst_ts=worst["day"].to_pydatetime(),
        worst_score=float(worst["anom_score"]),
        p95=p95,
        p99=p99,
        severity=sev,
        details={
            "connected": float(worst.get("connected", np.nan)),
            "visitors": float(worst.get("visitors", np.nan)),
            "passersby": float(worst.get("passersby", np.nan)),
            "dow": int(worst.get("dow", -1)),
        },
    )


# -----------------------
# MX uplink (hourly, API)
# -----------------------
def fetch_mx_uplink_hourly(
    dashboard: meraki.DashboardAPI,
    network_id: str,
    hours: int = 24,
) -> pd.DataFrame:
    # Endpoint: /networks/{networkId}/appliance/uplinks/usageHistory
    # Use timespan + resolution=3600 for hourly buckets.
    timespan = hours * 3600
    resp = dashboard.appliance.getNetworkApplianceUplinksUsageHistory(
        network_id,
        timespan=timespan,
        resolution=3600,
    )

    rows = []
    for b in resp or []:
        # example has startTime/endTime and byInterface list
        start = b.get("startTime")
        by_if = b.get("byInterface") or []
        sent = 0
        recv = 0
        uplinks = []
        for item in by_if:
            uplinks.append(item.get("interface"))
            sent += int(item.get("sent") or 0)
            recv += int(item.get("received") or 0)
        rows.append(
            {
                "ts": pd.to_datetime(start, utc=True, errors="coerce"),
                "uplinks": ",".join([u for u in uplinks if u]),
                "sent": float(sent),
                "received": float(recv),
                "total": float(sent + recv),
                "hour": pd.to_datetime(start, utc=True, errors="coerce"),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def eval_mx_uplink_24h(model, train_scores: np.ndarray, df: pd.DataFrame) -> SignalResult:
    if df.empty:
        die("MX uplink hourly df was empty")

    # add hour-of-day + dow for optional seasonality if model was trained that way
    df["hod"] = df["ts"].dt.hour.astype(float)
    df["dow"] = df["ts"].dt.dayofweek.astype(float)

    candidate = ["sent", "received", "total", "hod", "dow"]
    X = coerce_feature_frame(model, df, candidate)

    a_score = anomaly_score_from_model(model, X)
    df = df.assign(anom_score=a_score)

    worst = df.sort_values("anom_score", ascending=False).iloc[0]
    p95, p99 = thresholds_from_scores(train_scores)
    sev = severity_for(float(worst["anom_score"]), p95, p99)

    return SignalResult(
        name="mx_uplink_hourly",
        window_start=df["ts"].min().to_pydatetime(),
        window_end=df["ts"].max().to_pydatetime(),
        worst_ts=worst["ts"].to_pydatetime(),
        worst_score=float(worst["anom_score"]),
        p95=p95,
        p99=p99,
        severity=sev,
        details={
            "sent": float(worst["sent"]),
            "received": float(worst["received"]),
            "total": float(worst["total"]),
            "uplinks": str(worst.get("uplinks", "")),
        },
    )


# -----------------------
# Wireless events -> hourly assoc/disassoc (API)
# -----------------------
from datetime import datetime, timedelta, timezone
import time

def fetch_wireless_events_last_hours(
    dashboard,
    network_id: str,
    hours: int = 24,
    per_page: int = 1000,
    max_pages: int = 10,
    sleep_sec: float = 0.15,
):
    t1 = datetime.now(timezone.utc)
    t0 = t1 - timedelta(hours=hours)

    ending_before = None
    all_events = []

    for page in range(1, max_pages + 1):
        resp = dashboard.networks.getNetworkEvents(
            network_id,
            productType="wireless",
            perPage=per_page,
            t0=t0.isoformat().replace("+00:00", "Z"),
            t1=t1.isoformat().replace("+00:00", "Z"),
            **({"endingBefore": ending_before} if ending_before else {}),
        )

        events = resp.get("events") or []
        page_start = resp.get("pageStartAt")
        page_end = resp.get("pageEndAt")

        print(f"[wireless] page {page}: count={len(events)} start={page_start} end={page_end}")

        if not events:
            break

        all_events.extend(events)

        # If we got fewer than perPage, we're done
        if len(events) < per_page:
            break

        # Walk backward in time
        ending_before = page_start
        if not ending_before:
            break

        time.sleep(sleep_sec)

    return all_events, t0, t1

def wireless_hourly_assoc_disassoc(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in events:
        t = e.get("occurredAt") or e.get("t") or e.get("timestamp")
        typ = e.get("type") or "unknown"
        if not t:
            continue
        ts = pd.to_datetime(t, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append({"ts": ts, "type": typ})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["hour"] = df["ts"].dt.floor("h")
    # keep only assoc/disassoc
    df = df[df["type"].isin(["association", "disassociation"])].copy()

    if df.empty:
        # return an empty hourly frame
        return pd.DataFrame(columns=["hour", "association", "disassociation", "total"])

    pivot = df.groupby(["hour", "type"]).size().unstack(fill_value=0).reset_index()
    if "association" not in pivot.columns:
        pivot["association"] = 0
    if "disassociation" not in pivot.columns:
        pivot["disassociation"] = 0
    pivot["total"] = pivot["association"] + pivot["disassociation"]
    pivot = pivot.sort_values("hour").reset_index(drop=True)
    return pivot


def eval_wireless_24h(model, train_scores: np.ndarray, df: pd.DataFrame) -> SignalResult:
    if df.empty:
        die("Wireless hourly assoc/disassoc df was empty")

    df["hod"] = pd.to_datetime(df["hour"], utc=True).dt.hour.astype(float)
    df["dow"] = pd.to_datetime(df["hour"], utc=True).dt.dayofweek.astype(float)

    candidate = ["association", "disassociation", "total", "hod", "dow"]
    X = coerce_feature_frame(model, df, candidate)

    a_score = anomaly_score_from_model(model, X)
    df = df.assign(anom_score=a_score)

    worst = df.sort_values("anom_score", ascending=False).iloc[0]
    p95, p99 = thresholds_from_scores(train_scores)
    sev = severity_for(float(worst["anom_score"]), p95, p99)

    assoc_24h = int(df["association"].sum())
    disassoc_24h = int(df["disassociation"].sum())
    total_24h = int(df["total"].sum())

    return SignalResult(
        name="wireless_assoc_disassoc_hourly",
        window_start=pd.to_datetime(df["hour"].min(), utc=True).to_pydatetime(),
        window_end=pd.to_datetime(df["hour"].max(), utc=True).to_pydatetime(),
        worst_ts=pd.to_datetime(worst["hour"], utc=True).to_pydatetime(),
        worst_score=float(worst["anom_score"]),
        p95=p95,
        p99=p99,
        severity=sev,
        details={
            # worst hour (behavioral anomaly)
            "association": int(worst["association"]),
            "disassociation": int(worst["disassociation"]),
            "total": int(worst["total"]),

            # context (sanity / operator-visible)
            "association_24h": assoc_24h,
            "disassociation_24h": disassoc_24h,
            "total_24h": total_24h,
        }
    )


# -----------------------
# Alert roll-up
# -----------------------
def summarize(results: List[SignalResult]) -> Dict[str, Any]:
    n_p95 = sum(1 for r in results if r.severity in ("P95", "P99"))
    n_p99 = sum(1 for r in results if r.severity == "P99")

    overall = "OK"
    if n_p99 >= 1:
        overall = "CRITICAL"
    elif n_p95 >= 2:
        overall = "ALERT"
    elif n_p95 == 1:
        overall = "WARN"

    return {
        "overall": overall,
        "signals_p95_or_worse": n_p95,
        "signals_p99": n_p99,
        "signals": [
            {
                "name": r.name,
                "severity": r.severity,
                "worst_ts": r.worst_ts.isoformat(),
                "worst_score": r.worst_score,
                "p95": r.p95,
                "p99": r.p99,
                "window_start": r.window_start.isoformat(),
                "window_end": r.window_end.isoformat(),
                "details": r.details,
            }
            for r in results
        ],
    }
def wireless_assoc_disassoc_hourly_from_events(events, t0, t1):
    rows = []

    for e in events:
        ts = pd.to_datetime(e.get("occurredAt"), utc=True)
        etype = e.get("type", "").lower()

        if etype not in ("association", "disassociation"):
            continue

        rows.append({
            "hour": ts.floor("h"),
            "association": 1 if etype == "association" else 0,
            "disassociation": 1 if etype == "disassociation" else 0,
        })

    if not rows:
        return pd.DataFrame(columns=["hour", "association", "disassociation", "total"])

    df = pd.DataFrame(rows)

    agg = (
        df.groupby("hour", as_index=False)
          .sum()
    )

    agg["total"] = agg["association"] + agg["disassociation"]
    return agg

def main() -> None:
    check_exists(
        IF_LOCATION, IF_MX, IF_WIRELESS,
        LOC_SCORES, MX_SCORES, WIRELESS_SCORES,
        LOCATION_DAILY_CSV,
    )

    api_key = os.environ.get("MERAKI_DASHBOARD_API_KEY")
    network_id = os.environ.get("NETWORK_ID")
    if not api_key or not network_id:
        die("Set MERAKI_DASHBOARD_API_KEY and NETWORK_ID in your environment.")

    # load models + training score distributions
    if_loc = joblib.load(IF_LOCATION)
    if_mx = joblib.load(IF_MX)
    if_wl = joblib.load(IF_WIRELESS)

    loc_scores = np.load(LOC_SCORES)
    mx_scores = np.load(MX_SCORES)
    wl_scores = np.load(WIRELESS_SCORES)

    dashboard = meraki.DashboardAPI(api_key=api_key, suppress_logging=True)

    results: List[SignalResult] = []

    # Location (daily) from CSV
    results.append(
        eval_location_daily(if_loc, loc_scores, LOCATION_DAILY_CSV, lookback_days=60)
    )

    # MX uplink (last 24h)
    mx_df = fetch_mx_uplink_hourly(dashboard, network_id, hours=24)
    results.append(eval_mx_uplink_24h(if_mx, mx_scores, mx_df))

    # Wireless assoc/disassoc (last 24h)
    wl_df = pd.DataFrame()   # ? FIX 1

    events, t0, t1 = fetch_wireless_events_last_hours(
        dashboard, network_id, hours=24
    )
    print(f"[wireless] total fetched events: {len(events)} window {t0} -> {t1}")

    from collections import Counter

    def _wireless_debug_events(events, n=5):
        # show field presence
        if not events:
            print("[wireless] no events to debug")
            return

        sample = events[0]
        print("[wireless] sample keys:", sorted(sample.keys()))

        # count common "type-like" fields
        type_counts = Counter()
        eventtype_counts = Counter()
        category_counts = Counter()
        desc_hits = 0

        for e in events:
            t = str(e.get("type", "")).strip()
            et = str(e.get("eventType", "")).strip()
            cat = str(e.get("category", "")).strip()
            d = str(e.get("description", "")).lower()

            if t:
                type_counts[t] += 1
            if et:
                eventtype_counts[et] += 1
            if cat:
                category_counts[cat] += 1
            if ("disassoc" in d) or ("association" in d):
                desc_hits += 1

        print("[wireless] top types:", type_counts.most_common(15))
        print("[wireless] top eventType:", eventtype_counts.most_common(15))
        print("[wireless] top categories:", category_counts.most_common(15))
        print("[wireless] description contains association/disassoc:", desc_hits)

    _wireless_debug_events(events)

    if events:
        wl_df = wireless_assoc_disassoc_hourly_from_events(events, t0, t1)
        # DEBUG: persist raw hourly aggregation BEFORE any resampling/windowing/fillna
        debug_path = os.path.join(os.path.dirname(__file__), "debug_wireless_24h.csv")
        wl_df.to_csv(debug_path, index=False)
        print(f"[debug] wrote {debug_path} rows={len(wl_df)} cols={wl_df.columns.tolist()}")

        wl_df["hour"] = pd.to_datetime(wl_df["hour"], utc=True).dt.floor("h")

        end = wl_df["hour"].max()
        start = end - pd.Timedelta(hours=23)

        full = pd.DataFrame({
            "hour": pd.date_range(start=start, end=end, freq="h", tz="UTC")
        })

        wl_df = (
            full.merge(wl_df, on="hour", how="left")
                .fillna(0)
        )

        wl_df[["association", "disassociation", "total"]] = (
            wl_df[["association", "disassociation", "total"]].astype(int)
        )

    results.append(eval_wireless_24h(if_wl, wl_scores, wl_df))

    out = summarize(results)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()






