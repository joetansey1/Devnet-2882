import os
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import meraki


BASE_CSV = "wireless_events_last30d.csv"
BASE_JSONL = "wireless_events_last30d.jsonl"

PER_PAGE = 1000
PRODUCT_TYPE = "wireless"

# Safety limits so you don't accidentally fetch forever
MAX_PAGES = 500          # 50k events max
SLEEP_SEC = 0.20        # gentle pacing


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_dt(s: str) -> datetime:
    # s like "2026-01-03T06:54:27.188087Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_page(
    dashboard: meraki.DashboardAPI,
    network_id: str,
    ending_before: Optional[str],
    t0: datetime,
    t1: datetime,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    Returns: (events, next_ending_before, meta)
    next_ending_before is pageStartAt to walk backward in time.
    """
    params = {
        "perPage": PER_PAGE,
        "productType": PRODUCT_TYPE,
        "t0": iso(t0),
        "t1": iso(t1),
    }
    if ending_before:
        params["endingBefore"] = ending_before

    resp = dashboard.networks.getNetworkEvents(network_id, **params)

    # meraki python lib returns dict with keys: message, pageStartAt, pageEndAt, events
    events = resp.get("events") or []
    next_cursor = resp.get("pageStartAt") or None

    meta = {
        "pageStartAt": resp.get("pageStartAt"),
        "pageEndAt": resp.get("pageEndAt"),
        "count": len(events),
    }
    return events, next_cursor, meta


def flatten_events(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append({
            "occurredAt": e.get("occurredAt"),
            "type": e.get("type"),
            "category": e.get("category"),
            "description": e.get("description"),
            "clientId": e.get("clientId"),
            "clientMac": e.get("clientMac"),
            "clientDescription": e.get("clientDescription"),
            "deviceSerial": e.get("deviceSerial"),
            "deviceName": e.get("deviceName"),
            "ssidName": e.get("ssidName"),
            "ssidNumber": e.get("ssidNumber"),
        })
    df = pd.DataFrame(rows)
    if "occurredAt" in df.columns:
        df["occurredAt"] = pd.to_datetime(df["occurredAt"], utc=True, errors="coerce")
    return df


def main() -> None:
    api_key = os.environ["MERAKI_DASHBOARD_API_KEY"]
    network_id = os.environ["NETWORK_ID"]

    dashboard = meraki.DashboardAPI(
        api_key=api_key,
        suppress_logging=True,
    )

    # Choose your training window here
    t1 = datetime.now(timezone.utc)
    t0 = t1 - timedelta(days=180)

    ending_before = None
    all_events: List[Dict[str, Any]] = []

    print(f"NETWORK_ID: {network_id}")
    print(f"window: {iso(t0)} -> {iso(t1)}")
    print(f"perPage={PER_PAGE} productType={PRODUCT_TYPE}")

    for page in range(1, MAX_PAGES + 1):
        events, next_cursor, meta = fetch_page(
            dashboard=dashboard,
            network_id=network_id,
            ending_before=ending_before,
            t0=t0,
            t1=t1,
        )

        print(
            f"page {page:02d}: count={meta['count']} "
            f"pageStartAt={meta['pageStartAt']} pageEndAt={meta['pageEndAt']}"
        )

        if not events:
            break

        all_events.extend(events)

        # stop if this is the last page
        if len(events) < PER_PAGE:
            break

        # move backward in time
        ending_before = next_cursor
        if not ending_before:
            break

        time.sleep(SLEEP_SEC)

    # de-dupe by (occurredAt, type, clientId, deviceSerial) to be safe
    df = flatten_events(all_events)
    if not df.empty:
        df = df.drop_duplicates(subset=["occurredAt", "type", "clientId", "deviceSerial"], keep="first")
        df = df.sort_values("occurredAt").reset_index(drop=True)

    print(f"\nfinal events: {len(df)}")
    if len(df):
        print("earliest:", df["occurredAt"].min())
        print("latest:  ", df["occurredAt"].max())

    df.to_csv(BASE_CSV, index=False)
    print(f"wrote {BASE_CSV}")

    # raw JSONL too (handy for LLM / audits)
    with open(BASE_JSONL, "w", encoding="utf-8") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {BASE_JSONL}")


if __name__ == "__main__":
    main()

