# build_features.py
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Enable FastF1 cache for faster repeat runs
fastf1.Cache.enable_cache("fastf1_cache")

# Choose season(s) and GP(s) — you can add more later
SEASON = 2024
RACES = ["Bahrain", "Monaco", "Austria"]

def build_features(year, gp, session_type="R"):
    print(f"📡 Loading {year} {gp} {session_type} session...")
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=False)
    laps = session.laps.reset_index(drop=True)

    # --- Core Features ---
    laps["lap_sec"] = laps["LapTime"].dt.total_seconds()
    laps["tyre_age"] = laps.groupby(["Driver", "Stint"]).cumcount() + 1
    laps["lap_delta"] = laps.groupby("Driver")["lap_sec"].diff()
    laps["rolling_mean_3"] = (
        laps.groupby("Driver")["lap_sec"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    )

    # --- Pit Next Label ---
    pits = session.laps[session.laps["PitOutTime"].notna() | session.laps["PitInTime"].notna()]
    pit_laps = set(zip(pits["Driver"], pits["LapNumber"]))
    laps["pit_next"] = laps.apply(
        lambda r: 1 if (r["Driver"], int(r["LapNumber"]) + 1) in pit_laps else 0, axis=1
    )

    # --- Keep useful columns ---
    keep_cols = [
        "Driver",
        "Team",
        "LapNumber",
        "Stint",
        "Compound",
        "lap_sec",
        "tyre_age",
        "lap_delta",
        "rolling_mean_3",
        "pit_next",
    ]
    laps = laps[keep_cols].dropna(subset=["lap_sec"])
    laps["Season"] = year
    laps["GrandPrix"] = gp
    return laps


# Combine multiple races
dfs = []
for gp in RACES:
    try:
        df = build_features(SEASON, gp)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Skipped {gp}: {e}")

if dfs:
    all_data = pd.concat(dfs, ignore_index=True)
    out_path = DATA_DIR / "features_all.parquet"
    all_data.to_parquet(out_path)
    print(f"✅ Saved features to {out_path} (shape={all_data.shape})")
else:
    print("❌ No sessions loaded successfully. Check internet or race names.")
