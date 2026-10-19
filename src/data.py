from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_INTERACTION_COLUMNS = {
    "user_id", "app_id", "event_time", "event_name"
}

def load_interactions(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load interactions and apply explicit, auditable data-quality rules."""
    df = pd.read_csv(path)
    missing = REQUIRED_INTERACTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    report = {
        "rows_loaded": len(df),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "invalid_timestamps": 0,
        "missing_user_or_app": 0,
    }
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    report["invalid_timestamps"] = int(df["event_time"].isna().sum())
    report["missing_user_or_app"] = int(
        (df["user_id"].isna() | df["app_id"].isna()).sum()
    )

    # Exact duplicates are retained by default: repeated events may be meaningful.
    # Invalid timestamps cannot participate in a time-based split.
    df = df.dropna(subset=["event_time", "user_id", "app_id"]).copy()
    df["user_id"] = df["user_id"].astype(str)
    df["app_id"] = df["app_id"].astype(str)
    df["event_name"] = df["event_name"].fillna("unknown").astype(str)
    df = df.sort_values(["user_id", "event_time"], kind="mergesort")
    return df.reset_index(drop=True), report


def load_dimensions(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / "users.csv")
    apps = pd.read_csv(data_dir / "apps.csv")
    for frame, key, label in [(users, "user_id", "users"), (apps, "app_id", "apps")]:
        if key not in frame:
            raise ValueError(f"{label}.csv is missing {key!r}")
        if frame[key].duplicated().any():
            raise ValueError(f"{label}.csv contains duplicate {key!r} values")
    return users, apps


def audit_dimensions(
    interactions: pd.DataFrame, users: pd.DataFrame, apps: pd.DataFrame
) -> dict[str, int]:
    return {
        "interaction_users_missing_from_users_dimension": int(
            (~interactions["user_id"].isin(users["user_id"])).sum()
        ),
        "interaction_apps_missing_from_apps_dimension": int(
            (~interactions["app_id"].isin(apps["app_id"])).sum()
        ),
    }
