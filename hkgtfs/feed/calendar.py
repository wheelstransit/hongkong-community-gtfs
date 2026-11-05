import pandas as pd

from ..db import Store

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DEFAULT = {**{d: 1 for d in DAYS}, "start_date": "20230101", "end_date": "20991231"}


def build_calendar(db: Store, trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gov_calendar = db.read("SELECT * FROM gov_gtfs_calendar")
    gov_calendar["service_id"] = gov_calendar["service_id"].astype(str)
    gov_dates = db.read("SELECT * FROM gov_gtfs_calendar_dates")
    gov_dates["service_id"] = gov_dates["service_id"].astype(str)
    services = trips[["service_id", "original_service_id"]].drop_duplicates()

    calendar = services.merge(gov_calendar.set_index("service_id"), left_on="original_service_id", right_index=True, how="left")
    for col, val in DEFAULT.items():
        calendar[col] = calendar[col].fillna(val)
    for col in ["start_date", "end_date", *DAYS]:
        calendar[col] = pd.to_numeric(calendar[col]).astype(int)
    calendar = calendar.drop(columns=["original_service_id"]).drop_duplicates(subset=["service_id"])

    dates = services.merge(gov_dates.set_index("service_id"), left_on="original_service_id", right_index=True)
    dates = dates.drop(columns=["original_service_id"]).drop_duplicates(subset=["service_id", "date"])

    never = trips.loc[trips["original_service_id"] == "NEVER", "service_id"].unique()
    if len(never):
        dummy = pd.DataFrame([{"service_id": s, **{d: 0 for d in DAYS}, "start_date": 19700101, "end_date": 19700101} for s in never])
        calendar = pd.concat([calendar, dummy], ignore_index=True).drop_duplicates(subset=["service_id"])
    return calendar, dates
