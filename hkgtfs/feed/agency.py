import pandas as pd

AGENCIES = [
    ("KMB", "Kowloon Motor Bus", "https://kmb.hk"),
    ("CTB", "Citybus", "https://www.citybus.com.hk"),
    ("MTRB", "MTR Bus", "https://www.mtr.com.hk"),
    ("GMB", "Green Minibus", "https://td.gov.hk"),
    ("NLB", "New Lantao Bus", "https://www.nlb.com.hk"),
    ("FERRY", "Ferry Services", "https://td.gov.hk"),
    ("MTRR", "MTR Rail", "https://www.mtr.com.hk"),
    ("LR", "Light Rail", "https://www.mtr.com.hk"),
    ("AE", "Airport Express", "https://www.mtr.com.hk"),
    ("PT", "Peak Tram", "https://thepeak.com.hk"),
    ("TRAM", "Tramways", "https://www.hktramways.com"),
]


def build_feed_info(version: str, calendar: pd.DataFrame) -> pd.DataFrame:
    active = calendar[calendar["end_date"] > 19700101]
    return pd.DataFrame([{"feed_publisher_name": "Hong Kong Community GTFS",
                          "feed_publisher_url": "https://github.com/wheelstransit/hongkong-community-gtfs",
                          "feed_lang": "en", "feed_version": version,
                          "feed_start_date": active["start_date"].min(), "feed_end_date": active["end_date"].max(),
                          "feed_contact_url": "https://github.com/wheelstransit/hongkong-community-gtfs/issues"}])


def build_agency() -> pd.DataFrame:
    return pd.DataFrame([{"agency_id": i, "agency_name": n, "agency_url": u, "agency_timezone": "Asia/Hong_Kong", "agency_lang": "en"}
                         for i, n, u in AGENCIES])
