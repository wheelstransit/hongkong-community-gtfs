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
    ("FERRY", "Ferry Services", "https://www.td.gov.hk"),
]


def build_agency() -> pd.DataFrame:
    return pd.DataFrame([{"agency_id": i, "agency_name": n, "agency_url": u, "agency_timezone": "Asia/Hong_Kong", "agency_lang": "en"}
                         for i, n, u in AGENCIES])
