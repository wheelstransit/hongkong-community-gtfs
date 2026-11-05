import io

import pandas as pd

from ..http import get

BASE_URL = "https://static.data.gov.hk/td/pt-headway-en"
FILES = ("frequencies", "trips", "routes", "calendar", "calendar_dates", "stops", "stop_times", "fare_attributes", "fare_rules")


def fetch():
    return {name: pd.read_csv(io.BytesIO(get(f"{BASE_URL}/{name}.txt", timeout=300).content)) for name in FILES}
