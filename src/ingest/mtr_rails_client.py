import requests
import csv
import io
from typing import List, Dict, Optional

MTR_LINES_AND_STATIONS_URL = "https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv"
MTR_LINES_FARES_URL = "https://opendata.mtr.com.hk/data/mtr_lines_fares.csv"
LIGHT_RAIL_ROUTES_AND_STOPS_URL = "https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv"
LIGHT_RAIL_FARES_URL = "https://opendata.mtr.com.hk/data/light_rail_fares.csv"
AIRPORT_EXPRESS_FARES_URL = "https://opendata.mtr.com.hk/data/airport_express_fares.csv"

def fetch_and_parse_csv(url: str, encoding: str = "utf-8-sig", silent=False) -> List[Dict]:
    if not silent:
        print(f"Fetching CSV from {url} ...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw_text = response.content.decode(encoding)
        csv_file = io.StringIO(raw_text)
        reader = csv.DictReader(csv_file)
        data = list(reader)
        if not silent:
            print(f"Fetched and parsed {len(data)} records from {url}")
        return data
    except Exception as e:
        if not silent:
            print("bruh")
            print(f"Error fetching/parsing CSV from {url}: {e}")
        return []

def fetch_mtr_lines_and_stations(silent=False) -> List[Dict]:
    return fetch_and_parse_csv(MTR_LINES_AND_STATIONS_URL, silent=silent)

def fetch_mtr_lines_fares(silent=False) -> List[Dict]:
    return fetch_and_parse_csv(MTR_LINES_FARES_URL, silent=silent)

def fetch_light_rail_routes_and_stops(silent=False) -> List[Dict]:
    return fetch_and_parse_csv(LIGHT_RAIL_ROUTES_AND_STOPS_URL, silent=silent)

def fetch_light_rail_fares(silent=False) -> List[Dict]:
    return fetch_and_parse_csv(LIGHT_RAIL_FARES_URL, silent=silent)

def fetch_airport_express_fares(silent=False) -> List[Dict]:
    return fetch_and_parse_csv(AIRPORT_EXPRESS_FARES_URL, silent=silent)


if __name__ == "__main__":
    print("Testing MTR Rails Client...")

    mtr_lines_stations = fetch_mtr_lines_and_stations()
    if mtr_lines_stations:
        print("Sample MTR Line/Station record:")
        print(mtr_lines_stations[0])

    mtr_fares = fetch_mtr_lines_fares()
    if mtr_fares:
        print("Sample MTR Fare record:")
        print(mtr_fares[0])

    lrt_routes_stops = fetch_light_rail_routes_and_stops()
    if lrt_routes_stops:
        print("Sample Light Rail Route/Stop record:")
        print(lrt_routes_stops[0])

    lrt_fares = fetch_light_rail_fares()
    if lrt_fares:
        print("Sample Light Rail Fare record:")
        print(lrt_fares[0])

    ae_fares = fetch_airport_express_fares()
    if ae_fares:
        print("Sample Airport Express Fare record:")
        print(ae_fares[0])

    print("Done testing MTR Rails Client.")
