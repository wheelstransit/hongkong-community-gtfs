import asyncio
import csv
import io
import httpx
from pyproj import Transformer
from typing import List, Dict

# constants
# constants
MTR_LINES_AND_STATIONS_URL = "https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv"
MTR_LINES_FARES_URL = "https://opendata.mtr.com.hk/data/mtr_lines_fares.csv"
LIGHT_RAIL_ROUTES_AND_STOPS_URL = "https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv"
LIGHT_RAIL_FARES_URL = "https://opendata.mtr.com.hk/data/light_rail_fares.csv"
AIRPORT_EXPRESS_FARES_URL = "https://opendata.mtr.com.hk/data/airport_express_fares.csv"
GEODATA_API_URL = "https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q="

async def fetch_station_location(client, station_name_tc, epsg_transformer):
    try:
        query = f"港鐵{station_name_tc}站"
        url = f"{GEODATA_API_URL}{query}"
        response = await client.get(url, headers={'Accept': 'application/json'})
        response.raise_for_status()
        data = response.json()
        if data:
            y, x = data[0]['y'], data[0]['x']
            lat, lon = epsg_transformer.transform(y, x)
            return lat, lon
    except (httpx.RequestError, IndexError, KeyError, ValueError):
        pass
    return None, None

async def fetch_mtr_lines_and_stations_with_locations(silent=False) -> List[Dict]:
    if not silent:
        print(f"fetching mtr lines and stations from {MTR_LINES_AND_STATIONS_URL}...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(MTR_LINES_AND_STATIONS_URL, timeout=30)
            response.raise_for_status()
            raw_text = response.content.decode("utf-8-sig")
            csv_file = io.StringIO(raw_text)
            reader = csv.DictReader(csv_file)
            stations = [row for row in reader if row.get('Line Code') and row.get('Station ID')]
    except Exception as e:
        if not silent:
            print(f"error fetching or parsing base station csv: {e}")
        return []

    if not silent:
        print(f"fetched {len(stations)} station records. now fetching locations...")

    epsg_transformer = Transformer.from_crs('epsg:2326', 'epsg:4326')
    enriched_stations = []
    station_locations = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, pool=None)) as client:
        unique_stations_to_fetch = {row['Station ID']: row for row in stations}

        for station_id, station_data in unique_stations_to_fetch.items():
            lat, lon = await fetch_station_location(client, station_data['Chinese Name'], epsg_transformer)
            if lat and lon:
                station_locations[station_id] = {'latitude': lat, 'longitude': lon}

    for station in stations:
        location = station_locations.get(station['Station ID'])
        if location:
            station.update(location)
        else:
            station['latitude'] = None
            station['longitude'] = None
        enriched_stations.append(station)

    if not silent:
        success_count = len(station_locations)
        print(f"successfully fetched locations for {success_count} out of {len(unique_stations_to_fetch)} unique stations.")

    return enriched_stations

def fetch_and_parse_csv_sync(url: str, silent=False) -> List[Dict]:
    if not silent:
        print(f"fetching csv from {url} ...")
    try:
        response = httpx.get(url, timeout=30)
        response.raise_for_status()
        raw_text = response.content.decode("utf-8-sig")
        csv_file = io.StringIO(raw_text)
        reader = csv.DictReader(csv_file)
        data = list(reader)
        if not silent:
            print(f"fetched and parsed {len(data)} records from {url}")
        return data
    except Exception as e:
        if not silent:
            print(f"error fetching/parsing csv from {url}: {e}")
        return []

def fetch_mtr_lines_fares(silent=False) -> List[Dict]:
    return fetch_and_parse_csv_sync(MTR_LINES_FARES_URL, silent=silent)

def fetch_light_rail_routes_and_stops(silent=False) -> List[Dict]:
    return fetch_and_parse_csv_sync(LIGHT_RAIL_ROUTES_AND_STOPS_URL, silent=silent)

def fetch_light_rail_fares(silent=False) -> List[Dict]:
    return fetch_and_parse_csv_sync(LIGHT_RAIL_FARES_URL, silent=silent)

def fetch_airport_express_fares(silent=False) -> List[Dict]:
    return fetch_and_parse_csv_sync(AIRPORT_EXPRESS_FARES_URL, silent=silent)

if __name__ == "__main__":
    async def main():
        print("testing mtr rails client...")
        mtr_lines_stations = await fetch_mtr_lines_and_stations_with_locations()
        if mtr_lines_stations:
            print("sample mtr line/station record with location:")
            print(mtr_lines_stations[0])

        print("done testing mtr rails client.")

    asyncio.run(main())
