import requests
import csv
import io
import html

ROUTES_URL = "https://opendata.mtr.com.hk/data/mtr_bus_routes.csv"
STOPS_URL = "https://opendata.mtr.com.hk/data/mtr_bus_stops.csv"
FARES_URL = "https://opendata.mtr.com.hk/data/mtr_bus_fares.csv"

def fetch_and_parse_csv(url, data_name="data"):
    print(f"Fetching MTR Bus {data_name} from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # WHYJANKAKSJDNKJASdnkn

        raw_text = response.content.decode('utf-8-sig')

        decoded_content = html.unescape(raw_text)

        csv_file = io.StringIO(decoded_content)

        reader = csv.DictReader(csv_file)
        data = list(reader)

        print(f"Successfully fetched and parsed {len(data)} records for {data_name}.")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching MTR {data_name}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the {data_name} CSV: {e}")
        return None


def fetch_all_routes():
    return fetch_and_parse_csv(ROUTES_URL, "routes")


def fetch_all_fares():
    return fetch_and_parse_csv(FARES_URL, "fares")


def fetch_all_route_stops():
    return fetch_and_parse_csv(STOPS_URL, "route-stops")


def fetch_all_stops():
    print("\nProcessing raw data to generate a list of unique stops...")
    route_stops_data = fetch_all_route_stops()
    if not route_stops_data:
        print("Could not fetch route-stop data, cannot generate unique stops list.")
        return None

    unique_stops = {}
    for row in route_stops_data:
        station_id = row.get('STATION_ID')
        if station_id and station_id not in unique_stops:
            unique_stops[station_id] = {
                'stop_id': station_id,
                'name_en': row.get('STATION_NAME_ENG'),
                'name_zh': row.get('STATION_NAME_CHI'),
                'lat': row.get('STATION_LATITUDE'),
                'long': row.get('STATION_LONGITUDE'),
            }

    unique_stops_list = list(unique_stops.values())
    print(f"Successfully identified {len(unique_stops_list)} unique stops.")
    return unique_stops_list


if __name__ == '__main__':
    print("testing :)")

    routes_data = fetch_all_routes()
    if routes_data:
        print("\nSample route data:")
        print(routes_data[0])
    print("-" * 30)

    stops_data = fetch_all_stops()
    if stops_data:
        print("\nSample unique stop data:")
        print(stops_data[0])
    print("-" * 30)

    route_stops_data = fetch_all_route_stops()
    if route_stops_data:
        print("\nSample route-stop data (raw from CSV):")
        print(route_stops_data[0])
    print("-" * 30)

    fares_data = fetch_all_fares()
    if fares_data:
        print("\nSample fare data:")
        print(fares_data[0])
    print("-" * 30)

    print("done testing :)")

"""
testing :)
Fetching MTR Bus routes from https://opendata.mtr.com.hk/data/mtr_bus_routes.csv...
Successfully fetched and parsed 22 records for routes.

Sample route data:
{'ROUTE_ID': '506', 'ROUTE_NAME_CHI': '屯門碼頭至兆麟', 'ROUTE_NAME_ENG': 'Tuen Mun Ferry Pier to Siu Lun', 'IS_CIRCULAR': ''}
------------------------------

Processing raw data to generate a list of unique stops...
Fetching MTR Bus route-stops from https://opendata.mtr.com.hk/data/mtr_bus_stops.csv...
Successfully fetched and parsed 628 records for route-stops.
Successfully identified 628 unique stops.

Sample unique stop data:
{'stop_id': 'K12-U010', 'name_en': 'Tai Po Market Station', 'name_zh': '大埔墟站', 'lat': '22.444414', 'long': '114.169471'}
------------------------------
Fetching MTR Bus route-stops from https://opendata.mtr.com.hk/data/mtr_bus_stops.csv...
Successfully fetched and parsed 628 records for route-stops.

Sample route-stop data (raw from CSV):
{'ROUTE_ID': 'K12', 'DIRECTION': 'O', 'STATION_SEQNO': '1', 'STATION_ID': 'K12-U010', 'STATION_LATITUDE': '22.444414', 'STATION_LONGITUDE': '114.169471', 'STATION_NAME_CHI': '大埔墟站', 'STATION_NAME_ENG': 'Tai Po Market Station'}
------------------------------
Fetching MTR Bus fares from https://opendata.mtr.com.hk/data/mtr_bus_fares.csv...
Successfully fetched and parsed 22 records for fares.

Sample fare data:
{'ROUTE_ID': 'K12', 'FARE_OCTO_ADULT': '4.8', 'FARE_OCTO_CHILD': '2.4', 'FARE_OCTO_ELDERLY': '2', 'FARE_OCTO_JOYU': '2', 'FARE_OCTO_PWD': '2', 'FARE_OCTO_STUDENT': '4.8', 'FARE_SINGLE_ADULT': '4.8', 'FARE_SINGLE_CHILD': '2.4', 'FARE_SINGLE_ELDERLY': '2.4', 'FARE_SINGLE_JOYU': '-', 'FARE_SINGLE_PWD': '-', 'FARE_SINGLE_STUDENT': '-'}
------------------------------
done testing :)
"""
