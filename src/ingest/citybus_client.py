import requests
import time
import concurrent.futures
from tqdm import tqdm

BASE_URL = "https://rt.data.gov.hk/v2/transport/citybus"

def fetch_all_routes():
    endpoint = f"{BASE_URL}/route/CTB"
    print("Fetching all Citybus routes...")
    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json().get('data')
        if data is not None:
            print(f"Successfully fetched {len(data)} routes.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Citybus routes: {e}")
        return None
    except (KeyError, ValueError):
        print("Error: 'data' key not found or invalid JSON in the response.")
        return None

def fetch_route_stops(route_id, direction): #can't just do it all at once :(
    endpoint = f"{BASE_URL}/route-stop/CTB/{route_id}/{direction}"
    try:
        response = requests.get(endpoint, timeout=15)
        response.raise_for_status()
        return response.json().get('data')
    except (requests.exceptions.RequestException, KeyError, ValueError):
        return None

def worker_fetch_stops(task):
    route_id, direction = task
    return fetch_route_stops(route_id, direction)

def fetch_all_stops_threaded(max_workers=20):
    print(f"\nFetching all unique stop IDs with up to {max_workers} threads:>")
    all_routes = fetch_all_routes()
    if not all_routes:
        print("Could not fetch routes, cannot proceed to fetch stops.")
        return None
    tasks = []
    for route_info in all_routes:
        route_id = route_info.get('route')
        if route_id:
            tasks.append((route_id, 'inbound'))
            tasks.append((route_id, 'outbound'))

    unique_stop_ids = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #it's 3am and idk what i'm doing
        results_iterator = executor.map(worker_fetch_stops, tasks)

        results_iterator = tqdm(results_iterator, total=len(tasks), desc="Processing routes")

        for stops_on_route in results_iterator:
            if stops_on_route:
                for stop_data in stops_on_route:
                    if stop_data and 'stop' in stop_data:
                        unique_stop_ids.add(stop_data['stop'])

    print(f"\nSuccessfully found {len(unique_stop_ids)} unique stop IDs across all routes.")
    return sorted(list(unique_stop_ids))


def fetch_stop_details(stop_id):
    endpoint = f"{BASE_URL}/stop/{stop_id}"
    # print(f"\nFetching details for stop ID: {stop_id}...")
    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json().get('data')
        return data
    except (requests.exceptions.RequestException, KeyError, ValueError):
        return None

def worker_fetch_stop_details(stop_id):
    return fetch_stop_details(stop_id)

def fetch_all_stop_details_threaded(stop_ids, max_workers=20):
    print(f"\nFetching details for {len(stop_ids)} stops with up to {max_workers} threads...")
    if not stop_ids:
        print("No stop IDs provided.")
        return []

    stop_details = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(worker_fetch_stop_details, stop_ids)

        results_iterator = tqdm(results_iterator, total=len(stop_ids), desc="Fetching stop details")

        for stop_detail in results_iterator:
            if stop_detail:
                stop_details.append(stop_detail)

    print(f"\nSuccessfully fetched details for {len(stop_details)} stops.")
    return stop_details


if __name__ == '__main__':
    start_time = time.time()
    print("testing :)")

    routes_data = fetch_all_routes()
    if routes_data:
        print("\nSample route data:")
        print(routes_data[0])
    print("-" * 20)

    all_stop_ids = fetch_all_stops_threaded(max_workers=30)

    if all_stop_ids:
        print("\nSample of fetched stop IDs:")
        print(all_stop_ids[:5])

        print("\nFetching details for first 5 stops using threaded method...")
        sample_stop_details = fetch_all_stop_details_threaded(all_stop_ids[:5], max_workers=5)
        if sample_stop_details:
            print("Sample stop detail data:")
            print(sample_stop_details[0])

    end_time = time.time()
    print("done testing :)")

"""
testing :)
Fetching all Citybus routes...
Successfully fetched 400 routes.

Sample route data:
{'co': 'CTB', 'route': '1', 'orig_tc': '中環 (港澳碼頭)', 'orig_en': 'Central (Macau Ferry)', 'dest_tc': '跑馬地 (上)', 'dest_en': 'Happy Valley (Upper)', 'orig_sc': '中环 (港澳码头)', 'dest_sc': '跑马地 (上)', 'data_timestamp': '2025-06-14T05:00:02+08:00'}
--------------------

Fetching all unique stop IDs with up to 30 threads:>
Fetching all Citybus routes...
Successfully fetched 400 routes.
Processing routes: 100%

Successfully found 2558 unique stop IDs across all routes.

Sample of fetched stop IDs:
['001001', '001002', '001003', '001004', '001005']

Fetching details for a sample stop...
Sample stop detail data:
{'stop': '001001', 'name_tc': '中央廣場, 亞畢諾道', 'name_en': 'The Centrium, Arbuthnot Road', 'lat': '22.279916732091', 'long': '114.15458450053', 'name_sc': '中央广场, 亚毕诺道', 'data_timestamp': '2025-06-14T05:00:02+08:00'}
done testing :)
"""
