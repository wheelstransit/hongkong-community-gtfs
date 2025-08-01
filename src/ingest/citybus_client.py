import requests
import time
import concurrent.futures
from tqdm import tqdm

BASE_URL = "https://rt.data.gov.hk/v2/transport/citybus"

def fetch_all_routes(silent=False):
    endpoint = f"{BASE_URL}/route/CTB"
    if not silent:
        print("Fetching all Citybus routes...")
    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json().get('data')
        if data is not None and not silent:
            print(f"Successfully fetched {len(data)} routes.")
        return data
    except requests.exceptions.RequestException as e:
        if not silent:
            print(f"Error fetching Citybus routes: {e}")
        return None
    except (KeyError, ValueError):
        if not silent:
            print("Error: 'data' key not found or invalid JSON in the response.")
        return None

def fetch_route_stops(route_id, direction): #can't just do it all at once :(
    # you're getting something like:
    # {"type": "RouteStop", "version": "2.0", "generated_timestamp": "2025-06-25T23:12:23+08:00", "data": [{"co": "CTB", "route": "1", "dir": "I", "seq": 1, "stop": "002403", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 2, "stop": "002402", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 3, "stop": "002492", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 4, "stop": "002493", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 5, "stop": "002453", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 6, "stop": "002552", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 7, "stop": "002553", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 8, "stop": "002467", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 9, "stop": "002566", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 10, "stop": "002537", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 11, "stop": "002446", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 12, "stop": "002449", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 13, "stop": "001140", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 14, "stop": "001142", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 15, "stop": "001054", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "dir": "I", "seq": 16, "stop": "001056", "data_timestamp": "2025-06-25T05:00:04+08:00"}, {"co": "CTB", "route": "1", "d...
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

def fetch_all_stops_threaded(all_routes, max_workers=20, silent=False):
    if not silent:
        print(f"\nFetching all unique stop IDs and route sequences with up to {max_workers} threads:>")
    if not all_routes:
        if not silent:
            print("Could not fetch routes, cannot proceed to fetch stops.")
        return None, None
    tasks = []
    for route_info in all_routes:
        route_id = route_info.get('route')
        if route_id:
            tasks.append((route_id, 'inbound'))
            tasks.append((route_id, 'outbound'))

    unique_stop_ids = set()
    route_stop_sequences = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #it's 3am and idk what i'm doing
        results_iterator = executor.map(worker_fetch_stops, tasks)

        results_iterator = tqdm(results_iterator, total=len(tasks), desc="Processing routes", disable=silent)

        for i, stops_on_route in enumerate(results_iterator):
            route_id, direction = tasks[i]
            if stops_on_route:
                # Store simplified route stop sequence with just stop IDs in order
                stop_ids_in_sequence = [stop_data['stop'] for stop_data in stops_on_route if stop_data and 'stop' in stop_data]
                route_sequence = {
                    'route_id': route_id,
                    'direction': direction,
                    'stop_ids': stop_ids_in_sequence
                }
                route_stop_sequences.append(route_sequence)

                # Also collect unique stop IDs
                for stop_data in stops_on_route:
                    if stop_data and 'stop' in stop_data:
                        unique_stop_ids.add(stop_data['stop'])
    if not silent:
        print(f"\nSuccessfully found {len(unique_stop_ids)} unique stop IDs across {len(route_stop_sequences)} route directions.")
    return sorted(list(unique_stop_ids)), route_stop_sequences

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

def fetch_all_stop_details_threaded(list_of_route_stops, max_workers=20, silent=False):
    # list of route stops should be the result of fetch_all_stops_threaded
    if not silent:
        print(f"\nFetching stop details with up to {max_workers} threads...")
    if not list_of_route_stops:
        if not silent:
            print("No stop IDs provided, cannot fetch stop details.")
        return None

    stop_details = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(worker_fetch_stop_details, list_of_route_stops)

        results_iterator = tqdm(results_iterator, total=len(list_of_route_stops), desc="Fetching stop details", disable=silent)

        for stop_detail in results_iterator:
            if stop_detail:
                stop_details.append(stop_detail)
    if not silent:
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

    all_stop_ids, route_sequences = fetch_all_stops_threaded(routes_data, max_workers=30)

    all_stop_details = fetch_all_stop_details_threaded(all_stop_ids, max_workers=30)

    if all_stop_ids:
        print("\nSample of fetched stop IDs:")
        print(all_stop_ids[:5])

        print("\nFetching details for first 5 stops using threaded method...")
        sample_stop_details = fetch_all_stop_details_threaded(all_stop_ids[:5], max_workers=5)
        if sample_stop_details:
            print("Sample stop detail data:")
            print(sample_stop_details[0])

    if route_sequences:
        print(f"\nSample route sequence (first route):")
        print(f"Route ID: {route_sequences[0]['route_id']}, Direction: {route_sequences[0]['direction']}")
        print(f"First 3 stop IDs: {route_sequences[0]['stop_ids'][:3]}")

    if all_stop_details:
        print("\nSample of fetched stop details:")
        print(all_stop_details[:5])

    end_time = time.time()
    print("done testing :)")