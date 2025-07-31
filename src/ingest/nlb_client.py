import requests
import time
import json
import concurrent.futures
from tqdm import tqdm

BASE_URL = "https://rt.data.gov.hk/v2/transport/nlb/"

def fetch_all_routes():
    endpoint = f"{BASE_URL}route.php?action=list"
    print("Fetching all NLB routes...")

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json().get('routes')
        if data is not None:
            print(f"Successfully fetched {len(data)} routes.")
        else:
            print("Warning: 'routes' key not found in the response, but request was successful.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NLB routes: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from response.")
        return None

def fetch_stops_for_route(route):
    route_id = route.get('routeId')
    if not route_id:
        return route, None

    endpoint = f"{BASE_URL}stop.php?action=list&routeId={route_id}"
    try:
        response = requests.get(endpoint, timeout=15)
        response.raise_for_status()
        stops_data = response.json().get('stops')
        return route, stops_data
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return route, None

def fetch_all_stops_and_route_stops_threaded(routes, max_workers=20, silent=False):
    if not silent:
        print(f"\nFetching all NLB stops and route-stop sequences with up to {max_workers} threads...")
    
    if not routes:
        print("Could not fetch routes, aborting stop fetching.")
        return None, None


    all_stops_dict = {}
    all_route_stops = []

    tasks = routes # idk if this is the right way to do it or not someone please educate me
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(fetch_stops_for_route, tasks)
        
        progress_bar = tqdm(results_iterator, total=len(tasks), desc="Processing routes")

        for route, stops_for_route in progress_bar:
            if not stops_for_route:
                continue

            for seq, stop in enumerate(stops_for_route):
                stop_id = stop.get('stopId')
                if not stop_id:
                    continue
                
                if stop_id not in all_stops_dict:
                    all_stops_dict[stop_id] = stop
                
                all_route_stops.append({
                    "routeId": route.get("routeId"),
                    "routeNo": route.get("routeNo"),
                    "stopId": stop_id,
                    "sequence": seq + 1,
                })
    
    unique_stops_list = list(all_stops_dict.values())
    print(f"\nSuccessfully fetched {len(unique_stops_list)} unique stops.")
    print(f"Successfully fetched {len(all_route_stops)} route-stop records.")
    
    return unique_stops_list, all_route_stops


if __name__ == '__main__':
    start_time = time.time()
    print("testing :)")

    routes_data = fetch_all_routes()
    if routes_data:
        print("\nSample route data:")
        print(routes_data[0])
    print("-" * 30)

    stops_data, route_stops_data = fetch_all_stops_and_route_stops_threaded(max_workers=30)
    
    if stops_data:
        print("\nSample unique stop data (details are included):")
        print(stops_data[0])
    print("-" * 30)

    if route_stops_data:
        print("\nSample route-stop data:")
        sample_rs = next((rs for rs in route_stops_data if rs.get('sequence', 0) > 1), route_stops_data[0])
        print(sample_rs)
    
    end_time = time.time()
    print("done testing :)")

"""
testing :)
Fetching all NLB routes...
Successfully fetched 62 routes.

Sample route data:
{'routeId': '2', 'routeNo': '1', 'routeName_c': '大澳 > 梅窩碼頭', 'routeName_s': '大澳 > 梅窝码头', 'routeName_e': 'Tai O > Mui Wo Ferry Pier', 'overnightRoute': 0, 'specialRoute': 0}
------------------------------

Fetching all NLB stops and route-stop sequences with up to 30 threads...
Fetching all NLB routes...
Successfully fetched 62 routes.
Processing routes: 100%
Successfully fetched 284 unique stops.
Successfully fetched 1395 route-stop records.

Sample unique stop data (details are included):
{'stopId': '221', 'stopName_c': '大澳', 'stopName_s': '大澳', 'stopName_e': 'Tai O', 'stopLocation_c': '大澳道', 'stopLocation_s': '大澳道', 'stopLocation_e': 'Tai O Road', 'latitude': '22.25278300', 'longitude': '113.86216000', 'fare': '12.7', 'fareHoliday': '21.4', 'someDepartureObserveOnly': 0}
------------------------------

Sample route-stop data:
{'routeId': '2', 'routeNo': '1', 'stopId': '222', 'sequence': 2}
done testing :)

"""