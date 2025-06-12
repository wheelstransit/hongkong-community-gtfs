import requests

BASE_URL = "https://data.etabus.gov.hk/v1/transport/kmb"

def fetch_all_routes():
    endpoint = f"{BASE_URL}/route"
    print("Fetching all KMB routes...")

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json().get('data')
        if data is not None:
            print(f"Successfully fetched {len(data)} routes.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KMB routes: {e}")
        return None
    except KeyError:
        print("Error: 'data' key not found in the response JSON.")
        return None

def fetch_all_stops():
    endpoint = f"{BASE_URL}/stop"
    print("Fetching all KMB stops...")

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json().get('data')
        if data is not None:
            print(f"Successfully fetched {len(data)} stops.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KMB stops: {e}")
        return None
    except KeyError:
        print("Error: 'data' key not found in the response JSON.")
        return None

def fetch_all_route_stops():
    endpoint = f"{BASE_URL}/route-stop"
    print("Fetching all KMB route-stop sequences...")

    try:
        response = requests.get(endpoint, timeout=60) # longer timeout bc of large data
        response.raise_for_status()
        data = response.json().get('data')
        if data is not None:
            print(f"Successfully fetched {len(data)} route-stop records.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KMB route-stops: {e}")
        return None
    except KeyError:
        print("Error: 'data' key not found in the response JSON.")
        return None

if __name__ == '__main__':
    print("testing :)")

    routes_data = fetch_all_routes()
    if routes_data:
        print("Sample route data:")
        print(routes_data[0])
    print("-" * 20)

    stops_data = fetch_all_stops()
    if stops_data:
        print("Sample stop data:")
        print(stops_data[0])
    print("-" * 20)

    route_stops_data = fetch_all_route_stops()
    if route_stops_data:
        print("Sample route-stop data:")
        print(route_stops_data[0])
    print("-" * 20)

    print("done testing :)")