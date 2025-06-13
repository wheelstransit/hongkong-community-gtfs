import requests
import json

#i have no idea why i decided to write this in OOP

class GMBClient:
    BASE_URL = "https://data.etagmb.gov.hk"

    def __init__(self, timeout=30):
        self.session = requests.Session()
        self.timeout = timeout

    def _make_request(self, endpoint):
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get('data')
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None
        except (json.JSONDecodeError, KeyError):
            print(f"Error: Could not parse JSON or 'data' key not found in response from {url}.")
            return None

    def get_all_routes(self, region=None):
        endpoint = "/route"
        if region:
            if region.upper() not in ['HKI', 'KLN', 'NT']:
                raise ValueError("Region must be one of 'HKI', 'KLN', or 'NT'.")
            endpoint = f"/route/{region.upper()}"
        
        data = self._make_request(endpoint)
        #help i don't know what i'm doing
        if data and region:
            return data.get('routes')
        elif data:
            return data.get('routes')
        return None

    def get_route_details(self, route_id=None, region=None, route_code=None):
        if route_id:
            endpoint = f"/route/{route_id}"
        elif region and route_code:
            endpoint = f"/route/{region.upper()}/{route_code}"
        else:
            raise ValueError("You must provide either 'route_id' or both 'region' and 'route_code'.")
        
        return self._make_request(endpoint)

    def get_stop_details(self, stop_id):
        return self._make_request(f"/stop/{stop_id}")

    def get_route_stops(self, route_id, route_seq):
        return self._make_request(f"/route-stop/{route_id}/{route_seq}")

    def get_routes_for_stop(self, stop_id):
        return self._make_request(f"/stop-route/{stop_id}")

    def get_last_update_time(self, entity, **kwargs):
        endpoint = f"/last-update/{entity}"
        path_parts = []
        if 'region' in kwargs:
            path_parts.append(str(kwargs['region']))
        if 'route_code' in kwargs:
            path_parts.append(str(kwargs['route_code']))
        if 'route_id' in kwargs:
            path_parts.append(str(kwargs['route_id']))
        if 'route_seq' in kwargs:
            path_parts.append(str(kwargs['route_seq']))
        if 'stop_id' in kwargs:
            path_parts.append(str(kwargs['stop_id']))
        
        if path_parts:
            endpoint += "/" + "/".join(path_parts)
            
        return self._make_request(endpoint)

if __name__ == '__main__':
    client = GMBClient()

    print("--- 1. Fetching All Routes (Grouped by Region) ---")
    all_routes = client.get_all_routes()
    if all_routes:
        print(f"Found regions: {list(all_routes.keys())}")
        sample_region = 'HKI'
        sample_route_code = all_routes[sample_region][0]
        print(f"Sample route from {sample_region}: {sample_route_code}")
    print("-" * 40)

    print(f"--- 2. Fetching Route Details for {sample_region} {sample_route_code} ---")
    route_details = client.get_route_details(region=sample_region, route_code=sample_route_code)
    if route_details:
        sample_route_id = route_details[0].get('route_id')
        sample_route_seq = route_details[0]['directions'][0].get('route_seq')
        print(f"Route Name: {route_details[0]['description_en']}")
        print(f"Route ID found: {sample_route_id}")
    print("-" * 40)

    if sample_route_id and sample_route_seq:
        print(f"--- 3. Fetching Stops for Route ID {sample_route_id} (Sequence {sample_route_seq}) ---")
        route_stops = client.get_route_stops(route_id=sample_route_id, route_seq=sample_route_seq)
        if route_stops:
            all_stops_on_route = route_stops.get('route_stops')
            sample_stop_id = all_stops_on_route[0].get('stop_id')
            print(f"Found {len(all_stops_on_route)} stops on this route.")
            print(f"First stop is '{all_stops_on_route[0]['name_en']}' (ID: {sample_stop_id})")
        print("-" * 40)
        
        if sample_stop_id:
            print(f"--- 4. Fetching Details for Stop ID {sample_stop_id} ---")
            stop_details = client.get_stop_details(stop_id=sample_stop_id)
            if stop_details:
                wgs84 = stop_details['coordinates'].get('wgs84')
                print(f"Stop is enabled: {stop_details.get('enabled')}")
                print(f"Coordinates (WGS84): Lat={wgs84.get('latitude')}, Lon={wgs84.get('longitude')}")
            print("-" * 40)
            
            print(f"--- 5. Fetching All Routes that Service Stop ID {sample_stop_id} ---")
            routes_at_stop = client.get_routes_for_stop(stop_id=sample_stop_id)
            if routes_at_stop:
                print(f"Found {len(routes_at_stop)} route variations servicing this stop.")
                print(f"Example route ID servicing this stop: {routes_at_stop[0].get('route_id')}")