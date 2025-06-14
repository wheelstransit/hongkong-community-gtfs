import requests
import json
import concurrent.futures
import time
from tqdm import tqdm
import random

class GMBClient:
    BASE_URL = "https://data.etagmb.gov.hk"

    def __init__(self, timeout=30):
        self.session = requests.Session()
        self.timeout = timeout

    def _make_request(self, endpoint, max_retries=5):
        """Makes a request with exponential backoff for retries."""
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.json().get('data')
                elif response.status_code == 404:
                    print(f"HTTP 404 Not Found for {url}. The resource does not exist.")
                    return None
                else:
                    print(f"HTTP {response.status_code} error for {url}. Attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** (attempt + 1)) + random.uniform(0, 1)
                        print(f"Waiting {wait_time:.2f} seconds before retry...")
                        time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"Request exception for {url}: {e}. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** (attempt + 1)) + random.uniform(0, 1)
                    print(f"Waiting {wait_time:.2f} seconds before retry...")
                    time.sleep(wait_time)
                continue
            except (json.JSONDecodeError, KeyError):
                print(f"Error: Could not parse JSON or 'data' key not found in response from {url}.")
                return None
        
        print(f"Failed to get successful response from {url} after {max_retries} attempts")
        return None

    def get_all_routes(self, region=None):
        endpoint = "/route"
        if region:
            if region.upper() not in ['HKI', 'KLN', 'NT']:
                raise ValueError("Region must be one of 'HKI', 'KLN', or 'NT'.")
            endpoint = f"/route/{region.upper()}"
        
        data = self._make_request(endpoint)
        return data.get('routes') if data else None

    def get_route_details(self, route_id=None, region=None, route_code=None):
        if route_id:
            endpoint = f"/route/{route_id}"
        elif region and route_code:
            endpoint = f"/route/{region.upper()}/{route_code}"
        else:
            raise ValueError("You must provide either 'route_id' or both 'region' and 'route_code'.")
        
        return self._make_request(endpoint)

    def get_stop_details(self, stop_id):
        data = self._make_request(f"/stop/{stop_id}")
        if data:
            # Add stop_id to the response data for easier mapping
            data['stop_id'] = stop_id
        return data

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

    def get_all_stops_and_route_stops(self, max_workers=10):
        all_routes_by_region = self.get_all_routes()
        if not all_routes_by_region:
            print("Could not fetch initial route list. Aborting.")
            return [], []

        all_route_stops = []
        unique_stop_ids = set()
        
        tasks = []
        for region, route_codes in all_routes_by_region.items():
            for route_code in route_codes:
                tasks.append({'region': region, 'route_code': route_code})

        for task in tqdm(tasks, desc="Processing routes"):
            region = task['region']
            route_code = task['route_code']
            route_details = self.get_route_details(region=region, route_code=route_code)
            if not route_details:
                continue

            for route_variant in route_details:
                route_id = route_variant.get('route_id')
                if not route_id:
                    continue
                
                for direction in route_variant.get('directions', []):
                    route_seq = direction.get('route_seq')
                    if not route_seq:
                        continue
                    
                    route_stops_data = self.get_route_stops(route_id, route_seq)
                    if not route_stops_data or 'route_stops' not in route_stops_data:
                        continue
                        
                    for stop_info in route_stops_data['route_stops']:
                        stop_id = stop_info.get('stop_id')
                        if not stop_id:
                            continue
                        
                        unique_stop_ids.add(stop_id)
                        all_route_stops.append({
                            'route_id': route_id,
                            'route_seq': route_seq,
                            'region': region,
                            'route_code': route_code,
                            'stop_id': stop_id,
                            'sequence': stop_info.get('stop_seq'), 
                            'stop_name_en': stop_info.get('name_en'),
                            'stop_name_tc': stop_info.get('name_tc'),
                            'stop_name_sc': stop_info.get('name_sc'),
                        })
        
        print(f"\nFound {len(all_route_stops)} route-stop records.")
        print(f"Found {len(unique_stop_ids)} unique stops to fetch details for.")

        print(f"\n--- Phase 2: Fetching details for {len(unique_stop_ids)} unique stops (multithreaded) ---")
        all_stops_details = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stop = {executor.submit(self.get_stop_details, stop_id): stop_id for stop_id in unique_stop_ids}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_stop), total=len(unique_stop_ids), desc="Fetching stop details"):
                stop_details = future.result()
                if stop_details:
                    all_stops_details.append(stop_details)
        
        print(f"\nSuccessfully fetched details for {len(all_stops_details)} unique stops.")
        return all_stops_details, all_route_stops


if __name__ == '__main__':
    client = GMBClient()
    all_stops, all_route_stops = client.get_all_stops_and_route_stops(max_workers=10)
    
    print("\n--- Summary ---")
    print(f"Total unique stops with details: {len(all_stops)}")
    print(f"Total route-stop records: {len(all_route_stops)}")
    
    if all_stops:
        print("\nSample stop data:")
        print(json.dumps(all_stops[0], indent=2, ensure_ascii=False))
        
    if all_route_stops:
        print("\nSample route-stop record:")
        print(json.dumps(all_route_stops[0], indent=2, ensure_ascii=False))