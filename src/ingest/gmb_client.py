import requests
import json
import concurrent.futures
import time
from tqdm import tqdm
import random

class GMBClient:
    BASE_URL = "https://data.etagmb.gov.hk"

    def __init__(self, timeout=30, silent=False):
        self.session = requests.Session()
        self.timeout = timeout
        self.silent = silent

    def _make_request(self, endpoint, max_retries=35):
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code == 200:
                    return response.json().get('data')
                elif response.status_code == 404:
                    if not self.silent:
                        print(f"HTTP 404 Not Found for {url}. The resource does not exist.")
                    return None
                else:
                    if not self.silent:
                        print(f"HTTP {response.status_code} error for {url}. Attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        wait_time = 240 if attempt == 0 else 30
                        if not self.silent:
                            print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    continue

            except requests.exceptions.RequestException as e:
                if not self.silent:
                    print(f"Request exception for {url}: {e}. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = 240 if attempt == 0 else 30
                    if not self.silent:
                        print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                continue
            except (json.JSONDecodeError, KeyError):
                if not self.silent:
                    print(f"Error: Could not parse JSON or 'data' key not found in response from {url}.")
                return None
        if not self.silent:
            print(f"Failed to get successful response from {url} after {max_retries} attempts")
        return None

    def get_all_routes(self, region=None, silent=False):
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

    def get_all_stops_and_route_stops(self, max_workers=10, silent=False):
        all_routes_by_region = self.get_all_routes()
        if not all_routes_by_region:
            if not silent:
                print("Could not fetch initial route list. Aborting.")
            return [], []
        all_route_stops = []
        route_directions = []  # capture per-direction origin/destination data
        unique_stop_ids = set()

        tasks = []
        for region, route_codes in all_routes_by_region.items():
            for route_code in route_codes:
                tasks.append({'region': region, 'route_code': route_code})

        # Track duplication stats
        duplicate_exact = 0  # same (route_id, route_seq, stop_seq, stop_id)
        duplicate_seq_conflict = 0  # same (route_id, route_seq, stop_seq) but different stop_id

        for task in tqdm(tasks, desc="Processing routes", disable=silent):
            region = task['region']
            route_code = task['route_code']
            route_details = self.get_route_details(region=region, route_code=route_code)
            if not route_details:
                continue

            variant_metadata = []
            for route_variant in route_details:
                route_id = route_variant.get('route_id')
                if not route_id:
                    continue
                desc_en = (route_variant.get('description_en') or '').strip()
                desc_tc = (route_variant.get('description_tc') or '').strip()
                desc_sc = (route_variant.get('description_sc') or '').strip()
                is_normal = (
                    ('normal' in desc_en.lower()) or
                    ('正常' in desc_tc) or
                    ('正常' in desc_sc)
                )
                variant_metadata.append({
                    'route_variant': route_variant,
                    'route_id': route_id,
                    'desc_en': desc_en,
                    'desc_tc': desc_tc,
                    'desc_sc': desc_sc,
                    'is_normal': is_normal
                })

            normal_exists = any(meta['is_normal'] for meta in variant_metadata)
            if not normal_exists and not silent:
                print(f"No 'Normal' variant found for {region}-{route_code}; including available variants instead.")

            for meta in variant_metadata:
                if normal_exists and not meta['is_normal']:
                    continue

                route_variant = meta['route_variant']
                route_id = meta['route_id']
                desc_en = meta['desc_en']
                desc_tc = meta['desc_tc']
                desc_sc = meta['desc_sc']

                for direction in route_variant.get('directions', []):
                    route_seq = direction.get('route_seq')
                    if not route_seq:
                        continue
                    orig_en = (direction.get('orig_en') or '').strip()
                    dest_en = (direction.get('dest_en') or '').strip()
                    orig_tc = (direction.get('orig_tc') or '').strip()
                    dest_tc = (direction.get('dest_tc') or '').strip()
                    orig_sc = (direction.get('orig_sc') or '').strip()
                    dest_sc = (direction.get('dest_sc') or '').strip()
                    circular_flag = False
                    for txt in [dest_en.lower(), dest_tc, dest_sc]:
                        if txt and any(token in txt.lower() for token in ['circular', '循環', '环线', '循環線', '循環線)', '循環線)']):
                            circular_flag = True
                            break
                    route_directions.append({
                        'route_id': route_id,
                        'route_seq': route_seq,
                        'region': region,
                        'route_code': route_code,
                        'orig_en': orig_en,
                        'dest_en': dest_en,
                        'orig_tc': orig_tc,
                        'dest_tc': dest_tc,
                        'orig_sc': orig_sc,
                        'dest_sc': dest_sc,
                        'is_circular': circular_flag,
                        'variant_description_en': desc_en,
                    })

                    route_stops_data = self.get_route_stops(route_id, route_seq)
                    if not route_stops_data or 'route_stops' not in route_stops_data:
                        continue

                    seen_full_keys = set()
                    chosen_seq_stop = {}

                    for stop_info in route_stops_data['route_stops']:
                        stop_id = stop_info.get('stop_id')
                        stop_seq = stop_info.get('stop_seq')
                        if not stop_id or stop_seq is None:
                            continue
                        full_key = (route_id, route_seq, stop_seq, stop_id)
                        if full_key in seen_full_keys:
                            duplicate_exact += 1
                            continue
                        seq_key = (route_id, route_seq, stop_seq)
                        if seq_key in chosen_seq_stop and chosen_seq_stop[seq_key] != stop_id:
                            duplicate_seq_conflict += 1
                            continue
                        seen_full_keys.add(full_key)
                        chosen_seq_stop.setdefault(seq_key, stop_id)
                        unique_stop_ids.add(stop_id)
                        all_route_stops.append({
                            'route_id': route_id,
                            'route_seq': route_seq,
                            'region': region,
                            'route_code': route_code,
                            'stop_id': stop_id,
                            'sequence': stop_seq,
                            'stop_name_en': stop_info.get('name_en'),
                            'stop_name_tc': stop_info.get('name_tc'),
                            'stop_name_sc': stop_info.get('name_sc'),
                            'variant_description_en': desc_en,
                            'variant_description_tc': desc_tc,
                            'variant_description_sc': desc_sc,
                        })

        if not silent:
            print(f"\nDedup stats: exact dup rows skipped={duplicate_exact}, conflicting sequence choices skipped={duplicate_seq_conflict}")
            print(f"found {len(all_route_stops)} route-stop records.")
            print(f"found {len(unique_stop_ids)} unique stops to fetch details for.")
            print(f"\nfetching details for {len(unique_stop_ids)} unique stops (multithreaded)")
        all_stops_details = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stop = {executor.submit(self.get_stop_details, stop_id): stop_id for stop_id in unique_stop_ids}
            progress_bar = tqdm(concurrent.futures.as_completed(future_to_stop), total=len(unique_stop_ids), desc="Fetching stop details", disable=silent)
            for future in progress_bar:
                stop_details = future.result()
                if stop_details:
                    all_stops_details.append(stop_details)
        if not silent:
            print(f"\nsuccessfully fetched details for {len(all_stops_details)} unique stops.")
        return all_stops_details, all_route_stops, route_directions



if __name__ == '__main__':
    client = GMBClient()
    all_stops, all_route_stops, route_dirs = client.get_all_stops_and_route_stops(max_workers=10)

    # summary
    print(f"total unique stops with details: {len(all_stops)}")
    print(f"total route-stop records: {len(all_route_stops)}")
    print(f"total route direction records: {len(route_dirs)}")

    if all_stops:
        # sample stop data
        print(json.dumps(all_stops[0], indent=2, ensure_ascii=False))

    if all_route_stops:
        print(json.dumps(all_route_stops[0], indent=2, ensure_ascii=False))
    if route_dirs:
        print(json.dumps(route_dirs[0], indent=2, ensure_ascii=False))
