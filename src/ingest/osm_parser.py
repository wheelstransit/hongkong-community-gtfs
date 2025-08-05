import requests
import json

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# hong kong bounding box (south, west, north, east)
HONG_KONG_BBOX = (22.15, 113.83, 22.56, 114.42)

def build_overpass_query_routes(bbox):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""
    [out:json][timeout:300];
    (
      relation["type"="route"]["route"~"^(bus|trolleybus|minibus|train|tram|subway|ferry)$"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    return query

def fetch_osm_routes(silent=False):
    if not silent:
        print("building overpass query for hong kong public transit routes...")
    query = build_overpass_query_routes(HONG_KONG_BBOX)
    if not silent:
        print("fetching route data from overpass api (may take a few minutes)")
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
        response.raise_for_status()
        if not silent:
            print("successfully fetched route data from overpass api")
        return response.json()
    except requests.exceptions.RequestException as e:
        if not silent:
            print(f"error fetching route data from overpass api: {e}")
        return None
    except json.JSONDecodeError as e:
        if not silent:
            print(f"error decoding json response: {e}")
        return None

def build_overpass_query_station_relations(bbox):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""
    [out:json][timeout:300];
    (
      relation[name]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    return query

def fetch_osm_station_relations(silent=False):
    if not silent:
        print("building overpass query for hong kong station relations...")
    query = build_overpass_query_station_relations(HONG_KONG_BBOX)
    if not silent:
        print("fetching station relation data from overpass api (may take a few minutes)")
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
        response.raise_for_status()
        if not silent:
            print("successfully fetched station relation data from overpass api")
        return response.json()
    except requests.exceptions.RequestException as e:
        if not silent:
            print(f"error fetching station relation data from overpass api: {e}")
        return None
    except json.JSONDecodeError as e:
        if not silent:
            print(f"error decoding json response: {e}")
        return None


def build_overpass_query_subway_entrances(bbox):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""
    [out:json][timeout:300];
    (
      node["railway"="subway_entrance"]({bbox_str});
    );
    out body geom;
    """
    return query

def fetch_osm_subway_entrances(silent=False):
    if not silent:
        print("building overpass query for subway entrances...")
    query = build_overpass_query_subway_entrances(HONG_KONG_BBOX)
    if not silent:
        print("fetching subway entrance data from overpass api...")
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
        response.raise_for_status()
        if not silent:
            print("successfully fetched subway entrance data from overpass api")
        return response.json()
    except requests.exceptions.RequestException as e:
        if not silent:
            print(f"error fetching subway entrance data from overpass api: {e}")
        return None
    except json.JSONDecodeError as e:
        if not silent:
            print(f"error decoding json response: {e}")
        return None

if __name__ == '__main__':
    print("running osm parser as a standalone script...")
    osm_routes = fetch_osm_routes()
    if osm_routes:
        print(f"successfully retrieved {len(osm_routes.get('elements', []))} route elements from osm")
        with open("osm_routes.json", "w") as f:
            json.dump(osm_routes, f, indent=2)
        print("route data saved to osm_routes.json")

    osm_stations = fetch_osm_station_relations()
    if osm_stations:
        print(f"successfully retrieved {len(osm_stations.get('elements', []))} station relation elements from osm")
        with open("osm_station_relations.json", "w") as f:
            json.dump(osm_stations, f, indent=2)
        print("station relation data saved to osm_station_relations.json")

    osm_subway_entrances = fetch_osm_subway_entrances()
    if osm_subway_entrances:
        print(f"successfully retrieved {len(osm_subway_entrances.get('elements', []))} subway entrances from osm")
        with open("osm_subway_entrances.json", "w") as f:
            json.dump(osm_subway_entrances, f, indent=2)
        print("subway entrance data saved to osm_subway_entrances.json")
