import requests
import json

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box for Hong Kong
# (south, west, north, east)
HONG_KONG_BBOX = (22.15, 113.83, 22.56, 114.42)

def build_overpass_query(bbox):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    # This query looks for relations that are tagged as public transport routes.
    # It includes buses, minibuses, trains, trams, subways, and ferries.
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
        print("Building Overpass query for Hong Kong public transit routes...")
    query = build_overpass_query(HONG_KONG_BBOX)

    if not silent:
        print("Fetching data from Overpass API... (This may take a few minutes)")
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
        response.raise_for_status()
        print("Successfully fetched data from Overpass API.")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Overpass API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Response text: {response.text}")
        return None

if __name__ == '__main__':
    print("Running OSM parser as a standalone script...")
    osm_data = fetch_osm_routes()
    if osm_data:
        print(f"Successfully retrieved {len(osm_data.get('elements', []))} elements from OSM.")
        with open("osm_routes.json", "w") as f:
            json.dump(osm_data, f, indent=2)
        print("Data saved to osm_routes.json")
