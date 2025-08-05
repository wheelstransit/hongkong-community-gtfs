# orig: github.com/hkbus/hk-bus-crawling

import asyncio
import csv
import json
from pyproj import Transformer
import logging
import httpx

async def fetch_light_rail_stops(silent=False):
    # Fetch Light Rail stop data and their locations
    if not silent:
        print("Fetching Light Rail stops...")

    a_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, pool=None))
    epsg_transformer = Transformer.from_crs('epsg:2326', 'epsg:4326')
    stop_list = {}

    try:
        r = await a_client.get('https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv')
        r.raise_for_status()
    except httpx.RequestError as e:
        logging.error(f"Error fetching Light Rail data: {e}")
        return {}

    reader = csv.reader(r.text.splitlines())
    headers = next(reader, None)
    routes = [route for route in reader if len(route) == 7]

    for _, _, _, stop_id, chn, eng, _ in routes:
        light_rail_id = "LR" + stop_id
        if light_rail_id not in stop_list:
            url = f'https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q={chn}輕鐵站'
            try:
                r_geo = await a_client.get(url, headers={'Accept': 'application/json'})
                r_geo.raise_for_status()
                data = r_geo.json()
                if data and len(data) > 0:
                    lat, lng = epsg_transformer.transform(data[0]['y'], data[0]['x'])
                    stop_list[light_rail_id] = {
                        "stop_id": light_rail_id,
                        "name_en": eng,
                        "name_tc": chn,
                        "lat": lat,
                        "lon": lng
                    }
                else:
                    if not silent:
                        print(f"Warning: No location found for {chn} Light Rail Station.")
            except (httpx.RequestError, json.JSONDecodeError, IndexError, KeyError) as e:
                logging.error(f"Error processing geo data for {chn}: {e}")

    await a_client.aclose()
    return stop_list

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    stops = asyncio.run(fetch_light_rail_stops())
    with open('light_rail_stops.json', 'w', encoding='utf-8') as f:
        json.dump(stops, f, ensure_ascii=False, indent=4)
    print(f"Successfully fetched {len(stops)} Light Rail stops.")
