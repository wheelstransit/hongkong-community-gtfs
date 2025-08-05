# github.com/hkbus/hk-bus-crawling

import asyncio
import logging
from pyproj import Transformer
import json
import string
import httpx
import csv
import re

# HK1980 Grid to WGS84 transformer
epsgTransformer = Transformer.from_crs('epsg:2326', 'epsg:4326')

def check_and_add_result(results, query_name, stop_info, exit_char, barrier_free, final_res_list):
    # Add a formatted entry to the final list if a geocoding match is found
    for result in results:
        if result.get('nameZH') == query_name:
            lat, lng = epsgTransformer.transform(result['y'], result['x'])
            final_res_list.append({
                "station_name_en": stop_info["name_en"],
                "station_name_zh": stop_info["name_tc"],
                "exit": exit_char,
                "lat": lat,
                "lon": lng,
                "barrier_free": barrier_free,
            })
            return True
    return False

async def fetch_mtr_exits(silent=False):
    if not silent:
        print("Fetching MTR exit data from opendata.mtr.com.hk and geodata.gov.hk...")

    final_results = []
    mtr_stops = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, pool=None)) as client:
        # Fetch MTR stations
        try:
            stations_res = await client.get('https://opendata.mtr.com.hk/data/mtr_lines_and_stations.csv')
            stations_res.raise_for_status()
            stations_res.encoding = 'utf-8'
            reader = csv.reader(stations_res.text.strip().split("\n"))
            next(reader, None)
            for entry in reader:
                mtr_stops[entry[3]] = {"name_tc": entry[4], "name_en": entry[5]}
        except httpx.RequestError as e:
            if not silent:
                print(f"Error fetching MTR station list: {e}")
            return []

        # Fetch barrier-free (lift) info
        try:
            facilities_res = await client.get("https://opendata.mtr.com.hk/data/barrier_free_facilities.csv")
            facilities_res.raise_for_status()
            facilities_res.encoding = 'utf-8'
            reader = csv.reader(facilities_res.text.strip().split("\n"))
            for entry in reader:
                if entry[2] == 'Y' and entry[3] != '' and entry[0] in mtr_stops:
                    for exit_code in re.findall(r"[A-Z][0-9]*", entry[3]):
                        mtr_stops[entry[0]][exit_code.strip()] = True
        except httpx.RequestError as e:
            if not silent:
                print(f"Error fetching barrier-free facilities: {e}")

        # Crawl exit geolocations
        for key, stop in mtr_stops.items():
            try:
                geo_query = '港鐵' + stop['name_tc'] + '站進出口'
                geo_res = await client.get("https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q=" + geo_query)
                geo_res.raise_for_status()
                geo_results = geo_res.json()

                for char in string.ascii_uppercase:
                    q = '港鐵' + stop['name_tc'] + '站-' + str(char) + '進出口'
                    check_and_add_result(geo_results, q, stop, char, char in stop, final_results)

                    for i in range(1, 10):
                        exit_code = char + str(i)
                        q = '港鐵' + stop['name_tc'] + '站-' + exit_code + '進出口'
                        check_and_add_result(geo_results, q, stop, exit_code, exit_code in stop, final_results)
            except httpx.RequestError as e:
                if not silent:
                    print(f"Could not fetch geodata for {stop['name_tc']}: {e}")
                continue
            except json.JSONDecodeError:
                 if not silent:
                    print(f"Could not decode geodata for {stop['name_tc']}")
                 continue

    # Deduplicate results
    deduped_results = list({(v['station_name_zh'] + v['exit']): v for v in final_results}.values())

    if not silent:
        print(f"Successfully fetched and processed {len(deduped_results)} unique MTR station exits.")

    return deduped_results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    asyncio.run(fetch_mtr_exits())
