# -*- coding: utf-8 -*-
# MTR Light Rail GTFS Generator (Frequency-Based)

import asyncio
import csv
import json
from pyproj import Transformer
import logging
import httpx
import os
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

# List of Circular Routes
circularRoutes = ("705", "706")

# --- Typical Frequency Data ---
# This data is based on general information about Light Rail services.
# To set a constant 10-minute frequency, you can modify this section.
DEFAULT_FREQUENCIES = {
    "all_day": [
        # Set a single frequency for the entire service day (e.g., 5am to 1:30am next day)
        {"start_time": "05:00:00", "end_time": "25:30:00", "headway_secs": "600"}, # 10 minutes
    ]
}


def getBound(route, bound):
  if route in circularRoutes:
    return "O"
  else:
    return "O" if bound == "1" else "I"

def routeKey(route, bound):
  if route in circularRoutes:
    return f"{route}_O"
  return f"{route}_{bound}"

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

async def get_route_and_stop_data():
    """Fetches route and stop data from MTR and GeoData HK."""
    a_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, pool=None))
    epsg_transformer = Transformer.from_crs('epsg:2326', 'epsg:4326')

    route_list = {}
    stop_list = {}

    # Fetch main route data
    try:
        r = await a_client.get('https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv')
        r.raise_for_status()
    except httpx.RequestError as e:
        logging.error(f"Could not fetch route data: {e}")
        return {}, {}

    reader = csv.reader(r.text.splitlines())
    next(reader, None)  # Skip headers
    routes_data = [row for row in reader if len(row) == 7]

    for route, bound, _, stop_id, chn, eng, seq in routes_data:
        key = routeKey(route, bound)
        light_rail_id = "LR" + stop_id

        if key not in route_list:
            route_list[key] = {
                "gtfsId": None,
                "route": route,
                "bound": getBound(route, bound),
                "orig_tc": chn,
                "orig_en": eng,
                "dest_tc": chn,
                "dest_en": eng,
                "stops": [light_rail_id]
            }
        else:
            route_list[key]["dest_tc"] = chn + (" (循環線)" if route in circularRoutes else "")
            route_list[key]["dest_en"] = eng + (" (Circular)" if route in circularRoutes else "")
            if light_rail_id not in route_list[key]["stops"]:
                if route in circularRoutes and seq != "1.00" and light_rail_id == route_list[key]["stops"][0]:
                    continue
                route_list[key]["stops"].append(light_rail_id)

        if light_rail_id not in stop_list:
            url = f'https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q={chn}輕鐵站'
            try:
                r_geo = await a_client.get(url, headers={'Accept': 'application/json'})
                r_geo.raise_for_status()
                data = r_geo.json()
                if data:
                    lat, lng = epsg_transformer.transform(data[0]['y'], data[0]['x'])
                    stop_list[light_rail_id] = {
                        "stop": light_rail_id,
                        "name_en": eng,
                        "name_tc": chn,
                        "lat": lat,
                        "long": lng
                    }
            except (httpx.RequestError, KeyError, IndexError) as e:
                logging.error(f"Error processing geodata for {chn}: {e}")

    return route_list, stop_list

def generate_gtfs(route_list, stop_list, schedule_data):
    """Generates frequency-based GTFS files."""
    gtfs_dir = 'gtfs_frequency'
    if not os.path.exists(gtfs_dir):
        os.makedirs(gtfs_dir)
        
    # --- NEW: Define a fallback constant for time between stops ---
    FALLBACK_SECONDS_PER_STOP = 180 # 3 minutes

    # 1. agency.txt
    with open(os.path.join(gtfs_dir, 'agency.txt'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['agency_id', 'agency_name', 'agency_url', 'agency_timezone', 'agency_lang'])
        writer.writerow(['MTR', 'MTR Corporation', 'https://www.mtr.com.hk', 'Asia/Hong_Kong', 'zh'])

    # 2. stops.txt
    with open(os.path.join(gtfs_dir, 'stops.txt'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['stop_id', 'stop_name', 'stop_lat', 'stop_lon'])
        for stop_id, stop_info in stop_list.items():
            writer.writerow([stop_id, stop_info['name_tc'], stop_info['lat'], stop_info['long']])

    # 3. routes.txt
    with open(os.path.join(gtfs_dir, 'routes.txt'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type'])
        processed_routes = set()
        for route_info in route_list.values():
            if route_info['route'] not in processed_routes:
                route_name = f"{route_info['route']}"
                writer.writerow([route_info['route'], 'MTR', route_info['route'], route_name, 0]) # 0 for Tram/Light Rail
                processed_routes.add(route_info['route'])

    # 4. calendar.txt
    with open(os.path.join(gtfs_dir, 'calendar.txt'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['service_id', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'start_date', 'end_date'])
        # A single service_id for all days, as frequency is the same.
        writer.writerow(['all_days', '1', '1', '1', '1', '1', '1', '1', '20240101', '20241231'])


    # 5. trips.txt, stop_times.txt, and frequencies.txt
    with open(os.path.join(gtfs_dir, 'trips.txt'), 'w', newline='', encoding='utf-8') as f_trips, \
         open(os.path.join(gtfs_dir, 'stop_times.txt'), 'w', newline='', encoding='utf-8') as f_stop_times, \
         open(os.path.join(gtfs_dir, 'frequencies.txt'), 'w', newline='', encoding='utf-8') as f_frequencies:
        
        trips_writer = csv.writer(f_trips)
        stop_times_writer = csv.writer(f_stop_times)
        frequencies_writer = csv.writer(f_frequencies)
        
        trips_writer.writerow(['route_id', 'service_id', 'trip_id', 'trip_headsign', 'direction_id'])
        stop_times_writer.writerow(['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])
        frequencies_writer.writerow(['trip_id', 'start_time', 'end_time', 'headway_secs', 'exact_times'])

        for key, route_info in route_list.items():
            trip_id = f"{key}_trip"
            direction_id = '0' if route_info['bound'] == 'O' else '1'
            
            trips_writer.writerow([route_info['route'], 'all_days', trip_id, route_info['dest_tc'], direction_id])

            for freq_info in DEFAULT_FREQUENCIES['all_day']:
                 frequencies_writer.writerow([trip_id, freq_info['start_time'], freq_info['end_time'], freq_info['headway_secs'], '0'])

            # --- REVISED: Robust Stop Time Calculation ---
            # Default to the fallback value. Try to calculate a better value if data is available and valid.
            time_per_stop = FALLBACK_SECONDS_PER_STOP
            
            try:
                schedule_for_route = schedule_data.get(route_info['route'])
                if schedule_for_route:
                    direction_key = "circular" if route_info['route'] in circularRoutes else next((k for k in schedule_for_route.keys() if route_info['dest_en'] in k), None)
                    if direction_key and schedule_for_route[direction_key]:
                        stop_times_list = schedule_for_route[direction_key]
                        
                        start_time_str = stop_times_list[0]['first_train']
                        end_time_str = stop_times_list[-1]['first_train']
                        
                        start_dt = datetime.strptime(start_time_str, '%H:%M')
                        end_dt = datetime.strptime(end_time_str, '%H:%M')

                        if end_dt < start_dt: # Handle services past midnight
                            end_dt += timedelta(days=1)
                            
                        total_duration = (end_dt - start_dt).total_seconds()
                        num_stops = len(route_info['stops'])

                        # Validate the calculated average time
                        if num_stops > 1:
                            avg_time_per_stop = total_duration / (num_stops - 1)
                            # Check if calculated time is valid and realistic (e.g., > 0 and < 15 mins)
                            if 0 < avg_time_per_stop < 900:
                                time_per_stop = avg_time_per_stop # Use calculated value
                            else:
                                logging.warning(f"Invalid calculated time ({avg_time_per_stop:.2f}s) for {key}. Using fallback.")
                        else:
                             logging.warning(f"Only one stop for {key}. Using fallback time.")
            except (ValueError, IndexError, KeyError) as e:
                # If any error occurs during parsing, use the fallback and log it.
                logging.warning(f"Could not calculate trip duration for {key} due to '{e}'. Using fallback time.")

            # Write stop_times
            # Use schedule-based time_per_stop if available, otherwise calculate from distance
            use_distance_formula = (time_per_stop == FALLBACK_SECONDS_PER_STOP)
            
            cumulative_seconds = 0
            for i, stop_id in enumerate(route_info['stops']):
                if i > 0:
                    if use_distance_formula:
                        # Use distance-based calculation when schedule data unavailable
                        # Formula: travel_time_seconds = (distance_km * 1.5 / 25) * 3600
                        # where 1.5 is route factor, 25 is speed in km/h
                        prev_stop_id = route_info['stops'][i - 1]
                        if prev_stop_id in stop_list and stop_id in stop_list:
                            prev_stop = stop_list[prev_stop_id]
                            curr_stop = stop_list[stop_id]
                            
                            # Calculate distance between stops
                            distance_km = haversine(
                                prev_stop['long'], prev_stop['lat'],
                                curr_stop['long'], curr_stop['lat']
                            )
                            
                            # Calculate travel time: (distance * 1.5 / 25) * 3600 seconds
                            segment_time = (distance_km * 1.5 / 25) * 3600
                            cumulative_seconds += segment_time
                        else:
                            # Fallback if coordinates not available
                            cumulative_seconds += time_per_stop
                    else:
                        # Use schedule-based average time per stop
                        cumulative_seconds += time_per_stop
                
                elapsed_seconds = int(cumulative_seconds)
                # Format time as H:MM:SS from the start of the trip
                travel_time = str(timedelta(seconds=elapsed_seconds))
                if len(travel_time.split(":")[0]) == 1: # pad hour if needed
                    travel_time = "0" + travel_time

                stop_times_writer.writerow([trip_id, travel_time, travel_time, stop_id, i + 1])

async def main():
    """Main function to run the GTFS generation process."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)

    route_list, stop_list = await get_route_and_stop_data()

    try:
        with open('lightrailschedule.json', 'r', encoding='utf-8') as f:
            schedule_data = json.load(f)
    except FileNotFoundError:
        logging.error("lightrailschedule.json not found. Please run the `parse_schedule.py` script first.")
        return
        
    if route_list and stop_list:
        generate_gtfs(route_list, stop_list, schedule_data)
        logging.info("Frequency-based GTFS files generated successfully in the 'gtfs_frequency' directory.")

if __name__ == '__main__':
    asyncio.run(main())