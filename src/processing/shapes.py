import pandas as pd
from fuzzywuzzy import fuzz
import os
import json
from math import radians, sin, cos, sqrt, atan2
from collections import OrderedDict
import re
from tqdm import tqdm

def lat_long_dist(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    return c * r * 1000

def generate_shapes_from_csdi_files(output_path, silent=False):
    waypoints_dir = "waypoints"
    if not os.path.exists(waypoints_dir):
        if not silent:
            print(f"Directory not found: {waypoints_dir}")
        return False, []

    shape_info_list = []
    files_to_process = [f for f in os.listdir(waypoints_dir) if f.endswith('.json') and f != '0versions.json']
    
    with open(output_path, 'w') as f_out:
        f_out.write("shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n")

        for filename in tqdm(files_to_process, desc="Generating shapes from CSDI files", unit="file", disable=silent):
            filepath = os.path.join(waypoints_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    if not silent:
                        print(f"Warning: Could not decode JSON from {filename}")
                    continue

            if not data.get('features'):
                continue

            feature = data['features'][0]
            properties = feature['properties']
            
            route_id = properties.get('ROUTE_ID')
            route_seq = properties.get('ROUTE_SEQ')
            if route_id is None or route_seq is None:
                continue
            
            direction_char = 'O' if route_seq == 1 else 'I'
            shape_id = f"CSDI-{route_id}-{direction_char}"

            geom_type = feature['geometry']['type']
            coords = feature['geometry']['coordinates']

            if geom_type == 'MultiLineString':
                all_coords = [item for sublist in coords for item in sublist]
            else:
                all_coords = coords

            # Hard cut: remove first and last points
            if len(all_coords) > 2:
                all_coords = all_coords[1:-1]

            dist_traveled = 0
            prev_lat, prev_lon = None, None
            for i, (lon, lat) in enumerate(all_coords):
                if prev_lat is not None:
                    dist_traveled += lat_long_dist(prev_lat, prev_lon, lat, lon)
                f_out.write(f"{shape_id},{lat},{lon},{i+1},{dist_traveled}\n")
                prev_lat, prev_lon = lat, lon

            route_no_match = re.search(r'([a-zA-Z0-9]+)', properties.get('ROUTE_NAMEE', ''))
            route_no = route_no_match.group(1) if route_no_match else None

            shape_info_list.append({
                'shape_id': shape_id,
                'agency_id': properties.get('COMPANY_CODE'),
                'route_short_name': route_no,
                'origin_en': properties.get('ST_STOP_NAMEE'),
                'destination_en': properties.get('ED_STOP_NAMEE'),
            })

    return True, shape_info_list

def match_trips_to_csdi_shapes(trips_df, shape_info, silent=False):
    if not shape_info:
        if not silent:
            print("No shape information available to match.")
        trips_df['shape_id'] = None
        return trips_df

    shape_info_df = pd.DataFrame(shape_info)
    shape_info_df.dropna(subset=['agency_id', 'route_short_name', 'origin_en', 'destination_en'], inplace=True)

    # Handle co-operated routes by splitting the agency_id string and exploding the DataFrame
    shape_info_df['agency_id'] = shape_info_df['agency_id'].str.split('+')
    shape_info_df = shape_info_df.explode('agency_id')

    agency_map = {
        'KMB': 'KMB', 'LWB': 'KMB', 'CTB': 'CTB',
        'NWFB': 'CTB', 'NLB': 'NLB', 'MTRB': 'MTRB'
    }
    shape_info_df['agency_id'] = shape_info_df['agency_id'].map(agency_map)
    
    trips_df_with_agency = trips_df.assign(
        agency_id=trips_df['route_id'].apply(lambda x: x.split('-')[0])
    )

    merged_df = pd.merge(
        trips_df_with_agency,
        shape_info_df,
        on=['agency_id', 'route_short_name'],
        how='left',
        suffixes=['_trip', '_shape']
    )

    def calculate_match_score(row):
        if pd.notna(row['origin_en_trip']) and pd.notna(row['origin_en_shape']) and \
           pd.notna(row['destination_en_trip']) and pd.notna(row['destination_en_shape']):
            origin_score = fuzz.ratio(str(row['origin_en_trip']).lower(), str(row['origin_en_shape']).lower())
            dest_score = fuzz.ratio(str(row['destination_en_trip']).lower(), str(row['destination_en_shape']).lower())
            return (origin_score + dest_score) / 2
        return 0

    merged_df['match_score'] = merged_df.apply(calculate_match_score, axis=1)
    
    merged_df = merged_df.sort_values(by='match_score', ascending=False)
    
    best_matches = merged_df.drop_duplicates(subset=['trip_id'], keep='first')
    
    final_trips = trips_df.merge(best_matches[['trip_id', 'shape_id']], on='trip_id', how='left')
    
    if not silent:
        matched_count = final_trips['shape_id'].notna().sum()
        total_trips = len(final_trips)
        print(f"Matched {matched_count} out of {total_trips} trips to CSDI shapes ({matched_count/total_trips:.2%}).")

    return final_trips
