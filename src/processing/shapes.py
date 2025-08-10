import pandas as pd
import os
import json
from math import radians, sin, cos, sqrt, atan2
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
            
            try:
                base_filename = os.path.splitext(filename)[0]
                gov_gtfs_id, bound = base_filename.split('-')
            except ValueError:
                if not silent:
                    print(f"Warning: Could not parse filename {filename}")
                continue

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
            
            shape_id = f"CSDI-{gov_gtfs_id}-{bound}"

            geom_type = feature['geometry']['type']
            coords = feature['geometry']['coordinates']

            if geom_type == 'MultiLineString':
                all_coords = [item for sublist in coords for item in sublist]
            else:
                all_coords = coords

            dist_traveled = 0
            prev_lat, prev_lon = None, None
            for i, (lon, lat) in enumerate(all_coords):
                if prev_lat is not None:
                    dist_traveled += lat_long_dist(prev_lat, prev_lon, lat, lon)
                f_out.write(f"{shape_id},{lat},{lon},{i+1},{dist_traveled}\n")
                prev_lat, prev_lon = lat, lon

            shape_info_list.append({
                'shape_id': shape_id,
                'gov_route_id': properties.get('ROUTE_ID'),
                'bound': bound
            })

    return True, shape_info_list

def match_trips_to_csdi_shapes(trips_df, shape_info, engine, silent=False):
    if not shape_info:
        if not silent:
            print("No shape information available to match.")
        trips_df['shape_id'] = None
        return trips_df

    shape_info_df = pd.DataFrame(shape_info)

    gov_trips_df = pd.read_sql("SELECT trip_id, route_id, service_id FROM gov_gtfs_trips", engine)
    gov_routes_df = pd.read_sql("SELECT route_id, route_short_name FROM gov_gtfs_routes", engine)

    gov_df = pd.merge(gov_trips_df, gov_routes_df, on='route_id')
    gov_df['service_id'] = gov_df['service_id'].astype(str)
    # Parse direction_id and convert to match our mapping: 0=outbound, 1=inbound
    parsed_direction = gov_df['trip_id'].str.split('-').str[1].astype(int)
    gov_df['direction_id'] = (parsed_direction == 2).astype(int)
    gov_df['bound'] = gov_df['direction_id'].apply(lambda x: 'I' if x == 1 else 'O')

    # Create a mapping from our trip_id to the government's route_id
    trip_to_gov_route_map = trips_df.merge(
        gov_df,
        left_on=['original_service_id', 'route_short_name', 'direction_id'],
        right_on=['service_id', 'route_short_name', 'direction_id'],
        suffixes=('', '_gov')
    )

    # Merge with shape_info_df to get the shape_id
    # Need to match on route_id_gov (from gov data) to gov_route_id (from shapes)
    trips_with_shapes = trip_to_gov_route_map.merge(
        shape_info_df,
        left_on=['route_id_gov', 'bound'],
        right_on=['gov_route_id', 'bound'],
        how='left'
    )

    # Merge back to the original trips_df
    final_trips = trips_df.merge(trips_with_shapes[['trip_id', 'shape_id']], on='trip_id', how='left')

    if not silent:
        matched_count = final_trips['shape_id'].notna().sum()
        total_trips = len(final_trips)
        print(f"Matched {matched_count} out of {total_trips} trips to CSDI shapes ({matched_count/total_trips:.2%}).")

    return final_trips
