import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
import os
import zipfile
from src.processing.stop_unification import unify_stops_by_name_and_distance
from src.processing.stop_times import generate_stop_times_for_agency_optimized as generate_stop_times_for_agency
from src.processing.shapes import generate_shapes_from_csdi_files, match_trips_to_csdi_shapes
from src.processing.utils import get_direction
from src.processing.gtfs_route_matcher import match_operator_routes_to_government_gtfs, match_operator_routes_with_coop_fallback
from datetime import timedelta
import re
from typing import Union
import math

def format_timedelta(td): #timedelta object turning into HH:MM:SS
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def parse_headway_to_avg_secs(headway_str: str) -> Union[int, None]:
    #shitty function to parse headway strings like '2.1', '3.6-5', '2.5 / 4' into average seconds because MTR doesn't have detailed headway data :sneeze:
    if not headway_str or headway_str == '-':
        return None

    # remove any explanatory text in brackets or after special characters
    headway_str = re.sub(r'\s*\(.*\)$', '', headway_str).strip()
    headway_str = re.sub(r'\s*#.*$', '', headway_str).strip()
    headway_str = re.sub(r'\s*~.*$', '', headway_str).strip()

    try:
        if '-' in headway_str:
            low, high = map(float, headway_str.split('-'))
            avg_mins = (low + high) / 2
        elif '/' in headway_str:
            parts = [float(p.strip()) for p in headway_str.split('/')]
            avg_mins = sum(parts) / len(parts)
        else:
            avg_mins = float(headway_str)

        return int(avg_mins * 60)
    except (ValueError, TypeError):
        return None

def resolve_overlapping_frequencies(frequencies_df: pd.DataFrame) -> pd.DataFrame:
    # resolves overlapping frequency intervals for the same trip_id
    if frequencies_df.empty:
        return frequencies_df

    # thank you claude
    # drop exact duplicates on the composite key that causes validation errors
    frequencies_df = frequencies_df.drop_duplicates(subset=['trip_id', 'start_time'], keep='first').copy()

    # convert time strings to timedelta for comparison
    frequencies_df['start_time_td'] = pd.to_timedelta(frequencies_df['start_time'])
    frequencies_df['end_time_td'] = pd.to_timedelta(frequencies_df['end_time'])

    # sort by trip_id and start_time
    frequencies_df = frequencies_df.sort_values(by=['trip_id', 'start_time_td']).reset_index(drop=True)

    resolved_frequencies = []
    for trip_id, group in frequencies_df.groupby('trip_id'):
        if len(group) <= 1:
            resolved_frequencies.append(group)
            continue

        merged_group = []
        current_entry = group.iloc[0].to_dict()

        for i in range(1, len(group)):
            next_entry = group.iloc[i].to_dict()

            # check for overlap
            if next_entry['start_time_td'] < current_entry['end_time_td']:
                # if headway is the same, we can merge by extending the end_time.
                if next_entry['headway_secs'] == current_entry['headway_secs']:
                    current_entry['end_time_td'] = max(current_entry['end_time_td'], next_entry['end_time_td'])
                # if headway is different, we must truncate the current entry to avoid overlap.
                else:
                    current_entry['end_time_td'] = next_entry['start_time_td']
                    # add the truncated current entry to the list, if it's still a valid interval
                    if current_entry['start_time_td'] < current_entry['end_time_td']:
                        merged_group.append(current_entry)
                    current_entry = next_entry
            # check for continuous intervals with same headway to merge
            elif next_entry['start_time_td'] == current_entry['end_time_td'] and next_entry['headway_secs'] == current_entry['headway_secs']:
                 current_entry['end_time_td'] = next_entry['end_time_td']
            # no overlap, so we finalize current_entry
            else:
                merged_group.append(current_entry)
                current_entry = next_entry

        # add the very last entry ^-^
        merged_group.append(current_entry)
        if merged_group:
            resolved_frequencies.append(pd.DataFrame(merged_group))

    if not resolved_frequencies:
        return pd.DataFrame(columns=frequencies_df.columns)

    final_df = pd.concat(resolved_frequencies, ignore_index=True)
    # convert timedelta back to string format HH:MM:SS
    final_df['start_time'] = final_df['start_time_td'].apply(lambda td: format_timedelta(td))
    final_df['end_time'] = final_df['end_time_td'].apply(lambda td: format_timedelta(td))

    return final_df.drop(columns=['start_time_td', 'end_time_td'])

def export_unified_feed(engine: Engine, output_dir: str, journey_time_data: dict, mtr_headway_data: dict, osm_data: dict, silent: bool = False, no_regenerate_shapes: bool = False):
    print("==========================================")
    print("ENTERING export_unified_feed")
    print("==========================================")
    if not silent:
        print("--- Starting Unified GTFS Export Process ---")

    final_output_dir = os.path.join(output_dir, "unified_feed")
    os.makedirs(final_output_dir, exist_ok=True)

    if not silent:
        print("Building agency.txt...")
    agencies = [
        {'agency_id': 'KMB', 'agency_name': 'Kowloon Motor Bus', 'agency_url': 'https://kmb.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'CTB', 'agency_name': 'Citybus', 'agency_url': 'https://www.citybus.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'MTRB', 'agency_name': 'MTR Bus', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'GMB', 'agency_name': 'Green Minibus', 'agency_url': 'https://td.gov.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'NLB', 'agency_name': 'New Lantao Bus', 'agency_url': 'https://www.nlb.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'MTRR', 'agency_name': 'MTR Rail', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'LR', 'agency_name': 'Light Rail', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'AE', 'agency_name': 'Airport Express', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'PT', 'agency_name': 'Peak Tram', 'agency_url': 'https://thepeak.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'TRAM', 'agency_name': 'Tramways', 'agency_url': 'https://www.hktramways.com', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
    ]
    # we have zh-hant translations, check the end
    agency_df = pd.DataFrame(agencies)
    agency_df.to_csv(os.path.join(final_output_dir, 'agency.txt'), index=False)

    if not silent:
        print("Building stops.txt...")

    # KMB
    kmb_stops_gdf = gpd.read_postgis("SELECT * FROM kmb_stops", engine, geom_col='geometry')
    kmb_stops_gdf['stop_id'] = 'KMB-' + kmb_stops_gdf['stop'].astype(str)
    kmb_stops_gdf['stop_name'] = (
        kmb_stops_gdf['name_en']
        .str.replace(r'\s*\([A-Za-z0-9]{5}\)', '', regex=True)
        .str.replace(r'\s*-\s*', ' - ', regex=True)
        .str.replace(r'([^\s])(\([A-Za-z0-9]+\))', r'\1 \2', regex=True)
    )
    kmb_stops_gdf, kmb_duplicates_map = (kmb_stops_gdf, {})
    kmb_stops_gdf['stop_lat'] = kmb_stops_gdf.geometry.y
    kmb_stops_gdf['stop_lon'] = kmb_stops_gdf.geometry.x
    kmb_stops_final = kmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # Citybus
    ctb_stops_gdf = gpd.read_postgis("SELECT * FROM citybus_stops", engine, geom_col='geometry')
    ctb_stops_gdf['stop_id'] = 'CTB-' + ctb_stops_gdf['stop'].astype(str)
    ctb_stops_gdf['stop_name'] = ctb_stops_gdf['name_en']
    ctb_duplicates_map = {}
    ctb_stops_gdf['stop_lat'] = ctb_stops_gdf.geometry.y
    ctb_stops_gdf['stop_lon'] = ctb_stops_gdf.geometry.x
    ctb_stops_final = ctb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # GMB
    gmb_stops_gdf = gpd.read_postgis("SELECT * FROM gmb_stops", engine, geom_col='geometry')
    gmb_stops_gdf['stop_id'] = 'GMB-' + gmb_stops_gdf['stop_id'].astype(str)
    gmb_stops_gdf['stop_name'] = gmb_stops_gdf['stop_name_en']
    gmb_stops_gdf, gmb_duplicates_map = (gmb_stops_gdf, {})
    gmb_stops_gdf['stop_lat'] = gmb_stops_gdf.geometry.y
    gmb_stops_gdf['stop_lon'] = gmb_stops_gdf.geometry.x
    gmb_stops_final = gmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    #TODO: Fix GMB trips for some reason it only includes the first one it finds on the government GTFS

    # MTR Bus
    mtrbus_stops_gdf = gpd.read_postgis("SELECT * FROM mtrbus_stops", engine, geom_col='geometry')
    mtrbus_stops_gdf['stop_id'] = 'MTRB-' + mtrbus_stops_gdf['stop_id'].astype(str)
    mtrbus_stops_gdf['stop_name'] = mtrbus_stops_gdf['name_en']
    mtrbus_stops_gdf, mtrbus_duplicates_map = (mtrbus_stops_gdf, {})
    mtrbus_stops_gdf['stop_lat'] = mtrbus_stops_gdf.geometry.y
    mtrbus_stops_gdf['stop_lon'] = mtrbus_stops_gdf.geometry.x
    mtrbus_stops_final = mtrbus_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # NLB
    nlb_stops_gdf = gpd.read_postgis("SELECT * FROM nlb_stops", engine, geom_col='geometry')
    nlb_stops_gdf['stop_id'] = 'NLB-' + nlb_stops_gdf['stopId'].astype(str)
    nlb_stops_gdf['stop_name'] = nlb_stops_gdf['stopName_e']
    nlb_stops_gdf, nlb_duplicates_map = (nlb_stops_gdf, {})
    nlb_stops_gdf['stop_lat'] = nlb_stops_gdf.geometry.y
    nlb_stops_gdf['stop_lon'] = nlb_stops_gdf.geometry.x
    nlb_stops_final = nlb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # MTR Rails
    mtr_stations_gdf = gpd.read_postgis("SELECT * FROM mtr_lines_and_stations", engine, geom_col='geometry')
    # use Station Code (e.g. WHA, HOM) so journey_time_data can map
    mtr_stops_df = mtr_stations_gdf[['Station Code', 'English Name', 'geometry']].drop_duplicates(subset=['Station Code'])
    mtr_stops_df.rename(columns={'Station Code': 'station_code', 'English Name': 'stop_name'}, inplace=True)
    mtr_stops_df['stop_id'] = 'MTR-' + mtr_stops_df['station_code'].astype(str)
    mtr_stops_df['stop_lat'] = mtr_stops_df.geometry.y
    mtr_stops_df['stop_lon'] = mtr_stops_df.geometry.x
    mtr_stops_df['location_type'] = 1 
    mtr_stops_df['parent_station'] = None

    # MTR Entrances
    # thank you hkbus
    mtr_exits_gdf = gpd.read_postgis("SELECT * FROM mtr_exits", engine, geom_col='geometry')
    if not mtr_exits_gdf.empty:
        mtr_exits_gdf['stop_id'] = 'MTR-ENTRANCE-' + mtr_exits_gdf['station_name_en'] + '-' + mtr_exits_gdf['exit']
        mtr_exits_gdf['stop_name'] = mtr_exits_gdf['exit']
        mtr_exits_gdf['stop_lat'] = mtr_exits_gdf.geometry.y
        mtr_exits_gdf['stop_lon'] = mtr_exits_gdf.geometry.x
        mtr_exits_gdf['location_type'] = 2

        station_name_to_id_map = mtr_stops_df.set_index('stop_name')['stop_id'].to_dict()
        mtr_exits_gdf['parent_station'] = mtr_exits_gdf['station_name_en'].map(station_name_to_id_map)

        mtr_entrances_df = mtr_exits_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']]
    else:
        mtr_entrances_df = pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station'])

    # Light Rail
    lr_stops_gdf = gpd.read_postgis("SELECT * FROM light_rail_stops", engine, geom_col='geometry')
    lr_stops_gdf.rename(columns={'name_en': 'stop_name'}, inplace=True)
    lr_stops_gdf['stop_lat'] = lr_stops_gdf.geometry.y
    lr_stops_gdf['stop_lon'] = lr_stops_gdf.geometry.x
    lr_stops_gdf['location_type'] = 0
    lr_stops_gdf['parent_station'] = None

    # Combine all agencies
    all_stops_df = pd.concat([
        kmb_stops_final,
        ctb_stops_final,
        gmb_stops_final,
        mtrbus_stops_final,
        nlb_stops_final,
        mtr_stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']],
        mtr_entrances_df,
        lr_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']]
    ], ignore_index=True)

    gtfs_stops_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']
    for col in gtfs_stops_cols:
        if col not in all_stops_df.columns:
            all_stops_df[col] = None

    # Ensure location_type is an integer, defaulting missing values to 0 (stop)
    all_stops_df['location_type'] = all_stops_df['location_type'].fillna(0).astype(int)

    all_stops_output_df = all_stops_df[gtfs_stops_cols]
    all_stops_output_df.to_csv(os.path.join(final_output_dir, 'stops.txt'), index=False)

    if not silent:
        print(f"Generated stops.txt with {len(all_stops_output_df)} total stops.")

    # --- 3. Build `routes.txt`, `trips.txt`, `stop_times.txt` ---
    gov_routes_df = pd.read_sql("SELECT * FROM gov_gtfs_routes", engine)
    gov_trips_df = pd.read_sql("SELECT * FROM gov_gtfs_trips", engine)
    gov_frequencies_df = pd.read_sql("SELECT * FROM gov_gtfs_frequencies", engine)
    
    try:
        gov_stops_df = pd.read_sql("SELECT * FROM gov_gtfs_stops", engine)
        gov_stop_times_df = pd.read_sql("SELECT * FROM gov_gtfs_stop_times", engine)
        if not silent:
            print(f"Loaded government stops ({len(gov_stops_df)}) and stop_times ({len(gov_stop_times_df)}) for stop-sequence matching.")
    except Exception as e:
        if not silent:
            print(f"Warning: Could not load government stops/stop_times data: {e}")
        gov_stops_df = pd.DataFrame()
        gov_stop_times_df = pd.DataFrame()
    
    try:
        parsed_direction = gov_trips_df['trip_id'].str.split('-').str[1].astype(int)
        # Government uses: 1=outbound, 2=inbound like wtf
        # We want: 0=outbound, 1=inbound (to match KMB I/O mapping)
        gov_trips_df['direction_id'] = (parsed_direction == 2).astype(int)
        if not silent:
            print("Successfully parsed 'direction_id' from government trip_id.")
    except (IndexError, ValueError, TypeError):
        if not silent:
            print("Warning: Could not parse 'direction_id' from government trip_id.")
        gov_trips_df['direction_id'] = -1

    # standardize data types before merge
    gov_trips_df['service_id'] = gov_trips_df['service_id'].astype(str)
    gov_routes_df['route_short_name'] = gov_routes_df['route_short_name'].astype(str)

    gov_trips_with_route_info = gov_trips_df.merge(
        gov_routes_df[['route_id', 'route_short_name', 'agency_id', 'route_long_name']], on='route_id'
    )

    # -- KMB --
    if not silent:
        print("Processing KMB routes, trips, and stop_times...")
    kmb_routes_df = pd.read_sql("SELECT * FROM kmb_routes", engine)
    kmb_routes_df['agency_id'] = 'KMB'
    kmb_routes_df[['route', 'bound', 'service_type']] = kmb_routes_df['unique_route_id'].str.split('_', expand=True)
    kmb_routes_df['direction_id'] = kmb_routes_df['bound'].map({'O': 0, 'I': 1}).fillna(-1).astype(int)

    kmb_stoptimes_df = pd.read_sql("SELECT * FROM kmb_stop_sequences", engine)
    kmb_stoptimes_df[['route', 'bound', 'service_type']] = kmb_stoptimes_df['unique_route_id'].str.split('_', expand=True)
    kmb_stoptimes_df.dropna(subset=['service_type'], inplace=True)
    kmb_stoptimes_df = kmb_stoptimes_df.drop_duplicates(subset=['unique_route_id', 'seq'])
    kmb_stoptimes_df['stop_id'] = 'KMB-' + kmb_stoptimes_df['stop'].astype(str)
    kmb_stoptimes_df['stop_id'] = kmb_stoptimes_df['stop_id'].replace(kmb_duplicates_map)

    final_kmb_routes_list = []
    for route_num, group in kmb_routes_df.groupby('route'):
        first_outbound = group[group['bound'] == 'O'].iloc[0] if not group[group['bound'] == 'O'].empty else None
        first_inbound = group[group['bound'] == 'I'].iloc[0] if not group[group['bound'] == 'I'].empty else None

        if first_outbound is not None and first_inbound is not None:
            route_long_name = f"{first_outbound['orig_en']} - {first_inbound['orig_en']}"
        elif first_outbound is not None:
            route_long_name = f"{first_outbound['orig_en']} - {first_outbound['dest_en']}"
        elif first_inbound is not None:
            route_long_name = f"{first_inbound['orig_en']} - {first_inbound['dest_en']}"
        else:
            route_long_name = f"{group.iloc[0]['orig_en']} - {group.iloc[0]['dest_en']}" if not group.empty else ""

        final_kmb_routes_list.append({
            'route_id': f"KMB-{route_num}",
            'agency_id': 'KMB',
            'route_short_name': route_num,
            'route_long_name': route_long_name,
            'route_type': 3
        })
    final_kmb_routes = pd.DataFrame(final_kmb_routes_list)

    # Use enhanced matching for KMB routes
    if not silent:
        print("Using enhanced stop-count-based matching for KMB routes...")
    
    kmb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="KMB",
        debug=not silent
    )
    
    kmb_trips_list = []
    if kmb_route_matches:
        for route_key, route_matches in kmb_route_matches.items():
            route_short_name, bound, service_type = route_key.split('-')
            direction_id = 1 if bound == 'I' else 0
            
            # Find the corresponding database route info
            matching_db_routes = kmb_routes_df[
                (kmb_routes_df['route'] == route_short_name) & 
                (kmb_routes_df['bound'] == bound) &
                (kmb_routes_df['service_type'] == service_type)
            ]
            
            if not matching_db_routes.empty:
                route_info = matching_db_routes.iloc[0]
                for match in route_matches:
                    kmb_trips_list.append({
                        'route_id': f"KMB-{route_short_name}",
                        'service_id': f"KMB-{route_short_name}-{bound}-{match['gov_service_id']}",
                        'trip_id': f"KMB-{route_short_name}-{bound}-{service_type}-{match['gov_service_id']}",
                        'direction_id': direction_id,
                        'bound': bound,
                        'route_short_name': route_short_name,
                        'route_long_name': f"{route_info.get('orig_en', '')} - {route_info.get('dest_en', '')}",
                        'original_service_id': match['gov_service_id'],
                        'gov_route_id': match['gov_route_id'],  # Add this for proper shape matching
                        'service_type': service_type,
                        'origin_en': route_info.get('orig_en', ''),
                        'destination_en': route_info.get('dest_en', ''),
                        'agency_id': 'KMB'
                    })
    
    # Create DataFrame with required columns even if empty
    if kmb_trips_list:
        kmb_trips_df = pd.DataFrame(kmb_trips_list)
    else:
        if not silent:
            print("Warning: Enhanced KMB matching failed, creating empty DataFrame")
        kmb_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'bound', 'route_short_name', 
            'route_long_name', 'original_service_id', 'gov_route_id', 'service_type', 'origin_en', 'destination_en'
        ])
    
    # Ensure required columns exist for downstream processing
    if 'original_service_id' not in kmb_trips_df.columns:
        kmb_trips_df['original_service_id'] = 'DEFAULT'
    if 'route_short_name' not in kmb_trips_df.columns:
        kmb_trips_df['route_short_name'] = ''
    kmb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)

    # -- Citybus --
    if not silent:
        print("Processing Citybus routes, trips, and stop_times...")

    # Get co-op routes to exclude from Citybus processing
    co_op_routes_df = pd.read_sql("SELECT DISTINCT route_short_name FROM gov_gtfs_routes WHERE agency_id = 'KMB+CTB'", engine)
    co_op_routes_to_exclude = co_op_routes_df['route_short_name'].tolist()

    ctb_routes_df = pd.read_sql("SELECT * FROM citybus_routes", engine)

    # Exclude co-op routes
    ctb_routes_df = ctb_routes_df[~ctb_routes_df['route'].isin(co_op_routes_to_exclude)]

    ctb_routes_df['route_id'] = 'CTB-' + ctb_routes_df['route']
    ctb_routes_df['agency_id'] = 'CTB'
    ctb_routes_df['route_short_name'] = ctb_routes_df['route']
    ctb_routes_df['route_long_name'] = ctb_routes_df['orig_en'] + ' - ' + ctb_routes_df['dest_en']
    ctb_routes_df['route_type'] = 3
    ctb_routes_df['dir'] = ctb_routes_df['unique_route_id'].str.split('-').str[-1]
    final_ctb_routes_list = []
    for route_num, group in ctb_routes_df.groupby('route'):
        first_outbound = group[group['dir'] == 'outbound'].iloc[0] if not group[group['dir'] == 'outbound'].empty else group.iloc[0]
        first_inbound = group[group['dir'] == 'inbound'].iloc[0] if not group[group['dir'] == 'inbound'].empty else group.iloc[0]
        final_ctb_routes_list.append({
            'route_id': f"CTB-{route_num}",
            'agency_id': 'CTB',
            'route_short_name': route_num,
            'route_long_name': f"{first_outbound['orig_en']} - {first_inbound['orig_en']}",
            'route_type': 3
        })
    final_ctb_routes = pd.DataFrame(final_ctb_routes_list)

    # Use enhanced stop-count-based matching for CTB routes with co-op route handling
    if not silent:
        print("Using enhanced stop-count-based matching for CTB routes with co-op handling...")
    
    ctb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="CTB",
        debug=not silent
    )
    if not silent:
        print(f"Found stop-count matches for {len(ctb_route_matches)} CTB routes.")

    ctb_trips_list = []
    
    # Use enhanced matching results to create trips
    if ctb_route_matches:
        for route_short_name, gov_route_id in ctb_route_matches.items():
            # Find the corresponding database route info
            matching_db_routes = ctb_routes_df[ctb_routes_df['route'] == route_short_name]
            
            if not matching_db_routes.empty:
                # Create trips for both directions
                for direction in ['inbound', 'outbound']:
                    direction_id = 1 if direction == 'inbound' else 0
                    bound = 'I' if direction == 'inbound' else 'O'
                    
                    route = matching_db_routes[matching_db_routes['dir'] == direction]
                    if not route.empty:
                        route = route.iloc[0]
                        route_long_name = f"{route['orig_en']} - {route['dest_en']}"
                        
                        # Get all service ids for the government route
                        gov_trips_with_route_info['route_id'] = gov_trips_with_route_info['route_id'].astype(str)
                        gov_trips = gov_trips_with_route_info[gov_trips_with_route_info['route_id'] == gov_route_id]
                        service_ids = gov_trips['service_id'].unique()

                        for service_id in service_ids:
                            ctb_trips_list.append({
                                'route_id': f"CTB-{route_short_name}",
                                'service_id': f"CTB-{route_short_name}-{direction}-{service_id}",
                                'trip_id': f"CTB-{route_short_name}-{direction}-{service_id}",
                                'direction_id': direction_id,
                                'bound': bound,
                                'route_short_name': route_short_name,
                                'route_long_name': route_long_name,
                                'original_service_id': service_id,
                                'unique_route_id': route['unique_route_id'],
                                'origin_en': route['orig_en'],
                                'destination_en': route['dest_en'],
                                'gov_route_id': gov_route_id,
                                'agency_id': 'CTB'
                            })
    
    ctb_trips_df = pd.DataFrame(ctb_trips_list)
    if ctb_trips_df.empty:
        ctb_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'bound', 'route_short_name', 
            'route_long_name', 'original_service_id', 'unique_route_id', 'origin_en', 'destination_en', 
            'gov_route_id', 'agency_id'
        ])
    # Remove duplicate trips that might be created from multiple citybus route entries
    ctb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)
    
    # Load CTB stop sequences with special handling for circular routes
    if not silent:
        print("Loading CTB stop sequences with circular route handling...")
    
    # First, detect circular routes using the same logic as our matcher
    circular_routes_query = """
        SELECT DISTINCT gr.route_short_name as route
        FROM gov_gtfs_routes gr
        JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
        WHERE gr.agency_id = 'CTB'
        GROUP BY gr.route_short_name
        HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
    """
    circular_routes_df = pd.read_sql(circular_routes_query, engine)
    circular_routes = circular_routes_df['route'].tolist() if not circular_routes_df.empty else []
    
    if not silent:
        print(f"Detected {len(circular_routes)} circular CTB routes: {circular_routes}")
        print(f"Is 22M circular? {'22M' in circular_routes}")
    
    # Debug: Check what stop sequences exist for 22M
    debug_22m_query = """
        SELECT 
            cr.route,
            cr.direction,
            COUNT(css.sequence) as stop_count
        FROM citybus_routes cr
        JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
        WHERE cr.route = '22M'
        GROUP BY cr.route, cr.direction
        ORDER BY cr.route, cr.direction
    """
    debug_22m_df = pd.read_sql(debug_22m_query, engine)
    if not silent:
        print("22M stop counts by direction:")
        for _, row in debug_22m_df.iterrows():
            print(f"  {row['route']} {row['direction']}: {row['stop_count']} stops")
    
    # Load stop sequences with circular route merging
    if circular_routes:
        # For circular routes, merge outbound and inbound sequences
        circular_routes_list = "','".join(circular_routes)
        ctb_stop_sequences_query = f"""
            WITH outbound_sequences AS (
                -- Get outbound sequences for circular routes
                SELECT 
                    cr.route,
                    cr.unique_route_id,
                    css.stop_id,
                    css.sequence,
                    'outbound' as original_direction
                FROM citybus_routes cr
                JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
                WHERE cr.route IN ('{circular_routes_list}') 
                    AND cr.direction = 'outbound'
            ),
            inbound_sequences AS (
                -- Get inbound sequences for circular routes
                SELECT 
                    cr.route,
                    cr.unique_route_id,
                    css.stop_id,
                    css.sequence,
                    'inbound' as original_direction
                FROM citybus_routes cr
                JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
                WHERE cr.route IN ('{circular_routes_list}') 
                    AND cr.direction = 'inbound'
            ),
            circular_merged AS (
                -- Create proper circular route: skip inbound stops until we find non-overlapping pattern
                SELECT 
                    route,
                    'outbound' as direction,
                    unique_route_id,
                    stop_id,
                    sequence as merged_sequence,
                    'from_outbound' as source
                FROM outbound_sequences
                
                UNION ALL
                
                -- Add inbound stops starting from the first stop that doesn't exist in outbound
                -- This creates proper circular: [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,b,a]
                SELECT 
                    ins.route,
                    'outbound' as direction,
                    ins.unique_route_id,
                    ins.stop_id,
                    (SELECT MAX(sequence) FROM outbound_sequences os WHERE os.route = ins.route) + 
                    ROW_NUMBER() OVER (PARTITION BY ins.route ORDER BY ins.sequence) as merged_sequence,
                    'from_inbound' as source
                FROM inbound_sequences ins
                WHERE ins.sequence >= (
                    -- Find the first inbound sequence where the stop doesn't exist in outbound
                    SELECT COALESCE(MIN(i2.sequence), 1)
                    FROM inbound_sequences i2
                    WHERE i2.route = ins.route
                    AND NOT EXISTS (
                        SELECT 1 FROM outbound_sequences obs
                        WHERE obs.route = i2.route AND obs.stop_id = i2.stop_id
                    )
                )
            ),
            regular_routes AS (
                -- For regular routes, keep normal outbound/inbound separation
                SELECT 
                    cr.route,
                    cr.direction,
                    cr.unique_route_id,
                    css.stop_id,
                    css.sequence as merged_sequence,
                    'regular' as source
                FROM citybus_routes cr
                JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
                WHERE cr.route NOT IN ('{circular_routes_list}')
            )
            SELECT 
                route,
                direction,
                unique_route_id,
                stop_id,
                merged_sequence as sequence,
                source
            FROM circular_merged
            
            UNION ALL
            
            SELECT 
                route,
                direction,
                unique_route_id,
                stop_id,
                merged_sequence as sequence,
                source
            FROM regular_routes
            
            ORDER BY route, direction, sequence
        """
        ctb_stoptimes_df = pd.read_sql(ctb_stop_sequences_query, engine)
        
        if not silent:
            route_22m_stops = ctb_stoptimes_df[ctb_stoptimes_df['route'] == '22M']
            if not route_22m_stops.empty:
                print(f"22M after intelligent circular merging: {len(route_22m_stops)} stops")
                print(f"22M direction: {route_22m_stops['direction'].unique()}")
                print(f"22M sequence range: {route_22m_stops['sequence'].min()} to {route_22m_stops['sequence'].max()}")
                
                # Show merge source breakdown
                if 'source' in route_22m_stops.columns:
                    source_counts = route_22m_stops['source'].value_counts()
                    print(f"22M merge sources: {dict(source_counts)}")
                
                # Show first few and last few stops to verify merging
                sorted_stops = route_22m_stops.sort_values('sequence')
                print("First 5 stops:", sorted_stops.head(5)['sequence'].tolist())
                print("Last 5 stops:", sorted_stops.tail(5)['sequence'].tolist())
                
                # Show total unique stops vs government GTFS expectation
                print(f"22M total stops: {len(route_22m_stops)} (expected ~39 from gov GTFS)")
            else:
                print("No 22M stops found after merging!")
                
            # Also check if we have any circular routes processed
            circular_processed = ctb_stoptimes_df[ctb_stoptimes_df['route'].isin(circular_routes)]
            print(f"Total stops for all circular routes: {len(circular_processed)}")
            
            # Sample a few circular routes to check merging effectiveness
            sample_routes = circular_routes[:5] if len(circular_routes) >= 5 else circular_routes
            for route in sample_routes:
                route_stops = ctb_stoptimes_df[ctb_stoptimes_df['route'] == route]
                if len(route_stops) > 0:
                    if 'source' in route_stops.columns:
                        source_counts = route_stops['source'].value_counts()
                        print(f"  {route}: {len(route_stops)} stops {dict(source_counts)}")
                    else:
                        print(f"  {route}: {len(route_stops)} stops")
    else:
        if not silent:
            print("No circular routes detected, using normal loading...")
        # No circular routes detected, use normal loading
        ctb_stoptimes_df = pd.read_sql("SELECT * FROM citybus_stop_sequences", engine)
    ctb_stoptimes_df['stop_id'] = 'CTB-' + ctb_stoptimes_df['stop_id'].astype(str)
    ctb_stoptimes_df['stop_id'] = ctb_stoptimes_df['stop_id'].replace(ctb_duplicates_map)

    # -- GMB --
    if not silent:
        print("Processing GMB routes, trips, and stop_times...")

    gmb_routes_base_df = pd.read_sql("SELECT * FROM gmb_routes", engine)
    gmb_stoptimes_df = pd.read_sql("SELECT * FROM gmb_stop_sequences", engine)

    gmb_routes_base_df['agency_id'] = 'GMB'
    gmb_routes_base_df['route_type'] = 3
    gmb_routes_base_df['route_id'] = 'GMB-' + gmb_routes_base_df['region'] + '-' + gmb_routes_base_df['route_code']
    gmb_routes_base_df['route_short_name'] = gmb_routes_base_df['region'] + '-' + gmb_routes_base_df['route_code']
    gmb_routes_base_df['route_long_name'] = gmb_routes_base_df['region'] + ' - ' + gmb_routes_base_df['route_code']
    final_gmb_routes = gmb_routes_base_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()

    # Use enhanced matching for GMB routes
    if not silent:
        print("Using enhanced location-based matching for GMB routes...")
    
    gmb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="GMB",
        debug=not silent
    )
    
    gmb_trips_list = []
    if gmb_route_matches:
        for route_key, route_matches in gmb_route_matches.items():
            # route_key format: "HKI-1-O-1" (outbound) or "HKI-1-I-1" (inbound)
            route_parts = route_key.split('-')
            region = route_parts[0]
            route_code = route_parts[1]
            bound = route_parts[2]  # 'O' for outbound, 'I' for inbound
            # Determine which route_seq corresponds to this direction
            if bound == 'O':
                target_route_seq = 1  # Outbound
            else:
                target_route_seq = 2  # Inbound
            actual_route_seqs = gmb_stoptimes_df[
                (gmb_stoptimes_df['region'] == region) &
                (gmb_stoptimes_df['route_code'] == route_code) &
                (gmb_stoptimes_df['route_seq'] == target_route_seq)
            ]['route_seq'].unique()
            for match in route_matches:
                for route_seq in actual_route_seqs:
                    gmb_trips_list.append({
                        'route_id': f"GMB-{region}-{route_code}",
                        'service_id': f"GMB-{region}-{route_code}-{match['gov_service_id']}",
                        'trip_id': f"GMB-{region}-{route_code}-{match['gov_service_id']}-{route_seq}",
                        'direction_id': int(route_seq) - 1,
                        'route_short_name': f"{region}-{route_code}",
                        'route_seq': int(route_seq),
                        'route_code': route_code,
                        'region': region,  # ADDED region for disambiguation
                        'agency_id': 'GMB',
                        'original_service_id': match['gov_service_id'],
                        'gov_route_id': match['gov_route_id']
                    })
    
    # Create DataFrame with required columns even if empty
    if gmb_trips_list:
        gmb_trips_df = pd.DataFrame(gmb_trips_list)
    else:
        if not silent:
            print("Warning: Enhanced GMB matching failed, falling back to default logic")
        # Fallback to old logic
        gmb_trips_source = gmb_stoptimes_df[['route_code', 'route_seq', 'region']].drop_duplicates().copy()
        gmb_trips_source = pd.merge(gmb_trips_source, gmb_routes_base_df[['route_code', 'region', 'route_long_name', 'agency_id']].drop_duplicates(), on=['route_code', 'region'], how='left')
        gmb_trips_source['orig_en'] = gmb_trips_source['route_long_name'].str.split(' - ').str[0]
        gmb_trips_source['dest_en'] = gmb_trips_source['route_long_name'].str.split(' - ').str[1]
        gmb_trips_source['route_seq'] = pd.to_numeric(gmb_trips_source['route_seq'])
        gmb_trips_source['route_id'] = 'GMB-' + gmb_trips_source['region'] + '-' + gmb_trips_source['route_code']
        gmb_trips_source['direction_id'] = gmb_trips_source['route_seq'] - 1
        gmb_trips_source['service_id'] = 'GMB_DEFAULT_SERVICE'
        gmb_trips_source['trip_id'] = gmb_trips_source['route_id'] + '-' + gmb_trips_source['route_seq'].astype(str)
        gmb_trips_df = gmb_trips_source[['route_id', 'service_id', 'trip_id', 'direction_id', 'route_seq', 'route_code', 'orig_en', 'dest_en', 'agency_id']].copy()
        gmb_trips_df.rename(columns={'orig_en': 'origin_en', 'dest_en': 'destination_en'}, inplace=True)
        gmb_trips_df['original_service_id'] = 'GMB_DEFAULT_SERVICE'
        gmb_trips_df['route_short_name'] = gmb_trips_df['route_id'].str.replace('GMB-', '')

    gmb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)

    gmb_stoptimes_df['route_id'] = 'GMB-' + gmb_stoptimes_df['region'] + '-' + gmb_stoptimes_df['route_code']
    gmb_stoptimes_df['route_seq'] = pd.to_numeric(gmb_stoptimes_df['route_seq'])
    gmb_stoptimes_df = gmb_stoptimes_df.merge(
        gmb_trips_df[['route_id', 'route_seq', 'trip_id']],
        on=['route_id', 'route_seq']
    )
    gmb_stoptimes_df['stop_id'] = 'GMB-' + gmb_stoptimes_df['stop_id'].astype(str)

    # Defensive dedupe for GMB stop_times
    if 'sequence' in gmb_stoptimes_df.columns:
        _before = len(gmb_stoptimes_df)
        gmb_stoptimes_df = (
            gmb_stoptimes_df
            .sort_values(['trip_id', 'sequence', 'stop_id'])
            .drop_duplicates(subset=['trip_id', 'sequence'], keep='first')
        )
        if not silent:
            print(f"GMB stop_times deduped (export): {_before}->{len(gmb_stoptimes_df)}")
    else:
        if not silent:
            print("Warning: 'sequence' column missing in gmb_stoptimes_df; skipping GMB dedupe")

    # -- MTR Bus --
    if not silent:
        print("Processing MTR Bus routes, trips, and stop_times...")
    mtrbus_routes_df = pd.read_sql("SELECT * FROM mtrbus_routes", engine)
    mtrbus_routes_df['agency_id'] = 'MTRB'
    mtrbus_routes_df['route_type'] = 3
    mtrbus_routes_df['route_long_name'] = mtrbus_routes_df['route_name_eng']
    mtrbus_routes_df['route_short_name'] = mtrbus_routes_df['route_id']
    mtrbus_routes_df['route_id'] = 'MTRB-' + mtrbus_routes_df['route_id']
    final_mtrbus_routes = mtrbus_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].drop_duplicates(subset=['route_id']).copy()
    # Use enhanced matching for MTRB routes  
    if not silent:
        print("Using enhanced stop-count-based matching for MTRB routes...")
    
    mtrb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="MTRB", 
        debug=not silent
    )
    
    mtrbus_trips_list = []
    if mtrb_route_matches:
        for route_key, route_matches in mtrb_route_matches.items():
            route_short_name, bound, service_type = route_key.split('-')
            direction_id = 0 if bound == 'O' else 1
            
            # Find the corresponding database route info
            matching_db_routes = mtrbus_routes_df[mtrbus_routes_df['route_short_name'] == route_short_name]
            
            if not matching_db_routes.empty:
                route_info = matching_db_routes.iloc[0]
                
                # Extract origin and destination from route name
                origin_en = ''
                destination_en = ''
                if isinstance(route_info['route_name_eng'], str) and ' to ' in route_info['route_name_eng']:
                    parts = route_info['route_name_eng'].split(' to ')
                    origin_en = parts[0]
                    destination_en = parts[1]
                
                for match in route_matches:
                    mtrbus_trips_list.append({
                        'route_id': f"MTRB-{route_short_name}",
                        'service_id': f"MTRB-{route_short_name}-{bound}-{match['gov_service_id']}",
                        'trip_id': f"MTRB-{route_short_name}-{bound}-{match['gov_service_id']}",
                        'direction_id': direction_id,
                        'route_short_name': route_short_name,
                        'original_service_id': match['gov_service_id'],
                        'origin_en': origin_en,
                        'destination_en': destination_en,
                        'agency_id': 'MTRB'
                    })
    
    # Create DataFrame with required columns even if empty
    if mtrbus_trips_list:
        mtrbus_trips_df = pd.DataFrame(mtrbus_trips_list)
    else:
        if not silent:
            print("Warning: Enhanced MTRB matching failed, creating empty DataFrame")
        mtrbus_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'route_short_name', 
            'original_service_id', 'origin_en', 'destination_en'
        ])
    mtrbus_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)
    
    # Ensure required columns exist for downstream processing
    if 'original_service_id' not in mtrbus_trips_df.columns:
        mtrbus_trips_df['original_service_id'] = 'DEFAULT'
    if 'route_short_name' not in mtrbus_trips_df.columns:
        mtrbus_trips_df['route_short_name'] = ''
    mtrbus_stoptimes_df = pd.read_sql("SELECT * FROM mtrbus_stop_sequences", engine)
    mtrbus_stoptimes_df['trip_id'] = 'MTRB-' + mtrbus_stoptimes_df['route_id'] + '-' + mtrbus_stoptimes_df['direction']
    mtrbus_stoptimes_df['stop_id'] = 'MTRB-' + mtrbus_stoptimes_df['stop_id'].astype(str)
    mtrbus_stoptimes_df['stop_id'] = mtrbus_stoptimes_df['stop_id'].replace(mtrbus_duplicates_map)

    # -- NLB --
    if not silent:
        print("Processing NLB routes, trips, and stop_times...")
    nlb_routes_df = pd.read_sql("SELECT * FROM nlb_routes", engine)
    nlb_routes_df[['orig_en', 'dest_en']] = nlb_routes_df['routeName_e'].str.split(' > ', expand=True)
    final_nlb_routes_list = []
    for route_no, group in nlb_routes_df.groupby('routeNo'):
        all_dests = group['dest_en'].unique()
        long_name = ' - '.join(all_dests)
        final_nlb_routes_list.append({
            'route_id': f'NLB-{route_no}',
            'agency_id': 'NLB',
            'route_short_name': route_no,
            'route_long_name': long_name,
            'route_type': 3
        })
    final_nlb_routes = pd.DataFrame(final_nlb_routes_list)
    nlb_routes_df['direction_id'] = nlb_routes_df.apply(lambda row: get_direction(row['routeName_e'], row['orig_en']), axis=1)

    # Use enhanced matching for NLB routes
    if not silent:
        print("Using enhanced stop-count-based matching for NLB routes...")
    
    nlb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="NLB",
        debug=not silent
    )
    if not silent:
        print(f"NLB matching produced {len(nlb_route_matches)} route keys")
        if len(nlb_route_matches)==0:
            # diagnostics
            sample_nlb = pd.read_sql("SELECT routeId, \"routeNo\", \"routeName_e\" FROM nlb_routes ORDER BY routeId LIMIT 10", engine)
            print("Sample NLB routes (first 10):")
            print(sample_nlb.to_string(index=False))
            print("Creating fallback NLB trips (one per directionId heuristic)...")
    
    nlb_trips_list = []
    if nlb_route_matches:
        seen_trip_keys = set()
        for route_key, route_matches in nlb_route_matches.items():
            # route_key like "1-O-1" or "1-I-1"; we ignore service_type portion for NLB direction mapping
            parts = route_key.split('-')
            if len(parts) < 2:
                continue
            route_short_name = parts[0]
            for match in route_matches:
                gov_bound = match.get('gov_bound')
                if gov_bound not in ('O','I'):
                    continue
                direction_id = 0 if gov_bound == 'O' else 1
                routeid = match.get('routeid')
                if routeid is None:
                    try:
                        q = f"SELECT \"routeId\" FROM nlb_routes WHERE \"routeNo\"='{route_short_name}' ORDER BY \"routeId\" LIMIT 1"
                        routeid_df = pd.read_sql(q, engine)
                        if not routeid_df.empty:
                            routeid = routeid_df.iloc[0]['routeId']
                    except Exception:
                        pass
                # Use combination to prevent duplicates
                trip_key = (route_short_name, routeid, match['gov_service_id'], gov_bound)
                if trip_key in seen_trip_keys:
                    continue
                seen_trip_keys.add(trip_key)
                trip_id = f"NLB-{route_short_name}-{routeid}-{match['gov_service_id']}"
                nlb_trips_list.append({
                    'route_id': f"NLB-{route_short_name}",
                    'agency_id': 'NLB',
                    'service_id': f"NLB-{route_short_name}-{match['gov_service_id']}",
                    'trip_id': trip_id,
                    'direction_id': direction_id,
                    'bound': gov_bound,
                    'route_short_name': route_short_name,
                    'original_service_id': match['gov_service_id'],
                    'route_long_name': match.get('gov_route_long_name',''),
                    'routeId': routeid,
                    'origin_en': '',
                    'destination_en': '',
                    'gov_route_id': match['gov_route_id']
                })
    else:
        # Fallback removed: we require matches now for NLB
        if not silent:
            print("Warning: No NLB matches found; no NLB trips will be exported")
    
    nlb_trips_df = pd.DataFrame(nlb_trips_list)
    if nlb_trips_df.empty:
        nlb_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'route_short_name', 
            'original_service_id', 'route_long_name', 'routeId', 'origin_en', 'destination_en', 'gov_route_id'
        ])
    nlb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)
    
    # Ensure required columns exist for downstream processing  
    if 'original_service_id' not in nlb_trips_df.columns:
        nlb_trips_df['original_service_id'] = 'DEFAULT'
    if 'route_short_name' not in nlb_trips_df.columns:
        nlb_trips_df['route_short_name'] = ''
    nlb_stoptimes_df = pd.read_sql("SELECT * FROM nlb_stop_sequences", engine)
    nlb_stoptimes_df['trip_id'] = 'NLB-' + nlb_stoptimes_df['routeNo'] + '-' + nlb_stoptimes_df['routeId'].astype(str)
    nlb_stoptimes_df['stop_id'] = 'NLB-' + nlb_stoptimes_df['stopId'].astype(str)
    nlb_stoptimes_df['stop_id'] = nlb_stoptimes_df['stop_id'].replace(nlb_duplicates_map)

    # -- MTR Rail --
    if not silent:
        print("==========================================")
        print("Processing MTR Rail routes, trips, stop_times...")
        print("==========================================")
    mtr_lines_and_stations_df = pd.read_sql("SELECT * FROM mtr_lines_and_stations", engine)

    # Prepare journey time lookup (from Station Code to Station Code)
    jt_lookup = {}
    try:
        if isinstance(journey_time_data, (list, tuple)):
            jt_df = pd.DataFrame(journey_time_data)
        elif isinstance(journey_time_data, dict):
            # Attempt to coerce dict-of-lists or list-of-dicts
            if {'from_stop_id','to_stop_id','travel_time_seconds'}.issubset(set(journey_time_data.keys())):
                jt_df = pd.DataFrame(journey_time_data)
            else:
                jt_df = pd.DataFrame(list(journey_time_data))
        else:
            jt_df = pd.DataFrame()
        if not jt_df.empty and {'from_stop_id','to_stop_id','travel_time_seconds'}.issubset(jt_df.columns):
            # Keep only plausible rail codes (alphanumeric <=4 chars) to reduce noise
            jt_df = jt_df[jt_df['from_stop_id'].str.len() <= 4]
            jt_df = jt_df[jt_df['to_stop_id'].str.len() <= 4]
            jt_lookup = {(r.from_stop_id, r.to_stop_id): float(r.travel_time_seconds) for r in jt_df.itertuples()}
            if not silent:
                print(f"Loaded {len(jt_lookup)} journey-time edges for potential MTR timing.")
    except Exception as e:
        if not silent:
            print(f"Journey time integration skipped (error building lookup): {e}")
        jt_lookup = {}

    # Build route metadata (one GTFS route per line code)
    line_code_to_name = {
        'EAL': 'East Rail Line',
        'TML': 'Tuen Ma Line',
        'TWL': 'Tsuen Wan Line',
        'KTL': 'Kwun Tong Line',
        'ISL': 'Island Line',
        'TKL': 'Tseung Kwan O Line',
        'SIL': 'South Island Line',
        'AEL': 'Airport Express',
        'TCL': 'Tung Chung Line',
        'DRL': 'Disneyland Resort Line'
    }
    # Derive terminal pairs from base DT direction (or UT if DT missing)
    terminals = []
    for lc, group in mtr_lines_and_stations_df.groupby('Line Code'):
        direction_order_candidates = [d for d in group['Direction'].unique() if d.endswith('DT')]
        chosen_dir = 'DT'
        if 'DT' not in group['Direction'].unique() and direction_order_candidates:
            chosen_dir = direction_order_candidates[0]
        elif 'DT' not in group['Direction'].unique() and 'UT' in group['Direction'].unique():
            chosen_dir = 'UT'
        seq_subset = group[group['Direction'] == chosen_dir].copy()
        if seq_subset.empty:
            seq_subset = group.copy()
        seq_subset['seq_num'] = seq_subset['Sequence'].str.extract(r'^(\d+)').astype(int)
        seq_subset = seq_subset.sort_values('seq_num')
        if not seq_subset.empty:
            first_station = seq_subset.iloc[0]['English Name']
            last_station = seq_subset.iloc[-1]['English Name']
            terminals.append({'line_code': lc, 'terminal_pair': f"{first_station} - {last_station}"})
    terminals_df = pd.DataFrame(terminals)
    mtr_routes_df = terminals_df.copy()
    mtr_routes_df['route_id'] = 'MTR-' + mtr_routes_df['line_code']
    mtr_routes_df['agency_id'] = 'MTRR'
    mtr_routes_df['route_short_name'] = mtr_routes_df['line_code']
    mtr_routes_df['route_long_name'] = mtr_routes_df.apply(lambda r: f"{line_code_to_name.get(r['line_code'], r['line_code'])} ({r['terminal_pair']})", axis=1)
    mtr_routes_df['route_type'] = 1
    mtr_routes_df = mtr_routes_df[['route_id','agency_id','route_short_name','route_long_name','route_type']]

    direction_variants = mtr_lines_and_stations_df[['Line Code','Direction']].drop_duplicates()
    mtr_trips_list = []
    for _, row in direction_variants.iterrows():
        line_code = row['Line Code']
        variant = row['Direction']
        direction_id = 0 if variant.endswith('UT') else 1
        trip_id = f"MTR-{line_code}-{variant}"
        mtr_trips_list.append({
            'route_id': f"MTR-{line_code}",
            'agency_id': 'MTRR',
            'service_id': f"MTR-{line_code}-SERVICE",
            'trip_id': trip_id,
            'direction_id': direction_id,
            'original_service_id': f"MTR-{line_code}-SERVICE"
        })
    mtr_trips_df = pd.DataFrame(mtr_trips_list)

    # Stop times using journey_time_data where possible
    mtr_stoptimes_rows = []
    # Compute a robust default from journey_time_data (median) else fallback 120s
    mtr_edge_times = [v for (a,b), v in jt_lookup.items() if len(a)<=4 and len(b)<=4]
    import statistics
    try:
        median_edge = statistics.median(mtr_edge_times) if mtr_edge_times else 120.0
    except statistics.StatisticsError:
        median_edge = 120.0
    DEFAULT_SEGMENT = median_edge if 30 <= median_edge <= 600 else 120.0
    if not silent:
        print(f"MTR default segment time set to {int(DEFAULT_SEGMENT)}s (median of journey_time_data)" )
    for _, row in direction_variants.iterrows():
        line_code = row['Line Code']
        variant = row['Direction']
        trip_id = f"MTR-{line_code}-{variant}"
        seg_df = mtr_lines_and_stations_df[(mtr_lines_and_stations_df['Line Code']==line_code) & (mtr_lines_and_stations_df['Direction']==variant)].copy()
        if seg_df.empty:
            continue
        seg_df['seq_num'] = seg_df['Sequence'].str.extract(r'^(\d+)').astype(int)
        seg_df = seg_df.sort_values('seq_num')
        cumulative = 0.0
        prev_code = None
        for idx, r in enumerate(seg_df.itertuples(index=False), start=1):
            # safe station_code access
            if hasattr(r, 'Station_Code'):
                station_code = getattr(r, 'Station_Code')
            else:
                station_code = seg_df.iloc[idx-1]['Station Code']
            if prev_code is not None:
                tt = jt_lookup.get((prev_code, station_code))
                if tt is None:
                    # try reverse (assume symmetric)
                    rev = jt_lookup.get((station_code, prev_code))
                    tt = rev
                if tt is None:
                    tt = DEFAULT_SEGMENT
                # sanity clamp
                if tt <= 0:
                    tt = DEFAULT_SEGMENT
                if tt > 900:  # improbable long dwell between adjacent stations
                    tt = DEFAULT_SEGMENT
                cumulative += tt
            else:
                cumulative = 0.0
            hh = int(cumulative)//3600; mm=(int(cumulative)%3600)//60; ss=int(cumulative)%60
            t_str = f"{hh:02}:{mm:02}:{ss:02}"
            mtr_stoptimes_rows.append({
                'trip_id': trip_id,
                'arrival_time': t_str,
                'departure_time': t_str,
                'stop_id': 'MTR-' + station_code,
                'stop_sequence': idx
            })
            prev_code = station_code
    mtr_stoptimes_df = pd.DataFrame(mtr_stoptimes_rows)

    # -- Light Rail --
    if not silent:
        print("Processing Light Rail routes, trips, and stop_times...")
    light_rail_routes_and_stops_df = pd.read_sql("SELECT * FROM light_rail_routes_and_stops", engine)
    lr_routes_df = light_rail_routes_and_stops_df[['Line Code', 'English Name']].drop_duplicates(subset=['Line Code'])
    lr_routes_df.rename(columns={'Line Code': 'route_id', 'English Name': 'route_long_name'}, inplace=True)
    lr_routes_df['route_id'] = 'LR-' + lr_routes_df['route_id']
    lr_routes_df['agency_id'] = 'LR'
    lr_routes_df['route_short_name'] = lr_routes_df['route_id'].str.replace('LR-', '')
    lr_routes_df['route_type'] = 0 # Tram

    lr_trips_list = []
    for _, route in lr_routes_df.iterrows():
        lr_trips_list.append({
            'route_id': route['route_id'],
            'agency_id': 'LR',
            'service_id': f"{route['route_id']}-SERVICE",
            'trip_id': f"{route['route_id']}-TRIP",
            'direction_id': 0
        })
        lr_trips_list.append({
            'route_id': route['route_id'],
            'agency_id': 'LR',
            'service_id': f"{route['route_id']}-SERVICE",
            'trip_id': f"{route['route_id']}-TRIP-2",
            'direction_id': 1
        })
    lr_trips_df = pd.DataFrame(lr_trips_list)

    lr_stoptimes_df = light_rail_routes_and_stops_df.copy()
    lr_stoptimes_df['trip_id'] = 'LR-' + lr_stoptimes_df['Line Code'] + '-TRIP'
    lr_stoptimes_df.loc[lr_stoptimes_df['Direction'] == '2', 'trip_id'] = 'LR-' + lr_stoptimes_df['Line Code'] + '-TRIP-2'
    lr_stoptimes_df['stop_id'] = 'LR-' + lr_stoptimes_df['Stop ID'].astype(str)
    lr_stoptimes_df.rename(columns={'Sequence': 'stop_sequence'}, inplace=True)
    lr_stoptimes_df['stop_sequence'] = pd.to_numeric(lr_stoptimes_df['stop_sequence'], errors='coerce').astype('Int64')
    # Estimate journey time between stations (e.g., 90 seconds)
    lr_stoptimes_df['travel_time'] = 90
    lr_stoptimes_df['arrival_time'] = lr_stoptimes_df.groupby('trip_id')['travel_time'].cumsum() - 90
    lr_stoptimes_df['departure_time'] = lr_stoptimes_df['arrival_time']
    lr_stoptimes_df['arrival_time'] = lr_stoptimes_df['arrival_time'].apply(lambda x: format_timedelta(timedelta(seconds=x)))
    lr_stoptimes_df['departure_time'] = lr_stoptimes_df['departure_time'].apply(lambda x: format_timedelta(timedelta(seconds=x)))

    # -- Frequency Processing --
    # Standardize merge keys to string to prevent type errors
    kmb_trips_df['original_service_id'] = kmb_trips_df['original_service_id'].astype(str)
    kmb_trips_df['route_short_name'] = kmb_trips_df['route_short_name'].astype(str)
    ctb_trips_df['original_service_id'] = ctb_trips_df['original_service_id'].astype(str)
    ctb_trips_df['route_short_name'] = ctb_trips_df['route_short_name'].astype(str)
    gmb_trips_df['original_service_id'] = gmb_trips_df['original_service_id'].astype(str)
    gmb_trips_df['route_short_name'] = gmb_trips_df['route_short_name'].astype(str)
    mtrbus_trips_df['original_service_id'] = mtrbus_trips_df['original_service_id'].astype(str)
    mtrbus_trips_df['route_short_name'] = mtrbus_trips_df['route_short_name'].astype(str)
    nlb_trips_df['original_service_id'] = nlb_trips_df['original_service_id'].astype(str)
    nlb_trips_df['route_short_name'] = nlb_trips_df['route_short_name'].astype(str)

    master_duplicates_map = {
        **kmb_duplicates_map,
        **ctb_duplicates_map,
        **gmb_duplicates_map,
        **mtrbus_duplicates_map,
        **nlb_duplicates_map
    }

    if not silent:
        print("Creating a reverse map for unified stops to original stops...")
    unified_to_original_map = {}
    for original, unified in master_duplicates_map.items():
        if unified not in unified_to_original_map:
            unified_to_original_map[unified] = []
        unified_to_original_map[unified].append(original)


    kmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('KMB', na=False)]
    kmb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(kmb_gov_routes_df['route_id'])]
    kmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(kmb_gov_trips_df['trip_id'])]
    kmb_stoptimes_df = generate_stop_times_for_agency(
        'KMB', kmb_trips_df, kmb_stoptimes_df, kmb_gov_routes_df, kmb_gov_trips_df, kmb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
    )

    ctb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('CTB', na=False)]
    ctb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(ctb_gov_routes_df['route_id'])]
    ctb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(ctb_gov_trips_df['trip_id'])]
    
    ctb_stoptimes_df = generate_stop_times_for_agency(
        'CTB', ctb_trips_df, ctb_stoptimes_df, ctb_gov_routes_df, ctb_gov_trips_df, ctb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
    )

    gmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('GMB', na=False)]
    gmb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(gmb_gov_routes_df['route_id'])]
    gmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(gmb_gov_trips_df['trip_id'])]
    gmb_stoptimes_df = generate_stop_times_for_agency(
        'GMB', gmb_trips_df, gmb_stoptimes_df, gmb_gov_routes_df, gmb_gov_trips_df, gmb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
    )
    print(gmb_stoptimes_df.head())

    mtrbus_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('LRTFeeder', na=False)]
    mtrbus_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(mtrbus_gov_routes_df['route_id'])]
    mtrbus_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(mtrbus_gov_trips_df['trip_id'])]
    mtrbus_stoptimes_df = generate_stop_times_for_agency(
        'MTRB', mtrbus_trips_df, mtrbus_stoptimes_df, mtrbus_gov_routes_df, mtrbus_gov_trips_df, mtrbus_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
    )

    nlb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('NLB', na=False)]
    nlb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(nlb_gov_routes_df['route_id'])]
    nlb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(nlb_gov_trips_df['trip_id'])]
    nlb_stoptimes_df = generate_stop_times_for_agency(
        'NLB', nlb_trips_df, nlb_stoptimes_df, nlb_gov_routes_df, nlb_gov_trips_df, nlb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
    )

    # --- Combine & Standardize--
    if not silent:
        print("Combining and standardizing data for final GTFS files...")
    final_routes_df = pd.concat([final_kmb_routes, final_ctb_routes, final_gmb_routes, final_mtrbus_routes, final_nlb_routes, mtr_routes_df, lr_routes_df], ignore_index=True)
    color_map = {'KMB': 'EE171F', 'CTB': '0053B9', 'NLB': '8AB666', 'MTRB': 'AE2A42', 'GMB': '34C759', 'MTRR': '003DA5', 'LR': 'FF8800'}
    
    def get_route_color(agency_id):
        if not isinstance(agency_id, str):
            return None
        # For co-op routes, use the color of the first agency listed.
        first_agency = agency_id.split('+')[0]
        return color_map.get(first_agency)

    final_routes_df['route_color'] = final_routes_df['agency_id'].apply(get_route_color)
    final_routes_df['route_text_color'] = 'FFFFFF'
    all_trips_df = pd.concat([kmb_trips_df, ctb_trips_df, gmb_trips_df, mtrbus_trips_df, nlb_trips_df, mtr_trips_df, lr_trips_df], ignore_index=True)

    # Create a mapping from the original government trip_id to our new trip_id
    trip_id_mapping = all_trips_df.merge(
        gov_trips_with_route_info,
        left_on=['original_service_id', 'route_short_name', 'direction_id'],
        right_on=['service_id', 'route_short_name', 'direction_id']
    )[['trip_id_x', 'trip_id_y']].rename(columns={'trip_id_x': 'new_trip_id', 'trip_id_y': 'original_trip_id'})

    # Special handling for GMB routes - their route_short_name includes region (HKI-1 vs 1)
    gmb_trips = all_trips_df[all_trips_df['trip_id'].str.startswith('GMB-')]
    if not gmb_trips.empty:
        # Extract base route number from GMB route_short_name (HKI-1 -> 1)
        gmb_trips_copy = gmb_trips.copy()
        gmb_trips_copy['base_route_name'] = gmb_trips_copy['route_short_name'].str.split('-').str[1]
        
        gmb_trip_mapping = gmb_trips_copy.merge(
            gov_trips_with_route_info,
            left_on=['original_service_id', 'base_route_name', 'direction_id'],
            right_on=['service_id', 'route_short_name', 'direction_id']
        )[['trip_id_x', 'trip_id_y']].rename(columns={'trip_id_x': 'new_trip_id', 'trip_id_y': 'original_trip_id'})
        
        # Add GMB mappings to the main mapping
        trip_id_mapping = pd.concat([trip_id_mapping, gmb_trip_mapping], ignore_index=True)

    # Update the trip_id in the frequencies dataframe
    final_frequencies_df = gov_frequencies_df.merge(trip_id_mapping, left_on='trip_id', right_on='original_trip_id', how='inner')
    final_frequencies_df['trip_id'] = final_frequencies_df['new_trip_id']
    final_frequencies_df = final_frequencies_df.drop(columns=['new_trip_id', 'original_trip_id'])

    # --- MTR/LR Frequencies ---
    try:
        # Remove any pre-existing MTR/LR frequencies from government feed (they don't apply to our synthetic rail trips)
        if not final_frequencies_df.empty:
            final_frequencies_df = final_frequencies_df[~final_frequencies_df['trip_id'].str.startswith('MTR-')]
            final_frequencies_df = final_frequencies_df[~final_frequencies_df['trip_id'].str.startswith('LR-')]

        # Collect trips per line
        mtr_line_trips = {}
       
        if not mtr_trips_df.empty:
            tdf = mtr_trips_df.copy()
            tdf['line_code'] = tdf['route_id'].str.replace('MTR-','', regex=False)
            for lc, grp in tdf.groupby('line_code'):
                mtr_line_trips[lc] = grp['trip_id'].unique().tolist()
        lr_line_trips = {}
        if not lr_trips_df.empty:
            ldf = lr_trips_df.copy()
            ldf['line_code'] = ldf['route_id'].str.replace('LR-','', regex=False)
            for lc, grp in ldf.groupby('line_code'):
                lr_line_trips[lc] = grp['trip_id'].unique().tolist()

        abbr_map = {
            'EAL': 'East Rail', 'TML': 'Tuen Ma', 'TWL': 'Tsuen Wan', 'KTL': 'Kwun Tong',
            'ISL': 'Island', 'TKL': 'Tseung Kwan O', 'SIL': 'South Island', 'AEL': 'Airport Express',
            'TCL': 'Tung Chung', 'DRL': 'Disneyland Resort'
        }

        # Normalize scraped headway data
        scraped_info = {}
        if mtr_headway_data:
            for raw_key, data in mtr_headway_data.items():
                if not isinstance(data, dict):
                    continue
                key_norm = raw_key.lower().replace(' line','').strip()
                target = None
                for abbr, base in abbr_map.items():
                    bl = base.lower()
                    if key_norm == abbr.lower() or key_norm == bl or bl in key_norm or abbr.lower() in key_norm:
                        target = abbr
                        break
                if not target:
                    continue
                scraped_info[target] = {
                    'morning_peak': parse_headway_to_avg_secs(data.get('weekdays', {}).get('morning_peak')),
                    'non_peak': parse_headway_to_avg_secs(data.get('weekdays', {}).get('non_peak')),
                    'saturdays': parse_headway_to_avg_secs(data.get('saturdays'))
                }

        # Default daily slices (extended 25:00 for after midnight)
        default_slices = [
            ('05:30:00','07:00:00',360),
            ('07:00:00','10:00:00',180),
            ('10:00:00','17:00:00',300),
            ('17:00:00','20:00:00',180),
            ('20:00:00','23:00:00',360),
            ('23:00:00','25:00:00',600)
        ]

        def build_slices(line_code):
            info = scraped_info.get(line_code, {})
            slices = []
            for start, end, hw in default_slices:
                adj = hw
                # Override peaks
                if start in ('07:00:00','17:00:00') and info.get('morning_peak'):
                    adj = info['morning_peak']
                # Override mid / evening non-peak
                if start in ('10:00:00','20:00:00') and info.get('non_peak'):
                    adj = info['non_peak']
                # Optionally shorten late night if scraped has Saturday spec (ignore for now)
                slices.append((start, end, int(adj)))
            return slices

        new_rows = []
        # Heavy rail
        for lc, trips in mtr_line_trips.items():
            slices = build_slices(lc)
            for trip in trips:
                for start, end, headway in slices:
                    new_rows.append({'trip_id': trip, 'start_time': start, 'end_time': end, 'headway_secs': headway})
        # Light Rail (same defaults)
        for lc, trips in lr_line_trips.items():
            slices = build_slices(lc)  # no scraped overrides expected
            for trip in trips:
                for start, end, headway in slices:
                    new_rows.append({'trip_id': trip, 'start_time': start, 'end_time': end, 'headway_secs': headway})

        if new_rows:
            mtr_lr_freq_df = pd.DataFrame(new_rows)
            final_frequencies_df = pd.concat([final_frequencies_df, mtr_lr_freq_df], ignore_index=True)
            if not silent:
                missing_defaults = [lc for lc in mtr_line_trips.keys() if lc not in scraped_info]
                print(f"Applied frequencies for MTR lines {sorted(mtr_line_trips.keys())}; defaults used for {missing_defaults}")
    except Exception as e:
        if not silent:
            print(f"Warning: MTR/LR frequency generation failed: {e}")
    # --- END MTR/LR Frequencies ---

    # Remove true duplicates only - preserve different services for the same route
    final_trips_df = all_trips_df.drop_duplicates(subset=['trip_id'], keep='first')

    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
    final_kmb_stoptimes = kmb_stoptimes_df.rename(columns={'seq': 'stop_sequence'})
    final_ctb_stoptimes = ctb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_gmb_stoptimes = gmb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtrbus_stoptimes = mtrbus_stoptimes_df.rename(columns={'station_seqno': 'stop_sequence'})
    final_nlb_stoptimes = nlb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtr_stoptimes = mtr_stoptimes_df
    final_lr_stoptimes = lr_stoptimes_df
    final_stop_times_df = pd.concat([final_kmb_stoptimes, final_ctb_stoptimes, final_gmb_stoptimes, final_mtrbus_stoptimes, final_nlb_stoptimes, final_mtr_stoptimes, final_lr_stoptimes], ignore_index=True)
    final_stop_times_df['stop_id'] = final_stop_times_df['stop_id'].replace(master_duplicates_map)

    final_stop_times_output_df = final_stop_times_df[stop_times_cols].copy()

    # Enforce unique stop_sequence for each trip
    final_stop_times_output_df.sort_values(['trip_id', 'stop_sequence'], inplace=True)
    final_stop_times_output_df['stop_sequence'] = final_stop_times_output_df.groupby('trip_id').cumcount() + 1
    final_stop_times_output_df.drop_duplicates(subset=['trip_id', 'stop_sequence'], keep='first', inplace=True)

    # Filter stop_times to ensure foreign key constraint with stops.txt
    valid_stop_ids = set(all_stops_output_df['stop_id'])
    original_stop_times_count = len(final_stop_times_output_df)
    final_stop_times_output_df = final_stop_times_output_df[final_stop_times_output_df['stop_id'].isin(valid_stop_ids)]
    if not silent:
        filtered_count = original_stop_times_count - len(final_stop_times_output_df)
        if filtered_count > 0:
            print(f"Warning: Removed {filtered_count} stop_times records that referenced non-existent stops.")

    # --- Shapes --- 
    if not silent:
        print("Generating shapes from CSDI data...")
    if no_regenerate_shapes and os.path.exists(os.path.join(final_output_dir, 'shapes.txt')):
        if not silent:
            print("Skipping shape generation as --no-regenerate-shapes is set and shapes.txt already exists.")
        success = True
        with open(os.path.join(final_output_dir, 'shapes.txt'), 'r') as f:
            # this is a hack to get the shape_info
            shape_info = []
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                shape_id = parts[0]
                if shape_id not in [s['shape_id'] for s in shape_info]:
                    gov_route_id, bound = shape_id.split('-')[1:]
                    shape_info.append({'shape_id': shape_id, 'gov_route_id': gov_route_id, 'bound': bound})
    else:
        success, shape_info = generate_shapes_from_csdi_files(os.path.join(final_output_dir, 'shapes.txt'), engine, silent=silent)
    if success:
        final_trips_df = match_trips_to_csdi_shapes(final_trips_df, shape_info, engine, silent=silent)

    # Standardize trips.txt output
    final_trips_df['trip_headsign'] = ''  # Add empty column as per GTFS spec
    gtfs_trips_cols = ['route_id', 'service_id', 'trip_id', 'trip_headsign', 'direction_id', 'shape_id']
    # Ensure all columns exist, fill with None if they don't
    for col in gtfs_trips_cols:
        if col not in final_trips_df.columns:
            final_trips_df[col] = None
    final_trips_output_df = final_trips_df[gtfs_trips_cols]


    final_routes_df.to_csv(os.path.join(final_output_dir, 'routes.txt'), index=False)
    final_trips_output_df.to_csv(os.path.join(final_output_dir, 'trips.txt'), index=False)
    final_stop_times_output_df.to_csv(os.path.join(final_output_dir, 'stop_times.txt'), index=False)

    # Resolve overlapping frequencies before saving
    final_frequencies_df = resolve_overlapping_frequencies(final_frequencies_df)
    frequencies_cols = ['trip_id', 'start_time', 'end_time', 'headway_secs']
    final_frequencies_output_df = final_frequencies_df[frequencies_cols]
    final_frequencies_output_df.to_csv(os.path.join(final_output_dir, 'frequencies.txt'), index=False)

    # --- Pathways & Transfers (MTR Entrances to Stations) ---
    try:
        entrances_df = all_stops_output_df[all_stops_output_df['stop_id'].str.startswith('MTR-ENTRANCE')].copy()
        stations_df = all_stops_output_df[(all_stops_output_df['stop_id'].str.startswith('MTR-')) & (~all_stops_output_df['stop_id'].str.startswith('MTR-ENTRANCE'))].copy()
        if not entrances_df.empty and not stations_df.empty:
            if not silent:
                print(f"Generating pathways/transfers for {len(entrances_df)} MTR entrances...")
            import math
            station_coords = stations_df.set_index('stop_id')[['stop_lat','stop_lon']].to_dict('index')
            def haversine(lat1, lon1, lat2, lon2):
                R=6371000.0
                dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
                a=math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
                return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))
            pathways=[]; transfers=[]
            for _, ent in entrances_df.iterrows():
                parent = ent.get('parent_station')
                if not parent or parent not in station_coords:
                    continue
                lat1, lon1 = ent['stop_lat'], ent['stop_lon']
                lat2, lon2 = station_coords[parent]['stop_lat'], station_coords[parent]['stop_lon']
                dist = haversine(lat1, lon1, lat2, lon2)
                if not dist or dist == 0:
                    dist = 30.0  # nominal distance
                traversal_time = max(10, int(round(dist / 1.25)))  # 1.25 m/s
                pid = f"PATH-{ent['stop_id']}__{parent}"
                pathways.append({
                    'pathway_id': pid,
                    'from_stop_id': ent['stop_id'],
                    'to_stop_id': parent,
                    'pathway_mode': 1,  # walkway
                    'is_bidirectional': 1,
                    'length': int(round(dist)),
                    'traversal_time': traversal_time
                })
                transfers.append({'from_stop_id': ent['stop_id'], 'to_stop_id': parent, 'transfer_type': 2, 'min_transfer_time': traversal_time})
                transfers.append({'from_stop_id': parent, 'to_stop_id': ent['stop_id'], 'transfer_type': 2, 'min_transfer_time': traversal_time})
            if pathways:
                pd.DataFrame(pathways)[['pathway_id','from_stop_id','to_stop_id','pathway_mode','is_bidirectional','length','traversal_time']].to_csv(os.path.join(final_output_dir,'pathways.txt'), index=False)
            if transfers:
                pd.DataFrame(transfers)[['from_stop_id','to_stop_id','transfer_type','min_transfer_time']].to_csv(os.path.join(final_output_dir,'transfers.txt'), index=False)
            if not silent:
                print(f"Wrote pathways.txt ({len(pathways)}) and transfers.txt ({len(transfers)})")
    except Exception as e:
        if not silent:
            print(f"Warning generating pathways/transfers: {e}")

    # --- 4. Handle `calendar.txt` and `calendar_dates.txt` ---
    if not silent:
        print("Processing calendar and calendar_dates...")

    # Get the base calendar and calendar_dates data
    gov_calendar_df = pd.read_sql("SELECT * FROM gov_gtfs_calendar", engine)
    gov_calendar_df['service_id'] = gov_calendar_df['service_id'].astype(str)
    gov_calendar_df = gov_calendar_df.set_index('service_id')

    gov_calendar_dates_df = pd.read_sql("SELECT * FROM gov_gtfs_calendar_dates", engine)
    gov_calendar_dates_df['service_id'] = gov_calendar_dates_df['service_id'].astype(str)
    gov_calendar_dates_df = gov_calendar_dates_df.set_index('service_id')

    # Create a dataframe with the mapping between new and original service IDs
    service_id_mapping_df = final_trips_df[['service_id', 'original_service_id']].drop_duplicates()

    # Merge the mapping with the calendar data
    # Use a left merge to ensure all service_ids from trips.txt are kept.
    final_calendar_df = service_id_mapping_df.merge(
        gov_calendar_df,
        left_on='original_service_id',
        right_index=True,
        how='left'
    )

    # For trips that didn't have a matching calendar entry (i.e., our default services),
    # create a default calendar entry that runs every day.
    default_calendar_values = {
        'monday': 1, 'tuesday': 1, 'wednesday': 1, 'thursday': 1,
        'friday': 1, 'saturday': 1, 'sunday': 1,
        'start_date': '20230101', 'end_date': '20251231'
    }
    # Fill NaN values for the default services
    for col, val in default_calendar_values.items():
        final_calendar_df[col] = final_calendar_df[col].fillna(val)

    # Ensure integer types for date and day columns
    date_cols = ['start_date', 'end_date']
    day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for col in date_cols + day_cols:
        # Convert to numeric first to handle potential floats from NaN fill, then to int
        final_calendar_df[col] = pd.to_numeric(final_calendar_df[col]).astype(int)

    final_calendar_df = final_calendar_df.drop(columns=['original_service_id'])

    # For calendar_dates, we only want entries that actually existed in the gov data.
    # An inner join is correct here, as we don't want to create default exception dates.
    final_calendar_dates_df = service_id_mapping_df.merge(
        gov_calendar_dates_df,
        left_on='original_service_id',
        right_index=True,
        how='inner'
    )
    final_calendar_dates_df = final_calendar_dates_df.drop(columns=['original_service_id'])

    # why are there dupes??
    final_calendar_df = final_calendar_df.drop_duplicates(subset=['service_id'])
    final_calendar_dates_df = final_calendar_dates_df.drop_duplicates(subset=['service_id', 'date'])


    final_calendar_df.to_csv(os.path.join(final_output_dir, 'calendar.txt'), index=False)
    final_calendar_dates_df.to_csv(os.path.join(final_output_dir, 'calendar_dates.txt'), index=False)

    ## --- Translations (Traditional Chinese) ---
    if not silent:
        print("Building translations.txt (Traditional Chinese)...")
    translations = []
    lang_tc = "zh-Hant"
    
    # Stop name translations
    try:
        if 'kmb_stops_gdf' in locals() and 'name_tc' in kmb_stops_gdf.columns:
            kmb_tc_clean = (
                kmb_stops_gdf['name_tc']
                .str.replace(r'\s*\([A-Za-z0-9]{5}\)', '', regex=True)
                .str.replace(r'\s*-\s*', ' - ', regex=True)
                .str.replace(r'([^\s])(\([A-Za-z0-9]+\))', r'\1 \2', regex=True)
            )
            for stop_id, name_tc in zip(kmb_stops_gdf['stop_id'], kmb_tc_clean):
                if pd.notna(name_tc) and str(name_tc).strip():
                    translations.append({
                        'table_name':'stops.txt',
                        'field_name':'stop_name',
                        'language':lang_tc,
                        'record_id':stop_id,
                        'translation':name_tc
                    })
        if 'ctb_stops_gdf' in locals() and 'name_tc' in ctb_stops_gdf.columns:
            for r in ctb_stops_gdf[['stop_id','name_tc']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.name_tc})
        if 'gmb_stops_gdf' in locals() and 'stop_name_tc' in gmb_stops_gdf.columns:
            for r in gmb_stops_gdf[['stop_id','stop_name_tc']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.stop_name_tc})
        if 'nlb_stops_gdf' in locals() and 'stopName_c' in nlb_stops_gdf.columns:
            for r in nlb_stops_gdf[['stop_id','stopName_c']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.stopName_c})
        if 'mtr_stations_gdf' in locals() and 'Chinese Name' in mtr_stations_gdf.columns:
            for r in (mtr_stations_gdf[['Station Code','Chinese Name']].drop_duplicates('Station Code').dropna().itertuples()):
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':f"MTR-{r.Station_Code}",'translation':r.Chinese_Name})
        if 'mtr_exits_gdf' in locals() and 'station_name_zh' in mtr_exits_gdf.columns:
            for r in mtr_exits_gdf[['stop_id','station_name_zh','exit']].dropna(subset=['station_name_zh']).itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':f"{r.station_name_zh} 出口 {r.exit}"})
        if 'lr_stops_gdf' in locals() and 'name_tc' in lr_stops_gdf.columns:
            for r in lr_stops_gdf[['stop_id','name_tc']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.name_tc})
    except Exception as e:
        if not silent:
            print(f"Stop translations warning: {e}")
    
    # Route long name translations
    try:
        if 'kmb_routes_df' in locals() and not kmb_routes_df.empty:
            for route_num, grp in kmb_routes_df.groupby('route'):
                fo = grp[grp['bound']=='O']
                fi = grp[grp['bound']=='I']
                fo_row = fo.iloc[0] if not fo.empty else (grp.iloc[0] if not grp.empty else None)
                fi_row = fi.iloc[0] if not fi.empty else None
                if fo_row is not None and fi_row is not None:
                    cn_long = f"{fo_row.get('orig_tc','')} - {fi_row.get('orig_tc','')}"
                elif fo_row is not None:
                    cn_long = f"{fo_row.get('orig_tc','')} - {fo_row.get('dest_tc','')}"
                elif fi_row is not None:
                    cn_long = f"{fi_row.get('orig_tc','')} - {fi_row.get('dest_tc','')}"
                else:
                    cn_long = ''
                translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':f"KMB-{route_num}",'translation':cn_long})
        if 'ctb_routes_df' in locals() and not ctb_routes_df.empty:
            for route_num, grp in ctb_routes_df.groupby('route'):
                fo = grp[grp['dir']=='outbound']
                fi = grp[grp['dir']=='inbound']
                fo_row = fo.iloc[0] if not fo.empty else grp.iloc[0]
                fi_row = fi.iloc[0] if not fi.empty else fo_row
                cn_long = f"{fo_row.get('orig_tc','')} - {fi_row.get('orig_tc','')}"
                translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':f"CTB-{route_num}",'translation':cn_long})
        if 'nlb_routes_df' in locals() and not nlb_routes_df.empty and 'routeName_c' in nlb_routes_df.columns:
            for r in nlb_routes_df[['routeNo','routeName_c']].dropna().itertuples():
                translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':f"NLB-{r.routeNo}",'translation':r.routeName_c.replace(' > ', ' - ')})
        if 'mtr_routes_df' in locals() and 'mtr_stations_gdf' in locals() and 'Chinese Name' in mtr_stations_gdf.columns:
            try:
                cn_pairs = {}
                for lc, grp in mtr_stations_gdf.groupby('Line Code'):
                    ordered = grp.sort_values('Sequence', key=lambda s: s.astype(int) if s.dtype==object else s)
                    if ordered.empty:
                        continue
                    first_cn = ordered.iloc[0]['Chinese Name']
                    last_cn = ordered.iloc[-1]['Chinese Name'] if len(ordered)>1 else first_cn
                    cn_pairs[lc] = f"{first_cn} - {last_cn}"
                for r in mtr_routes_df.itertuples():
                    translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':r.route_id,'translation':cn_pairs.get(r.route_short_name, r.route_long_name)})
            except Exception:
                pass
    except Exception as e:
        if not silent:
            print(f"Route translations warning: {e}")
    
    if translations:
        translations_df = pd.DataFrame(translations).drop_duplicates(subset=['table_name','field_name','language','record_id'])
        translations_df.to_csv(os.path.join(final_output_dir,'translations.txt'), index=False)
        if not silent:
            print(f"Generated translations.txt with {len(translations_df)} records.")
    else:
        if not silent:
            print("No translation records generated.")
    # --- 5. Zip the feed ---
    if not silent:
        print("Zipping the unified GTFS feed...")
    zip_path = os.path.join(output_dir, 'unified-agency-specific-stops.gtfs.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(final_output_dir):
            zf.write(os.path.join(final_output_dir, filename), arcname=filename)

    # i'll implament gtfs dense encoding later

    if not silent:
        print(f"--- Unified GTFS Build Complete. Output at {zip_path} ---")
