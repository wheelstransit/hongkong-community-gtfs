import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
import os
import zipfile
from src.processing.stop_unification import unify_stops_by_name_and_distance
from src.processing.stop_times import generate_stop_times_for_agency_optimized as generate_stop_times_for_agency
from src.processing.utils import get_direction

def export_unified_feed(engine: Engine, output_dir: str, journey_time_data: dict, mtr_headway_data: dict, osm_subway_entrances_data: dict, silent: bool = False):
    if not silent:
        print("--- Starting Unified GTFS Export Process ---")

    final_output_dir = os.path.join(output_dir, "unified_feed")
    os.makedirs(final_output_dir, exist_ok=True)

    if not silent:
        print("Building agency.txt...")
    agencies = [
        {'agency_id': 'KMB', 'agency_name': 'Kowloon Motor Bus', 'agency_url': 'https://kmb.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'CTB', 'agency_name': 'Citybus', 'agency_url': 'https://www.citybus.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'MTRB', 'agency_name': 'MTR Bus', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'GMB', 'agency_name': 'Green Minibus', 'agency_url': 'https://td.gov.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'NLB', 'agency_name': 'New Lantao Bus', 'agency_url': 'https://www.nlb.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'MTRR', 'agency_name': 'MTR Rail', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'LR', 'agency_name': 'Light Rail', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'AE', 'agency_name': 'Airport Express', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'PT', 'agency_name': 'Peak Tram', 'agency_url': 'https://thepeak.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'TRAM', 'agency_name': 'Tramways', 'agency_url': 'https://www.hktramways.com', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
        {'agency_id': 'FERRY', 'agency_name': 'Ferry', 'agency_url': 'https://td.gov.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'zh-Hant'},
    ]
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
    kmb_stops_gdf, kmb_duplicates_map = unify_stops_by_name_and_distance(kmb_stops_gdf, 'stop_name', 'stop_id', silent=silent)
    kmb_stops_gdf['stop_lat'] = kmb_stops_gdf.geometry.y
    kmb_stops_gdf['stop_lon'] = kmb_stops_gdf.geometry.x
    kmb_stops_final = kmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # Citybus
    ctb_stops_gdf = gpd.read_postgis("SELECT * FROM citybus_stops", engine, geom_col='geometry')
    ctb_stops_gdf['stop_id'] = 'CTB-' + ctb_stops_gdf['stop'].astype(str)
    ctb_stops_gdf['stop_name'] = ctb_stops_gdf['name_en']
    ctb_stops_gdf, ctb_duplicates_map = unify_stops_by_name_and_distance(ctb_stops_gdf, 'stop_name', 'stop_id', silent=silent)
    ctb_stops_gdf['stop_lat'] = ctb_stops_gdf.geometry.y
    ctb_stops_gdf['stop_lon'] = ctb_stops_gdf.geometry.x
    ctb_stops_final = ctb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # GMB
    gmb_stops_gdf = gpd.read_postgis("SELECT * FROM gmb_stops", engine, geom_col='geometry')
    gmb_stops_gdf['stop_id'] = 'GMB-' + gmb_stops_gdf['stop_id'].astype(str)
    gmb_stops_gdf['stop_name'] = gmb_stops_gdf['stop_name_en']
    gmb_stops_gdf, gmb_duplicates_map = unify_stops_by_name_and_distance(gmb_stops_gdf, 'stop_name', 'stop_id', silent=silent)
    gmb_stops_gdf['stop_lat'] = gmb_stops_gdf.geometry.y
    gmb_stops_gdf['stop_lon'] = gmb_stops_gdf.geometry.x
    gmb_stops_final = gmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # MTR Bus
    mtrbus_stops_gdf = gpd.read_postgis("SELECT * FROM mtrbus_stops", engine, geom_col='geometry')
    mtrbus_stops_gdf['stop_id'] = 'MTRB-' + mtrbus_stops_gdf['stop_id'].astype(str)
    mtrbus_stops_gdf['stop_name'] = mtrbus_stops_gdf['name_en']
    mtrbus_stops_gdf, mtrbus_duplicates_map = unify_stops_by_name_and_distance(mtrbus_stops_gdf, 'stop_name', 'stop_id', silent=silent)
    mtrbus_stops_gdf['stop_lat'] = mtrbus_stops_gdf.geometry.y
    mtrbus_stops_gdf['stop_lon'] = mtrbus_stops_gdf.geometry.x
    mtrbus_stops_final = mtrbus_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # NLB
    nlb_stops_gdf = gpd.read_postgis("SELECT * FROM nlb_stops", engine, geom_col='geometry')
    nlb_stops_gdf['stop_id'] = 'NLB-' + nlb_stops_gdf['stopId'].astype(str)
    nlb_stops_gdf['stop_name'] = nlb_stops_gdf['stopName_e']
    nlb_stops_gdf, nlb_duplicates_map = unify_stops_by_name_and_distance(nlb_stops_gdf, 'stop_name', 'stop_id', silent=silent)
    nlb_stops_gdf['stop_lat'] = nlb_stops_gdf.geometry.y
    nlb_stops_gdf['stop_lon'] = nlb_stops_gdf.geometry.x
    nlb_stops_final = nlb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # MTR Rails
    mtr_lines_and_stations_df = pd.read_sql("SELECT * FROM mtr_lines_and_stations", engine)
    mtr_stops_df = mtr_lines_and_stations_df[['Station ID', 'English Name']].drop_duplicates()
    mtr_stops_df.rename(columns={'Station ID': 'stop_id', 'English Name': 'stop_name'}, inplace=True)
    mtr_stops_df['stop_id'] = 'MTR-' + mtr_stops_df['stop_id'].astype(str)
    mtr_stops_df['location_type'] = 1
    mtr_stops_df['parent_station'] = None

    # MTR Entrances
    mtr_entrances = []
    for element in osm_subway_entrances_data.get('elements', []):
        if element['type'] == 'node':
            mtr_entrances.append({
                'stop_id': f"MTR-ENTRANCE-{element['id']}",
                'stop_name': element.get('tags', {}).get('name', 'Subway Entrance'),
                'stop_lat': element['lat'],
                'stop_lon': element['lon'],
                'location_type': 2,
                'parent_station': None # To be filled later
            })
    mtr_entrances_df = pd.DataFrame(mtr_entrances)

    # Light Rail
    light_rail_routes_and_stops_df = pd.read_sql("SELECT * FROM light_rail_routes_and_stops", engine)
    lr_stops_df = light_rail_routes_and_stops_df[['Stop ID', 'English Name']].drop_duplicates()
    lr_stops_df.rename(columns={'Stop ID': 'stop_id', 'English Name': 'stop_name'}, inplace=True)
    lr_stops_df['stop_id'] = 'LR-' + lr_stops_df['stop_id'].astype(str)
    lr_stops_df['location_type'] = 0
    lr_stops_df['parent_station'] = None

    # Combine all agencies
    all_stops_df = pd.concat([
        kmb_stops_final,
        ctb_stops_final,
        gmb_stops_final,
        mtrbus_stops_final,
        nlb_stops_final,
        mtr_stops_df,
        mtr_entrances_df,
        lr_stops_df
    ], ignore_index=True)
    all_stops_df.to_csv(os.path.join(final_output_dir, 'stops.txt'), index=False)
    if not silent:
        print(f"Generated stops.txt with {len(all_stops_df)} total stops.")

    # --- 3. Build `routes.txt`, `trips.txt`, `stop_times.txt` ---
    gov_routes_df = pd.read_sql("SELECT * FROM gov_gtfs_routes", engine)
    gov_trips_df = pd.read_sql("SELECT * FROM gov_gtfs_trips", engine)
    gov_frequencies_df = pd.read_sql("SELECT * FROM gov_gtfs_frequencies", engine)
    try:
        parsed_direction = gov_trips_df['trip_id'].str.split('-').str[1].astype(int)
        gov_trips_df['direction_id'] = parsed_direction - 1
        if not silent:
            print("Successfully parsed 'direction_id' from government trip_id.")
    except (IndexError, ValueError, TypeError):
        if not silent:
            print("Warning: Could not parse 'direction_id' from government trip_id.")
        gov_trips_df['direction_id'] = -1

    # Standardize data types before merge
    gov_trips_df['service_id'] = gov_trips_df['service_id'].astype(str)
    gov_routes_df['route_short_name'] = gov_routes_df['route_short_name'].astype(str)

    gov_trips_with_route_info = gov_trips_df.merge(
        gov_routes_df[['route_id', 'route_short_name', 'agency_id']], on='route_id'
    )

    # -- KMB --
    if not silent:
        print("Processing KMB routes, trips, and stop_times...")
    kmb_routes_df = pd.read_sql("SELECT * FROM kmb_routes", engine)
    kmb_routes_df[['route', 'bound', 'service_type']] = kmb_routes_df['unique_route_id'].str.split('_', expand=True)
    kmb_routes_df['direction_id'] = kmb_routes_df['bound'].map({'O': 0, 'I': 1}).fillna(-1).astype(int)
    final_kmb_routes_list = []
    for route_num, group in kmb_routes_df.groupby('route'):
        first_outbound = group[group['bound'] == 'O'].iloc[0] if not group[group['bound'] == 'O'].empty else group.iloc[0]
        final_kmb_routes_list.append({
            'route_id': f"KMB-{route_num}",
            'agency_id': 'KMB',
            'route_short_name': route_num,
            'route_long_name': f"{first_outbound['orig_en']} - {first_outbound['dest_en']}",
            'route_type': 3
        })
    final_kmb_routes = pd.DataFrame(final_kmb_routes_list)

    kmb_trips_list = []
    for _, route in kmb_routes_df.drop_duplicates(subset=['route', 'bound']).iterrows():
        route_short_name = route['route']
        direction_id = route['direction_id']
        bound = route['bound']
        agency_id = 'KMB'
        matching_gov_services = gov_trips_with_route_info[
            (gov_trips_with_route_info['agency_id'].str.contains('KMB|LWB', na=False)) &
            (gov_trips_with_route_info['route_short_name'] == route_short_name) &
            (gov_trips_with_route_info['direction_id'] == direction_id)
        ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'] == route_short_name)
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'].str.startswith(route_short_name, na=False))
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = [f"DEFAULT_{route['service_type']}"]
        for service_id in matching_gov_services:
            kmb_trips_list.append({
                'route_id': f"KMB-{route_short_name}",
                'service_id': f"KMB-{route_short_name}-{service_id}",
                'trip_id': f"KMB-{route_short_name}-{bound}-{service_id}",
                'direction_id': direction_id,
                'bound': bound,
                'route_short_name': route_short_name,
                'original_service_id': service_id
            })
    kmb_trips_df = pd.DataFrame(kmb_trips_list)
    kmb_stoptimes_df = pd.read_sql("SELECT * FROM kmb_stop_sequences", engine)
    kmb_stoptimes_df[['route', 'bound', 'service_type']] = kmb_stoptimes_df['unique_route_id'].str.split('_', expand=True)
    kmb_stoptimes_df['stop_id'] = 'KMB-' + kmb_stoptimes_df['stop'].astype(str)
    kmb_stoptimes_df['stop_id'] = kmb_stoptimes_df['stop_id'].replace(kmb_duplicates_map)

    # -- Citybus --
    if not silent:
        print("Processing Citybus routes, trips, and stop_times...")
    ctb_routes_df = pd.read_sql("SELECT * FROM citybus_routes", engine)
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

    ctb_trips_list = []
    for _, route in ctb_routes_df.iterrows():
        route_short_name = route['route']
        direction_id = 1 if route['dir'] == 'inbound' else 0
        agency_id = 'CTB'
        matching_gov_services = gov_trips_with_route_info[
            (gov_trips_with_route_info['agency_id'].str.contains('CTB|NWFB', na=False)) &
            (gov_trips_with_route_info['route_short_name'] == route_short_name) &
            (gov_trips_with_route_info['direction_id'] == direction_id)
        ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'] == route_short_name)
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'].str.startswith(route_short_name, na=False))
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = ["DEFAULT"]
        for service_id in matching_gov_services:
            ctb_trips_list.append({
                'route_id': f"CTB-{route_short_name}",
                'service_id': f"CTB-{route_short_name}-{service_id}",
                'trip_id': f"CTB-{route['unique_route_id']}-{service_id}",
                'direction_id': direction_id,
                'route_short_name': route_short_name,
                'original_service_id': service_id,
                'unique_route_id': route['unique_route_id']
            })
    ctb_trips_df = pd.DataFrame(ctb_trips_list)
    ctb_stoptimes_df = pd.read_sql("SELECT * FROM citybus_stop_sequences", engine)
    ctb_stoptimes_df['stop_id'] = 'CTB-' + ctb_stoptimes_df['stop_id'].astype(str)
    ctb_stoptimes_df['stop_id'] = ctb_stoptimes_df['stop_id'].replace(ctb_duplicates_map)

    # -- GMB --
    if not silent:
        print("Processing GMB routes, trips, and stop_times...")
    gmb_routes_base_df = pd.read_sql("SELECT * FROM gmb_routes", engine)
    gmb_routes_base_df['agency_id'] = 'GMB'
    gmb_routes_base_df['route_type'] = 3
    gmb_routes_base_df['route_id'] = 'GMB-' + gmb_routes_base_df['route_code']
    gmb_routes_base_df['route_short_name'] = gmb_routes_base_df['route_code']
    gmb_routes_base_df['route_long_name'] = gmb_routes_base_df['region'] + ' - ' + gmb_routes_base_df['route_code']
    final_gmb_routes = gmb_routes_base_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()
    gmb_stoptimes_df = pd.read_sql("SELECT * FROM gmb_stop_sequences", engine)
    gmb_trips_source = gmb_stoptimes_df[['route_code', 'route_seq']].drop_duplicates()
    gmb_trips_source['route_seq'] = pd.to_numeric(gmb_trips_source['route_seq'])

    gmb_trips_list = []
    for _, trip_info in gmb_trips_source.iterrows():
        route_short_name = trip_info['route_code']
        direction_id = trip_info['route_seq'] - 1
        agency_id = 'GMB'

        matching_gov_services = gov_trips_with_route_info[
            (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
            (gov_trips_with_route_info['route_short_name'] == route_short_name) &
            (gov_trips_with_route_info['direction_id'] == direction_id)
        ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'] == route_short_name)
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'].str.startswith(route_short_name, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = ["DEFAULT"]

        for service_id in matching_gov_services:
            gmb_trips_list.append({
                'route_id': f"GMB-{route_short_name}",
                'service_id': f"GMB-{route_short_name}-{service_id}",
                'trip_id': f"GMB-{route_short_name}-{trip_info['route_seq']}-{service_id}",
                'direction_id': direction_id,
                'route_short_name': route_short_name,
                'original_service_id': service_id
            })
    gmb_trips_df = pd.DataFrame(gmb_trips_list)
    gmb_stoptimes_df['trip_id'] = 'GMB-' + gmb_stoptimes_df['region'] + '-' + gmb_stoptimes_df['route_code'] + '-' + gmb_stoptimes_df['route_seq'].astype(str)
    gmb_stoptimes_df['stop_id'] = 'GMB-' + gmb_stoptimes_df['stop_id'].astype(str)
    gmb_stoptimes_df['stop_id'] = gmb_stoptimes_df['stop_id'].replace(gmb_duplicates_map)

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
    mtrbus_trips_source = pd.read_sql("SELECT DISTINCT route_id, direction FROM mtrbus_stop_sequences", engine)

    mtrbus_trips_list = []
    for _, trip_info in mtrbus_trips_source.iterrows():
        route_short_name = trip_info['route_id']
        agency_id = 'LRTFeeder'
        direction_id = 0 if trip_info['direction'] == 'O' else 1

        matching_gov_services = gov_trips_with_route_info[
            (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
            (gov_trips_with_route_info['route_short_name'] == route_short_name) &
            (gov_trips_with_route_info['direction_id'] == direction_id)
        ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'] == route_short_name)
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'].str.startswith(route_short_name, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = ["DEFAULT"]

        for service_id in matching_gov_services:
            mtrbus_trips_list.append({
                'route_id': f"MTRB-{route_short_name}",
                'service_id': f"MTRB-{route_short_name}-{service_id}",
                'trip_id': f"MTRB-{route_short_name}-{trip_info['direction']}-{service_id}",
                'direction_id': direction_id,
                'route_short_name': route_short_name,
                'original_service_id': service_id
            })
    mtrbus_trips_df = pd.DataFrame(mtrbus_trips_list)
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

    nlb_trips_list = []
    for _, route_info in nlb_routes_df.iterrows():
        route_short_name = route_info['routeNo']
        direction_id = route_info['direction_id']
        agency_id = 'NLB'

        matching_gov_services = gov_trips_with_route_info[
            (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
            (gov_trips_with_route_info['route_short_name'] == route_short_name) &
            (gov_trips_with_route_info['direction_id'] == direction_id)
        ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'] == route_short_name)
            ]['service_id'].unique()
        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False)) &
                (gov_trips_with_route_info['route_short_name'].str.startswith(route_short_name, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = gov_trips_with_route_info[
                (gov_trips_with_route_info['agency_id'].str.contains(agency_id, na=False))
            ]['service_id'].unique()

        if len(matching_gov_services) == 0:
            matching_gov_services = ["DEFAULT"]

        for service_id in matching_gov_services:
            nlb_trips_list.append({
                'route_id': f"NLB-{route_short_name}",
                'service_id': f"NLB-{route_short_name}-{service_id}",
                'trip_id': f"NLB-{route_info['routeId']}-{service_id}",
                'direction_id': direction_id,
                'route_short_name': route_short_name,
                'original_service_id': service_id,
                'route_long_name': route_info['routeName_e'],
                'routeId': route_info['routeId']
            })
    nlb_trips_df = pd.DataFrame(nlb_trips_list)
    nlb_stoptimes_df = pd.read_sql("SELECT * FROM nlb_stop_sequences", engine)
    nlb_stoptimes_df['trip_id'] = 'NLB-' + nlb_stoptimes_df['routeId'].astype(str)
    nlb_stoptimes_df['stop_id'] = 'NLB-' + nlb_stoptimes_df['stopId'].astype(str)
    nlb_stoptimes_df['stop_id'] = nlb_stoptimes_df['stop_id'].replace(nlb_duplicates_map)
    nlb_stoptimes_df = nlb_stoptimes_df.merge(nlb_routes_df[['routeId', 'routeNo', 'direction_id', 'routeName_e']], on='routeId')

    # -- MTR Rail --
    if not silent:
        print("Processing MTR Rail routes, trips, and stop_times...")
    mtr_lines_and_stations_df = pd.read_sql("SELECT * FROM mtr_lines_and_stations", engine)
    mtr_routes_df = mtr_lines_and_stations_df[['Line Code', 'English Name']].drop_duplicates(subset=['Line Code'])
    mtr_routes_df.rename(columns={'Line Code': 'route_id', 'English Name': 'route_long_name'}, inplace=True)
    mtr_routes_df['route_id'] = 'MTR-' + mtr_routes_df['route_id']
    mtr_routes_df['agency_id'] = 'MTRR'
    mtr_routes_df['route_short_name'] = mtr_routes_df['route_id'].str.replace('MTR-', '')
    mtr_routes_df['route_type'] = 1 # Subway

    mtr_trips_list = []
    for _, route in mtr_routes_df.iterrows():
        mtr_trips_list.append({
            'route_id': route['route_id'],
            'service_id': f"{route['route_id']}-SERVICE",
            'trip_id': f"{route['route_id']}-TRIP",
            'direction_id': 0
        })
        mtr_trips_list.append({
            'route_id': route['route_id'],
            'service_id': f"{route['route_id']}-SERVICE",
            'trip_id': f"{route['route_id']}-TRIP-2",
            'direction_id': 1
        })
    mtr_trips_df = pd.DataFrame(mtr_trips_list)

    mtr_stoptimes_df = mtr_lines_and_stations_df.copy()
    mtr_stoptimes_df['trip_id'] = 'MTR-' + mtr_stoptimes_df['Line Code'] + '-TRIP'
    mtr_stoptimes_df.loc[mtr_stoptimes_df['Direction'] == 'D', 'trip_id'] = 'MTR-' + mtr_stoptimes_df['Line Code'] + '-TRIP-2'
    mtr_stoptimes_df['stop_id'] = 'MTR-' + mtr_stoptimes_df['Station ID'].astype(str)
    mtr_stoptimes_df.rename(columns={'Sequence': 'stop_sequence'}, inplace=True)
    mtr_stoptimes_df['arrival_time'] = None
    mtr_stoptimes_df['departure_time'] = None

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
            'service_id': f"{route['route_id']}-SERVICE",
            'trip_id': f"{route['route_id']}-TRIP",
            'direction_id': 0
        })
        lr_trips_list.append({
            'route_id': route['route_id'],
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
    lr_stoptimes_df['arrival_time'] = None
    lr_stoptimes_df['departure_time'] = None

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

    # -- Combine & Standardize--
    if not silent:
        print("Combining and standardizing data for final GTFS files...")
    final_routes_df = pd.concat([final_kmb_routes, final_ctb_routes, final_gmb_routes, final_mtrbus_routes, final_nlb_routes, mtr_routes_df, lr_routes_df], ignore_index=True)
    all_trips_df = pd.concat([kmb_trips_df, ctb_trips_df, gmb_trips_df, mtrbus_trips_df, nlb_trips_df, mtr_trips_df, lr_trips_df], ignore_index=True)

    # Create a mapping from the original government trip_id to our new trip_id
    trip_id_mapping = all_trips_df.merge(
        gov_trips_with_route_info,
        left_on=['original_service_id', 'route_short_name', 'direction_id'],
        right_on=['service_id', 'route_short_name', 'direction_id']
    )[['trip_id_x', 'trip_id_y']].rename(columns={'trip_id_x': 'new_trip_id', 'trip_id_y': 'original_trip_id'})

    # Update the trip_id in the frequencies dataframe
    final_frequencies_df = gov_frequencies_df.merge(trip_id_mapping, left_on='trip_id', right_on='original_trip_id', how='left')
    final_frequencies_df['trip_id'] = final_frequencies_df['new_trip_id'].fillna(final_frequencies_df['trip_id'])
    final_frequencies_df = final_frequencies_df.drop(columns=['new_trip_id', 'original_trip_id'])

    # For trips with frequency info, keep only one representative trip per service
    final_trips_df = all_trips_df.drop_duplicates(subset=['service_id'])

    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
    final_kmb_stoptimes = kmb_stoptimes_df.rename(columns={'seq': 'stop_sequence'})
    final_ctb_stoptimes = ctb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_gmb_stoptimes = gmb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtrbus_stoptimes = mtrbus_stoptimes_df.rename(columns={'station_seqno': 'stop_sequence'})
    final_nlb_stoptimes = nlb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtr_stoptimes = mtr_stoptimes_df
    final_lr_stoptimes = lr_stoptimes_df
    final_stop_times_df = pd.concat([final_kmb_stoptimes, final_ctb_stoptimes, final_gmb_stoptimes, final_mtrbus_stoptimes, final_nlb_stoptimes, final_mtr_stoptimes, final_lr_stoptimes], ignore_index=True)

    final_routes_df.to_csv(os.path.join(final_output_dir, 'routes.txt'), index=False)
    final_trips_df.to_csv(os.path.join(final_output_dir, 'trips.txt'), index=False)
    final_stop_times_df.to_csv(os.path.join(final_output_dir, 'stop_times.txt'), index=False)
    final_frequencies_df.to_csv(os.path.join(final_output_dir, 'frequencies.txt'), index=False)

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
    # This will create a new row for each new service_id that shares an original_service_id
    final_calendar_df = service_id_mapping_df.merge(
        gov_calendar_df,
        left_on='original_service_id',
        right_index=True,
        how='inner'
    )
    # Now, we just need to select the columns for the final calendar.txt, using the new service_id

    final_calendar_df = final_calendar_df.drop(columns=['original_service_id'])

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