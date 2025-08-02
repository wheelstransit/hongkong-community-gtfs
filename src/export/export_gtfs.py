import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
import os
import zipfile
from src.processing.stop_unification import unify_stops_by_name_and_distance
from src.processing.stop_times import generate_stop_times_for_agency_optimized as generate_stop_times_for_agency
from src.processing.utils import get_direction

def export_unified_feed(engine: Engine, output_dir: str, journey_time_data: dict, silent: bool = False):
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

    # Combine all agencies
    all_stops_df = pd.concat([
        kmb_stops_final,
        ctb_stops_final,
        gmb_stops_final,
        mtrbus_stops_final,
        nlb_stops_final
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
    ctb_routes_df['route_id'] = 'CTB-' + ctb_routes_df['unique_route_id']
    ctb_routes_df['agency_id'] = 'CTB'
    ctb_routes_df['route_short_name'] = ctb_routes_df['route']
    ctb_routes_df['route_long_name'] = ctb_routes_df['orig_en'] + ' - ' + ctb_routes_df['dest_en']
    ctb_routes_df['route_type'] = 3
    ctb_routes_df['dir'] = ctb_routes_df['unique_route_id'].str.split('-').str[-1]
    final_ctb_routes = ctb_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].drop_duplicates(subset=['route_id']).copy()

    ctb_trips_list = []
    for _, route in ctb_routes_df.iterrows():
        route_short_name = route['route']
        direction_id = 1 if route['dir'] == 'inbound' else 0
        agency_id = 'CTB'
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
            matching_gov_services = ["DEFAULT"]
        for service_id in matching_gov_services:
            ctb_trips_list.append({
                'route_id': route['route_id'],
                'service_id': f"CTB-{route_short_name}-{service_id}",
                'trip_id': f"CTB-{route['unique_route_id']}-{service_id}",
                'direction_id': direction_id,
                'route_short_name': route_short_name,
                'original_service_id': service_id
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
    final_mtrbus_routes = mtrbus_routes_df.drop_duplicates(subset=['route_id'])
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
    nlb_routes_df['direction_id'] = nlb_routes_df['routeName_e'].apply(get_direction)

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
            matching_gov_services = ["DEFAULT"]

        for service_id in matching_gov_services:
            nlb_trips_list.append({
                'route_id': f"NLB-{route_short_name}",
                'service_id': f"NLB-{route_short_name}-{service_id}",
                'trip_id': f"NLB-{route_info['routeId']}-{service_id}",
                'direction_id': direction_id,
                'route_short_name': route_short_name,
                'original_service_id': service_id,
                'route_long_name': route_info['routeName_e']
            })
    nlb_trips_df = pd.DataFrame(nlb_trips_list)
    nlb_stoptimes_df = pd.read_sql("SELECT * FROM nlb_stop_sequences", engine)
    nlb_stoptimes_df['trip_id'] = 'NLB-' + nlb_stoptimes_df['routeId'].astype(str)
    nlb_stoptimes_df['stop_id'] = 'NLB-' + nlb_stoptimes_df['stopId'].astype(str)
    nlb_stoptimes_df['stop_id'] = nlb_stoptimes_df['stop_id'].replace(nlb_duplicates_map)
    nlb_stoptimes_df = nlb_stoptimes_df.merge(nlb_routes_df[['routeId', 'routeNo', 'direction_id', 'routeName_e']], on='routeId')

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

    kmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('KMB', na=False)]
    kmb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(kmb_gov_routes_df['route_id'])]
    kmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(kmb_gov_trips_df['trip_id'])]
    kmb_stoptimes_df = generate_stop_times_for_agency(
        'KMB', kmb_trips_df, kmb_stoptimes_df, kmb_gov_routes_df, kmb_gov_trips_df, kmb_gov_frequencies_df, journey_time_data, silent=silent
    )

    ctb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('CTB', na=False)]
    ctb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(ctb_gov_routes_df['route_id'])]
    ctb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(ctb_gov_trips_df['trip_id'])]
    ctb_stoptimes_df = generate_stop_times_for_agency(
        'CTB', ctb_trips_df, ctb_stoptimes_df, ctb_gov_routes_df, ctb_gov_trips_df, ctb_gov_frequencies_df, journey_time_data, silent=silent
    )

    gmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('GMB', na=False)]
    gmb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(gmb_gov_routes_df['route_id'])]
    gmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(gmb_gov_trips_df['trip_id'])]
    gmb_stoptimes_df = generate_stop_times_for_agency(
        'GMB', gmb_trips_df, gmb_stoptimes_df, gmb_gov_routes_df, gmb_gov_trips_df, gmb_gov_frequencies_df, journey_time_data, silent=silent
    )

    mtrbus_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('LRTFeeder', na=False)]
    mtrbus_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(mtrbus_gov_routes_df['route_id'])]
    mtrbus_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(mtrbus_gov_trips_df['trip_id'])]
    mtrbus_stoptimes_df = generate_stop_times_for_agency(
        'MTRB', mtrbus_trips_df, mtrbus_stoptimes_df, mtrbus_gov_routes_df, mtrbus_gov_trips_df, mtrbus_gov_frequencies_df, journey_time_data, silent=silent
    )

    nlb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('NLB', na=False)]
    nlb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(nlb_gov_routes_df['route_id'])]
    nlb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(nlb_gov_trips_df['trip_id'])]
    nlb_stoptimes_df = generate_stop_times_for_agency(
        'NLB', nlb_trips_df, nlb_stoptimes_df, nlb_gov_routes_df, nlb_gov_trips_df, nlb_gov_frequencies_df, journey_time_data, silent=silent
    )

    # -- Combine & Standardize--
    if not silent:
        print("Combining and standardizing data for final GTFS files...")
    final_routes_df = pd.concat([final_kmb_routes, final_ctb_routes, final_gmb_routes, final_mtrbus_routes, final_nlb_routes], ignore_index=True)
    final_trips_df = pd.concat([kmb_trips_df, ctb_trips_df, gmb_trips_df, mtrbus_trips_df, nlb_trips_df], ignore_index=True)

    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
    final_kmb_stoptimes = kmb_stoptimes_df.rename(columns={'seq': 'stop_sequence'})[stop_times_cols]
    final_ctb_stoptimes = ctb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_gmb_stoptimes = gmb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtrbus_stoptimes = mtrbus_stoptimes_df.rename(columns={'station_seqno': 'stop_sequence'})
    final_nlb_stoptimes = nlb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_stop_times_df = pd.concat([final_kmb_stoptimes, final_ctb_stoptimes, final_gmb_stoptimes, final_mtrbus_stoptimes, final_nlb_stoptimes], ignore_index=True)

    final_routes_df.to_csv(os.path.join(final_output_dir, 'routes.txt'), index=False)
    final_trips_df.to_csv(os.path.join(final_output_dir, 'trips.txt'), index=False)
    final_stop_times_df.to_csv(os.path.join(final_output_dir, 'stop_times.txt'), index=False)
    gov_frequencies_df.to_csv(os.path.join(final_output_dir, 'frequencies.txt'), index=False)

    # --- 4. Handle `calendar.txt` and `calendar_dates.txt` ---
    if not silent:
        print("Processing calendar and calendar_dates...")
    service_id_map = final_trips_df[final_trips_df['original_service_id'].notna()].set_index('service_id')['original_service_id'].to_dict()
    gov_calendar_df = pd.read_sql("SELECT * FROM gov_gtfs_calendar", engine)
    gov_calendar_dates_df = pd.read_sql("SELECT * FROM gov_gtfs_calendar_dates", engine)
    mapped_gov_service_ids = set(service_id_map.values())
    final_calendar_df = gov_calendar_df[gov_calendar_df['service_id'].isin(mapped_gov_service_ids)].copy()
    final_calendar_dates_df = gov_calendar_dates_df[gov_calendar_dates_df['service_id'].isin(mapped_gov_service_ids)].copy()

    reverse_service_id_map = {}
    for new_id, orig_id in service_id_map.items():
        if orig_id not in reverse_service_id_map:
            reverse_service_id_map[orig_id] = []
        reverse_service_id_map[orig_id].append(new_id)

    new_calendar_rows = []
    final_calendar_df['service_id'] = final_calendar_df['service_id'].astype(str)
    for _, row in final_calendar_df.iterrows():
        original_id = str(row['service_id'])
        if original_id in reverse_service_id_map:
            for new_id in reverse_service_id_map[original_id]:
                new_row = row.copy()
                new_row['service_id'] = new_id
                new_calendar_rows.append(new_row)

    new_calendar_dates_rows = []
    final_calendar_dates_df['service_id'] = final_calendar_dates_df['service_id'].astype(str)
    for _, row in final_calendar_dates_df.iterrows():
        original_id = str(row['service_id'])
        if original_id in reverse_service_id_map:
            for new_id in reverse_service_id_map[original_id]:
                new_row = row.copy()
                new_row['service_id'] = new_id
                new_calendar_dates_rows.append(new_row)

    if new_calendar_rows:
        final_calendar_df = pd.DataFrame(new_calendar_rows)
    else:
        final_calendar_df = pd.DataFrame(columns=gov_calendar_df.columns)

    if new_calendar_dates_rows:
        final_calendar_dates_df = pd.DataFrame(new_calendar_dates_rows)
    else:
        final_calendar_dates_df = pd.DataFrame(columns=gov_calendar_dates_df.columns)

    final_calendar_df.to_csv(os.path.join(final_output_dir, 'calendar.txt'), index=False)
    final_calendar_dates_df.to_csv(os.path.join(final_output_dir, 'calendar_dates.txt'), index=False)

    # --- 5. Zip the feed ---
    if not silent:
        print("Zipping the unified GTFS feed...")
    zip_path = os.path.join(output_dir, 'unified-agency-specific-stops.gtfs.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(final_output_dir):
            zf.write(os.path.join(final_output_dir, filename), arcname=filename)

    if not silent:
        print(f"--- Unified GTFS Build Complete. Output at {zip_path} ---")
