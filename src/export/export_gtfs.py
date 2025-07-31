import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
import os
import zipfile
from src.processing.stop_unification import unify_stops_by_name_and_distance
from src.processing.stop_times import generate_stop_times_for_agency_optimized as generate_stop_times_for_agency

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
    # This process will be repeated for each agency, then concatenated.

    # -- KMB --
    if not silent:
        print("Processing KMB routes, trips, and stop_times...")
    kmb_routes_df = pd.read_sql("SELECT * FROM kmb_routes", engine)

    # --- Parse unique_route_id into route, bound, service_type ---
    # unique_route_id format: {route}_{bound}_{service_type}
    kmb_routes_df[['route', 'bound', 'service_type']] = kmb_routes_df['unique_route_id'].str.split('_', expand=True)

    # Map bound to direction_id (O=0, I=1, fallback -1)
    kmb_routes_df['direction_id'] = kmb_routes_df['bound'].map({'O': 0, 'I': 1}).fillna(-1).astype(int)

    # Compose route_id and trip_id with all components for uniqueness
    kmb_routes_df['route_id'] = 'KMB-' + kmb_routes_df['route'] + '-' + kmb_routes_df['bound'] + '-' + kmb_routes_df['service_type']
    kmb_routes_df['agency_id'] = 'KMB'
    kmb_routes_df['route_short_name'] = kmb_routes_df['route']
    kmb_routes_df['route_long_name'] = kmb_routes_df['orig_en'] + ' - ' + kmb_routes_df['dest_en']
    kmb_routes_df['route_type'] = 3

    # Add service_type as a custom field in routes.txt
    # Build final routes.txt DataFrame
    final_kmb_routes = kmb_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type', 'service_type']].copy()

    # --- Build trips.txt ---
    # Each trip is uniquely identified by route_id, direction_id, and service_type
    kmb_trips_df = pd.DataFrame({
        'route_id': kmb_routes_df['route_id'],
        'service_id': 'KMB-' + kmb_routes_df['route_id'],
        'trip_id': kmb_routes_df['route_id'],  # For simplicity, trip_id = route_id
        'direction_id': kmb_routes_df['direction_id'],
        'service_type': kmb_routes_df['service_type'],
        'route_short_name': kmb_routes_df['route_short_name'],
    })

    # --- Build stop_times.txt ---
    kmb_stoptimes_df = pd.read_sql("SELECT * FROM kmb_stop_sequences", engine)
    # Parse unique_route_id in stop_times as well
    kmb_stoptimes_df[['route', 'bound', 'service_type']] = kmb_stoptimes_df['unique_route_id'].str.split('_', expand=True)
    kmb_stoptimes_df['trip_id'] = 'KMB-' + kmb_stoptimes_df['route'] + '-' + kmb_stoptimes_df['bound'] + '-' + kmb_stoptimes_df['service_type']
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
    final_ctb_routes = ctb_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()

    ctb_trips_df = pd.DataFrame({
        'route_id': ctb_routes_df['route_id'],
        'service_id': 'CTB-' + ctb_routes_df['route_id'],
        'trip_id': ctb_routes_df['route_id'],
        'direction_id': ctb_routes_df['dir'].map({'outbound': 0, 'inbound': 1}).fillna(-1).astype(int),
        'route_short_name': ctb_routes_df['route']
    })

    ctb_stoptimes_df = pd.read_sql("SELECT * FROM citybus_stop_sequences", engine)
    ctb_stoptimes_df['trip_id'] = 'CTB-' + ctb_stoptimes_df['unique_route_id']
    ctb_stoptimes_df['stop_id'] = 'CTB-' + ctb_stoptimes_df['stop_id'].astype(str)
    ctb_stoptimes_df['stop_id'] = ctb_stoptimes_df['stop_id'].replace(ctb_duplicates_map)

    # -- GMB --
    if not silent:
        print("Processing GMB routes, trips, and stop_times...")
    gmb_routes_df = pd.read_sql("SELECT * FROM gmb_routes", engine)
    gmb_routes_df['route_id'] = 'GMB-' + gmb_routes_df['unique_route_id']
    gmb_routes_df['agency_id'] = 'GMB'
    gmb_routes_df['route_short_name'] = gmb_routes_df['route_code']
    gmb_routes_df['route_long_name'] = gmb_routes_df['region'] + ' - ' + gmb_routes_df['route_code']
    gmb_routes_df['route_type'] = 3
    final_gmb_routes = gmb_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()

    gmb_trips_df = pd.DataFrame({
        'route_id': gmb_routes_df['route_id'],
        'service_id': 'GMB-' + gmb_routes_df['route_id'],
        'trip_id': gmb_routes_df['route_id'],
        'direction_id': 0,  # Placeholder, GMB data does not have direction
        'route_short_name': gmb_routes_df['route_code']
    })

    gmb_stoptimes_df = pd.read_sql("SELECT * FROM gmb_stop_sequences", engine)
    gmb_stoptimes_df['trip_id'] = 'GMB-' + gmb_stoptimes_df['route_id'].astype(str) + '-' + gmb_stoptimes_df['route_seq'].astype(str)
    gmb_stoptimes_df['stop_id'] = 'GMB-' + gmb_stoptimes_df['stop_id'].astype(str)
    gmb_stoptimes_df['stop_id'] = gmb_stoptimes_df['stop_id'].replace(gmb_duplicates_map)

    print("GMB Trips DF")
    print(gmb_trips_df.head())
    print("GMB Stoptimes DF")
    print(gmb_stoptimes_df.head())

    # -- MTR Bus --
    if not silent:
        print("Processing MTR Bus routes, trips, and stop_times...")
    mtrbus_routes_df = pd.read_sql("SELECT * FROM mtrbus_routes", engine)
    mtrbus_stop_sequences_df = pd.read_sql("SELECT * FROM mtrbus_stop_sequences", engine)
    mtrbus_routes_df = mtrbus_routes_df.merge(mtrbus_stop_sequences_df[['route_id', 'direction']].drop_duplicates(), on='route_id')
    mtrbus_routes_df['unique_route_id'] = mtrbus_routes_df['route_id']
    mtrbus_routes_df['route_id'] = 'MTRB-' + mtrbus_routes_df['unique_route_id'] + '-' + mtrbus_routes_df['direction']
    mtrbus_routes_df['agency_id'] = 'MTRB'
    mtrbus_routes_df['route_short_name'] = mtrbus_routes_df['unique_route_id']
    mtrbus_routes_df['route_long_name'] = mtrbus_routes_df['route_name_eng']
    mtrbus_routes_df['route_type'] = 3
    final_mtrbus_routes = mtrbus_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()

    mtrbus_trips_df = pd.DataFrame({
        'route_id': mtrbus_routes_df['route_id'],
        'service_id': 'MTRB-' + mtrbus_routes_df['route_id'],
        'trip_id': mtrbus_routes_df['route_id'],
        'route_short_name': mtrbus_routes_df['route_short_name'],
        'direction_id': mtrbus_routes_df['direction'].map({'O': 0, 'I': 1}).fillna(-1).astype(int)
    })

    mtrbus_stoptimes_df = pd.read_sql("SELECT * FROM mtrbus_stop_sequences", engine)
    mtrbus_stoptimes_df['trip_id'] = 'MTRB-' + mtrbus_stoptimes_df['route_id'] + '-' + mtrbus_stoptimes_df['direction']
    mtrbus_stoptimes_df['stop_id'] = 'MTRB-' + mtrbus_stoptimes_df['stop_id'].astype(str)
    mtrbus_stoptimes_df['stop_id'] = mtrbus_stoptimes_df['stop_id'].replace(mtrbus_duplicates_map)

    # -- NLB --
    if not silent:
        print("Processing NLB routes, trips, and stop_times...")
    nlb_routes_df = pd.read_sql("SELECT * FROM nlb_routes", engine)
    nlb_routes_df['route_id'] = 'NLB-' + nlb_routes_df['routeId'].astype(str)
    nlb_routes_df['agency_id'] = 'NLB'
    nlb_routes_df['route_short_name'] = nlb_routes_df['routeNo']
    nlb_routes_df['route_long_name'] = nlb_routes_df['routeName_e']
    nlb_routes_df['route_type'] = 3
    final_nlb_routes = nlb_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()

    nlb_trips_df = pd.DataFrame({
        'route_id': nlb_routes_df['route_id'],
        'service_id': 'NLB-' + nlb_routes_df['route_id'],
        'trip_id': nlb_routes_df['route_id'],
        'direction_id': 0, # Placeholder, NLB data does not have direction
        'route_short_name': nlb_routes_df['routeNo']
    })

    nlb_stoptimes_df = pd.read_sql("SELECT * FROM nlb_stop_sequences", engine)
    nlb_stoptimes_df['trip_id'] = 'NLB-' + nlb_stoptimes_df['routeId'].astype(str)
    nlb_stoptimes_df['stop_id'] = 'NLB-' + nlb_stoptimes_df['stopId'].astype(str)
    nlb_stoptimes_df['stop_id'] = nlb_stoptimes_df['stop_id'].replace(nlb_duplicates_map)

    # -- KMB Frequency --
    gov_routes_df = pd.read_sql("SELECT * FROM gov_gtfs_routes", engine)
    gov_trips_df = pd.read_sql("SELECT * FROM gov_gtfs_trips", engine)
    gov_frequencies_df = pd.read_sql("SELECT * FROM gov_gtfs_frequencies", engine)

    try:
        # The direction in the gov data seems to be 1-based (1 for outbound, 2 for inbound)
        # GTFS standard is 0-based (0 for outbound, 1 for inbound). We convert it.
        parsed_direction = gov_trips_df['trip_id'].str.split('-').str[1].astype(int)
        gov_trips_df['direction_id'] = parsed_direction - 1
        print("Successfully parsed 'direction_id' from government trip_id.")
    except (IndexError, ValueError, TypeError):
        print("Warning: Could not parse 'direction_id' from government trip_id. Stop times for KMB may be incorrect.")
        gov_trips_df['direction_id'] = -1 # Add placeholder to prevent KeyError

    kmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'] == 'KMB']
    kmb_gov_trips_df = gov_trips_df[gov_trips_df['route_id'].isin(kmb_gov_routes_df['route_id'])]
    kmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(kmb_gov_trips_df['trip_id'])]
    kmb_stoptimes_df = generate_stop_times_for_agency(
        'KMB',
        kmb_trips_df,
        kmb_stoptimes_df,
        kmb_gov_routes_df,
        kmb_gov_trips_df,
        kmb_gov_frequencies_df,
        journey_time_data,
        silent=silent
    )

    # -- Citybus Frequency --
    ctb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'] == 'CTB']
    ctb_gov_trips_df = gov_trips_df[gov_trips_df['route_id'].isin(ctb_gov_routes_df['route_id'])]
    ctb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(ctb_gov_trips_df['trip_id'])]
    ctb_stoptimes_df = generate_stop_times_for_agency(
        'CTB',
        ctb_trips_df,
        ctb_stoptimes_df,
        ctb_gov_routes_df,
        ctb_gov_trips_df,
        ctb_gov_frequencies_df,
        journey_time_data,
        silent=silent
    )

    # -- GMB Frequency --
    gmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'] == 'GMB']
    gmb_gov_trips_df = gov_trips_df[gov_trips_df['route_id'].isin(gmb_gov_routes_df['route_id'])]
    gmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(gmb_gov_trips_df['trip_id'])]
    gmb_stoptimes_df = generate_stop_times_for_agency(
        'GMB',
        gmb_trips_df,
        gmb_stoptimes_df,
        gmb_gov_routes_df,
        gmb_gov_trips_df,
        gmb_gov_frequencies_df,
        journey_time_data,
        silent=silent
    )

    # -- MTR Bus Frequency --
    mtrbus_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'] == 'LRTFeeder']
    mtrbus_gov_trips_df = gov_trips_df[gov_trips_df['route_id'].isin(mtrbus_gov_routes_df['route_id'])]
    mtrbus_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(mtrbus_gov_trips_df['trip_id'])]
    mtrbus_stoptimes_df = generate_stop_times_for_agency(
        'MTRB',
        mtrbus_trips_df,
        mtrbus_stoptimes_df,
        mtrbus_gov_routes_df,
        mtrbus_gov_trips_df,
        mtrbus_gov_frequencies_df,
        journey_time_data,
        silent=silent
    )

    # -- NLB Frequency --
    nlb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'] == 'NLB']
    nlb_gov_trips_df = gov_trips_df[gov_trips_df['route_id'].isin(nlb_gov_routes_df['route_id'])]
    nlb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(nlb_gov_trips_df['trip_id'])]
    nlb_stoptimes_df = generate_stop_times_for_agency(
        'NLB',
        nlb_trips_df,
        nlb_stoptimes_df,
        nlb_gov_routes_df,
        nlb_gov_trips_df,
        nlb_gov_frequencies_df,
        journey_time_data,
        silent=silent
    )

    # -- Combine & Standardize--
    if not silent:
        print("Combining and standardizing data for final GTFS files...")
    final_routes_df = pd.concat([final_kmb_routes, final_ctb_routes, final_gmb_routes, final_mtrbus_routes, final_nlb_routes], ignore_index=True)
    final_trips_df = pd.concat([kmb_trips_df, ctb_trips_df, gmb_trips_df, mtrbus_trips_df, nlb_trips_df], ignore_index=True)

    # --- Standardize stop_times.txt before combining ---
    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']

    # KMB is already processed and has the correct columns + times
    final_kmb_stoptimes = kmb_stoptimes_df[stop_times_cols]

    # Standardize Citybus
    final_ctb_stoptimes = ctb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})

    # Standardize GMB
    final_gmb_stoptimes = gmb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})

    # Standardize MTR Bus
    final_mtrbus_stoptimes = mtrbus_stoptimes_df.rename(columns={'station_seqno': 'stop_sequence'})

    # Standardize NLB
    final_nlb_stoptimes = nlb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})

    final_stop_times_df = pd.concat([final_kmb_stoptimes, final_ctb_stoptimes, final_gmb_stoptimes, final_mtrbus_stoptimes, final_nlb_stoptimes], ignore_index=True)

    final_routes_df.to_csv(os.path.join(final_output_dir, 'routes.txt'), index=False)
    final_trips_df.to_csv(os.path.join(final_output_dir, 'trips.txt'), index=False)
    final_stop_times_df.to_csv(os.path.join(final_output_dir, 'stop_times.txt'), index=False)
    kmb_gov_frequencies_df.to_csv(os.path.join(final_output_dir, 'frequencies.txt'), index=False)


    # --- 4. Handle `calendar.txt` and `frequencies.txt` ---
    # These will be read from the gov_gtfs tables and mapped to the new prefixed IDs.

    # --- 5. Zip the feed ---
    if not silent:
        print("Zipping the unified GTFS feed...")
    zip_path = os.path.join(output_dir, 'unified-agency-specific-stops.gtfs.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(final_output_dir):
            zf.write(os.path.join(final_output_dir, filename), arcname=filename)

    if not silent:
        print(f"--- Unified GTFS Build Complete. Output at {zip_path} ---")
