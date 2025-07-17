import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
import os
import zipfile

def export_unified_feed(engine: Engine, output_dir: str):
    print("--- Starting Unified GTFS Export Process ---")

    final_output_dir = os.path.join(output_dir, "unified_feed")
    os.makedirs(final_output_dir, exist_ok=True)

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

    print("Building stops.txt...")

    # KMB
    kmb_stops_gdf = gpd.read_postgis("SELECT * FROM kmb_stops", engine, geom_col='geometry')
    kmb_stops_gdf['stop_id'] = 'KMB-' + kmb_stops_gdf['stop'].astype(str)

    # remove all 5-character parentheticals (ig they're ids?) and normalize dashes to have exactly one space before and after
    # fix ur data KMB
    kmb_stops_gdf['stop_name'] = (
        kmb_stops_gdf['name_en']
        .str.replace(r'\s*\([A-Za-z0-9]{5}\)', '', regex=True)  # remove all 5-char parentheticals
        .str.replace(r'\s*-\s*', ' - ', regex=True)             # normalize dashes
        .str.replace(r'([^\s])(\([A-Za-z0-9]+\))', r'\1 \2', regex=True)  # ensure space before any parenthetical
    )
    kmb_stops_gdf['stop_lat'] = kmb_stops_gdf.geometry.y
    kmb_stops_gdf['stop_lon'] = kmb_stops_gdf.geometry.x
    kmb_stops_final = kmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # Citybus
    ctb_stops_gdf = gpd.read_postgis("SELECT * FROM citybus_stops", engine, geom_col='geometry')
    ctb_stops_gdf['stop_id'] = 'CTB-' + ctb_stops_gdf['stop'].astype(str)
    ctb_stops_gdf['stop_name'] = ctb_stops_gdf['name_en']
    ctb_stops_gdf['stop_lat'] = ctb_stops_gdf.geometry.y
    ctb_stops_gdf['stop_lon'] = ctb_stops_gdf.geometry.x
    ctb_stops_final = ctb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # GMB
    gmb_stops_gdf = gpd.read_postgis("SELECT * FROM gmb_stops", engine, geom_col='geometry')
    gmb_stops_gdf['stop_id'] = 'GMB-' + gmb_stops_gdf['stop'].astype(str)
    gmb_stops_gdf['stop_name'] = gmb_stops_gdf['name_en']
    gmb_stops_gdf['stop_lat'] = gmb_stops_gdf.geometry.y
    gmb_stops_gdf['stop_lon'] = gmb_stops_gdf.geometry.x
    gmb_stops_final = gmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # MTR Bus
    mtrbus_stops_gdf = gpd.read_postgis("SELECT * FROM mtrbus_stops", engine, geom_col='geometry')
    mtrbus_stops_gdf['stop_id'] = 'MTRB-' + mtrbus_stops_gdf['stop'].astype(str)
    mtrbus_stops_gdf['stop_name'] = mtrbus_stops_gdf['name_en']
    mtrbus_stops_gdf['stop_lat'] = mtrbus_stops_gdf.geometry.y
    mtrbus_stops_gdf['stop_lon'] = mtrbus_stops_gdf.geometry.x
    mtrbus_stops_final = mtrbus_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # NLB
    nlb_stops_gdf = gpd.read_postgis("SELECT * FROM nlb_stops", engine, geom_col='geometry')
    nlb_stops_gdf['stop_id'] = 'NLB-' + nlb_stops_gdf['stop'].astype(str)
    nlb_stops_gdf['stop_name'] = nlb_stops_gdf['name_en']
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
    print(f"Generated stops.txt with {len(all_stops_df)} total stops.")

    # --- 3. Build `routes.txt`, `trips.txt`, `stop_times.txt` ---
    # This process will be repeated for each agency, then concatenated.

    # -- KMB --
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
    })

    # --- Build stop_times.txt ---
    kmb_stoptimes_df = pd.read_sql("SELECT * FROM kmb_stop_sequences", engine)
    # Parse unique_route_id in stop_times as well
    kmb_stoptimes_df[['route', 'bound', 'service_type']] = kmb_stoptimes_df['unique_route_id'].str.split('_', expand=True)
    kmb_stoptimes_df['trip_id'] = 'KMB-' + kmb_stoptimes_df['route'] + '-' + kmb_stoptimes_df['bound'] + '-' + kmb_stoptimes_df['service_type']
    kmb_stoptimes_df['stop_id'] = 'KMB-' + kmb_stoptimes_df['stop_id'].astype(str)
    # ... (build final kmb_stoptimes_df)

    kmb_trips_df = pd.DataFrame({'route_id': kmb_routes_df['route_id']})
    kmb_trips_df['service_id'] = 'KMB-' + kmb_trips_df['route_id'] # Example service_id
    kmb_trips_df['trip_id'] = kmb_trips_df['route_id'] # For headway, trip_id can be simple

    kmb_stoptimes_df = pd.read_sql("SELECT * FROM kmb_stop_sequences", engine)
    kmb_stoptimes_df['trip_id'] = 'KMB-' + kmb_stoptimes_df['unique_route_id']
    kmb_stoptimes_df['stop_id'] = 'KMB-' + kmb_stoptimes_df['stop_id'].astype(str)
    # ... (build final kmb_stoptimes_df)

    # -- Citybus --
    print("Processing Citybus routes, trips, and stop_times...")
    # ... (repeat the same process for Citybus, prefixing all IDs with 'CTB-') ...

    # -- Combine --
    print("Combining data for final GTFS files...")
    # final_routes_df = pd.concat([final_kmb_routes, final_ctb_routes], ...)
    # final_trips_df = pd.concat([final_kmb_trips, final_ctb_trips], ...)
    # final_stop_times_df = pd.concat([final_kmb_stoptimes, final_ctb_stoptimes], ...)

    # final_routes_df.to_csv(os.path.join(final_output_dir, 'routes.txt'), index=False)
    # final_trips_df.to_csv(os.path.join(final_output_dir, 'trips.txt'), index=False)
    # final_stop_times_df.to_csv(os.path.join(final_output_dir, 'stop_times.txt'), index=False)

    # --- 4. Handle `calendar.txt` and `frequencies.txt` ---
    # These will be read from the gov_gtfs tables and mapped to the new prefixed IDs.

    # --- 5. Zip the feed ---
    print("Zipping the unified GTFS feed...")
    zip_path = os.path.join(output_dir, 'unified-agency-specific-stops.gtfs.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(final_output_dir):
            zf.write(os.path.join(final_output_dir, filename), arcname=filename)

    print(f"--- Unified GTFS Build Complete. Output at {zip_path} ---")
