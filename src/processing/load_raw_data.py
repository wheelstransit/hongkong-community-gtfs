import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
from shapely import wkt
import re

def process_and_load_kmb_data(raw_routes: list, raw_stops: list, raw_route_stops: list, engine: Engine, silent=False):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        if not silent:
            print("One or more raw data lists are empty. Aborting KMB data processing.")
        return

    if not silent:
        print("Processing KMB routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['route'] + '_' + routes_df['bound'] + '_' + routes_df['service_type']
    routes_df.to_sql(
        'kmb_routes',
        engine,
        if_exists='replace',
        index=False
    )
    if not silent:
        print(f"Loaded {len(routes_df)} records into 'kmb_routes' table.")

    if not silent:
        print("Processing KMB stops...")
    stops_df = pd.DataFrame(raw_stops)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.long, stops_df.lat),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis(
        'kmb_stops',
        engine,
        if_exists='replace',
        index=False
    )
    if not silent:
        print(f"Loaded {len(stops_gdf)} records into spatial table 'kmb_stops'.")

    if not silent:
        print("Processing KMB route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df['unique_route_id'] = route_stops_df['route'] + '_' + route_stops_df['bound'] + '_' + route_stops_df['service_type']
    route_stops_df.to_sql(
        'kmb_stop_sequences',
        engine,
        if_exists='replace',
        index=False
    )
    if not silent:
        print(f"Loaded {len(route_stops_df)} records into 'kmb_stop_sequences' table.")

def process_and_load_gmb_data(raw_routes: dict, raw_stops: list, raw_route_stops: list, engine: Engine, silent=False):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        if not silent:
            print("One or more GMB raw data lists are empty. Aborting GMB data processing.")
        return

    if not silent:
        print("Processing GMB routes...")
    # Flatten the nested routes structure
    flattened_routes = []
    for region, routes in raw_routes.items():
        for route_code in routes:
            flattened_routes.append({
                'region': region,
                'route_code': route_code,
                'unique_route_id': f"{region}_{route_code}"
            })

    routes_df = pd.DataFrame(flattened_routes)
    routes_df.to_sql('gmb_routes', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(routes_df)} records into 'gmb_routes' table.")

    if not silent:
        print("Processing GMB stops...")
    # Create a mapping from stop_id to stop names from the route_stops data
    route_stops_df_for_names = pd.DataFrame(raw_route_stops)
    stop_names_map = route_stops_df_for_names[['stop_id', 'stop_name_en', 'stop_name_tc', 'stop_name_sc']].drop_duplicates('stop_id').set_index('stop_id')

    stops_df = pd.DataFrame(raw_stops)

    # Join the names into the main stops dataframe
    stops_df = stops_df.join(stop_names_map, on='stop_id')

    # Extract coordinates from nested structure
    stops_df['lat'] = stops_df['coordinates'].apply(lambda x: x['wgs84']['latitude'] if x and 'wgs84' in x else None)
    stops_df['long'] = stops_df['coordinates'].apply(lambda x: x['wgs84']['longitude'] if x and 'wgs84' in x else None)
    stops_df = stops_df.drop(columns=['coordinates'])

    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.long, stops_df.lat),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis('gmb_stops', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(stops_gdf)} records into spatial table 'gmb_stops'.")

    if not silent:
        print("Processing GMB route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.to_sql('gmb_stop_sequences', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(route_stops_df)} records into 'gmb_stop_sequences' table.")

from .mtrbus_station_merging import unify_mtrbus_stops

def process_and_load_mtrbus_data(raw_routes: list, raw_stops: list, raw_route_stops: list, raw_fares: list, engine: Engine, silent=False):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        if not silent:
            print("One or more MTR Bus raw data lists are empty. Aborting MTR Bus data processing.")
        return

    if not silent:
        print("Processing MTR Bus routes...")
    routes_df = pd.DataFrame(raw_routes)
    # Standardize column names from CSV to lowercase for consistency
    routes_df.columns = routes_df.columns.str.lower()
    routes_df['unique_route_id'] = routes_df['route_id']
    routes_df.to_sql('mtrbus_routes', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(routes_df)} records into 'mtrbus_routes' table.")

    if not silent:
        print("Unifying MTR Bus stops by coordinates...")
    # The 'raw_stops' list already has the desired lowercase keys ('stop_id', 'lat', 'long')
    # from the client's fetch_all_stops() function.
    # Ensure the stop_id is a string for mapping.
    for stop in raw_stops:
        stop['stop_id'] = str(stop.get('stop_id', ''))

    unified_stops, orig_to_unified, unified_to_orig = unify_mtrbus_stops(raw_stops, stop_id_key="stop_id", lat_key="lat", lon_key="long", precision=7)
    if not silent:
        print(f"Unified {len(raw_stops)} stops into {len(unified_stops)} unique locations.")

    stops_df = pd.DataFrame(unified_stops)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df['long'].astype(float), stops_df['lat'].astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis('mtrbus_stops', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(stops_gdf)} records into spatial table 'mtrbus_stops'.")

    if not silent:
        print("Processing MTR Bus route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.columns = route_stops_df.columns.str.lower()

    if "station_id" in route_stops_df.columns:
        if not silent:
            print("Found 'station_id' column. Renaming to 'stop_id' and mapping to unified IDs...")

        route_stops_df = route_stops_df.rename(columns={"station_id": "stop_id"})
        original_ids = route_stops_df["stop_id"].copy()

        route_stops_df["stop_id"] = route_stops_df["stop_id"].astype(str).map(orig_to_unified)

        unmapped_count = route_stops_df["stop_id"].isna().sum()
        if unmapped_count > 0:
            if not silent:
                print(f"Warning: {unmapped_count} stop_ids could not be mapped. Reverting them to original IDs.")
            route_stops_df["stop_id"] = route_stops_df["stop_id"].fillna(original_ids)
        if not silent:
            print("Mapping complete.")
    else:
        if not silent:
            print("Warning: 'STATION_ID' column not found in route-stop data. Cannot map to unified IDs.")

    route_stops_df.to_sql('mtrbus_stop_sequences', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(route_stops_df)} records into 'mtrbus_stop_sequences' table.")

    if raw_fares:
        if not silent:
            print("Processing MTR Bus fares...")
        fares_df = pd.DataFrame(raw_fares)
        for col in ["origin_stop_id", "destination_stop_id"]:
            if col in fares_df.columns:
                fares_df[col] = fares_df[col].astype(str).map(orig_to_unified).fillna(fares_df[col])
        fares_df.to_sql('mtrbus_fares', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(fares_df)} records into 'mtrbus_fares' table.")

def process_and_load_citybus_data(raw_routes: list, raw_stop_details: list, raw_route_sequences: list, engine: Engine, silent=False):
    if not all([raw_routes, raw_stop_details, raw_route_sequences]):
        if not silent:
            print("One or more Citybus raw data lists are empty. Aborting Citybus data processing.")
        return

    if not silent:
        print("Processing Citybus routes...")
    routes_info_df = pd.DataFrame(raw_routes)
    route_directions_df = pd.DataFrame(raw_route_sequences)[['route_id', 'direction']].drop_duplicates()
    routes_df = pd.merge(
        routes_info_df,
        route_directions_df,
        left_on='route',
        right_on='route_id'
    )
    routes_df['unique_route_id'] = routes_df['route'] + '-' + routes_df['direction']
    routes_df.to_sql('citybus_routes', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(routes_df)} records into 'citybus_routes' table.")

    if not silent:
        print("Processing Citybus stops...")
    stops_df = pd.DataFrame(raw_stop_details)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.long.astype(float), stops_df.lat.astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis('citybus_stops', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(stops_gdf)} records into spatial table 'citybus_stops'.")

    if not silent:
        print("Processing Citybus route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_sequences)
    # Expand the stop_ids list into individual rows
    expanded_sequences = []
    for _, row in route_stops_df.iterrows():
        for seq, stop_id in enumerate(row['stop_ids'], 1):
            expanded_sequences.append({
                'route_id': row['route_id'],
                'direction': row['direction'],
                'stop_id': stop_id,
                'sequence': seq
            })

    sequences_df = pd.DataFrame(expanded_sequences)
    sequences_df['unique_route_id'] = sequences_df['route_id'] + '-' + sequences_df['direction']
    sequences_df.to_sql('citybus_stop_sequences', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(sequences_df)} records into 'citybus_stop_sequences' table.")

def process_and_load_nlb_data(raw_routes: list, raw_stops: list, raw_route_stops: list, engine: Engine, silent=False):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        if not silent:
            print("One or more NLB raw data lists are empty. Aborting NLB data processing.")
        return

    if not silent:
        print("Processing NLB routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['routeId']
    routes_df.to_sql('nlb_routes', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(routes_df)} records into 'nlb_routes' table.")

    if not silent:
        print("Processing NLB stops...")
    stops_df = pd.DataFrame(raw_stops)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.longitude.astype(float), stops_df.latitude.astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['latitude', 'longitude'])
    stops_gdf.to_postgis('nlb_stops', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(stops_gdf)} records into spatial table 'nlb_stops'.")

    if not silent:
        print("Processing NLB route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.to_sql('nlb_stop_sequences', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(route_stops_df)} records into 'nlb_stop_sequences' table.")

def process_and_load_gov_gtfs_data(raw_frequencies: list, raw_trips: list, raw_routes: list,
                                   raw_calendar: list, raw_calendar_dates: list, raw_fares: dict, engine: Engine, silent=False):
    """Process and load Government GTFS data into the database."""
    if not any([raw_frequencies, raw_trips, raw_routes, raw_calendar, raw_fares]):
        if not silent:
            print("All Government GTFS raw data lists are empty. Aborting Government GTFS data processing.")
        return

    if raw_frequencies:
        if not silent:
            print("Processing Government GTFS frequencies...")
        frequencies_df = pd.DataFrame(raw_frequencies)
        frequencies_df.to_sql('gov_gtfs_frequencies', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(frequencies_df)} records into 'gov_gtfs_frequencies' table.")

    if raw_trips:
        if not silent:
            print("Processing Government GTFS trips...")
        trips_df = pd.DataFrame(raw_trips)
        trips_df.to_sql('gov_gtfs_trips', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(trips_df)} records into 'gov_gtfs_trips' table.")

    if raw_routes:
        if not silent:
            print("Processing Government GTFS routes...")
        routes_df = pd.DataFrame(raw_routes)
        routes_df.to_sql('gov_gtfs_routes', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(routes_df)} records into 'gov_gtfs_routes' table.")

    if raw_calendar:
        if not silent:
            print("Processing Government GTFS calendar...")
        calendar_df = pd.DataFrame(raw_calendar)
        calendar_df.to_sql('gov_gtfs_calendar', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(calendar_df)} records into 'gov_gtfs_calendar' table.")

    if raw_calendar_dates:
        if not silent:
            print("Processing Government GTFS calendar dates...")
        calendar_dates_df = pd.DataFrame(raw_calendar_dates)
        calendar_dates_df.to_sql('gov_gtfs_calendar_dates', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(calendar_dates_df)} records into 'gov_gtfs_calendar_dates' table.")

    if raw_fares:
        if raw_fares.get('attributes'):
            if not silent:
                print("Processing Government GTFS fare attributes...")
            fare_attributes_df = pd.DataFrame(raw_fares['attributes'])
            fare_attributes_df.to_sql('gov_gtfs_fare_attributes', engine, if_exists='replace', index=False)
            if not silent:
                print(f"Loaded {len(fare_attributes_df)} records into 'gov_gtfs_fare_attributes' table.")

        if raw_fares.get('rules'):
            if not silent:
                print("Processing Government GTFS fare rules...")
            fare_rules_df = pd.DataFrame(raw_fares['rules'])
            fare_rules_df.to_sql('gov_gtfs_fare_rules', engine, if_exists='replace', index=False)
            if not silent:
                print(f"Loaded {len(fare_rules_df)} records into 'gov_gtfs_fare_rules' table.")

def process_and_load_csdi_data(raw_csdi_data: list, engine: Engine, silent=False):
    """Process and load CSDI bus routes data into the database."""
    from tqdm import tqdm

    if not raw_csdi_data:
        if not silent:
            print("CSDI raw data list is empty. Aborting CSDI data processing.")
        return

    try:
        if not silent:
            print(f"Processing CSDI bus routes data... ({len(raw_csdi_data)} records)")
        csdi_df = pd.DataFrame(raw_csdi_data)
        if not silent:
            print(f"Created DataFrame with columns: {list(csdi_df.columns)}")

        # Convert WKT geometry strings back to geometry objects if they exist
        if 'geometry' in csdi_df.columns:
            if not silent:
                print("Converting WKT geometry strings to geometry objects...")
            # Check for null/empty geometries first
            null_geoms = csdi_df['geometry'].isna().sum()
            if null_geoms > 0 and not silent:
                print(f"Warning: Found {null_geoms} null geometries")

            # Process geometries in smaller batches to avoid memory issues
            batch_size = 200
            total_batches = (len(csdi_df) + batch_size - 1) // batch_size

            processed_geometries = []
            for i in range(0, len(csdi_df), batch_size):
                batch_num = (i // batch_size) + 1
                if not silent:
                    print(f"Processing geometry batch {batch_num}/{total_batches}...")
                batch = csdi_df.iloc[i:i+batch_size]
                batch_geoms = batch['geometry'].apply(lambda x: wkt.loads(x) if pd.notna(x) and x else None)
                processed_geometries.extend(batch_geoms.tolist())

            csdi_df['geometry'] = processed_geometries
            if not silent:
                print("Creating GeoDataFrame...")
            csdi_gdf = gpd.GeoDataFrame(csdi_df, crs="EPSG:4326")

            if not silent:
                print("Saving to PostGIS...")
                print(f"GeoDataFrame has {len(csdi_gdf)} records")
            try:
                # Manually chunk and load data with a progress bar
                chunk_size = 200

                # Create the table with the correct schema by sending an empty GeoDataFrame
                csdi_gdf.head(0).to_postgis('csdi_bus_routes', engine, if_exists='replace', index=False)

                # Use tqdm for the progress bar
                with tqdm(total=len(csdi_gdf), desc="Loading CSDI data to PostGIS", disable=silent) as pbar:
                    for i in range(0, len(csdi_gdf), chunk_size):
                        chunk = csdi_gdf.iloc[i:i + chunk_size]
                        chunk.to_postgis(
                            'csdi_bus_routes',
                            engine,
                            if_exists='append',
                            index=False
                        )
                        pbar.update(len(chunk))
                if not silent:
                    print(f"Loaded {len(csdi_gdf)} records into spatial table 'csdi_bus_routes'.")
            except Exception as postgis_error:
                if not silent:
                    print(f"Error saving to PostGIS: {str(postgis_error)}")
                    print(f"PostGIS error type: {type(postgis_error).__name__}")
                    print("GeoDataFrame info:")
                    print(f"  - Columns: {list(csdi_gdf.columns)}")
                    print(f"  - CRS: {csdi_gdf.crs}")
                    print(f"  - Geometry column: {csdi_gdf.geometry.name}")
                    print(f"  - Valid geometries: {csdi_gdf.geometry.is_valid.sum()}/{len(csdi_gdf)}")
                    print(f"  - Null geometries: {csdi_gdf.geometry.isna().sum()}")

                # Try to save without geometry as fallback
                if not silent:
                    print("Attempting to save without geometry data as fallback...")
                try:
                    csdi_df_no_geom = csdi_gdf.drop(columns=['geometry'])
                    csdi_df_no_geom.to_sql('csdi_bus_routes_no_geom', engine, if_exists='replace', index=False)
                    if not silent:
                        print(f"Loaded {len(csdi_df_no_geom)} records into 'csdi_bus_routes_no_geom' table (without geometry).")
                except Exception as fallback_error:
                    if not silent:
                        print(f"Fallback also failed: {str(fallback_error)}")
                    raise postgis_error  # Re-raise the original PostGIS error
        else:
            if not silent:
                print("No geometry column found, saving as regular table...")
            csdi_df.to_sql('csdi_bus_routes', engine, if_exists='replace', index=False)
            if not silent:
                print(f"Loaded {len(csdi_df)} records into 'csdi_bus_routes' table.")
    except Exception as e:
        if not silent:
            print(f"Error processing CSDI data: {str(e)}")
            print(f"Error type: {type(e).__name__}")
        # Try to save without geometry as fallback
        try:
            if not silent:
                print("Attempting to save without geometry data as fallback...")
            csdi_df_no_geom = pd.DataFrame(raw_csdi_data)
            if 'geometry' in csdi_df_no_geom.columns:
                csdi_df_no_geom = csdi_df_no_geom.drop(columns=['geometry'])
            csdi_df_no_geom.to_sql('csdi_bus_routes_no_geom', engine, if_exists='replace', index=False)
            if not silent:
                print(f"Loaded {len(csdi_df_no_geom)} records into 'csdi_bus_routes_no_geom' table (without geometry).")
        except Exception as fallback_e:
            if not silent:
                print(f"Fallback also failed: {str(fallback_e)}")

def process_and_load_mtr_rails_data(
    raw_mtr_lines_and_stations: list,
    raw_mtr_lines_fares: list,
    raw_light_rail_routes_and_stops: list,
    raw_light_rail_fares: list,
    raw_airport_express_fares: list,
    engine: Engine,
    silent=False
):
    # --- MTR Heavy Rail ---
    if raw_mtr_lines_and_stations:
        if not silent:
            print("Processing MTR lines and stations...")

        stations_df = pd.DataFrame(raw_mtr_lines_and_stations)
        # Drop rows where location data could not be fetched
        stations_df.dropna(subset=['longitude', 'latitude'], inplace=True)

        # Create a GeoDataFrame to add the geometry column
        stations_gdf = gpd.GeoDataFrame(
            stations_df,
            geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
            crs="EPSG:4326"
        )
        stations_gdf.drop(columns=['latitude', 'longitude'], inplace=True)

        stations_gdf.to_postgis('mtr_lines_and_stations', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(stations_gdf)} records into 'mtr_lines_and_stations' table.")

    if raw_mtr_lines_fares:
        if not silent:
            print("Processing MTR heavy rail fares...")
        mtr_fares_df = pd.DataFrame(raw_mtr_lines_fares)
        mtr_fares_df.to_sql('mtr_lines_fares', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(mtr_fares_df)} records into 'mtr_lines_fares' table.")

    # --- Light Rail ---
    if raw_light_rail_routes_and_stops:
        if not silent:
            print("Processing Light Rail routes and stops...")
        lrt_routes_stops_df = pd.DataFrame(raw_light_rail_routes_and_stops)
        lrt_routes_stops_df.to_sql('light_rail_routes_and_stops', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(lrt_routes_stops_df)} records into 'light_rail_routes_and_stops' table.")

    if raw_light_rail_fares:
        if not silent:
            print("Processing Light Rail fares...")
        lrt_fares_df = pd.DataFrame(raw_light_rail_fares)
        lrt_fares_df.to_sql('light_rail_fares', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(lrt_fares_df)} records into 'light_rail_fares' table.")

    # --- Airport Express ---
    if raw_airport_express_fares:
        if not silent:
            print("Processing Airport Express fares...")
        ae_fares_df = pd.DataFrame(raw_airport_express_fares)
        ae_fares_df.to_sql('airport_express_fares', engine, if_exists='replace', index=False)
        if not silent:
            print(f"Loaded {len(ae_fares_df)} records into 'airport_express_fares' table.")

def process_and_load_journey_time_data(raw_journey_time_data: dict, raw_hourly_journey_time_data: dict, engine: Engine, silent=False):
    if not any([raw_journey_time_data, raw_hourly_journey_time_data]):
        if not silent:
            print("Journey time raw data is empty. Aborting journey time data processing.")
        return

    if raw_journey_time_data:
        if not silent:
            print("Processing journey time data...")
        # Flatten the nested structure: from_stop_id -> to_stop_id -> travel_time_seconds
        flattened_data = []
        for from_stop_id, destinations in raw_journey_time_data.items():
            if isinstance(destinations, dict):
                for to_stop_id, travel_time_seconds in destinations.items():
                    flattened_data.append({
                        'from_stop_id': from_stop_id,
                        'to_stop_id': to_stop_id,
                        'travel_time_seconds': travel_time_seconds
                    })

        if flattened_data:
            journey_time_df = pd.DataFrame(flattened_data)
            journey_time_df.to_sql('journey_time_data', engine, if_exists='replace', index=False)
            if not silent:
                print(f"Loaded {len(journey_time_df)} records into 'journey_time_data' table.")

    if raw_hourly_journey_time_data:
        if not silent:
            print("Processing hourly journey time data...")
        # Flatten the nested structure: weekday -> hour -> from_stop_id -> to_stop_id -> travel_time_seconds
        flattened_hourly_data = []
        for weekday, weekday_data in raw_hourly_journey_time_data.items():
            #if isinstance(weekday_data, dict):
            if False:
                for hour, hour_data in weekday_data.items():
                    if isinstance(hour_data, dict):
                        for from_stop_id, destinations in hour_data.items():
                            if isinstance(destinations, dict):
                                for to_stop_id, travel_time_seconds in destinations.items():
                                    flattened_hourly_data.append({
                                        'weekday': weekday,
                                        'hour': hour,
                                        'from_stop_id': from_stop_id,
                                        'to_stop_id': to_stop_id,
                                        'travel_time_seconds': travel_time_seconds
                                    })

        if flattened_hourly_data:
            hourly_journey_time_df = pd.DataFrame(flattened_hourly_data)
            hourly_journey_time_df.to_sql('hourly_journey_time_data', engine, if_exists='replace', index=False)
            if not silent:
                print(f"Loaded {len(hourly_journey_time_df)} records into 'hourly_journey_time_data' table.")

def process_and_load_mtr_exits_data(raw_mtr_exits_data: list, engine: Engine, silent=False):
    """Process and load MTR exit data into the database."""
    if not raw_mtr_exits_data:
        if not silent:
            print("MTR exits raw data is empty. Aborting MTR exits data processing.")
        return

    if not silent:
        print(f"Processing MTR exits data... ({len(raw_mtr_exits_data)} records)")

    exits_df = pd.DataFrame(raw_mtr_exits_data)

    # Create a GeoDataFrame to store the data spatially
    exits_gdf = gpd.GeoDataFrame(
        exits_df,
        geometry=gpd.points_from_xy(exits_df.lon, exits_df.lat),
        crs="EPSG:4326"
    )
    exits_gdf = exits_gdf.drop(columns=['lat', 'lon'])

    # Save to PostGIS
    exits_gdf.to_postgis('mtr_exits', engine, if_exists='replace', index=False)

    if not silent:
        print(f"Loaded {len(exits_gdf)} records into spatial table 'mtr_exits'.")

def process_and_load_light_rail_stops_data(raw_light_rail_stops: dict, engine: Engine, silent=False):
    """Process and load Light Rail stop data into the database."""
    if not raw_light_rail_stops:
        if not silent:
            print("Light Rail stops raw data is empty. Aborting.")
        return

    if not silent:
        print(f"Processing Light Rail stops... ({len(raw_light_rail_stops)} records)")

    stops_df = pd.DataFrame(raw_light_rail_stops.values())

    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.lon, stops_df.lat),
        crs="EPSG:4326"
    )
    stops_gdf.drop(columns=['lat', 'lon'], inplace=True)

    stops_gdf.to_postgis('light_rail_stops', engine, if_exists='replace', index=False)

    if not silent:
        print(f"Loaded {len(stops_gdf)} records into spatial table 'light_rail_stops'.")

def process_and_load_tramway_data(raw_routes: list, raw_stops: list, engine: Engine, silent=False):
    """
    Process and load Tramway routes and stops (with multilingual fields, no location data) into the database.
    """
    if not raw_routes or not raw_stops:
        if not silent:
            print("Tramway routes or stops data is empty. Aborting Tramway data processing.")
        return

    if not silent:
        print("Processing Tramway routes...")
    routes_df = pd.DataFrame(raw_routes)
    # Select and rename columns for clarity and multilingual support
    route_columns = [
        "Route ID" if "Route ID" in routes_df.columns else routes_df.columns[0],
        "route_name_en", "route_name_tc", "route_name_sc",
        "origin_en", "origin_tc", "origin_sc",
        "destination_en", "destination_tc", "destination_sc"
    ]
    route_columns = [col for col in route_columns if col in routes_df.columns]
    routes_df = routes_df[route_columns]
    routes_df = routes_df.rename(columns={
        route_columns[0]: "route_id",
        "origin_en": "start_en",
        "origin_tc": "start_tc",
        "origin_sc": "start_sc",
        "destination_en": "end_en",
        "destination_tc": "end_tc",
        "destination_sc": "end_sc"
    })
    routes_df.to_sql('tramway_routes', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(routes_df)} records into 'tramway_routes' table.")

    if not silent:
        print("Processing Tramway stops...")
    stops_df = pd.DataFrame(raw_stops)
    # Select and rename columns for clarity and multilingual support
    stop_columns = [
        "Stops Code" if "Stops Code" in stops_df.columns else stops_df.columns[0],
        "Traveling Direction" if "Traveling Direction" in stops_df.columns else None,
        "stop_name_en", "stop_name_tc", "stop_name_sc"
    ]
    stop_columns = [col for col in stop_columns if col and col in stops_df.columns]
    stops_df = stops_df[stop_columns]
    stops_df = stops_df.rename(columns={
        stop_columns[0]: "stop_code",
        "Traveling Direction": "direction_en",  # Only English direction is available from the English CSV
        "stop_name_en": "stop_name_en",
        "stop_name_tc": "stop_name_tc",
        "stop_name_sc": "stop_name_sc"
    })
    stops_df.to_sql('tramway_stops', engine, if_exists='replace', index=False)
    if not silent:
        print(f"Loaded {len(stops_df)} records into 'tramway_stops' table.")

    if not silent:
        print("Tramway data processing and loading complete.")
