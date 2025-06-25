import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
from shapely import wkt

def process_and_load_kmb_data(raw_routes: list, raw_stops: list, raw_route_stops: list, engine: Engine):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        print("One or more raw data lists are empty. Aborting KMB data processing.")
        return

    print("Processing KMB routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['route'] + '_' + routes_df['bound'] + '_' + routes_df['service_type']
    routes_df.to_sql(
        'kmb_routes',
        engine,
        if_exists='replace',
        index=False
    )
    print(f"Loaded {len(routes_df)} records into 'kmb_routes' table.")

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
    print(f"Loaded {len(stops_gdf)} records into spatial table 'kmb_stops'.")

    print("Processing KMB route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df['unique_route_id'] = route_stops_df['route'] + '_' + route_stops_df['bound'] + '_' + route_stops_df['service_type']
    route_stops_df.to_sql(
        'kmb_stop_sequences',
        engine,
        if_exists='replace',
        index=False
    )
    print(f"Loaded {len(route_stops_df)} records into 'kmb_stop_sequences' table.")

def process_and_load_gmb_data(raw_routes: dict, raw_stops: list, raw_route_stops: list, engine: Engine):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        print("One or more GMB raw data lists are empty. Aborting GMB data processing.")
        return

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
    print(f"Loaded {len(routes_df)} records into 'gmb_routes' table.")

    print("Processing GMB stops...")
    stops_df = pd.DataFrame(raw_stops)
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
    print(f"Loaded {len(stops_gdf)} records into spatial table 'gmb_stops'.")

    print("Processing GMB route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.to_sql('gmb_stop_sequences', engine, if_exists='replace', index=False)
    print(f"Loaded {len(route_stops_df)} records into 'gmb_stop_sequences' table.")

def process_and_load_mtrbus_data(raw_routes: list, raw_stops: list, raw_route_stops: list, raw_fares: list, engine: Engine):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        print("One or more MTR Bus raw data lists are empty. Aborting MTR Bus data processing.")
        return

    print("Processing MTR Bus routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['ROUTE_ID']
    routes_df.to_sql('mtrbus_routes', engine, if_exists='replace', index=False)
    print(f"Loaded {len(routes_df)} records into 'mtrbus_routes' table.")

    print("Processing MTR Bus stops...")
    stops_df = pd.DataFrame(raw_stops)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df['long'].astype(float), stops_df['lat'].astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis('mtrbus_stops', engine, if_exists='replace', index=False)
    print(f"Loaded {len(stops_gdf)} records into spatial table 'mtrbus_stops'.")

    print("Processing MTR Bus route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.to_sql('mtrbus_stop_sequences', engine, if_exists='replace', index=False)
    print(f"Loaded {len(route_stops_df)} records into 'mtrbus_stop_sequences' table.")

    if raw_fares:
        print("Processing MTR Bus fares...")
        fares_df = pd.DataFrame(raw_fares)
        fares_df.to_sql('mtrbus_fares', engine, if_exists='replace', index=False)
        print(f"Loaded {len(fares_df)} records into 'mtrbus_fares' table.")

def process_and_load_citybus_data(raw_routes: list, raw_stop_details: list, raw_route_sequences: list, engine: Engine):
    if not all([raw_routes, raw_stop_details, raw_route_sequences]):
        print("One or more Citybus raw data lists are empty. Aborting Citybus data processing.")
        return

    print("Processing Citybus routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['route']
    routes_df.to_sql('citybus_routes', engine, if_exists='replace', index=False)
    print(f"Loaded {len(routes_df)} records into 'citybus_routes' table.")

    print("Processing Citybus stops...")
    stops_df = pd.DataFrame(raw_stop_details)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.long.astype(float), stops_df.lat.astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis('citybus_stops', engine, if_exists='replace', index=False)
    print(f"Loaded {len(stops_gdf)} records into spatial table 'citybus_stops'.")

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
    sequences_df['unique_route_id'] = sequences_df['route_id']
    sequences_df.to_sql('citybus_stop_sequences', engine, if_exists='replace', index=False)
    print(f"Loaded {len(sequences_df)} records into 'citybus_stop_sequences' table.")

def process_and_load_nlb_data(raw_routes: list, raw_stops: list, raw_route_stops: list, engine: Engine):
    if not all([raw_routes, raw_stops, raw_route_stops]):
        print("One or more NLB raw data lists are empty. Aborting NLB data processing.")
        return

    print("Processing NLB routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['routeId']
    routes_df.to_sql('nlb_routes', engine, if_exists='replace', index=False)
    print(f"Loaded {len(routes_df)} records into 'nlb_routes' table.")

    print("Processing NLB stops...")
    stops_df = pd.DataFrame(raw_stops)
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.longitude.astype(float), stops_df.latitude.astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['latitude', 'longitude'])
    stops_gdf.to_postgis('nlb_stops', engine, if_exists='replace', index=False)
    print(f"Loaded {len(stops_gdf)} records into spatial table 'nlb_stops'.")

    print("Processing NLB route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.to_sql('nlb_stop_sequences', engine, if_exists='replace', index=False)
    print(f"Loaded {len(route_stops_df)} records into 'nlb_stop_sequences' table.")

def process_and_load_gov_gtfs_data(raw_frequencies: list, raw_trips: list, raw_routes: list,
                                   raw_calendar: list, raw_fares: dict, engine: Engine):
    """Process and load Government GTFS data into the database."""
    if not any([raw_frequencies, raw_trips, raw_routes, raw_calendar, raw_fares]):
        print("All Government GTFS raw data lists are empty. Aborting Government GTFS data processing.")
        return

    if raw_frequencies:
        print("Processing Government GTFS frequencies...")
        frequencies_df = pd.DataFrame(raw_frequencies)
        frequencies_df.to_sql('gov_gtfs_frequencies', engine, if_exists='replace', index=False)
        print(f"Loaded {len(frequencies_df)} records into 'gov_gtfs_frequencies' table.")

    if raw_trips:
        print("Processing Government GTFS trips...")
        trips_df = pd.DataFrame(raw_trips)
        trips_df.to_sql('gov_gtfs_trips', engine, if_exists='replace', index=False)
        print(f"Loaded {len(trips_df)} records into 'gov_gtfs_trips' table.")

    if raw_routes:
        print("Processing Government GTFS routes...")
        routes_df = pd.DataFrame(raw_routes)
        routes_df.to_sql('gov_gtfs_routes', engine, if_exists='replace', index=False)
        print(f"Loaded {len(routes_df)} records into 'gov_gtfs_routes' table.")

    if raw_calendar:
        print("Processing Government GTFS calendar...")
        calendar_df = pd.DataFrame(raw_calendar)
        calendar_df.to_sql('gov_gtfs_calendar', engine, if_exists='replace', index=False)
        print(f"Loaded {len(calendar_df)} records into 'gov_gtfs_calendar' table.")

    if raw_fares:
        if raw_fares.get('attributes'):
            print("Processing Government GTFS fare attributes...")
            fare_attributes_df = pd.DataFrame(raw_fares['attributes'])
            fare_attributes_df.to_sql('gov_gtfs_fare_attributes', engine, if_exists='replace', index=False)
            print(f"Loaded {len(fare_attributes_df)} records into 'gov_gtfs_fare_attributes' table.")

        if raw_fares.get('rules'):
            print("Processing Government GTFS fare rules...")
            fare_rules_df = pd.DataFrame(raw_fares['rules'])
            fare_rules_df.to_sql('gov_gtfs_fare_rules', engine, if_exists='replace', index=False)
            print(f"Loaded {len(fare_rules_df)} records into 'gov_gtfs_fare_rules' table.")

def process_and_load_csdi_data(raw_csdi_data: list, engine: Engine):
    """Process and load CSDI bus routes data into the database."""
    from tqdm import tqdm

    if not raw_csdi_data:
        print("CSDI raw data list is empty. Aborting CSDI data processing.")
        return

    try:
        print(f"Processing CSDI bus routes data... ({len(raw_csdi_data)} records)")
        csdi_df = pd.DataFrame(raw_csdi_data)
        print(f"Created DataFrame with columns: {list(csdi_df.columns)}")

        # Convert WKT geometry strings back to geometry objects if they exist
        if 'geometry' in csdi_df.columns:
            print("Converting WKT geometry strings to geometry objects...")
            # Check for null/empty geometries first
            null_geoms = csdi_df['geometry'].isna().sum()
            if null_geoms > 0:
                print(f"Warning: Found {null_geoms} null geometries")

            # Process geometries in smaller batches to avoid memory issues
            batch_size = 200
            total_batches = (len(csdi_df) + batch_size - 1) // batch_size

            processed_geometries = []
            for i in range(0, len(csdi_df), batch_size):
                batch_num = (i // batch_size) + 1
                print(f"Processing geometry batch {batch_num}/{total_batches}...")
                batch = csdi_df.iloc[i:i+batch_size]
                batch_geoms = batch['geometry'].apply(lambda x: wkt.loads(x) if pd.notna(x) and x else None)
                processed_geometries.extend(batch_geoms.tolist())

            csdi_df['geometry'] = processed_geometries
            print("Creating GeoDataFrame...")
            csdi_gdf = gpd.GeoDataFrame(csdi_df, crs="EPSG:4326")

            print("Saving to PostGIS...")
            print(f"GeoDataFrame has {len(csdi_gdf)} records")
            try:
                # Manually chunk and load data with a progress bar
                chunk_size = 200

                # Create the table with the correct schema by sending an empty GeoDataFrame
                csdi_gdf.head(0).to_postgis('csdi_bus_routes', engine, if_exists='replace', index=False)

                # Use tqdm for the progress bar
                with tqdm(total=len(csdi_gdf), desc="Loading CSDI data to PostGIS") as pbar:
                    for i in range(0, len(csdi_gdf), chunk_size):
                        chunk = csdi_gdf.iloc[i:i + chunk_size]
                        chunk.to_postgis(
                            'csdi_bus_routes',
                            engine,
                            if_exists='append',
                            index=False
                        )
                        pbar.update(len(chunk))

                print(f"Loaded {len(csdi_gdf)} records into spatial table 'csdi_bus_routes'.")
            except Exception as postgis_error:
                print(f"Error saving to PostGIS: {str(postgis_error)}")
                print(f"PostGIS error type: {type(postgis_error).__name__}")
                print("GeoDataFrame info:")
                print(f"  - Columns: {list(csdi_gdf.columns)}")
                print(f"  - CRS: {csdi_gdf.crs}")
                print(f"  - Geometry column: {csdi_gdf.geometry.name}")
                print(f"  - Valid geometries: {csdi_gdf.geometry.is_valid.sum()}/{len(csdi_gdf)}")
                print(f"  - Null geometries: {csdi_gdf.geometry.isna().sum()}")

                # Try to save without geometry as fallback
                print("Attempting to save without geometry data as fallback...")
                try:
                    csdi_df_no_geom = csdi_gdf.drop(columns=['geometry'])
                    csdi_df_no_geom.to_sql('csdi_bus_routes_no_geom', engine, if_exists='replace', index=False)
                    print(f"Loaded {len(csdi_df_no_geom)} records into 'csdi_bus_routes_no_geom' table (without geometry).")
                except Exception as fallback_error:
                    print(f"Fallback also failed: {str(fallback_error)}")
                    raise postgis_error  # Re-raise the original PostGIS error
        else:
            print("No geometry column found, saving as regular table...")
            csdi_df.to_sql('csdi_bus_routes', engine, if_exists='replace', index=False)
            print(f"Loaded {len(csdi_df)} records into 'csdi_bus_routes' table.")
    except Exception as e:
        print(f"Error processing CSDI data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        # Try to save without geometry as fallback
        try:
            print("Attempting to save without geometry data as fallback...")
            csdi_df_no_geom = pd.DataFrame(raw_csdi_data)
            if 'geometry' in csdi_df_no_geom.columns:
                csdi_df_no_geom = csdi_df_no_geom.drop(columns=['geometry'])
            csdi_df_no_geom.to_sql('csdi_bus_routes_no_geom', engine, if_exists='replace', index=False)
            print(f"Loaded {len(csdi_df_no_geom)} records into 'csdi_bus_routes_no_geom' table (without geometry).")
        except Exception as fallback_e:
            print(f"Fallback also failed: {str(fallback_e)}")

def process_and_load_journey_time_data(raw_journey_time_data: dict, raw_hourly_journey_time_data: dict, engine: Engine):
    """Process and load journey time data into the database."""
    if not any([raw_journey_time_data, raw_hourly_journey_time_data]):
        print("Journey time raw data is empty. Aborting journey time data processing.")
        return

    if raw_journey_time_data:
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
            print(f"Loaded {len(journey_time_df)} records into 'journey_time_data' table.")

    if raw_hourly_journey_time_data:
        print("Processing hourly journey time data...")
        # Flatten the nested structure: weekday -> hour -> from_stop_id -> to_stop_id -> travel_time_seconds
        flattened_hourly_data = []
        for weekday, weekday_data in raw_hourly_journey_time_data.items():
            if isinstance(weekday_data, dict):
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
            print(f"Loaded {len(hourly_journey_time_df)} records into 'hourly_journey_time_data' table.")
