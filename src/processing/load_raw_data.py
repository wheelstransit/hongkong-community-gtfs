import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine

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
    if not all([raw_routes, raw_stops, raw_route_stops, raw_fares]):
        print("One or more MTR Bus raw data lists are empty. Aborting MTR Bus data processing.")
        return

    print("Processing MTR Bus routes...")
    routes_df = pd.DataFrame(raw_routes)
    routes_df['unique_route_id'] = routes_df['ROUTE_ID']
    routes_df.to_sql('mtrbus_routes', engine, if_exists='replace', index=False)
    print(f"Loaded {len(routes_df)} records into 'mtrbus_routes' table.")

    print("Processing MTR Bus stops...")
    stops_df = pd.DataFrame(raw_stops)
    stops_df = stops_df.rename(columns={'stop_id': 'STATION_ID', 'name_en': 'STATION_NAME_ENG', 'name_zh': 'STATION_NAME_CHI'})
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.long.astype(float), stops_df.lat.astype(float)),
        crs="EPSG:4326"
    )
    stops_gdf = stops_gdf.drop(columns=['lat', 'long'])
    stops_gdf.to_postgis('mtrbus_stops', engine, if_exists='replace', index=False)
    print(f"Loaded {len(stops_gdf)} records into spatial table 'mtrbus_stops'.")

    print("Processing MTR Bus route-stop sequences...")
    route_stops_df = pd.DataFrame(raw_route_stops)
    route_stops_df.to_sql('mtrbus_stop_sequences', engine, if_exists='replace', index=False)
    print(f"Loaded {len(route_stops_df)} records into 'mtrbus_stop_sequences' table.")

    print("Processing MTR Bus fares...")
    fares_df = pd.DataFrame(raw_fares)
    fares_df.to_sql('mtrbus_fares', engine, if_exists='replace', index=False)
    print(f"Loaded {len(fares_df)} records into 'mtrbus_fares' table.")

def process_and_load_citybus_data(raw_routes: list, raw_stops: list, raw_stop_details: list, engine: Engine):
    if not all([raw_routes, raw_stops, raw_stop_details]):
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
    route_stops_df = pd.DataFrame(raw_stops)
    route_stops_df.to_sql('citybus_stop_sequences', engine, if_exists='replace', index=False)
    print(f"Loaded {len(route_stops_df)} records into 'citybus_stop_sequences' table.")

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

