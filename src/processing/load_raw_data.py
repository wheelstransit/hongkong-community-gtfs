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

