#!/usr/bin/env python3

import pandas as pd
from src.common.database import get_db_engine

def test_270b_matching():
    """Test the improved 270B matching logic"""
    engine = get_db_engine()
    
    # Get government data
    gov_trips_df = pd.read_sql("SELECT trip_id, route_id, service_id FROM gov_gtfs_trips", engine)
    gov_routes_df = pd.read_sql("SELECT route_id, route_short_name, route_long_name, agency_id FROM gov_gtfs_routes", engine)
    gov_trips_with_route_info = pd.merge(gov_trips_df, gov_routes_df, on='route_id')
    
    # Parse direction_id
    parsed_direction = gov_trips_with_route_info['trip_id'].str.split('-').str[1].astype(int)
    gov_trips_with_route_info['direction_id'] = (parsed_direction == 2).astype(int)
    
    # Get KMB 270B data
    kmb_routes_df = pd.read_sql("SELECT * FROM kmb_routes WHERE route = '270B'", engine)
    
    print("Testing improved matching for KMB 270B:")
    print("=" * 60)
    
    for _, trip_info in kmb_routes_df.iterrows():
        route_short_name = trip_info['route']
        bound = trip_info['bound']
        direction_id = 1 if bound == 'I' else 0
        
        print(f"\n{trip_info['unique_route_id']} ({bound}, dir_id={direction_id}):")
        print(f"  KMB: {trip_info['orig_en']} → {trip_info['dest_en']}")
        
        # Test the new matching logic
        matching_gov_routes = gov_trips_with_route_info[
            (gov_trips_with_route_info['agency_id'].str.contains('KMB|LWB', na=False)) &
            (gov_trips_with_route_info['route_short_name'] == route_short_name) &
            (gov_trips_with_route_info['direction_id'] == direction_id) &
            (
                # Exact directional match
                (gov_trips_with_route_info['route_long_name'].str.contains(trip_info['orig_en'], na=False) &
                 gov_trips_with_route_info['route_long_name'].str.contains(trip_info['dest_en'], na=False)) |
                # Bidirectional match
                (gov_trips_with_route_info['route_long_name'].str.contains(trip_info['dest_en'], na=False) &
                 gov_trips_with_route_info['route_long_name'].str.contains(trip_info['orig_en'], na=False))
            )
        ][['service_id', 'route_id', 'route_long_name', 'direction_id']].drop_duplicates()
        
        if len(matching_gov_routes) > 0:
            print("  Matches found:")
            for _, match in matching_gov_routes.iterrows():
                print(f"    Route {match['route_id']} (dir_id={match['direction_id']}): {match['route_long_name'][:50]}...")
        else:
            print("  No matches found")

if __name__ == "__main__":
    test_270b_matching()
