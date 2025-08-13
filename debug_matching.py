#!/usr/bin/env python3

import sys
import os
sys.path.append('/app')

import pandas as pd
from sqlalchemy import create_engine
from src.processing.gtfs_route_matcher import match_operator_routes_to_government_gtfs

# Database connection
engine = create_engine("postgresql://user:password@db:5432/gtfs_hk")

print("=== Testing KMB DataFrame and Matching ===")

# Get KMB routes DataFrame (same as in export_gtfs.py)
kmb_routes_df = pd.read_sql("SELECT * FROM kmb_routes", engine)
print(f"kmb_routes_df has {len(kmb_routes_df)} rows")
print(f"Route 98 in kmb_routes_df:")
route_98_df = kmb_routes_df[kmb_routes_df['route'] == '98']
print(route_98_df[['route', 'bound', 'service_type']].head())
print(f"service_type data type: {route_98_df['service_type'].dtype}")

# Get matcher results
kmb_route_matches = match_operator_routes_to_government_gtfs(
    engine=engine,
    operator_name="KMB",
    debug=False
)

print(f"\nMatcher returned {len(kmb_route_matches)} matches")

# Test the specific route 98 matching logic
route_98_matches = {k: v for k, v in kmb_route_matches.items() if k.startswith('98-O')}
print(f"Route 98 matches: {list(route_98_matches.keys())}")

for route_key, route_matches in route_98_matches.items():
    route_short_name, bound, service_type = route_key.split('-')
    print(f"\nProcessing route_key: {route_key}")
    print(f"  route_short_name: '{route_short_name}' (type: {type(route_short_name)})")
    print(f"  bound: '{bound}' (type: {type(bound)})")  
    print(f"  service_type: '{service_type}' (type: {type(service_type)})")
    
    # Test the DataFrame matching
    matching_db_routes = kmb_routes_df[
        (kmb_routes_df['route'] == route_short_name) & 
        (kmb_routes_df['bound'] == bound) &
        (kmb_routes_df['service_type'] == service_type)
    ]
    
    print(f"  Found {len(matching_db_routes)} matching database routes")
    if len(matching_db_routes) == 0:
        # Debug why no match
        route_match = kmb_routes_df[kmb_routes_df['route'] == route_short_name]
        bound_match = route_match[route_match['bound'] == bound] 
        print(f"  Debug: route matches: {len(route_match)}")
        print(f"  Debug: route+bound matches: {len(bound_match)}")
        if len(bound_match) > 0:
            print(f"  Debug: available service_types: {bound_match['service_type'].unique()}")
            print(f"  Debug: service_type comparison: {service_type} == {bound_match['service_type'].iloc[0]} -> {service_type == bound_match['service_type'].iloc[0]}")
