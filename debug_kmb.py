#!/usr/bin/env python3

from src.processing.gtfs_route_matcher import get_operator_route_stop_counts, match_operator_routes_to_government_gtfs
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:password@db:5432/gtfs_hk')

print("=== Testing KMB Route Query ===")
result = get_operator_route_stop_counts(engine, 'KMB')
print(f'Found {len(result)} KMB routes')

route_98 = result[result['route'] == '98']
print(f'\nRoute 98 variants: {len(route_98)}')
if len(route_98) > 0:
    print(route_98)

print("\n=== Testing KMB Matcher ===")
matches = match_operator_routes_to_government_gtfs(engine, 'KMB', debug=True)
print(f'Matcher returned {len(matches)} route matches')

route_98_matches = [k for k in matches.keys() if k.startswith('98-')]
print(f'Route 98 matches: {route_98_matches}')
