#!/usr/bin/env python3
"""
Test script for the new GTFS route matcher
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.database import get_db_engine
from src.processing.gtfs_route_matcher import match_operator_routes_to_government_gtfs

def test_matching():
    """Test the new matching algorithm."""
    print("Testing new GTFS route matching algorithm...")
    
    # Get database engine
    engine = get_db_engine()
    
    # Test KMB routes 
    print("\n=== Testing KMB routes ===")
    kmb_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name='KMB',
        route_filter=['1A', '3D', '118'],
        debug=True
    )
    
    print(f"\nKMB Matches found: {len(kmb_matches)}")
    for route_key, match in kmb_matches.items():
        print(f"  {route_key}: → gov route {match['gov_route_id']} "
              f"(stops: {match['operator_stops']} → {match['gov_stops']}, "
              f"diff: {match['stop_diff']})")
    
    # Test CTB routes
    print("\n=== Testing CTB routes ===")
    ctb_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name='CTB', 
        route_filter=['2', '5B'],
        debug=True
    )
    
    print(f"\nCTB Matches found: {len(ctb_matches)}")
    for route_key, match in ctb_matches.items():
        print(f"  {route_key}: → gov route {match['gov_route_id']} "
              f"(stops: {match['operator_stops']} → {match['gov_stops']}, "
              f"diff: {match['stop_diff']})")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_matching()
