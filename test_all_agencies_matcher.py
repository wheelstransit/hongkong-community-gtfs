#!/usr/bin/env python3
"""
Test the enhanced stop-count-based matcher for all agencies
to evaluate if they can benefit from the improved matching algorithm.
"""

import os
import sys
from sqlalchemy import create_engine
import pandas as pd

# Add the project root to the Python path
sys.path.append('/home/anscg/hongkong-community-gtfs')

from src.processing.gtfs_route_matcher import match_operator_routes_to_government_gtfs

def main():
    # Database connection - using Docker database
    engine = create_engine('postgresql://user:password@localhost:5432/gtfs_hk')
    
    print("=" * 80)
    print("TESTING ENHANCED MATCHER FOR ALL AGENCIES")
    print("=" * 80)
    
    agencies = ['KMB', 'CTB', 'MTRB', 'GMB', 'NLB']
    
    for agency in agencies:
        print(f"\n{'-' * 60}")
        print(f"TESTING {agency} ROUTES")
        print(f"{'-' * 60}")
        
        try:
            # Test the enhanced matcher
            matches = match_operator_routes_to_government_gtfs(
                engine=engine,
                operator_name=agency,
                debug=True
            )
            
            if matches:
                total_service_patterns = sum(len(service_list) for service_list in matches.values())
                print(f"\n✓ {agency}: {len(matches)} routes matched with {total_service_patterns} service patterns")
                
                # Show a few sample matches
                sample_count = 0
                for route_key, service_matches in matches.items():
                    if sample_count >= 3:  # Show first 3 matches as samples
                        break
                    print(f"  Sample {sample_count + 1}: {route_key}")
                    for i, match in enumerate(service_matches[:2]):  # Show max 2 services per route
                        print(f"    Service {i+1}: Gov Route {match['gov_route_id']}, Stop diff: {match['stop_diff']}")
                    sample_count += 1
            else:
                print(f"\n✗ {agency}: No matches found")
                
        except Exception as e:
            print(f"\n✗ {agency}: Error - {e}")
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("Check unmatched_routes_*.txt files for detailed analysis")
    print("=" * 80)

if __name__ == "__main__":
    main()
