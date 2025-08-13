#!/usr/bin/env python3
"""
Quick test script to verify the CTB route matching fixes
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.common.database import get_db_engine
    from src.processing.gtfs_route_matcher import get_operator_route_stop_counts
    import pandas as pd

    engine = get_db_engine()

    print("Testing CTB route matching fixes...")
    print("=" * 50)

    # Test the enhanced CTB query
    print("\n1. Testing CTB route stop counts (with circular route handling):")
    ctb_routes = get_operator_route_stop_counts(engine, 'CTB')
    
    # Filter for our test routes
    test_routes = ctb_routes[ctb_routes['route'].isin(['22M', '107'])]
    print(test_routes.to_string(index=False))

    print("\n2. Expected behavior:")
    print("- 22M should appear only once with bound='O' and stop_count close to 39-47")
    print("- 107 should appear twice (inbound and outbound) with normal stop counts")

    # Check specific routes
    route_22M = test_routes[test_routes['route'] == '22M']
    route_107 = test_routes[test_routes['route'] == '107']
    
    print(f"\n3. Results:")
    print(f"- 22M entries: {len(route_22M)}")
    if len(route_22M) > 0:
        print(f"  Bounds: {route_22M['bound'].tolist()}")
        print(f"  Stop counts: {route_22M['stop_count'].tolist()}")
    
    print(f"- 107 entries: {len(route_107)}")
    if len(route_107) > 0:
        print(f"  Bounds: {route_107['bound'].tolist()}")
        print(f"  Stop counts: {route_107['stop_count'].tolist()}")

    # Test co-op route detection
    print("\n4. Testing co-op route detection:")
    coop_query = """
        SELECT DISTINCT gr.route_short_name 
        FROM gov_gtfs_routes gr 
        WHERE gr.agency_id = 'KMB+CTB'
        AND gr.route_short_name IN ('107', '22M')
        ORDER BY gr.route_short_name
    """
    coop_routes = pd.read_sql(coop_query, engine)
    print(f"Co-op routes found: {coop_routes['route_short_name'].tolist()}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
