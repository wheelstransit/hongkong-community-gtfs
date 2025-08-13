#!/usr/bin/env python3
"""
Test CTB export integration for 22M circular route
"""

from src.common.database import get_db_engine
from src.processing.gtfs_route_matcher import match_operator_routes_with_coop_fallback
import pandas as pd

def test_ctb_export_integration():
    print('Testing full CTB export integration for 22M...')

    engine = get_db_engine()

    # 1. Load CTB routes from database
    print('1. Loading CTB routes from database...')
    ctb_routes_df = pd.read_sql('SELECT * FROM citybus_routes WHERE route = "22M"', engine)
    print(f'   Found {len(ctb_routes_df)} CTB 22M route entries')
    for _, route in ctb_routes_df.iterrows():
        print(f'     - {route["route"]} {route["direction"]}: {route["orig_en"]} -> {route["dest_en"]}')

    # 2. Load government data (required for enhanced matching)
    gov_stops_df = pd.read_sql('SELECT * FROM gov_gtfs_stops LIMIT 1', engine)
    gov_stop_times_df = pd.read_sql('SELECT * FROM gov_gtfs_stop_times LIMIT 1', engine)

    # 3. Run enhanced CTB matching
    print('\n2. Running enhanced CTB matching...')
    ctb_route_matches = match_operator_routes_with_coop_fallback(
        engine=engine,
        operator_name='CTB',
        route_filter=['22M'],
        debug=False
    )

    print(f'   Found {len(ctb_route_matches)} route matches')
    for route_key in ctb_route_matches.keys():
        print(f'     - {route_key}: {len(ctb_route_matches[route_key])} services')

    # 4. Simulate export trip creation logic
    print('\n3. Creating trips based on enhanced matching...')
    ctb_trips_list = []

    if ctb_route_matches:
        print('   Using enhanced matching results (GOOD!)')
        for route_key, route_matches in ctb_route_matches.items():
            route_short_name, bound, service_type = route_key.split('-')
            print(f'   Processing route_key: {route_key}')
            
            # Find corresponding database route
            matching_db_routes = ctb_routes_df[
                (ctb_routes_df['route'] == route_short_name) & 
                (ctb_routes_df['direction'] == ('inbound' if bound == 'I' else 'outbound'))
            ]
            
            if not matching_db_routes.empty:
                route = matching_db_routes.iloc[0]
                for match in route_matches:
                    trip_id = f'CTB-{route_short_name}-{route["direction"]}-{match["gov_service_id"]}'
                    ctb_trips_list.append({
                        'trip_id': trip_id,
                        'route_short_name': route_short_name,
                        'bound': bound,
                        'direction': route["direction"],
                        'stop_count_match': True
                    })
                    print(f'     Created trip: {trip_id}')
            else:
                print(f'     No matching database route found for {route_key}')
    else:
        print('   Enhanced matching failed - would use fallback (BAD!)')

    # 5. Analyze results
    print(f'\n4. Results Analysis:')
    print(f'   Total trips created: {len(ctb_trips_list)}')
    
    if len(ctb_trips_list) == 1:
        trip = ctb_trips_list[0]
        if '22M' in trip['trip_id'] and 'outbound' in trip['direction']:
            print('   ✅ SUCCESS: Only one 22M trip created (outbound only)')
            print(f'   ✅ Trip ID: {trip["trip_id"]}')
        else:
            print('   ❌ ISSUE: Wrong trip created')
            print(f'   ❌ Trip: {trip}')
    elif len(ctb_trips_list) == 2:
        print('   ❌ ISSUE: Two trips created (both inbound and outbound)')
        for trip in ctb_trips_list:
            print(f'   ❌ Trip: {trip["trip_id"]} ({trip["direction"]})')
    else:
        print(f'   ❌ ISSUE: Unexpected number of trips: {len(ctb_trips_list)}')

    print('\n5. Expected vs Actual:')
    print('   Expected: 1 trip (CTB-22M-outbound-XXX)')
    print('   Actual:   ', [t['trip_id'] for t in ctb_trips_list])

    # 6. Success criteria
    success = (
        len(ctb_route_matches) == 1 and 
        '22M-O-1' in ctb_route_matches and 
        '22M-I-1' not in ctb_route_matches and
        len(ctb_trips_list) == 1 and
        'outbound' in ctb_trips_list[0]['direction']
    )
    
    if success:
        print('\n🎉 OVERALL RESULT: ✅ SUCCESS - Circular route fix working correctly!')
        print('   - 22M correctly identified as circular (outbound only)')
        print('   - No inbound trip created')
        print('   - Enhanced matching integration working')
        return True
    else:
        print('\n❌ OVERALL RESULT: FAILED - Circular route fix not working')
        return False

if __name__ == "__main__":
    test_ctb_export_integration()
