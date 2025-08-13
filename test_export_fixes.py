#!/usr/bin/env python3
"""
Test that our export fixes work correctly
"""
print('Testing fixed export integration...')

try:
    from src.common.database import get_db_engine
    from src.processing.gtfs_route_matcher import match_operator_routes_with_coop_fallback
    
    engine = get_db_engine()
    print('✅ Successfully imported match_operator_routes_with_coop_fallback')
    
    # Test a quick run to make sure no import errors
    matches = match_operator_routes_with_coop_fallback(
        engine=engine,
        operator_name='CTB',
        route_filter=['22M'],
        debug=False
    )
    
    print(f'✅ Function works: Found {len(matches)} matches')
    for key in matches.keys():
        if '22M' in key:
            print(f'   {key}: {len(matches[key])} services')
    
    print('✅ Ready to run full export!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
