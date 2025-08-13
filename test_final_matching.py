#!/usr/bin/env python3
"""
Final test of CTB route matching with our fixes
"""

import subprocess
import sys

def test_final_ctb_matching():
    print("Testing final CTB route matching with our fixes...")
    
    # Test using the container directly to avoid environment issues
    cmd = [
        'docker', 'exec', 'hongkong-community-gtfs-app-1', 'python3', '-c',
        '''
from src.common.database import get_db_engine
from src.processing.gtfs_route_matcher import match_operator_routes_with_coop_fallback

engine = get_db_engine()
print("Testing enhanced CTB matching with co-op fallback...")

# Test the complete enhanced matching for CTB
matches = match_operator_routes_with_coop_fallback(
    engine=engine,
    operator_name="CTB",
    route_filter=["22M", "107"],
    debug=True
)

print("\\nFinal Results Summary:")
for route_key, services in matches.items():
    route_num = route_key.split("-")[0]
    direction = route_key.split("-")[1]
    print(f"Route {route_num} ({direction}): {len(services)} service patterns matched")
        '''
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print("Enhanced matching results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        # Analyze the results
        output = result.stdout
        if "22M (O):" in output and "22M (I):" not in output:
            print("✅ SUCCESS: 22M appears only once as outbound (circular route)")
        elif "22M (O):" in output and "22M (I):" in output:
            print("❌ ISSUE: 22M still appears as both inbound and outbound")
        else:
            print("⚠️  UNCLEAR: Could not determine 22M status from output")
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    test_final_ctb_matching()
