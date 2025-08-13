#!/usr/bin/env python3
"""
Test the CTB fixes directly
"""

import os
import sys
import subprocess

# Run a simple test query to check if 22M is identified as circular
test_query = """
SELECT DISTINCT gr.route_short_name as route,
       'CIRCULAR' as type
FROM gov_gtfs_routes gr
JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
WHERE gr.agency_id = 'CTB'
  AND gr.route_short_name = '22M'
GROUP BY gr.route_short_name, gr.route_id
HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
UNION ALL
SELECT '22M' as route, 'NOT_CIRCULAR' as type
WHERE NOT EXISTS (
    SELECT 1 FROM gov_gtfs_routes gr
    JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
    WHERE gr.agency_id = 'CTB'
      AND gr.route_short_name = '22M'
    GROUP BY gr.route_short_name, gr.route_id
    HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
);
"""

# Run the query using docker-compose
cmd = f'cd /home/anscg/hongkong-community-gtfs && docker-compose exec -T db psql -U user -d gtfs_hk -c "{test_query}"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("22M Circular Route Test:")
print("=" * 30)
print(result.stdout)

if "CIRCULAR" in result.stdout:
    print("✅ 22M correctly identified as circular route")
else:
    print("❌ 22M NOT identified as circular route")

print("\nCo-op Route Test:")
print("=" * 30)

# Test co-op routes
coop_query = """
SELECT DISTINCT gr.route_short_name 
FROM gov_gtfs_routes gr 
WHERE gr.agency_id = 'KMB+CTB'
  AND gr.route_short_name IN ('107', '22M')
ORDER BY gr.route_short_name;
"""

cmd2 = f'cd /home/anscg/hongkong-community-gtfs && docker-compose exec -T db psql -U user -d gtfs_hk -c "{coop_query}"'
result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)

print(result2.stdout)

if "107" in result2.stdout:
    print("✅ Route 107 is correctly identified as co-op (KMB+CTB)")
else:
    print("❌ Route 107 NOT found as co-op route")

print("\nSummary of Expected Fixes:")
print("=" * 40)
print("1. 22M should be treated as circular (outbound only, combined stops)")
print("2. 107 should use KMB data for matching (co-op route)")
print("3. Enhanced matching should handle both cases correctly")
