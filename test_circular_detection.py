#!/usr/bin/env python3
"""
Test script to verify CTB circular route handling
"""

import os
import sys
import subprocess

def test_ctb_circular_routes():
    print("Testing CTB circular route detection...")
    
    # Test via docker-compose exec to avoid Python environment issues
    cmd = [
        'docker-compose', 'exec', '-T', 'db', 'psql', '-U', 'user', '-d', 'gtfs_hk', '-c',
        """
        -- Test the CTB circular route detection query directly
        WITH circular_routes AS (
            SELECT DISTINCT 
                gr.route_short_name as route,
                COUNT(DISTINCT gst.stop_sequence) as gov_stop_count
            FROM gov_gtfs_routes gr
            JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
            JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
            WHERE gr.agency_id = 'CTB'
            AND gr.route_short_name IN ('22M', '107')
            GROUP BY gr.route_short_name, gr.route_id
            HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
        )
        SELECT 
            CASE WHEN route IS NOT NULL THEN route || ' is circular with ' || gov_stop_count || ' stops'
                 ELSE 'No circular routes found'
            END as result
        FROM circular_routes
        UNION ALL
        SELECT '22M detection: ' || CASE WHEN EXISTS(SELECT 1 FROM circular_routes WHERE route = '22M') THEN 'CIRCULAR' ELSE 'NOT CIRCULAR' END
        UNION ALL  
        SELECT '107 detection: ' || CASE WHEN EXISTS(SELECT 1 FROM circular_routes WHERE route = '107') THEN 'CIRCULAR' ELSE 'NOT CIRCULAR' END;
        """
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/anscg/hongkong-community-gtfs')
        print("Circular route detection results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running test: {e}")

if __name__ == "__main__":
    test_ctb_circular_routes()
