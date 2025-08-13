#!/usr/bin/env python3
"""
Simple test of our CTB route matching SQL
"""

import subprocess

def test_ctb_query():
    print("Testing our CTB route query directly...")
    
    # Test the exact query from our code
    cmd = [
        'docker-compose', 'exec', '-T', 'db', 'psql', '-U', 'user', '-d', 'gtfs_hk', '-c',
        """
        WITH route_stops AS (
            SELECT 
                cr.route,
                cr.direction,
                cr.orig_en,
                cr.dest_en,
                COUNT(css.sequence) as stop_count,
                cr.unique_route_id
            FROM citybus_routes cr 
            JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
            WHERE cr.route IN ('22M', '107')
            GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en, cr.unique_route_id
        ),
        circular_routes AS (
            SELECT DISTINCT 
                gr.route_short_name as route,
                COUNT(DISTINCT gst.stop_sequence) as gov_stop_count
            FROM gov_gtfs_routes gr
            JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
            JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
            WHERE gr.agency_id = 'CTB'
            GROUP BY gr.route_short_name, gr.route_id
            HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
        )
        SELECT 
            rs.route,
            CASE 
                WHEN rs.route IN (SELECT route FROM circular_routes) THEN 'O'
                WHEN rs.direction = 'outbound' THEN 'O' 
                WHEN rs.direction = 'inbound' THEN 'I'
                ELSE rs.direction 
            END as bound,
            '1' as service_type,
            CASE 
                WHEN rs.route IN (SELECT route FROM circular_routes) THEN 
                    (SELECT gov_stop_count FROM circular_routes WHERE route = rs.route)
                ELSE rs.stop_count
            END as stop_count,
            rs.route || '-' || CASE 
                WHEN rs.route IN (SELECT route FROM circular_routes) THEN 'O'
                WHEN rs.direction = 'outbound' THEN 'O' 
                WHEN rs.direction = 'inbound' THEN 'I'
                ELSE rs.direction 
            END || '-1' as route_key
        FROM route_stops rs
        WHERE rs.route NOT IN (SELECT route FROM circular_routes)
            OR (rs.route IN (SELECT route FROM circular_routes) AND rs.direction = 'outbound')
        ORDER BY rs.route, bound;
        """
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/anscg/hongkong-community-gtfs')
        print("CTB Query Results:")
        print(result.stdout)
        
        # Check if 22M appears only once
        output_lines = result.stdout.strip().split('\n')
        route_22m_lines = [line for line in output_lines if '22M' in line and '|' in line]
        route_107_lines = [line for line in output_lines if '107' in line and '|' in line]
        
        print(f"\n22M lines found: {len(route_22m_lines)}")
        for line in route_22m_lines:
            print(f"  {line.strip()}")
            
        print(f"\n107 lines found: {len(route_107_lines)}")
        for line in route_107_lines:
            print(f"  {line.strip()}")
            
        # Analysis
        if len(route_22m_lines) == 1:
            print("\n✅ SUCCESS: 22M appears only once (circular route fix working)")
        elif len(route_22m_lines) == 2:
            print("\n❌ ISSUE: 22M still appears twice (fix not working)")
        else:
            print(f"\n⚠️  UNCLEAR: 22M appears {len(route_22m_lines)} times")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ctb_query()
