-- Test the new GTFS route matching logic
-- This verifies that our rewritten matching algorithm works correctly

-- Test KMB route matching
WITH kmb_test AS (
    SELECT 
        'KMB' as operator,
        k.route,
        k.bound,
        k.service_type::text,
        COUNT(ks.seq) as stop_count,
        k.route || '-' || k.bound || '-' || k.service_type as route_key
    FROM kmb_routes k 
    JOIN kmb_stop_sequences ks ON k.route = ks.route 
        AND k.bound = ks.bound 
        AND k.service_type = ks.service_type
    WHERE k.route IN ('1A', '3D', '118')
    GROUP BY k.route, k.bound, k.service_type
),
gov_test AS (
    SELECT 
        gr.route_id,
        gr.route_short_name,
        gr.route_long_name,
        gr.agency_id,
        CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '1' THEN 'O' 
             WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 'I'
             ELSE 'UNKNOWN' END as bound,
        COUNT(DISTINCT gst.stop_sequence) as stop_count
    FROM gov_gtfs_routes gr
    JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
    JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
    WHERE gr.route_short_name IN ('1A', '3D', '118')
        AND gr.agency_id IN ('KMB', 'LWB', 'KMB+CTB')
    GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, gr.agency_id,
             SPLIT_PART(gt.trip_id, '-', 2)
),
matches AS (
    SELECT 
        k.route_key,
        k.route,
        k.bound,
        k.service_type,
        k.stop_count as operator_stops,
        g.route_id as gov_route_id,
        g.route_short_name as gov_route_name,
        g.bound as gov_bound,
        g.stop_count as gov_stops,
        g.agency_id as gov_agency,
        ABS(k.stop_count - g.stop_count) as stop_diff,
        ROW_NUMBER() OVER (
            PARTITION BY k.route_key 
            ORDER BY ABS(k.stop_count - g.stop_count) ASC, g.route_id ASC
        ) as match_rank
    FROM kmb_test k
    CROSS JOIN gov_test g
    WHERE k.route = g.route_short_name 
        AND k.bound = g.bound
        AND ABS(k.stop_count - g.stop_count) <= 10
)
SELECT 
    route_key,
    route,
    bound,
    service_type,
    operator_stops,
    gov_route_id,
    gov_route_name,
    gov_bound,
    gov_stops,
    gov_agency,
    stop_diff
FROM matches
WHERE match_rank = 1  -- Only best match per operator route
ORDER BY route, bound, service_type;
