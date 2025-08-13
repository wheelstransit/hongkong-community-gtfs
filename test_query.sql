-- Test matching logic for KMB routes
WITH kmb_operator_routes AS (
    SELECT 
        k.route,
        k.bound,
        k.service_type,
        k.orig_en,
        k.dest_en,
        COUNT(ks.seq) as stop_count
    FROM kmb_routes k 
    JOIN kmb_stop_sequences ks ON k.route = ks.route 
        AND k.bound = ks.bound 
        AND k.service_type = ks.service_type
    WHERE k.route IN ('1A', '3D')
    GROUP BY k.route, k.bound, k.service_type, k.orig_en, k.dest_en
),
gov_gtfs_routes AS (
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
    WHERE gr.agency_id IN ('KMB', 'LWB', 'KMB+CTB') 
        AND gr.route_short_name IN ('1A', '3D')
    GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, gr.agency_id,
             SPLIT_PART(gt.trip_id, '-', 2)
)
SELECT 
    o.route as operator_route,
    o.bound as operator_bound,
    o.service_type as operator_service_type,
    o.stop_count as operator_stops,
    g.route_id as gov_route_id,
    g.route_short_name as gov_route_name,
    g.bound as gov_bound,
    g.stop_count as gov_stops,
    ABS(o.stop_count - g.stop_count) as stop_diff
FROM kmb_operator_routes o
CROSS JOIN gov_gtfs_routes g
WHERE o.route = g.route_short_name 
    AND o.bound = g.bound
ORDER BY o.route, o.bound, o.service_type, stop_diff;
