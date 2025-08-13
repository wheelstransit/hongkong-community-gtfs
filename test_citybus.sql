-- Test matching logic for Citybus routes
WITH citybus_operator_routes AS (
    SELECT 
        cr.route,
        CASE 
            WHEN cr.direction = 'outbound' THEN 'O' 
            WHEN cr.direction = 'inbound' THEN 'I'
            ELSE cr.direction 
        END as bound,
        cr.orig_en,
        cr.dest_en,
        COUNT(css.sequence) as stop_count
    FROM citybus_routes cr 
    JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
    WHERE cr.route = '2'
    GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en
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
    WHERE gr.agency_id IN ('CTB', 'CTB+KMB') 
        AND gr.route_short_name = '2'
    GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, gr.agency_id,
             SPLIT_PART(gt.trip_id, '-', 2)
)
SELECT 
    o.route as operator_route,
    o.bound as operator_bound,
    o.stop_count as operator_stops,
    g.route_id as gov_route_id,
    g.route_short_name as gov_route_name,
    g.bound as gov_bound,
    g.stop_count as gov_stops,
    ABS(o.stop_count - g.stop_count) as stop_diff
FROM citybus_operator_routes o
CROSS JOIN gov_gtfs_routes g
WHERE o.route = g.route_short_name 
    AND o.bound = g.bound
ORDER BY o.route, o.bound, stop_diff;
