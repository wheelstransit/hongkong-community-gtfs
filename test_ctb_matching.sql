-- Test CTB route matching
WITH ctb_test AS (
    SELECT 
        'CTB' as operator,
        cr.route,
        CASE 
            WHEN cr.direction = 'outbound' THEN 'O' 
            WHEN cr.direction = 'inbound' THEN 'I'
            ELSE cr.direction 
        END as bound,
        '1' as service_type,
        COUNT(css.sequence) as stop_count,
        cr.route || '-' || CASE 
            WHEN cr.direction = 'outbound' THEN 'O' 
            WHEN cr.direction = 'inbound' THEN 'I'
            ELSE cr.direction 
        END || '-1' as route_key
    FROM citybus_routes cr 
    JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
    WHERE cr.route IN ('2', '5B', '6')
    GROUP BY cr.route, cr.direction
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
    WHERE gr.route_short_name IN ('2', '5B', '6')
        AND gr.agency_id IN ('CTB', 'CTB+KMB')
    GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, gr.agency_id,
             SPLIT_PART(gt.trip_id, '-', 2)
),
matches AS (
    SELECT 
        c.route_key,
        c.route,
        c.bound,
        c.service_type,
        c.stop_count as operator_stops,
        g.route_id as gov_route_id,
        g.route_short_name as gov_route_name,
        g.bound as gov_bound,
        g.stop_count as gov_stops,
        g.agency_id as gov_agency,
        ABS(c.stop_count - g.stop_count) as stop_diff,
        ROW_NUMBER() OVER (
            PARTITION BY c.route_key 
            ORDER BY ABS(c.stop_count - g.stop_count) ASC, g.route_id ASC
        ) as match_rank
    FROM ctb_test c
    CROSS JOIN gov_test g
    WHERE c.route = g.route_short_name 
        AND c.bound = g.bound
        AND ABS(c.stop_count - g.stop_count) <= 10
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
