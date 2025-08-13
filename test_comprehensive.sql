-- Comprehensive matching algorithm with ranking
WITH operator_routes AS (
    -- KMB routes
    SELECT 
        'KMB' as operator,
        k.route,
        k.bound,
        k.service_type::text,
        k.orig_en,
        k.dest_en,
        COUNT(ks.seq) as stop_count,
        k.route || '-' || k.bound || '-' || k.service_type as route_key
    FROM kmb_routes k 
    JOIN kmb_stop_sequences ks ON k.route = ks.route 
        AND k.bound = ks.bound 
        AND k.service_type = ks.service_type
    WHERE k.route IN ('1A', '3D', '118')
    GROUP BY k.route, k.bound, k.service_type, k.orig_en, k.dest_en
    
    UNION ALL
    
    -- Citybus routes
    SELECT 
        'CTB' as operator,
        cr.route,
        CASE 
            WHEN cr.direction = 'outbound' THEN 'O' 
            WHEN cr.direction = 'inbound' THEN 'I'
            ELSE cr.direction 
        END as bound,
        '1' as service_type,  -- Citybus doesn't have service types
        cr.orig_en,
        cr.dest_en,
        COUNT(css.sequence) as stop_count,
        cr.route || '-' || CASE 
            WHEN cr.direction = 'outbound' THEN 'O' 
            WHEN cr.direction = 'inbound' THEN 'I'
            ELSE cr.direction 
        END || '-1' as route_key
    FROM citybus_routes cr 
    JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
    WHERE cr.route = '2'
    GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en
),
gov_gtfs_candidate_routes AS (
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
    WHERE gr.route_short_name IN ('1A', '3D', '118', '2')
    GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, gr.agency_id,
             SPLIT_PART(gt.trip_id, '-', 2)
),
potential_matches AS (
    SELECT 
        o.operator,
        o.route_key,
        o.route,
        o.bound,
        o.service_type,
        o.stop_count as operator_stops,
        g.route_id as gov_route_id,
        g.route_short_name as gov_route_name,
        g.bound as gov_bound,
        g.stop_count as gov_stops,
        g.agency_id as gov_agency,
        ABS(o.stop_count - g.stop_count) as stop_diff
    FROM operator_routes o
    CROSS JOIN gov_gtfs_candidate_routes g
    WHERE o.route = g.route_short_name 
        AND o.bound = g.bound
        AND (
            (o.operator = 'KMB' AND g.agency_id IN ('KMB', 'LWB', 'KMB+CTB')) OR
            (o.operator = 'CTB' AND g.agency_id IN ('CTB', 'CTB+KMB')) OR
            (o.operator = 'GMB' AND g.agency_id = 'GMB') OR
            (o.operator = 'MTRB' AND g.agency_id = 'DB') OR
            (o.operator = 'NLB' AND g.agency_id = 'NLB')
        )
),
ranked_matches AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY route_key 
            ORDER BY stop_diff ASC, gov_route_id ASC
        ) as match_rank
    FROM potential_matches
)
SELECT 
    operator,
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
    stop_diff,
    match_rank
FROM ranked_matches
WHERE match_rank = 1  -- Only best match per operator route
ORDER BY operator, route, bound, service_type;
