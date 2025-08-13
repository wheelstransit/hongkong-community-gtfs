-- Test our current merging query for 22M
WITH outbound_sequences AS (
    SELECT 
        cr.route,
        cr.unique_route_id,
        css.stop_id,
        css.sequence,
        'outbound' as original_direction
    FROM citybus_routes cr
    JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
    WHERE cr.route = '22M' AND cr.direction = 'outbound'
),
inbound_sequences AS (
    SELECT 
        cr.route,
        cr.unique_route_id,
        css.stop_id,
        css.sequence,
        'inbound' as original_direction
    FROM citybus_routes cr
    JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
    WHERE cr.route = '22M' AND cr.direction = 'inbound'
),
circular_merged AS (
    SELECT 
        route,
        'outbound' as direction,
        unique_route_id,
        stop_id,
        CASE 
            WHEN original_direction = 'outbound' THEN sequence
            ELSE (SELECT MAX(sequence) FROM outbound_sequences os WHERE os.route = ins.route) + sequence
        END as merged_sequence
    FROM (
        SELECT * FROM outbound_sequences
        UNION ALL
        SELECT * FROM inbound_sequences
    ) ins
)
SELECT COUNT(*) as total_merged_stops, 
       MIN(merged_sequence) as min_seq, 
       MAX(merged_sequence) as max_seq
FROM circular_merged;
