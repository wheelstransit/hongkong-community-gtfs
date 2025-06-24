-- 🚌 Fun SQL Discoveries for Hong Kong Transit Data 🚌
-- A collection of interesting queries to explore the rich transit dataset

-- =============================================================================
-- 🎯 QUICK STATS & FUN FACTS
-- =============================================================================

-- 1. 📊 Transit System Overview
SELECT
    'KMB Routes' as operator, COUNT(*) as count FROM kmb_routes
UNION ALL
SELECT 'Citybus Routes', COUNT(*) FROM citybus_routes
UNION ALL
SELECT 'NLB Routes', COUNT(*) FROM nlb_routes
UNION ALL
SELECT 'MTR Bus Routes', COUNT(*) FROM mtrbus_routes
UNION ALL
SELECT 'GMB Routes', COUNT(*) FROM gmb_routes
UNION ALL
SELECT 'Government GTFS Routes', COUNT(*) FROM gov_gtfs_routes
ORDER BY count DESC;

-- 2. 🚏 Stop Density Champions
SELECT
    'KMB Stops' as operator, COUNT(*) as stops FROM kmb_stops
UNION ALL
SELECT 'Citybus Stops', COUNT(*) FROM citybus_stops
UNION ALL
SELECT 'NLB Stops', COUNT(*) FROM nlb_stops
UNION ALL
SELECT 'MTR Bus Stops', COUNT(*) FROM mtrbus_stops
UNION ALL
SELECT 'GMB Stops', COUNT(*) FROM gmb_stops
ORDER BY stops DESC;

-- =============================================================================
-- 🔍 ROUTE DISCOVERIES
-- =============================================================================

-- 3. 🏆 Longest Route Names (Who likes to be verbose?)
SELECT
    'KMB' as operator,
    route,
    orig_en,
    dest_en,
    LENGTH(orig_en || ' to ' || dest_en) as name_length
FROM kmb_routes
WHERE orig_en IS NOT NULL AND dest_en IS NOT NULL
ORDER BY name_length DESC
LIMIT 10;

-- 4. 🎭 Routes with Interesting Names
SELECT
    'KMB' as operator,
    route,
    orig_en as origin,
    dest_en as destination
FROM kmb_routes
WHERE orig_en ILIKE '%airport%'
   OR dest_en ILIKE '%airport%'
   OR orig_en ILIKE '%disneyland%'
   OR dest_en ILIKE '%disneyland%'
   OR orig_en ILIKE '%ocean park%'
   OR dest_en ILIKE '%ocean park%'
ORDER BY route;

-- 5. 🔄 Circular Routes (Same origin and destination)
SELECT
    'KMB' as operator,
    route,
    orig_en,
    dest_en,
    service_type
FROM kmb_routes
WHERE orig_en = dest_en
   AND orig_en IS NOT NULL
ORDER BY route;

-- =============================================================================
-- 🌍 SPATIAL ADVENTURES
-- =============================================================================

-- 6. 🗺️ Northernmost and Southernmost Stops
WITH stop_extremes AS (
    SELECT
        'KMB' as operator,
        stop,
        name_en,
        ST_Y(geometry) as latitude,
        'Northernmost' as position
    FROM kmb_stops
    ORDER BY ST_Y(geometry) DESC
    LIMIT 1

    UNION ALL

    SELECT
        'KMB' as operator,
        stop,
        name_en,
        ST_Y(geometry) as latitude,
        'Southernmost' as position
    FROM kmb_stops
    ORDER BY ST_Y(geometry) ASC
    LIMIT 1
)
SELECT * FROM stop_extremes;

-- 7. 🏝️ Stops on Different Islands (based on longitude)
SELECT
    CASE
        WHEN ST_X(geometry) < 114.0 THEN 'Lantau/Western Islands'
        WHEN ST_X(geometry) BETWEEN 114.0 AND 114.3 THEN 'Hong Kong Island/Kowloon'
        ELSE 'New Territories/Eastern'
    END as rough_area,
    COUNT(*) as stop_count,
    ROUND(AVG(ST_X(geometry))::numeric, 4) as avg_longitude,
    ROUND(AVG(ST_Y(geometry))::numeric, 4) as avg_latitude
FROM kmb_stops
GROUP BY
    CASE
        WHEN ST_X(geometry) < 114.0 THEN 'Lantau/Western Islands'
        WHEN ST_X(geometry) BETWEEN 114.0 AND 114.3 THEN 'Hong Kong Island/Kowloon'
        ELSE 'New Territories/Eastern'
    END
ORDER BY stop_count DESC;

-- =============================================================================
-- ⏱️ JOURNEY TIME MYSTERIES
-- =============================================================================

-- 8. 🐌 Slowest Journey Times (Where does time crawl?)
SELECT
    from_stop_id,
    to_stop_id,
    travel_time_seconds,
    ROUND(travel_time_seconds / 60.0, 2) as travel_time_minutes
FROM journey_time_data
WHERE travel_time_seconds > 0
ORDER BY travel_time_seconds DESC
LIMIT 10;

-- 9. 🚀 Lightning Fast Connections
SELECT
    from_stop_id,
    to_stop_id,
    travel_time_seconds,
    ROUND(travel_time_seconds, 2) as exact_seconds
FROM journey_time_data
WHERE travel_time_seconds > 0 AND travel_time_seconds < 60
ORDER BY travel_time_seconds ASC
LIMIT 10;

-- 10. 📊 Journey Time Distribution
SELECT
    CASE
        WHEN travel_time_seconds < 60 THEN '< 1 minute'
        WHEN travel_time_seconds < 300 THEN '1-5 minutes'
        WHEN travel_time_seconds < 600 THEN '5-10 minutes'
        WHEN travel_time_seconds < 1200 THEN '10-20 minutes'
        WHEN travel_time_seconds < 1800 THEN '20-30 minutes'
        ELSE '30+ minutes'
    END as time_bucket,
    COUNT(*) as journey_count,
    ROUND(AVG(travel_time_seconds), 2) as avg_seconds
FROM journey_time_data
WHERE travel_time_seconds > 0
GROUP BY
    CASE
        WHEN travel_time_seconds < 60 THEN '< 1 minute'
        WHEN travel_time_seconds < 300 THEN '1-5 minutes'
        WHEN travel_time_seconds < 600 THEN '5-10 minutes'
        WHEN travel_time_seconds < 1200 THEN '10-20 minutes'
        WHEN travel_time_seconds < 1800 THEN '20-30 minutes'
        ELSE '30+ minutes'
    END
ORDER BY
    CASE
        WHEN time_bucket = '< 1 minute' THEN 1
        WHEN time_bucket = '1-5 minutes' THEN 2
        WHEN time_bucket = '5-10 minutes' THEN 3
        WHEN time_bucket = '10-20 minutes' THEN 4
        WHEN time_bucket = '20-30 minutes' THEN 5
        ELSE 6
    END;

-- =============================================================================
-- 🕐 TEMPORAL PATTERNS
-- =============================================================================

-- 11. ⏰ Rush Hour vs Off-Peak Analysis
SELECT
    CASE
        WHEN hour IN (7, 8, 9, 17, 18, 19) THEN 'Rush Hour'
        WHEN hour IN (0, 1, 2, 3, 4, 5) THEN 'Late Night'
        ELSE 'Off-Peak'
    END as time_period,
    COUNT(*) as journey_count,
    ROUND(AVG(travel_time_seconds), 2) as avg_travel_time,
    ROUND(MIN(travel_time_seconds), 2) as min_travel_time,
    ROUND(MAX(travel_time_seconds), 2) as max_travel_time
FROM hourly_journey_time_data
WHERE travel_time_seconds > 0
GROUP BY
    CASE
        WHEN hour IN (7, 8, 9, 17, 18, 19) THEN 'Rush Hour'
        WHEN hour IN (0, 1, 2, 3, 4, 5) THEN 'Late Night'
        ELSE 'Off-Peak'
    END
ORDER BY avg_travel_time DESC;

-- 12. 📅 Weekend vs Weekday Patterns
SELECT
    CASE
        WHEN weekday IN (0, 1, 2, 3, 4) THEN 'Weekday'
        ELSE 'Weekend'
    END as day_type,
    COUNT(*) as journey_count,
    ROUND(AVG(travel_time_seconds), 2) as avg_travel_time
FROM hourly_journey_time_data
WHERE travel_time_seconds > 0
GROUP BY
    CASE
        WHEN weekday IN (0, 1, 2, 3, 4) THEN 'Weekday'
        ELSE 'Weekend'
    END;

-- =============================================================================
-- 🎲 RANDOM DISCOVERIES
-- =============================================================================

-- 13. 🎰 Random Route Explorer (Run multiple times for different results!)
SELECT
    'KMB' as operator,
    route,
    orig_en as from_location,
    dest_en as to_location,
    service_type
FROM kmb_routes
WHERE orig_en IS NOT NULL AND dest_en IS NOT NULL
ORDER BY RANDOM()
LIMIT 5;

-- 14. 🔢 Routes with Lucky Numbers
SELECT
    'KMB' as operator,
    route,
    orig_en,
    dest_en
FROM kmb_routes
WHERE route SIMILAR TO '%(8|88|888|168|888)%'
   OR route SIMILAR TO '%(7|77|777)%'
ORDER BY route;

-- 15. 🌟 Special Route Patterns
SELECT
    operator,
    route_pattern,
    example_routes,
    count
FROM (
    SELECT
        'KMB' as operator,
        CASE
            WHEN route ~ '^[0-9]+$' THEN 'Numeric Only'
            WHEN route ~ '^[0-9]+[A-Z]$' THEN 'Number + Letter'
            WHEN route ~ '^[A-Z][0-9]+$' THEN 'Letter + Number'
            WHEN route ~ '^[A-Z]+$' THEN 'Letters Only'
            ELSE 'Mixed/Special'
        END as route_pattern,
        STRING_AGG(route, ', ' ORDER BY route) FILTER (WHERE rn <= 3) as example_routes,
        COUNT(*) as count
    FROM (
        SELECT
            route,
            ROW_NUMBER() OVER (PARTITION BY
                CASE
                    WHEN route ~ '^[0-9]+$' THEN 'Numeric Only'
                    WHEN route ~ '^[0-9]+[A-Z]$' THEN 'Number + Letter'
                    WHEN route ~ '^[A-Z][0-9]+$' THEN 'Letter + Number'
                    WHEN route ~ '^[A-Z]+$' THEN 'Letters Only'
                    ELSE 'Mixed/Special'
                END
                ORDER BY route
            ) as rn
        FROM kmb_routes
    ) t
    GROUP BY route_pattern
) patterns
ORDER BY count DESC;

-- =============================================================================
-- 🔗 CROSS-OPERATOR CONNECTIONS
-- =============================================================================

-- 16. 🤝 Potential Transfer Points (Stops with similar names across operators)
SELECT
    k.name_en as kmb_stop_name,
    c.name_en as citybus_stop_name,
    ST_Distance(k.geometry, c.geometry) * 111000 as distance_meters
FROM kmb_stops k
JOIN citybus_stops c ON SIMILARITY(k.name_en, c.name_en) > 0.7
WHERE k.name_en IS NOT NULL
  AND c.name_en IS NOT NULL
  AND ST_Distance(k.geometry, c.geometry) * 111000 < 500  -- Within 500 meters
ORDER BY distance_meters
LIMIT 10;

-- =============================================================================
-- 🎯 CHALLENGE QUERIES (Advanced Fun!)
-- =============================================================================

-- 17. 🏆 The Most "Central" Stop (closest to average position of all stops)
WITH center_point AS (
    SELECT ST_Centroid(ST_Collect(geometry)) as center FROM kmb_stops
)
SELECT
    stop,
    name_en,
    ST_Distance(geometry, center.center) * 111000 as distance_to_center_meters
FROM kmb_stops, center_point center
ORDER BY distance_to_center_meters
LIMIT 5;

-- 18. 🌐 Route Diversity Index (How many different first characters do routes have?)
SELECT
    'Route Diversity' as metric,
    COUNT(DISTINCT LEFT(route, 1)) as unique_first_characters,
    COUNT(DISTINCT route) as total_routes,
    ROUND(
        COUNT(DISTINCT LEFT(route, 1))::numeric / COUNT(DISTINCT route) * 100, 2
    ) as diversity_percentage
FROM kmb_routes;

-- =============================================================================
-- 🎨 DATA QUALITY GEMS
-- =============================================================================

-- 19. 🔍 Missing Data Detective
SELECT
    'KMB Routes Missing English Origin' as issue,
    COUNT(*) as count
FROM kmb_routes WHERE orig_en IS NULL
UNION ALL
SELECT
    'KMB Routes Missing English Destination',
    COUNT(*)
FROM kmb_routes WHERE dest_en IS NULL
UNION ALL
SELECT
    'Journey Times with Zero Seconds',
    COUNT(*)
FROM journey_time_data WHERE travel_time_seconds = 0;

-- 20. 🎉 Fun Final Stats
SELECT
    'Total Bus Stops in Hong Kong' as statistic,
    (SELECT COUNT(*) FROM kmb_stops) +
    (SELECT COUNT(*) FROM citybus_stops) +
    (SELECT COUNT(*) FROM nlb_stops) +
    (SELECT COUNT(*) FROM mtrbus_stops) as value
UNION ALL
SELECT
    'Total Route Variations',
    (SELECT COUNT(*) FROM kmb_routes) +
    (SELECT COUNT(*) FROM citybus_routes) +
    (SELECT COUNT(*) FROM nlb_routes) +
    (SELECT COUNT(*) FROM mtrbus_routes) +
    (SELECT COUNT(*) FROM gmb_routes)
UNION ALL
SELECT
    'Journey Time Records',
    (SELECT COUNT(*) FROM journey_time_data)
UNION ALL
SELECT
    'Hourly Journey Time Records',
    (SELECT COUNT(*) FROM hourly_journey_time_data);

-- =============================================================================
-- 🚀 GETTING STARTED TIPS
-- =============================================================================

/*
🎯 HOW TO USE THIS FILE:

1. 📋 Copy any query above and run it in your SQL client
2. 🔄 Modify the LIMIT values to see more/fewer results
3. 🎲 Run query #13 multiple times for different random routes
4. 🔍 Use these as starting points for your own investigations
5. 📊 Export results to CSV for visualization in tools like Excel, Tableau, or Python

🌟 ADVANCED EXPLORATION IDEAS:

- Combine journey time data with stop locations for mapping
- Create time-based heatmaps using hourly data
- Build route optimization algorithms
- Analyze accessibility patterns across Hong Kong
- Find the busiest transit corridors
- Identify underserved areas

💡 PRO TIPS:

- Use ST_AsText(geometry) to see coordinates in readable format
- Try ST_Buffer() for finding stops within a radius
- Experiment with different SIMILARITY() thresholds for fuzzy matching
- Add WHERE clauses to filter by specific areas or route types

Happy exploring! 🚌✨
*/
