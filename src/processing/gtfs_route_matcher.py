"""
GTFS Route Matching Module

This module implements a robust stop-count-based matching algorithm 
for matching operator routes with government GTFS routes.

The matching algorithm works by:
1. Normalizing direction encoding (O/I vs 1/2 in trip_id)
2. Matching by route number, direction, and operator agency
3. Ranking matches by stop count similarity (closest match wins)
4. Handling complex agency mappings in government GTFS

This approach is much more reliable than geographic matching since
government GTFS uses the same route naming as operators, just with
different service type handling.
"""

import pandas as pd
from typing import List, Tuple, Dict, Optional
from sqlalchemy.engine import Engine

# Agency mapping from operator names to government GTFS agency IDs
AGENCY_MAPPING = {
    'KMB': ['KMB', 'LWB', 'KMB+CTB', 'KWB'],
    'CTB': ['CTB'], 
    'GMB': ['GMB'],
    'MTRB': ['LRTFeeder'],  # MTR Bus appears as LRTFeeder in government GTFS
    'NLB': ['NLB']
}

def normalize_direction(direction: str) -> str:
    """
    Normalize direction strings to O/I format.
    
    Args:
        direction: Direction string ('outbound', 'inbound', 'O', 'I', '1', '2')
        
    Returns:
        Normalized direction ('O' for outbound, 'I' for inbound)
    """
    direction = str(direction).lower().strip()
    if direction in ['outbound', 'o', '1']:
        return 'O'
    elif direction in ['inbound', 'i', '2']:
        return 'I'
    else:
        return direction.upper()

def extract_direction_from_trip_id(trip_id: str) -> str:
    """
    Extract direction from government GTFS trip_id.
    Trip IDs follow pattern: route_id-direction-service-time
    Direction 1 = outbound (O), Direction 2 = inbound (I)
    
    Args:
        trip_id: Government GTFS trip ID
        
    Returns:
        Direction ('O' or 'I')
    """
    try:
        direction_part = trip_id.split('-')[1]
        return 'O' if direction_part == '1' else 'I'
    except (IndexError, AttributeError):
        return 'UNKNOWN'

def get_operator_route_stop_counts(engine: Engine, operator_name: str) -> pd.DataFrame:
    """
    Get stop counts for all operator routes by route/bound/service_type.
    
    Args:
        engine: Database engine
        operator_name: Operator name ('KMB', 'CTB', 'GMB', 'MTRB', 'NLB')
        
    Returns:
        DataFrame with columns: route, bound, service_type, stop_count, route_key
    """
    
    if operator_name == 'KMB':
        query = """
            SELECT 
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
            GROUP BY k.route, k.bound, k.service_type, k.orig_en, k.dest_en
        """
    elif operator_name == 'CTB':
        # For CTB, handle circular routes by using government GTFS stop counts
        query = f"""
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
                GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en, cr.unique_route_id
            ),
            circular_routes_raw AS (
                -- First, detect circular routes and get stop counts per route_id
                SELECT 
                    gr.route_short_name as route,
                    COUNT(DISTINCT gst.stop_sequence) as gov_stop_count
                FROM gov_gtfs_routes gr
                JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
                JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
                WHERE gr.agency_id = 'CTB'
                GROUP BY gr.route_short_name, gr.route_id
                HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
            ),
            circular_routes AS (
                -- Then, get average stop count per route (in case of multiple route_ids)
                SELECT 
                    route,
                    ROUND(AVG(gov_stop_count))::int as avg_gov_stop_count
                FROM circular_routes_raw
                GROUP BY route
            )
            SELECT 
                rs.route,
                CASE 
                    WHEN rs.route IN (SELECT route FROM circular_routes) THEN 'O'  -- Circular routes: outbound only
                    WHEN rs.direction = 'outbound' THEN 'O' 
                    WHEN rs.direction = 'inbound' THEN 'I'
                    ELSE rs.direction 
                END as bound,
                '1' as service_type,
                rs.orig_en,
                rs.dest_en,
                CASE 
                    WHEN rs.route IN (SELECT route FROM circular_routes) THEN 
                        (SELECT avg_gov_stop_count FROM circular_routes WHERE route = rs.route)  -- Use government GTFS count for circular
                    ELSE rs.stop_count
                END as stop_count,
                rs.route || '-' || CASE 
                    WHEN rs.route IN (SELECT route FROM circular_routes) THEN 'O'
                    WHEN rs.direction = 'outbound' THEN 'O' 
                    WHEN rs.direction = 'inbound' THEN 'I'
                    ELSE rs.direction 
                END || '-1' as route_key
            FROM route_stops rs
            WHERE rs.route NOT IN (SELECT route FROM circular_routes)  -- Regular routes: both directions
                OR (rs.route IN (SELECT route FROM circular_routes) AND rs.direction = 'outbound')  -- Circular: outbound only
        """
    elif operator_name == 'GMB':
        # GMB has different structure - route_code is the route number
        query = """
            SELECT 
                gr.route_code as route,
                'O' as bound,  -- GMB doesn't have clear direction info
                '1' as service_type,
                '' as orig_en,
                '' as dest_en,
                COUNT(gss.sequence) as stop_count,
                gr.route_code || '-O-1' as route_key
            FROM gmb_routes gr 
            JOIN gmb_stop_sequences gss ON gr.route_code = gss.route_code 
                AND gr.region = gss.region
            GROUP BY gr.route_code, gr.region
        """
    elif operator_name == 'MTRB':
        query = """
            SELECT 
                mr.route_id as route,
                mss.direction as bound,
                '1' as service_type,
                '' as orig_en,
                '' as dest_en,
                COUNT(mss.station_seqno) as stop_count,
                mr.route_id || '-' || mss.direction || '-1' as route_key
            FROM mtrbus_routes mr 
            JOIN mtrbus_stop_sequences mss ON mr.route_id = mss.route_id 
            GROUP BY mr.route_id, mss.direction
        """
    elif operator_name == 'NLB':
        query = """
            SELECT 
                nr."routeNo" as route,
                'O' as bound,  -- NLB structure needs investigation
                '1' as service_type,
                '' as orig_en,
                '' as dest_en,
                COUNT(nss.sequence) as stop_count,
                nr."routeNo" || '-O-1' as route_key
            FROM nlb_routes nr 
            JOIN nlb_stop_sequences nss ON nr."routeId" = nss."routeId"
            GROUP BY nr."routeNo"
        """
    else:
        raise ValueError(f"Unsupported operator: {operator_name}")
    
    return pd.read_sql(query, engine)

def get_government_gtfs_route_stop_counts(engine: Engine, route_numbers: List[str], 
                                         agency_ids: List[str]) -> pd.DataFrame:
    """
    Get stop counts for government GTFS routes with service information.
    
    Args:
        engine: Database engine
        route_numbers: List of route numbers to match
        agency_ids: List of agency IDs to filter by
        
    Returns:
        DataFrame with gov route info, service patterns, and stop counts by direction
    """
    
    if not route_numbers or not agency_ids:
        return pd.DataFrame()
    
    route_list = "','".join(route_numbers)
    agency_list = "','".join(agency_ids)
    
    query = f"""
        SELECT 
            gr.route_id,
            gr.route_short_name,
            gr.route_long_name,
            gr.agency_id,
            gt.service_id,
            CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '1' THEN 'O' 
                 WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 'I'
                 ELSE 'UNKNOWN' END as bound,
            COUNT(DISTINCT gst.stop_sequence) as stop_count
        FROM gov_gtfs_routes gr
        JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
        JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
        WHERE gr.route_short_name IN ('{route_list}')
            AND gr.agency_id IN ('{agency_list}')
        GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, gr.agency_id,
                 gt.service_id, SPLIT_PART(gt.trip_id, '-', 2)
    """
    
    return pd.read_sql(query, engine)

def find_best_matches(operator_routes: pd.DataFrame, 
                     gov_routes: pd.DataFrame,
                     max_stop_diff: int = 10,
                     operator_name: str = None) -> Dict[str, List[Dict]]:
    """
    Find best matches between operator routes and government GTFS routes.
    
    Args:
        operator_routes: DataFrame with operator route stop counts
        gov_routes: DataFrame with government GTFS route stop counts
        max_stop_diff: Maximum allowed stop count difference
        operator_name: Name of the operator to prioritize matching
        
    Returns:
        Dictionary mapping route_key to list of best service matches only
    """
    from tqdm import tqdm
    
    matches = {}
    
    # Create all potential matches
    potential_matches = []
    
    same_agency_map = {
        'KMB': ['KMB', 'LWB', 'KMB+CTB', 'KWB'],
        'CTB': ['CTB'],
        'GMB': ['GMB'],
        'MTRB': ['LRTFeeder'],
        'NLB': ['NLB']
    }
    valid_agencies = same_agency_map.get(operator_name, [operator_name])

    # Use tqdm to show progress bar for route matching
    for _, op_route in tqdm(operator_routes.iterrows(), 
                           total=len(operator_routes), 
                           desc="Finding route matches", 
                           unit="routes"):
        for _, gov_route in gov_routes.iterrows():
            # Only match if route number and direction match
            if (op_route['route'] == gov_route['route_short_name'] and 
                op_route['bound'] == gov_route['bound']):
                
                stop_diff = abs(op_route['stop_count'] - gov_route['stop_count'])
                
                if stop_diff <= max_stop_diff:
                    if gov_route['agency_id'] in valid_agencies:
                        potential_matches.append({
                            'route_key': op_route['route_key'],
                            'operator_route': op_route['route'],
                            'operator_bound': op_route['bound'],
                            'operator_service_type': op_route['service_type'],
                            'operator_stops': op_route['stop_count'],
                            'gov_route_id': gov_route['route_id'],
                            'gov_route_name': gov_route['route_short_name'],
                            'gov_bound': gov_route['bound'],
                            'gov_stops': gov_route['stop_count'],
                            'gov_agency': gov_route['agency_id'],
                            'gov_service_id': gov_route['service_id'],
                            'stop_diff': stop_diff,
                            'gov_route_long_name': gov_route.get('route_long_name', '')
                        })
    
    # Group by route_key and keep only the best matches (lowest stop_diff) for each route
    if potential_matches:
        print(f"Processing {len(potential_matches)} potential matches...")
    
    potential_matches.sort(key=lambda x: (x['route_key'], x['stop_diff'], x['gov_service_id']))
    
    current_route_key = None
    best_stop_diff = None
    
    for match in potential_matches:
        route_key = match['route_key']
        
        # Starting a new route_key
        if route_key != current_route_key:
            current_route_key = route_key
            best_stop_diff = match['stop_diff']
            matches[route_key] = []
        
        # Only keep matches with the same (best) stop_diff for this route
        if match['stop_diff'] == best_stop_diff:
            matches[route_key].append(match)
    
    return matches

def _output_unmatched_routes(operator_routes: pd.DataFrame, 
                            gov_routes: pd.DataFrame, 
                            matches: Dict[str, List[Dict]], 
                            operator_name: str) -> None:
    """
    Output unmatched routes to a text file for analysis.
    
    Args:
        operator_routes: DataFrame with operator route stop counts
        gov_routes: DataFrame with government GTFS route stop counts  
        matches: Dictionary of successful matches (now list of services per route)
        operator_name: Name of the operator (for filename)
    """
    matched_operator_keys = set(matches.keys())
    # Flatten all matched government route IDs from all service patterns
    matched_gov_route_ids = set()
    for route_matches in matches.values():
        for match in route_matches:
            matched_gov_route_ids.add(match['gov_route_id'])
    
    # Find unmatched operator routes
    unmatched_operator = operator_routes[~operator_routes['route_key'].isin(matched_operator_keys)]
    
    # Find unmatched government routes
    unmatched_gov = gov_routes[~gov_routes['route_id'].isin(matched_gov_route_ids)]
    
    filename = f"unmatched_routes_{operator_name.lower()}.txt"
    
    # Count total service patterns matched
    total_service_patterns = sum(len(service_list) for service_list in matches.values())
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"UNMATCHED ROUTES ANALYSIS FOR {operator_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        f.write(f"Total {operator_name} routes: {len(operator_routes)}\n")
        f.write(f"Total government GTFS routes: {len(gov_routes)}\n") 
        f.write(f"Successful route matches: {len(matches)}\n")
        f.write(f"Total service patterns matched: {total_service_patterns}\n")
        f.write(f"Unmatched {operator_name} routes: {len(unmatched_operator)}\n")
        f.write(f"Unmatched government routes: {len(unmatched_gov)}\n\n")
        
        # Show matched routes with service patterns
        f.write(f"MATCHED {operator_name} ROUTES (with service patterns):\n")
        f.write("-" * 50 + "\n")
        for route_key, service_matches in matches.items():
            f.write(f"Route Key: {route_key}\n")
            for i, match in enumerate(service_matches):
                f.write(f"  Service {i+1}: ID {match['gov_service_id']}, "
                       f"Stop diff: {match['stop_diff']}, "
                       f"Gov Route: {match['gov_route_id']}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Unmatched operator routes
        f.write(f"UNMATCHED {operator_name} ROUTES:\n")
        f.write("-" * 30 + "\n")
        for _, route in unmatched_operator.iterrows():
            f.write(f"Route: {route['route']}, Direction: {route['bound']}, Service: {route['service_type']}, Stops: {route['stop_count']}\n")
            if 'orig_en' in route and 'dest_en' in route:
                f.write(f"  Origin: {route['orig_en']}\n")
                f.write(f"  Destination: {route['dest_en']}\n")
            f.write(f"  Route Key: {route['route_key']}\n\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        # Unmatched government routes
        f.write("UNMATCHED GOVERNMENT GTFS ROUTES:\n")
        f.write("-" * 35 + "\n")
        for _, route in unmatched_gov.iterrows():
            f.write(f"Route ID: {route['route_id']}, Name: {route['route_short_name']}, Direction: {route['bound']}, Stops: {route['stop_count']}\n")
            f.write(f"  Agency: {route['agency_id']}, Service ID: {route['service_id']}\n")
            if 'route_long_name' in route:
                f.write(f"  Long Name: {route['route_long_name']}\n")
            f.write("\n")
        
        # Potential matching issues analysis
        f.write("\n" + "=" * 50 + "\n")
        f.write("POTENTIAL MATCHING ISSUES ANALYSIS:\n")
        f.write("-" * 40 + "\n\n")
        
        # Group unmatched operator routes by route number
        unmatched_by_route = unmatched_operator.groupby('route')
        for route_num, group in unmatched_by_route:
            # Check if there are government routes with same number
            matching_gov = gov_routes[gov_routes['route_short_name'] == route_num]
            if not matching_gov.empty:
                f.write(f"Route {route_num} - {operator_name} unmatched but government routes exist:\n")
                f.write(f"  {operator_name} variants: {len(group)} routes\n")
                for _, op_route in group.iterrows():
                    f.write(f"    {op_route['bound']}-{op_route['service_type']}: {op_route['stop_count']} stops\n")
                f.write(f"  Government variants: {len(matching_gov)} routes\n")
                for _, gov_route in matching_gov.iterrows():
                    f.write(f"    {gov_route['bound']}: {gov_route['stop_count']} stops (Agency: {gov_route['agency_id']}, Service: {gov_route['service_id']})\n")
                f.write("\n")
    
    print(f"Unmatched routes analysis written to: {filename}")
    print(f"Matched {len(matches)} routes with {total_service_patterns} total service patterns")

def match_operator_routes_to_government_gtfs(
    engine: Engine,
    operator_name: str,
    route_filter: List[str] = None,
    debug: bool = False
) -> Dict[str, List[Dict]]:
    """
    Main function to match operator routes to government GTFS routes.
    
    Args:
        engine: Database engine
        operator_name: Operator name ('KMB', 'CTB', 'GMB', 'MTRB', 'NLB')
        route_filter: Optional list of route numbers to filter by
        debug: Enable debug output
        
    Returns:
        Dictionary mapping route_key to list of service matches
    """
    
    if debug:
        print(f"Matching {operator_name} routes to government GTFS...")
    
    # Get operator route stop counts
    try:
        operator_routes = get_operator_route_stop_counts(engine, operator_name)
        if debug:
            print(f"Found {len(operator_routes)} {operator_name} routes")
    except Exception as e:
        if debug:
            print(f"Error getting {operator_name} routes: {e}")
        return {}
    
    if operator_routes.empty:
        if debug:
            print(f"No {operator_name} routes found")
        return {}
    
    # Filter routes if specified
    if route_filter:
        operator_routes = operator_routes[operator_routes['route'].isin(route_filter)]
        if debug:
            print(f"Filtered to {len(operator_routes)} routes: {route_filter}")
    
    # Get unique route numbers
    route_numbers = operator_routes['route'].unique().tolist()
    
    # Get corresponding government agency IDs
    gov_agency_ids = AGENCY_MAPPING.get(operator_name, [])
    if not gov_agency_ids:
        if debug:
            print(f"No agency mapping found for {operator_name}")
        return {}
    
    # Get government GTFS route stop counts with service information
    try:
        gov_routes = get_government_gtfs_route_stop_counts(engine, route_numbers, gov_agency_ids)
        if debug and operator_name == 'KMB':
            kmb_90_gov_routes = gov_routes[gov_routes['route_short_name'] == '90']
            if not kmb_90_gov_routes.empty:
                print("DEBUG: KMB-90 gov_routes:")
                print(kmb_90_gov_routes)
        if debug:
            print(f"Found {len(gov_routes)} government GTFS route-service combinations")
    except Exception as e:
        if debug:
            print(f"Error getting government routes: {e}")
        return {}
    
    # Find best matches
    matches = find_best_matches(operator_routes, gov_routes, operator_name=operator_name)
    
    total_service_patterns = sum(len(service_list) for service_list in matches.values())
    if debug:
        print(f"Found {len(matches)} route matches with {total_service_patterns} total service patterns")
    
    # Output unmatched routes to file for analysis
    _output_unmatched_routes(operator_routes, gov_routes, matches, operator_name)
    
    return matches

def match_operator_routes_with_coop_fallback(
    engine: Engine,
    operator_name: str,
    route_filter: List[str] = None,
    debug: bool = False
) -> Dict[str, List[Dict]]:
    """
    Enhanced matching function that handles co-op routes by preferring KMB matching.
    For CTB routes that are operated jointly with KMB (agency_id = 'KMB+CTB'), 
    use KMB route data instead of CTB data for better accuracy.
    
    Args:
        engine: Database engine
        operator_name: Operator name ('KMB', 'CTB', 'GMB', 'MTRB', 'NLB')
        route_filter: Optional list of route numbers to filter by
        debug: Enable debug output
        
    Returns:
        Dictionary mapping route_key to list of service matches
    """
    
    # For CTB, check for co-op routes and use KMB data where applicable
    if operator_name == 'CTB':
        # Get CTB routes that are co-operated with KMB
        coop_routes_query = """
            SELECT DISTINCT gr.route_short_name 
            FROM gov_gtfs_routes gr 
            WHERE gr.agency_id = 'KMB+CTB'
        """
        coop_routes_df = pd.read_sql(coop_routes_query, engine)
        coop_routes = coop_routes_df['route_short_name'].tolist() if not coop_routes_df.empty else []
        
        if debug and coop_routes:
            print(f"Found {len(coop_routes)} co-op routes (KMB+CTB): {coop_routes[:10]}...")
        
        # Get regular CTB matches first
        ctb_matches = match_operator_routes_to_government_gtfs(
            engine=engine,
            operator_name="CTB", 
            route_filter=route_filter,
            debug=debug
        )
        
        # For co-op routes, also get KMB matches and prefer them
        if coop_routes:
            coop_route_filter = [r for r in coop_routes if not route_filter or r in route_filter]
            if coop_route_filter:
                if debug:
                    print(f"Getting KMB matches for co-op routes: {coop_route_filter}")
                
                kmb_matches = match_operator_routes_to_government_gtfs(
                    engine=engine,
                    operator_name="KMB",
                    route_filter=coop_route_filter,
                    debug=debug
                )
                
                # Replace CTB matches with KMB matches for co-op routes where KMB has better data
                for route_key, kmb_services in kmb_matches.items():
                    route_num = route_key.split('-')[0]
                    if route_num in coop_routes:
                        # Convert KMB route_key format to CTB format for compatibility
                        for bound in ['O', 'I']:
                            ctb_route_key = f"{route_num}-{bound}-1"
                            kmb_route_key = route_key
                            
                            if (kmb_route_key.endswith(f'-{bound}-1') or kmb_route_key.endswith(f'-{bound}-2')) and kmb_services:
                                if debug:
                                    pass  # Debug output commented out
                                    #print(f"Using KMB data for co-op route {ctb_route_key} (replacing CTB data)")
                                ctb_matches[ctb_route_key] = kmb_services
        
        return ctb_matches
    
    else:
        # For non-CTB operators, use standard matching
        return match_operator_routes_to_government_gtfs(
            engine=engine,
            operator_name=operator_name,
            route_filter=route_filter,
            debug=debug
        )

# Legacy function stubs for compatibility
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Legacy function - no longer used."""
    return 0.0

def calculate_route_name_similarity(operator_origin: str, operator_dest: str, gov_route_name: str) -> float:
    """Legacy function - no longer used."""
    return 0.0

def match_stops_by_dp(operator_stops: List[Dict], gtfs_stops: List[Dict], 
                     debug: bool = False, route_info: str = "") -> Tuple[List[Tuple[int, int]], float]:
    """Legacy function - no longer used."""
    return [], 0.0

def find_matching_government_routes(
    operator_route_stops: List[Dict],
    government_routes_data: pd.DataFrame,
    government_stops_data: pd.DataFrame,
    government_stop_times_data: pd.DataFrame,
    government_trips_data: pd.DataFrame,
    route_number: str,
    direction_id: int,
    operator_origin: str = "",
    operator_dest: str = "",
    operator_name: str = "KMB",
    debug: bool = False
) -> List[Dict]:
    """
    Legacy function - replaced by new stop-count based matching.
    
    This function is kept for compatibility but should not be used.
    Use match_operator_routes_to_government_gtfs instead.
    """
    if debug:
        print("Warning: find_matching_government_routes is deprecated")
        print("Use match_operator_routes_to_government_gtfs instead")
    
    return []

def get_best_government_route_matches(
    operator_routes_data: pd.DataFrame,
    operator_stops_data: pd.DataFrame, 
    operator_stop_sequences_data: pd.DataFrame,
    government_routes_data: pd.DataFrame,
    government_stops_data: pd.DataFrame,
    government_stop_times_data: pd.DataFrame,
    government_trips_data: pd.DataFrame,
    operator_name: str = "KMB",
    max_matches_per_route: int = 1,
    debug: bool = False
) -> Dict[str, List[Dict]]:
    """
    Legacy function - replaced by new stop-count based matching.
    
    This function is kept for compatibility but should not be used.
    Use match_operator_routes_to_government_gtfs instead.
    """
    if debug:
        print("Warning: get_best_government_route_matches is deprecated")
        print("Use match_operator_routes_to_government_gtfs instead")
    
    return {}
