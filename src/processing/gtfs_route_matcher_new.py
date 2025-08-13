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
    'KMB': ['KMB', 'LWB', 'KMB+CTB'],
    'CTB': ['CTB', 'CTB+KMB'], 
    'GMB': ['GMB'],
    'MTRB': ['DB'],  # MTR Bus appears as DB in government GTFS
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
        query = """
            SELECT 
                cr.route,
                CASE 
                    WHEN cr.direction = 'outbound' THEN 'O' 
                    WHEN cr.direction = 'inbound' THEN 'I'
                    ELSE cr.direction 
                END as bound,
                '1' as service_type,
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
            GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en
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
                mr.route_no as route,
                mr.direction_bound as bound,
                '1' as service_type,
                '' as orig_en,
                '' as dest_en,
                COUNT(mss.sequence) as stop_count,
                mr.route_no || '-' || mr.direction_bound || '-1' as route_key
            FROM mtrbus_routes mr 
            JOIN mtrbus_stop_sequences mss ON mr.route_no = mss.route_no 
                AND mr.direction_bound = mss.direction_bound
            GROUP BY mr.route_no, mr.direction_bound
        """
    elif operator_name == 'NLB':
        query = """
            SELECT 
                nr.routeNo as route,
                'O' as bound,  -- NLB structure needs investigation
                '1' as service_type,
                '' as orig_en,
                '' as dest_en,
                COUNT(nss.sequence) as stop_count,
                nr.routeNo || '-O-1' as route_key
            FROM nlb_routes nr 
            JOIN nlb_stop_sequences nss ON nr.routeNo = nss.routeNo
            GROUP BY nr.routeNo
        """
    else:
        raise ValueError(f"Unsupported operator: {operator_name}")
    
    return pd.read_sql(query, engine)

def get_government_gtfs_route_stop_counts(engine: Engine, route_numbers: List[str], 
                                         agency_ids: List[str]) -> pd.DataFrame:
    """
    Get stop counts for government GTFS routes.
    
    Args:
        engine: Database engine
        route_numbers: List of route numbers to match
        agency_ids: List of agency IDs to filter by
        
    Returns:
        DataFrame with gov route info and stop counts by direction
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
                 SPLIT_PART(gt.trip_id, '-', 2)
    """
    
    return pd.read_sql(query, engine)

def find_best_matches(operator_routes: pd.DataFrame, 
                     gov_routes: pd.DataFrame,
                     max_stop_diff: int = 10) -> Dict[str, Dict]:
    """
    Find best matches between operator routes and government GTFS routes.
    
    Args:
        operator_routes: DataFrame with operator route stop counts
        gov_routes: DataFrame with government GTFS route stop counts
        max_stop_diff: Maximum allowed stop count difference
        
    Returns:
        Dictionary mapping route_key to best match info
    """
    matches = {}
    
    # Create all potential matches
    potential_matches = []
    
    for _, op_route in operator_routes.iterrows():
        for _, gov_route in gov_routes.iterrows():
            # Only match if route number and direction match
            if (op_route['route'] == gov_route['route_short_name'] and 
                op_route['bound'] == gov_route['bound']):
                
                stop_diff = abs(op_route['stop_count'] - gov_route['stop_count'])
                
                if stop_diff <= max_stop_diff:
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
                        'stop_diff': stop_diff,
                        'gov_route_long_name': gov_route.get('route_long_name', '')
                    })
    
    # Sort by stop difference and pick best match for each route_key
    potential_matches.sort(key=lambda x: (x['route_key'], x['stop_diff'], x['gov_route_id']))
    
    for match in potential_matches:
        route_key = match['route_key']
        if route_key not in matches:
            matches[route_key] = match
    
    return matches

def match_operator_routes_to_government_gtfs(
    engine: Engine,
    operator_name: str,
    route_filter: List[str] = None,
    debug: bool = False
) -> Dict[str, Dict]:
    """
    Main function to match operator routes to government GTFS routes.
    
    Args:
        engine: Database engine
        operator_name: Operator name ('KMB', 'CTB', 'GMB', 'MTRB', 'NLB')
        route_filter: Optional list of route numbers to filter by
        debug: Enable debug output
        
    Returns:
        Dictionary mapping route_key to best match info
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
    
    # Get government GTFS route stop counts
    try:
        gov_routes = get_government_gtfs_route_stop_counts(engine, route_numbers, gov_agency_ids)
        if debug:
            print(f"Found {len(gov_routes)} government GTFS routes")
    except Exception as e:
        if debug:
            print(f"Error getting government routes: {e}")
        return {}
    
    # Find best matches
    matches = find_best_matches(operator_routes, gov_routes)
    
    if debug:
        print(f"Found {len(matches)} matches")
        for route_key, match in matches.items():
            print(f"  {route_key}: stops {match['operator_stops']} → "
                  f"gov route {match['gov_route_id']} stops {match['gov_stops']} "
                  f"(diff: {match['stop_diff']})")
    
    return matches

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
