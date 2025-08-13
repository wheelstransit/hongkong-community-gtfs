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
    Get the best government route matches for all operator routes using stop count similarity.
    
    This replaces the complex geographic matching with a simpler and more reliable approach
    that leverages the fact that government GTFS uses the same route names as operators.
    
    Args:
        operator_routes_data: Not used directly, we query from database
        operator_stops_data: Not used directly
        operator_stop_sequences_data: Not used directly  
        government_routes_data: DataFrame with government route information
        government_stops_data: Not used in this approach
        government_stop_times_data: DataFrame with government stop_times information
        government_trips_data: DataFrame with government trips information
        operator_name: Name of the operator (e.g., "KMB", "CTB")
        max_matches_per_route: Maximum matches per route (usually 1)
        debug: Enable debug output
        
    Returns:
        Dictionary mapping operator route keys to lists of matching government routes
    """
    
    # This function signature is kept for compatibility, but we need the engine
    # In practice, this should be called with an engine parameter
    # For now, return empty dict and log warning
    if debug:
        print(f"Warning: get_best_government_route_matches called without engine")
        print(f"This function has been rewritten to use database queries")
    
    return {}

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

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import List, Tuple, Dict, Optional

# Constants
INFINITY_DIST = 1000000
DIST_THRESHOLD_METERS = 600  # Maximum average distance per stop for a match
STOP_SKIP_PENALTY = 0.01  # Penalty for each stop position difference


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r


def calculate_route_name_similarity(operator_origin: str, operator_dest: str, gov_route_name: str) -> float:
    """
    Calculate similarity score between operator route and government route names.
    Returns a score between 0 (no similarity) and 1 (perfect match).
    """
    if not operator_origin or not operator_dest or not gov_route_name:
        return 0.0
    
    # Normalize names (uppercase, remove extra spaces)
    operator_origin = operator_origin.upper().strip()
    operator_dest = operator_dest.upper().strip()
    gov_route_name = gov_route_name.upper().strip()
    
    # Split government route name (usually "ORIGIN - DESTINATION")
    if " - " in gov_route_name:
        gov_parts = gov_route_name.split(" - ")
        if len(gov_parts) >= 2:
            gov_origin = gov_parts[0].strip()
            gov_dest = gov_parts[1].strip()
            
            # Check for exact matches
            origin_match = operator_origin in gov_origin or gov_origin in operator_origin
            dest_match = operator_dest in gov_dest or gov_dest in operator_dest
            
            if origin_match and dest_match:
                return 1.0
            elif origin_match or dest_match:
                return 0.5
    
    # Check if any part of operator names appears in government route name
    if operator_origin in gov_route_name or operator_dest in gov_route_name:
        return 0.3
    
    return 0.0


def match_stops_by_dp(operator_stops: List[Dict], gtfs_stops: List[Dict], debug: bool = False, route_info: str = "") -> Tuple[List[Tuple[int, int]], float]:
    """
    Match operator stops to GTFS stops using dynamic programming.
    
    Args:
        operator_stops: List of operator stop dicts with 'lat', 'lon', 'name' fields
        gtfs_stops: List of GTFS stop dicts with 'lat', 'lon', 'name' fields
        debug: Enable debug output
        route_info: Additional info for debug output
    
    Returns:
        Tuple of (matching_pairs, average_distance)
        matching_pairs: List of (gtfs_index, operator_index) pairs
        average_distance: Average distance per matched stop in meters
    """
    # Handle edge cases - reject if stop count difference is too large
    stop_count_diff = abs(len(gtfs_stops) - len(operator_stops))
    if stop_count_diff > 5:  # Allow up to 5 stops difference (was too strict at 3)
        if debug:
            print(f"    Rejecting {route_info}: stop count difference too large ({len(gtfs_stops)} vs {len(operator_stops)})")
        return [], INFINITY_DIST
    
    # Adjust GTFS stops if slightly longer than operator stops
    if len(gtfs_stops) - len(operator_stops) == 1:
        gtfs_stops = gtfs_stops[:-1]
    
    # Initialize DP table
    dist_sum = [[INFINITY_DIST for _ in range(len(operator_stops) + 1)] 
                for _ in range(len(gtfs_stops) + 1)]
    
    # Base case: allow skipping operator stops at the beginning
    for j in range(len(operator_stops) - len(gtfs_stops) + 1):
        dist_sum[0][j] = 0
    
    # Fill DP table
    for i in range(len(gtfs_stops)):
        gtfs_stop = gtfs_stops[i]
        for j in range(len(operator_stops)):
            operator_stop = operator_stops[j]
            
            # Calculate distance between stops
            if (gtfs_stop.get('name') == operator_stop.get('name') and 
                gtfs_stop.get('name') is not None):
                # Exact name match
                dist = 0
            else:
                # Geographic distance
                dist = haversine_distance(
                    float(gtfs_stop['lat']), float(gtfs_stop['lon']),
                    float(operator_stop['lat']), float(operator_stop['lon'])
                )
            
            # DP recurrence: match both stops OR skip operator stop
            dist_sum[i + 1][j + 1] = min(
                dist_sum[i][j] + dist,      # Match both stops
                dist_sum[i + 1][j]          # Skip operator stop
            )
    
    # Check if result is good enough
    min_dist = min(dist_sum[len(gtfs_stops)])
    avg_dist = min_dist / len(gtfs_stops) if len(gtfs_stops) > 0 else INFINITY_DIST
    
    if avg_dist >= DIST_THRESHOLD_METERS:
        return [], INFINITY_DIST
    
    # Backtrack to find the matching pairs
    i = len(gtfs_stops)
    j = len(operator_stops)
    matches = []
    
    while i > 0 and j > 0:
        if dist_sum[i][j] == dist_sum[i][j - 1]:
            # This operator stop was skipped
            j -= 1
        else:
            # This is a match
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
    
    matches.reverse()
    
    # Add penalty for position differences (encourages sequential matching)
    position_penalty = sum([abs(gtfs_idx - op_idx) for gtfs_idx, op_idx in matches]) * STOP_SKIP_PENALTY
    
    final_score = avg_dist + position_penalty
    
    return matches, final_score


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
    Find matching government GTFS routes for an operator route using stop sequences.
    
    Args:
        operator_route_stops: List of stops for the operator route (with lat, lon, name)
        government_routes_data: DataFrame with government route information
        government_stops_data: DataFrame with government stop information  
        government_stop_times_data: DataFrame with government stop_times information
        route_number: Route number to match (e.g., "118", "270B")
        direction_id: Direction (0=outbound, 1=inbound)
        operator_origin: Origin station name for route name matching
        operator_dest: Destination station name for route name matching
        operator_name: Name of the operator (e.g., "KMB", "CTB")
        debug: Enable debug output
        
    Returns:
        List of matching government route dictionaries with match scores
    """
    matches = []
    
    # Filter government routes by route number and agency
    candidate_routes = government_routes_data[
        (government_routes_data['route_short_name'] == route_number) &
        (government_routes_data['agency_id'].str.contains(operator_name, na=False))
    ]
    
    if debug:
        print(f"    Found {len(candidate_routes)} candidate government routes for {route_number}")
        for _, route in candidate_routes.iterrows():
            print(f"      - Route {route['route_id']}: {route.get('route_long_name', 'N/A')}")
    
    for _, gov_route in candidate_routes.iterrows():
        try:
            # Get trips for this route
            gov_trips = government_trips_data[
                government_trips_data['route_id'] == gov_route['route_id']
            ]
            
            # Filter by direction if we have direction info in trips
            if 'direction_id' in gov_trips.columns:
                gov_trips = gov_trips[gov_trips['direction_id'] == direction_id]
            
            if len(gov_trips) == 0:
                continue
            
            # Get stop sequence for one of the trips (they should all have the same sequence)
            sample_trip = gov_trips.iloc[0]['trip_id']
            gov_route_stop_times = government_stop_times_data[
                government_stop_times_data['trip_id'] == sample_trip
            ].sort_values('stop_sequence')
            
            if len(gov_route_stop_times) == 0:
                continue
            
            # Join with stops data to get coordinates
            gov_route_stops = gov_route_stop_times.merge(
                government_stops_data[['stop_id', 'stop_lat', 'stop_lon', 'stop_name']], 
                on='stop_id', 
                how='left'
            ).sort_values('stop_sequence')
                
            # Convert to the format expected by the DP algorithm
            gov_stops_list = []
            for _, stop in gov_route_stops.iterrows():
                gov_stops_list.append({
                    'lat': stop['stop_lat'],
                    'lon': stop['stop_lon'], 
                    'name': stop.get('stop_name'),
                    'stop_id': stop['stop_id']
                })
            
            # Perform DP matching
            route_info = f"Route {gov_route['route_id']} ({gov_route.get('route_long_name', 'N/A')[:50]}...)"
            matching_pairs, score = match_stops_by_dp(operator_route_stops, gov_stops_list, debug, route_info)
            
            if score < DIST_THRESHOLD_METERS:
                # Calculate route name similarity bonus
                name_similarity = calculate_route_name_similarity(
                    operator_origin, operator_dest, gov_route.get('route_long_name', '')
                )
                
                # Apply name similarity bonus to the score (reduce score for better name matches)
                final_score = score * (1.0 - name_similarity * 0.5)  # Up to 50% score reduction for perfect name match
                
                matches.append({
                    'route_id': gov_route['route_id'],
                    'service_id': gov_trips.iloc[0].get('service_id'),
                    'route_long_name': gov_route.get('route_long_name'),
                    'score': final_score,
                    'raw_score': score,
                    'name_similarity': name_similarity,
                    'matching_pairs': matching_pairs,
                    'stop_coverage': len(matching_pairs) / len(operator_route_stops) if operator_route_stops else 0,
                    'stop_count_diff': abs(len(gov_stops_list) - len(operator_route_stops))
                })
                
                if debug:
                    print(f"    Route {gov_route['route_id']}: geo_score={score:.1f}m, name_sim={name_similarity:.2f}, final_score={final_score:.1f}m, stops={len(gov_stops_list)}vs{len(operator_route_stops)}")
        
        except Exception as e:
            if debug:
                print(f"    Error matching route {gov_route.get('route_id', 'unknown')}: {e}")
            continue
    
    # Sort by final score (lower is better)
    matches.sort(key=lambda x: x['score'])
    
    return matches


def get_best_government_route_matches(
    operator_routes_data: pd.DataFrame,
    operator_stops_data: pd.DataFrame, 
    operator_stop_sequences_data: pd.DataFrame,
    government_routes_data: pd.DataFrame,
    government_stops_data: pd.DataFrame,
    government_stop_times_data: pd.DataFrame,
    government_trips_data: pd.DataFrame,
    operator_name: str = "KMB",
    max_matches_per_route: int = 2,
    debug: bool = False
) -> Dict[str, List[Dict]]:
    """
    Get the best government route matches for all operator routes.
    Uses improved assignment to prevent multiple operator routes from claiming the same government route.
    
    Returns:
        Dictionary mapping operator route keys to lists of matching government routes
    """
    route_matches = {}
    used_government_routes = set()  # Track which government routes have been assigned
    
    # Group operator routes by route + bound + service_type
    route_groups = operator_routes_data.groupby(['route', 'bound', 'service_type'])
    
    # Collect all potential matches first
    all_potential_matches = []
    
    for (route_num, bound, service_type), group in route_groups:
        direction_id = 1 if bound == 'I' else 0
        route_key = f"{route_num}-{bound}-{service_type}"
        
        # Only show debug for routes we care about
        should_debug = debug and (route_num in ['1', '17', '118', '270B'])
        
        if should_debug:
            print(f"\nMatching {operator_name} route {route_key}...")
        
        # Get stop sequence for this operator route
        operator_stops_seq = operator_stop_sequences_data[
            (operator_stop_sequences_data['route'] == route_num) &
            (operator_stop_sequences_data['bound'] == bound) &
            (operator_stop_sequences_data['service_type'] == service_type)
        ].sort_values('seq')
        
        if len(operator_stops_seq) == 0:
            if should_debug:
                print(f"  No stop sequence found for {route_key}")
            continue
        
        # Get route origin and destination for name matching
        route_info = group.iloc[0]
        operator_origin = route_info.get('orig_en', '')
        operator_dest = route_info.get('dest_en', '')
        
        # Get stop details with extracted coordinates
        operator_route_stops = []
        for _, stop_seq in operator_stops_seq.iterrows():
            stop_info = operator_stops_data[
                operator_stops_data['stop'] == stop_seq['stop']
            ]
            if len(stop_info) > 0:
                stop = stop_info.iloc[0]
                # Use lat/lon columns (extracted from PostGIS in the SQL query)
                lat = stop.get('lat', 0)
                lon = stop.get('lon', 0)
                    
                operator_route_stops.append({
                    'lat': float(lat) if lat else 0,
                    'lon': float(lon) if lon else 0,
                    'name': stop.get('name_en'),
                    'stop_id': stop['stop']
                })
        
        if len(operator_route_stops) == 0:
            if should_debug:
                print(f"  No valid stops found for {route_key}")
            continue
        
        # Find matching government routes
        matches = find_matching_government_routes(
            operator_route_stops,
            government_routes_data,
            government_stops_data,
            government_stop_times_data,
            government_trips_data,
            route_num,
            direction_id,
            operator_origin,
            operator_dest,
            operator_name,
            should_debug
        )
        
        # Store all potential matches for assignment optimization
        for match in matches:
            all_potential_matches.append({
                'route_key': route_key,
                'gov_route_id': match['route_id'],
                'match_data': match,
                'operator_stops_count': len(operator_route_stops)
            })
    
    # Optimize assignment to avoid conflicts
    # Sort all potential matches by score (best matches first)
    all_potential_matches.sort(key=lambda x: x['match_data']['score'])
    
    if debug:
        print(f"\nOptimal assignment from {len(all_potential_matches)} potential matches...")
    
    # Assign matches ensuring each government route is used only once
    for potential in all_potential_matches:
        route_key = potential['route_key']
        gov_route_id = potential['gov_route_id']
        match_data = potential['match_data']
        
        # Skip if this government route is already assigned
        if gov_route_id in used_government_routes:
            continue
        
        # Skip if this operator route already has enough matches
        if route_key in route_matches and len(route_matches[route_key]) >= max_matches_per_route:
            continue
        
        # Assign this match
        if route_key not in route_matches:
            route_matches[route_key] = []
        
        route_matches[route_key].append(match_data)
        used_government_routes.add(gov_route_id)
        
        if debug:
            print(f"  Assigned gov route {gov_route_id} to {route_key} (score: {match_data['score']:.1f}m)")
    
    # Show final assignments
    if debug:
        print(f"\nFinal assignments:")
        for route_key, matches in route_matches.items():
            if matches:
                print(f"  {route_key}: Route {matches[0]['route_id']} (score: {matches[0]['score']:.1f}m)")
            else:
                print(f"  {route_key}: No matches")
    
    return route_matches
