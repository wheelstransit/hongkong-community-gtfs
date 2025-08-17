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
from typing import List, Tuple, Dict, Optional
from sqlalchemy.engine import Engine
import pandas as pd
from fuzzywuzzy import process
from .shapes import lat_long_dist  # Import distance calculation function
import os

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
        query = f"""
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
            GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en, cr.unique_route_id
        """
    elif operator_name == 'GMB':
        # GMB has different structure - route_code is the route number, region matters!
        query = """
            SELECT 
                gr.region || '-' || gr.route_code as route,
                'O' as bound,  -- GMB doesn't have clear direction info
                '1' as service_type,
                '' as orig_en,
                '' as dest_en,
                COUNT(gss.sequence) as stop_count,
                gr.region || '-' || gr.route_code || '-O-1' as route_key
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
                nr."routeName_e" as route_name,
                nr."routeId" as routeid,
                COUNT(nss.sequence) as stop_count
            FROM nlb_routes nr 
            JOIN nlb_stop_sequences nss ON nr."routeId" = nss."routeId"
            GROUP BY nr."routeNo", nr."routeName_e", nr."routeId"
        """
        df = pd.read_sql(query, engine)
        df[['origin', 'destination']] = df['route_name'].str.split(' > ', expand=True)
        
        routes = []
        for route_no, group in df.groupby('route'):
            if len(group) == 2:
                r1 = group.iloc[0]
                r2 = group.iloc[1]
                if r1['origin'] == r2['destination'] and r1['destination'] == r2['origin']:
                    routes.append({'route': route_no, 'bound': 'O', 'service_type': '1', 'stop_count': r1['stop_count'], 'route_key': f"{route_no}-O-1", 'routeid': r1['routeid']})
                    routes.append({'route': route_no, 'bound': 'I', 'service_type': '1', 'stop_count': r2['stop_count'], 'route_key': f"{route_no}-I-1", 'routeid': r2['routeid']})
                else:
                    routes.append({'route': route_no, 'bound': 'O', 'service_type': '1', 'stop_count': r1['stop_count'], 'route_key': f"{route_no}-O-1", 'routeid': r1['routeid']})
            else:
                for _, r in group.iterrows():
                    routes.append({'route': route_no, 'bound': 'O', 'service_type': '1', 'stop_count': r['stop_count'], 'route_key': f"{route_no}-O-1", 'routeid': r['routeid']})
        return pd.DataFrame(routes)
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
                     engine: Engine, # Add engine for db access
                     max_stop_diff: int = 10,
                     operator_name: str = None) -> Dict[str, List[Dict]]:
    """
    Find best matches between operator routes and government GTFS routes.
    
    Args:
        operator_routes: DataFrame with operator route stop counts
        gov_routes: DataFrame with government GTFS route stop counts
        engine: Database engine
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
            if (op_route['route'] == gov_route['route_short_name']):
                # Enforce bound alignment for KMB to guarantee O=1 and I=2 mapping
                if operator_name == 'KMB' and op_route.get('bound') != gov_route.get('bound'):
                    continue
                # Skip special route variants for better matching
                route_long_name = str(gov_route.get('route_long_name', ''))
                if any(keyword in route_long_name.upper() for keyword in ['VIA', 'OMIT', 'SPECIAL', 'EXPRESS', 'SERVICE', 'CORRECTIONAL', 'HOSPITAL', 'SCHOOL']):
                    continue
                stop_diff = abs(op_route['stop_count'] - gov_route['stop_count'])
                if stop_diff <= max_stop_diff:
                    if gov_route['agency_id'] in valid_agencies:
                        match_record = {
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
                        }
                        # Preserve operator routeid for NLB (used later in export)
                        if 'routeid' in op_route.index:
                            match_record['routeid'] = op_route['routeid']
                        potential_matches.append(match_record)
    
    # Group by route_key and keep only the best matches (lowest stop_diff) for each route
    if potential_matches:
        print(f"Processing {len(potential_matches)} potential matches...")
    
    potential_matches.sort(key=lambda x: (x['route_key'], x['stop_diff'], x['gov_service_id']))
    
    # Group by route_key and keep all service patterns for the best stop_diff
    best_matches_by_route = {}
    for match in potential_matches:
        route_key = match['route_key']
        if route_key not in best_matches_by_route or match['stop_diff'] < best_matches_by_route[route_key][0]['stop_diff']:
            best_matches_by_route[route_key] = [match]
        elif match['stop_diff'] == best_matches_by_route[route_key][0]['stop_diff']:
            best_matches_by_route[route_key].append(match)

    # Keep ALL service patterns for each route_key at the best stop_diff
    for route_key, route_matches in best_matches_by_route.items():
        matches[route_key] = route_matches  # Keep all service patterns, not just the first

    # NLB specific dedupe: keep only one gov_route_id per route_key (direction)
    if operator_name == 'NLB':
        deduped = {}
        for rk, rm_list in matches.items():
            # Group by gov_route_id
            by_route_id = {}
            for m in rm_list:
                by_route_id.setdefault(m['gov_route_id'], []).append(m)
            # Rank: most service patterns, then lowest avg stop_diff, then lowest gov_route_id
            ranked = sorted(by_route_id.items(), key=lambda x: (-len(x[1]), sum(mm['stop_diff'] for mm in x[1]) / len(x[1]), x[0]))
            # Keep only top group's matches
            deduped[rk] = ranked[0][1]
        matches = deduped

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
    
    # Count total matches and service patterns
    total_matches = len(matches)
    total_service_patterns = sum(len(service_list) for service_list in matches.values())
    unique_routes_matched = len(set(match['operator_route'] for match_list in matches.values() for match in match_list))
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"UNMATCHED ROUTES ANALYSIS FOR {operator_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        f.write(f"Total {operator_name} routes: {len(operator_routes)}\n")
        f.write(f"Total government GTFS routes: {len(gov_routes)}\n") 
        f.write(f"Successful route matches: {unique_routes_matched} unique route(s) with {total_matches} direction matches\n")
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
    print(f"Matched {unique_routes_matched} unique route(s) with {total_matches} direction matches and {total_service_patterns} service patterns")

def get_government_gtfs_route_info_for_ctb(engine: Engine, route_numbers: List[str], agency_ids: List[str]) -> pd.DataFrame:
    """
    Get stop counts and direction info for government GTFS routes.
    """
    if not route_numbers or not agency_ids:
        return pd.DataFrame()
    
    route_list = "','".join(route_numbers)
    agency_list = "','".join(agency_ids)
    
    query = f"""
        SELECT 
            gr.route_id,
            gr.route_short_name,
            SPLIT_PART(gt.trip_id, '-', 2) as direction,
            COUNT(gst.stop_id) as stop_count
        FROM gov_gtfs_routes gr
        JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
        JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
        WHERE gr.route_short_name IN ('{route_list}')
            AND gr.agency_id IN ('{agency_list}')
        GROUP BY gr.route_id, gr.route_short_name, direction
    """
    
    df = pd.read_sql(query, engine)
    
    # Calculate directions and average stop counts
    gov_route_info = {}
    for route_id, group in df.groupby('route_id'):
        directions = group['direction'].unique()
        if '1' in directions and '2' in directions:
            is_bidirectional = True
        else:
            is_bidirectional = False
        
        gov_route_info[route_id] = {
            'route_short_name': group['route_short_name'].iloc[0],
            'is_bidirectional': is_bidirectional,
            'avg_stop_count_outbound': group[group['direction'] == '1']['stop_count'].mean(),
            'avg_stop_count_inbound': group[group['direction'] == '2']['stop_count'].mean()
        }
        
    return pd.DataFrame.from_dict(gov_route_info, orient='index')

def find_best_match_for_ctb_route(operator_route_variants: pd.DataFrame, gov_routes: pd.DataFrame) -> Optional[str]:
    """
    Finds the best government GTFS route match for a single operator route.
    """
    op_outbound = operator_route_variants[operator_route_variants['bound'] == 'O']
    op_inbound = operator_route_variants[operator_route_variants['bound'] == 'I']

    if op_outbound.empty or op_inbound.empty:
        return None # Requires both directions

    op_stop_count_outbound = op_outbound['stop_count'].iloc[0]
    op_stop_count_inbound = op_inbound['stop_count'].iloc[0]

    best_match = None
    min_diff = float('inf')

    for gov_route_id, gov_route in gov_routes.iterrows():
        if not gov_route['is_bidirectional']:
            continue

        diff_outbound = abs(op_stop_count_outbound - gov_route['avg_stop_count_outbound'])
        diff_inbound = abs(op_stop_count_inbound - gov_route['avg_stop_count_inbound'])
        total_diff = diff_outbound + diff_inbound

        if total_diff < min_diff:
            min_diff = total_diff
            best_match = gov_route_id
            
    return str(best_match) if best_match is not None else None

def get_government_gtfs_route_info_for_nlb(engine: Engine, route_numbers: List[str], agency_ids: List[str]) -> pd.DataFrame:
    """
    Get stop counts and direction info for government GTFS routes.
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
            SPLIT_PART(gt.trip_id, '-', 2) as direction,
            COUNT(gst.stop_id) as stop_count
        FROM gov_gtfs_routes gr
        JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
        JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id
        WHERE gr.route_short_name IN ('{route_list}')
            AND gr.agency_id IN ('{agency_list}')
        GROUP BY gr.route_id, gr.route_short_name, gr.route_long_name, direction
    """
    
    df = pd.read_sql(query, engine)
    
    # Calculate directions and average stop counts
    gov_route_info = {}
    for route_id, group in df.groupby('route_id'):
        directions = group['direction'].unique()
        if '1' in directions and '2' in directions:
            is_bidirectional = True
        else:
            is_bidirectional = False
        
        gov_route_info[route_id] = {
            'route_short_name': group['route_short_name'].iloc[0],
            'route_long_name': group['route_long_name'].iloc[0],
            'is_bidirectional': is_bidirectional,
            'avg_stop_count_outbound': group[group['direction'] == '1']['stop_count'].mean(),
            'avg_stop_count_inbound': group[group['direction'] == '2']['stop_count'].mean()
        }
        
    return pd.DataFrame.from_dict(gov_route_info, orient='index')

def find_best_match_for_nlb_route(operator_route_variants: pd.DataFrame, gov_routes: pd.DataFrame, engine: Engine) -> Optional[str]:
    """
    Finds the best government GTFS route match for a single operator route.
    """
    op_outbound = operator_route_variants[operator_routeVariants['bound'] == 'O']
    op_inbound = operator_route_variants[operator_routeVariants['bound'] == 'I']

    if op_outbound.empty or op_inbound.empty:
        return None # Requires both directions

    op_stop_count_outbound = op_outbound['stop_count'].iloc[0]
    op_stop_count_inbound = op_inbound['stop_count'].iloc[0]

    best_match = None
    min_diff = float('inf')

    for gov_route_id, gov_route in gov_routes.iterrows():
        if not gov_route['is_bidirectional']:
            continue

        # Get operator stop list
        op_stops_outbound_query = f"""
            SELECT ns."stopName_e" 
            FROM nlb_stop_sequences nss 
            JOIN nlb_stops ns ON nss."stopId"::int = ns."stopId"::int 
            WHERE nss."routeId"::int = {op_outbound['routeid'].iloc[0]} 
            ORDER BY nss.sequence
        """
        op_stops_inbound_query = f"""
            SELECT ns."stopName_e" 
            FROM nlb_stop_sequences nss 
            JOIN nlb_stops ns ON nss."stopId"::int = ns."stopId"::int 
            WHERE nss."routeId"::int = {op_inbound['routeid'].iloc[0]} 
            ORDER BY nss.sequence
        """
        op_stops_outbound = pd.read_sql(op_stops_outbound_query, engine)['stopName_e'].tolist()
        op_stops_inbound = pd.read_sql(op_stops_inbound_query, engine)['stopName_e'].tolist()

        # Get gov stop list
        gov_stops_outbound_query = f"""
            SELECT s.stop_name 
            FROM gov_gtfs_stop_times st 
            JOIN gov_gtfs_stops s ON st.stop_id = s.stop_id 
            WHERE st.trip_id = (
                SELECT trip_id 
                FROM gov_gtfs_trips 
                WHERE route_id = '{gov_route_id}' AND trip_id LIKE '%-%1-%' 
                LIMIT 1
            ) 
            ORDER BY st.stop_sequence
        """
        gov_stops_inbound_query = f"""
            SELECT s.stop_name 
            FROM gov_gtfs_stop_times st 
            JOIN gov_gtfs_stops s ON st.stop_id = s.stop_id 
            WHERE st.trip_id = (
                SELECT trip_id 
                FROM gov_gtfs_trips 
                WHERE route_id = '{gov_route_id}' AND trip_id LIKE '%-%2-%' 
                LIMIT 1
            ) 
            ORDER BY st.stop_sequence
        """
        
        try:
            gov_stops_outbound = pd.read_sql(gov_stops_outbound_query, engine)['stop_name'].str.replace('[NLB] ', '', regex=False).tolist()
            gov_stops_inbound = pd.read_sql(gov_stops_inbound_query, engine)['stop_name'].str.replace('[NLB] ', '', regex=False).tolist()
        except Exception:
            # Skip this government route if query fails
            continue

        # Fuzzy match scores
        score_outbound = 0
        for stop in op_stops_outbound:
            match = process.extractOne(stop, gov_stops_outbound)
            if match:
                score_outbound += match[1]
        avg_score_outbound = score_outbound / len(op_stops_outbound) if op_stops_outbound else 0

        score_inbound = 0
        for stop in op_stops_inbound:
            match = process.extractOne(stop, gov_stops_inbound)
            if match:
                score_inbound += match[1]
        avg_score_inbound = score_inbound / len(op_stops_inbound) if op_stops_inbound else 0

        # Combine scores and penalties
        total_diff = (100 - avg_score_outbound) + (100 - avg_score_inbound)
        total_diff += abs(op_stop_count_outbound - len(gov_stops_outbound)) + abs(op_stop_count_inbound - len(gov_stops_inbound))

        route_long_name = gov_route.get('route_long_name', '')
        if route_long_name and ('VIA' in route_long_name.upper() or 
           'OMIT' in route_long_name.upper() or 
           'SPECIAL' in route_long_name.upper()):
            total_diff += 100 # Increased penalty

        if total_diff < min_diff:
            min_diff = total_diff
            best_match = gov_route_id
            
    return str(best_match) if best_match is not None else None



def match_ctb_routes_to_government_gtfs(
    engine: Engine,
    operator_name: str,
    route_filter: List[str] = None,
    debug: bool = False
) -> Dict[str, str]:
    """
    Main function to match operator routes to government GTFS routes.
    """
    if debug:
        print(f"Matching {operator_name} routes to government GTFS...")

    operator_routes = get_operator_route_stop_counts(engine, operator_name)
    if route_filter:
        operator_routes = operator_routes[operator_routes['route'].isin(route_filter)]

    route_numbers = operator_routes['route'].unique().tolist()
    gov_agency_ids = AGENCY_MAPPING.get(operator_name, [])
    
    gov_routes_info = get_government_gtfs_route_info_for_ctb(engine, route_numbers, gov_agency_ids)

    matches = {}
    for route_num, group in operator_routes.groupby('route'):
        gov_candidates = gov_routes_info[gov_routes_info['route_short_name'] == route_num]
        best_match_id = find_best_match_for_ctb_route(group, gov_candidates)
        if best_match_id:
            matches[route_num] = best_match_id
            if debug and os.environ.get('VERBOSE_MATCH') == '1':
                print(f"Matched route {route_num} to {best_match_id}")

    return matches

def match_nlb_routes_to_government_gtfs(
    engine: Engine,
    operator_name: str,
    route_filter: List[str] = None,
    debug: bool = False
) -> Dict[str, str]:
    """
    Main function to match operator routes to government GTFS routes.
    """
    if debug:
        print(f"Matching {operator_name} routes to government GTFS...")

    operator_routes = get_operator_route_stop_counts(engine, operator_name)
    if route_filter:
        operator_routes = operator_routes[operator_routes['route'].isin(route_filter)]

    route_numbers = operator_routes['route'].unique().tolist()
    gov_agency_ids = AGENCY_MAPPING.get(operator_name, [])
    
    gov_routes_info = get_government_gtfs_route_info_for_nlb(engine, route_numbers, gov_agency_ids)

    matches = {}
    for route_num, group in operator_routes.groupby('route'):
        gov_candidates = gov_routes_info[gov_routes_info['route_short_name'] == route_num]
        best_match_id = find_best_match_for_nlb_route(group, gov_candidates, engine)
        if best_match_id:
            matches[route_num] = best_match_id
            if debug and os.environ.get('VERBOSE_MATCH') == '1':
                print(f"Matched route {route_num} to {best_match_id}")

    return matches

def match_gmb_routes_by_first_stop_location(
    engine: Engine,
    operator_name: str,
    route_filter: List[str] = None,
    debug: bool = False,
    max_distance_meters: float = 500.0
) -> Dict[str, List[Dict]]:
    """
    Match GMB routes to government GTFS routes based on first stop location proximity.
    
    This handles the regional separation (HKI-1 vs NT-1) by matching each regional route
    to the government route with the closest first stop location.
    NOTE: Previously this function incorrectly hard‑coded the route_key to outbound (O) only,
    collapsing both directions and causing only one bound to appear in trips.txt and mixing
    stop sequences across regions. We now preserve the bound derived from route_seq (1=O, 2=I).
    """
    if debug:
        print(f"Matching {operator_name} routes to government GTFS by first stop location...")

    # Get operator routes with first stop locations for each direction
    operator_query = """
        SELECT 
            gr.region || '-' || gr.route_code as route,
            gr.region,
            gr.route_code,
            gss.route_seq,
            gr.region || '-' || gr.route_code || '-' || 
                CASE WHEN gss.route_seq = 1 THEN 'O' ELSE 'I' END || '-1' as route_key,
            ST_X(gs.geometry) as first_stop_lng,
            ST_Y(gs.geometry) as first_stop_lat,
            gs.stop_name_en as first_stop_name
        FROM gmb_routes gr 
        JOIN gmb_stop_sequences gss ON gr.route_code = gss.route_code AND gr.region = gss.region
        JOIN gmb_stops gs ON gss.stop_id = gs.stop_id
        WHERE gss.sequence = 1
    """
    
    if route_filter:
        # Extract base route numbers from regional routes (HKI-1 -> 1)
        base_routes = [r.split('-')[-1] if '-' in r else r for r in route_filter]
        route_filter_str = "', '".join(base_routes)
        operator_query += f" AND gr.route_code IN ('{route_filter_str}')"
    
    operator_routes = pd.read_sql(operator_query, engine)
    
    if operator_routes.empty:
        if debug:
            print("No operator routes found")
        return {}

    # Get government GMB routes with first stop locations
    gov_query = """
        SELECT 
            gr.route_id,
            gr.route_short_name,
            gr.route_long_name,
            gr.agency_id,
            ST_X(gs.geometry) as first_stop_lng,
            ST_Y(gs.geometry) as first_stop_lat,
            gs.stop_name as first_stop_name,
            gt.service_id as gov_service_id
        FROM gov_gtfs_routes gr
        JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
        JOIN gov_gtfs_stop_times gst ON gt.trip_id = gst.trip_id AND gst.stop_sequence = 1
        JOIN gov_gtfs_stops gs ON gst.stop_id = gs.stop_id
        WHERE gr.agency_id = 'GMB'
    """
    
    gov_routes = pd.read_sql(gov_query, engine)
    
    # Filter out special route variants
    if not gov_routes.empty:
        gov_routes = gov_routes[~gov_routes['route_long_name'].str.upper().str.contains(
            'VIA|OMIT|SPECIAL|EXPRESS|SERVICE|CORRECTIONAL|HOSPITAL|SCHOOL', na=False)]
    
    if gov_routes.empty:
        if debug:
            print("No government GMB routes found")
        return {}

    matches: Dict[str, List[Dict]] = {}

    for _, op_route in operator_routes.iterrows():
        route_code = op_route['route_code']
        region = op_route['region']
        route_seq = int(op_route['route_seq']) if pd.notna(op_route['route_seq']) else 1
        bound = 'O' if route_seq == 1 else 'I'
        route_key = f"{region}-{route_code}-{bound}-1"  # preserve direction now
        
        gov_candidates = gov_routes[gov_routes['route_short_name'] == route_code]
        
        if gov_candidates.empty:
            if debug:
                print(f"No government routes found for route {region}-{route_code} ({bound})")
            continue
        
        best_match = None
        min_distance = float('inf')
        
        for _, gov_route in gov_candidates.iterrows():
            distance = lat_long_dist(
                op_route['first_stop_lat'], op_route['first_stop_lng'],
                gov_route['first_stop_lat'], gov_route['first_stop_lng']
            )
            if distance < min_distance and distance <= max_distance_meters:
                min_distance = distance
                best_match = gov_route
        
        if best_match is not None:
            if route_key not in matches:
                matches[route_key] = []
            matches[route_key].append({
                'gov_route_id': best_match['route_id'],
                'gov_service_id': best_match['gov_service_id'],
                'distance_meters': min_distance,
                'operator_first_stop': op_route['first_stop_name'],
                'gov_first_stop': best_match['first_stop_name'],
                'bound': bound
            })
            if debug and os.environ.get('VERBOSE_MATCH') == '1':
                print(f"Matched {region}-{route_code} {bound} to {best_match['route_id']} (distance: {min_distance:.1f}m)")
        else:
            if debug and os.environ.get('VERBOSE_MATCH') == '1':
                print(f"No close match within {max_distance_meters}m for {region}-{route_code} {bound} (closest {min_distance:.1f}m)")

    return matches

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
    
    if operator_name == 'CTB':
        return match_ctb_routes_to_government_gtfs(engine, operator_name, route_filter, debug)

    if operator_name == 'GMB':
        # Use location-based matching for GMB to handle regional routes properly
        return match_gmb_routes_by_first_stop_location(engine, operator_name, route_filter, debug)

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
    matches = find_best_matches(operator_routes, gov_routes, engine, operator_name=operator_name)
    
    total_matches = len(matches)
    total_service_patterns = sum(len(service_list) for service_list in matches.values())
    unique_routes_matched = len(set(match['operator_route'] for match_list in matches.values() for match in match_list))
    if debug:
        print(f"Found {unique_routes_matched} unique route(s) with {total_matches} direction matches and {total_service_patterns} service patterns")
    
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
