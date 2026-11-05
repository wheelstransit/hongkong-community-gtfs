import logging
import re
from typing import Dict, List

import pandas as pd

from ..db import Store
from .shapes import lat_long_dist

log = logging.getLogger(__name__)

AGENCY_MAPPING = {
    'KMB': ['KMB', 'LWB', 'KMB+CTB', 'KWB'],
    'CTB': ['CTB'], 
    'GMB': ['GMB'],
    'MTRB': ['LRTFeeder'],
    'NLB': ['NLB']
}

STOP_DIFF_TOLERANCE = {
    'KMB': 20,
    'CTB': 12,
    'GMB': 15,
    'MTRB': 12,
    'NLB': 15
}

def get_operator_route_stop_counts(db: Store, operator_name: str) -> pd.DataFrame:
    
    if operator_name == 'KMB':
        query = """
            SELECT 
                k.route,
                k.bound,
                k.service_type::text AS service_type,
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
            GROUP BY cr.route, cr.direction, cr.orig_en, cr.dest_en, cr.unique_route_id
        """
    elif operator_name == 'GMB':
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
        df = db.read(query)
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
    
    return db.read(query)

def get_government_gtfs_route_stop_counts(db: Store, route_numbers: List[str], 
                                         agency_ids: List[str]) -> pd.DataFrame:
    
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
    
    return db.read(query)

def find_best_matches(operator_routes: pd.DataFrame, 
                     gov_routes: pd.DataFrame,
                     db: Store,
                     max_stop_diff: int = 10,
                     operator_name: str = None) -> Dict[str, List[Dict]]:
    
    matches = {}
    
    potential_matches = []
    
    same_agency_map = {
        'KMB': ['KMB', 'LWB', 'KMB+CTB', 'KWB'],
        'CTB': ['CTB'],
        'GMB': ['GMB'],
        'MTRB': ['LRTFeeder'],
        'NLB': ['NLB']
    }
    valid_agencies = same_agency_map.get(operator_name, [operator_name])

    gov_grouped = {k: v for k, v in gov_routes.groupby('route_short_name')}

    for op in operator_routes.itertuples(index=False):
        candidate_gov = gov_grouped.get(op.route, [])
        if len(candidate_gov) == 0:
            continue
        for gov in candidate_gov.itertuples(index=False):
            if operator_name == 'KMB' and getattr(op, 'bound', None) != getattr(gov, 'bound', None):
                continue
            route_long_name = str(getattr(gov, 'route_long_name', '') or '')
            route_long_name_upper = route_long_name.upper()
            sanitized = re.sub(r"\([^\)]*\)", " ", route_long_name_upper)
            sanitized = re.sub(r"\s+", " ", sanitized).strip()
            if any(keyword in sanitized for keyword in ['VIA', 'OMIT', 'SPECIAL', 'EXPRESS', 'SERVICE', 'CORRECTIONAL']):
                continue
            stop_diff = abs(getattr(op, 'stop_count') - getattr(gov, 'stop_count'))
            if stop_diff > max_stop_diff:
                continue
            if getattr(gov, 'agency_id') not in valid_agencies:
                continue
            match_record = {
                'route_key': getattr(op, 'route_key'),
                'operator_route': getattr(op, 'route'),
                'operator_bound': getattr(op, 'bound'),
                'operator_service_type': getattr(op, 'service_type'),
                'operator_stops': getattr(op, 'stop_count'),
                'gov_route_id': getattr(gov, 'route_id'),
                'gov_route_name': getattr(gov, 'route_short_name'),
                'gov_bound': getattr(gov, 'bound'),
                'gov_stops': getattr(gov, 'stop_count'),
                'gov_agency': getattr(gov, 'agency_id'),
                'gov_service_id': getattr(gov, 'service_id'),
                'stop_diff': stop_diff,
                'gov_route_long_name': route_long_name
            }
            if hasattr(op, 'routeid'):
                match_record['routeid'] = getattr(op, 'routeid')
            potential_matches.append(match_record)
    
    potential_matches.sort(key=lambda x: (x['route_key'], x['stop_diff'], x['gov_service_id']))
    
    best_matches_by_route = {}
    for match in potential_matches:
        route_key = match['route_key']
        if route_key not in best_matches_by_route or match['stop_diff'] < best_matches_by_route[route_key][0]['stop_diff']:
            best_matches_by_route[route_key] = [match]
        elif match['stop_diff'] == best_matches_by_route[route_key][0]['stop_diff']:
            best_matches_by_route[route_key].append(match)

    for route_key, route_matches in best_matches_by_route.items():
        matches[route_key] = route_matches

    if operator_name == 'NLB':
        deduped = {}
        for rk, rm_list in matches.items():
            by_route_id = {}
            for m in rm_list:
                by_route_id.setdefault(m['gov_route_id'], []).append(m)
            ranked = sorted(by_route_id.items(), key=lambda x: (-len(x[1]), sum(mm['stop_diff'] for mm in x[1]) / len(x[1]), x[0]))
            deduped[rk] = ranked[0][1]
        matches = deduped

    return matches

def _ctb_route_id_map(route_df: pd.DataFrame) -> Dict[str, str]:
    route_df.columns = [c.strip().upper() for c in route_df.columns]
    ctb = route_df[route_df["COMPANY_CODE"] == "CTB"]
    standard = ctb[ctb["SPECIAL_TYPE"].astype(int) == 0]
    non_standard = ctb[~ctb["ROUTE_NAMEC"].isin(set(standard["ROUTE_NAMEC"].unique()))]
    firsts = [df.sort_values(["ROUTE_NAMEC", "ROUTE_ID"], kind="stable").groupby("ROUTE_NAMEC", as_index=False).first()
              for df in (standard, non_standard)]
    return {str(r["ROUTE_NAMEC"]).strip(): str(r["ROUTE_ID"]).strip() for _, r in pd.concat(firsts, ignore_index=True).iterrows()}

def match_ctb_routes_to_government_gtfs(
    db: Store,
    operator_name: str,
    route_filter: List[str] = None
) -> Dict[str, str]:

    mapping = _ctb_route_id_map(db.read("SELECT * FROM td_routes"))

    if route_filter:
        mapping = {k: v for k, v in mapping.items() if k in route_filter}

    return mapping

def match_gmb_routes_by_first_stop_location(
    db: Store,
    operator_name: str,
    route_filter: List[str] = None,
    max_distance_meters: float = 500.0
) -> Dict[str, List[Dict]]:

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
        base_routes = [r.split('-')[-1] if '-' in r else r for r in route_filter]
        route_filter_str = "', '".join(base_routes)
        operator_query += f" AND gr.route_code IN ('{route_filter_str}')"
    
    operator_routes = db.read(operator_query)
    
    if operator_routes.empty:
        return {}

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
    
    gov_routes = db.read(gov_query)
    
    if gov_routes.empty:
        return {}

    matches: Dict[str, List[Dict]] = {}

    for _, op_route in operator_routes.iterrows():
        route_code = op_route['route_code']
        region = op_route['region']
        route_seq = int(op_route['route_seq']) if pd.notna(op_route['route_seq']) else 1
        bound = 'O' if route_seq == 1 else 'I'
        route_key = f"{region}-{route_code}-{bound}-1"
        
        gov_candidates = gov_routes[gov_routes['route_short_name'] == route_code]
        
        if gov_candidates.empty:
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

    return matches

def match_operator_routes_to_government_gtfs(
    db: Store,
    operator_name: str,
    route_filter: List[str] = None
) -> Dict[str, List[Dict]]:
    
    if operator_name == 'CTB':
        return match_ctb_routes_to_government_gtfs(db, operator_name, route_filter)

    if operator_name == 'GMB':
        return match_gmb_routes_by_first_stop_location(db, operator_name, route_filter)
    
    try:
        operator_routes = get_operator_route_stop_counts(db, operator_name)
    except Exception as e:
        log.warning(f"Error getting {operator_name} routes: {e}")
        return {}
    
    if operator_routes.empty:
        return {}
    
    if route_filter:
        operator_routes = operator_routes[operator_routes['route'].isin(route_filter)]
    
    route_numbers = operator_routes['route'].unique().tolist()
    
    gov_agency_ids = AGENCY_MAPPING.get(operator_name, [])
    if not gov_agency_ids:
        return {}
    
    try:
        gov_routes = get_government_gtfs_route_stop_counts(db, route_numbers, gov_agency_ids)
    except Exception as e:
        log.warning(f"Error getting government routes: {e}")
        return {}
    
    stop_diff_limit = STOP_DIFF_TOLERANCE.get(operator_name, 10)
    matches = find_best_matches(
        operator_routes,
        gov_routes,
        db,
        max_stop_diff=stop_diff_limit,
        operator_name=operator_name
    )
    
    return matches

