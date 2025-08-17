import pandas as pd
import os
import json
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm

def lat_long_dist(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    return c * r * 1000

def generate_shapes_from_csdi_files(output_path, engine, silent=False, gov_route_ids=None):
    waypoints_dir = "waypoints"
    if not os.path.exists(waypoints_dir):
        if not silent:
            print(f"Directory not found: {waypoints_dir}")
        return False, []

    shape_info_list = []
    files_to_process = [f for f in os.listdir(waypoints_dir) if f.endswith('.json') and f != '0versions.json']
    if gov_route_ids:
        files_to_process = [f for f in files_to_process if f.split('-')[0] in gov_route_ids]
    
    with open(output_path, 'w') as f_out:
        f_out.write("shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n")

        for filename in tqdm(files_to_process, desc="Generating shapes from CSDI files", unit="file", disable=silent):
            filepath = os.path.join(waypoints_dir, filename)
            
            try:
                base_filename = os.path.splitext(filename)[0]
                gov_gtfs_id, bound = base_filename.split('-')
            except ValueError:
                if not silent:
                    print(f"Warning: Could not parse filename {filename}")
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    if not silent:
                        print(f"Warning: Could not decode JSON from {filename}")
                    continue

            if not data.get('features'):
                continue

            feature = data['features'][0]
            properties = feature['properties']
            
            shape_id = f"CSDI-{gov_gtfs_id}-{bound}"

            geom_type = feature['geometry']['type']
            coords = feature['geometry']['coordinates']

            if geom_type == 'MultiLineString':
                all_coords = [item for sublist in coords for item in sublist]
            else:
                all_coords = coords

            dist_traveled = 0
            prev_lat, prev_lon = None, None
            for i, (lon, lat) in enumerate(all_coords):
                if prev_lat is not None:
                    dist_traveled += lat_long_dist(prev_lat, prev_lon, lat, lon)
                f_out.write(f"{shape_id},{lat},{lon},{i+1},{dist_traveled}\n")
                prev_lat, prev_lon = lat, lon

            shape_info_list.append({
                'shape_id': shape_id,
                'gov_route_id': properties.get('ROUTE_ID'),
                'bound': bound
            })

    gov_route_ids_in_shapes = [str(s['gov_route_id']) for s in shape_info_list]
    gov_routes_df = pd.read_sql(f"SELECT route_id, agency_id FROM gov_gtfs_routes WHERE route_id IN ({','.join(gov_route_ids_in_shapes)})", engine)
    gov_routes_df['route_id'] = gov_routes_df['route_id'].astype(str)

    shape_info_df = pd.DataFrame(shape_info_list)
    shape_info_df['gov_route_id'] = shape_info_df['gov_route_id'].astype(str)

    shape_info_df = shape_info_df.merge(gov_routes_df, left_on='gov_route_id', right_on='route_id', how='left')
    shape_info_df.drop(columns=['route_id'], inplace=True)

    return True, shape_info_df.to_dict('records')

def match_trips_to_csdi_shapes(trips_df, shape_info, engine, silent=False):
    if not shape_info:
        if not silent:
            print("No shape information available to match.")
        trips_df['shape_id'] = None
        return trips_df

    shape_info_df = pd.DataFrame(shape_info)

    if 'gov_route_id' in trips_df.columns:
        trips_with_gov_route = trips_df[trips_df['gov_route_id'].notna()].copy()
        trips_without_gov_route = trips_df[trips_df['gov_route_id'].isna()].copy()
        
        if not trips_with_gov_route.empty:
            if not silent:
                print(f"Using direct gov_route_id matching for {len(trips_with_gov_route)} trips...")
            
            if 'direction_id' not in trips_with_gov_route.columns:
                trips_with_gov_route['direction_id'] = trips_with_gov_route['trip_id'].str.contains('-I-').astype(int)
            
            trips_with_gov_route['bound'] = trips_with_gov_route['direction_id'].apply(lambda x: 'I' if x == 1 else 'O')
            
            trips_with_gov_route['gov_route_id'] = pd.to_numeric(trips_with_gov_route['gov_route_id'], errors='coerce').astype('Int64').astype(str)
            shape_info_df['gov_route_id'] = pd.to_numeric(shape_info_df['gov_route_id'], errors='coerce').astype('Int64').astype(str)

            trips_direct_matched = trips_with_gov_route.merge(
                shape_info_df[['gov_route_id', 'bound', 'shape_id', 'agency_id']],
                on=['gov_route_id', 'bound'],
                how='left',
                suffixes=('', '_shape')
            )

            if not silent:
                print("DEBUG: Matched KMB route 10 trips:")
                print(trips_direct_matched[(trips_direct_matched['route_short_name'] == '10') & (trips_direct_matched['agency_id'] == 'KMB')][['agency_id', 'gov_route_id', 'bound', 'shape_id']].head())
                print("DEBUG: Matched CTB route 10 trips:")
                print(trips_direct_matched[(trips_direct_matched['route_short_name'] == '10') & (trips_direct_matched['agency_id'] == 'CTB')][['agency_id', 'gov_route_id', 'bound', 'shape_id']].head())

            agency_mapping = {
                'KMB': ['KMB', 'LWB', 'KMB+CTB', 'KWB'],
                'CTB': ['CTB', 'NWFB', 'KMB+CTB'],
                'GMB': ['GMB'],
                'MTRB': ['LRTFeeder'],
                'NLB': ['NLB']
            }

            def is_agency_match(row):
                if pd.isna(row['agency_id_shape']):
                    return True
                trip_agency = row['agency_id']
                shape_agency = row['agency_id_shape']
                
                valid_agencies = agency_mapping.get(trip_agency, [trip_agency])
                
                return shape_agency in valid_agencies

            trips_direct_matched = trips_direct_matched[trips_direct_matched.apply(is_agency_match, axis=1)]
            
            if not silent:
                print("DEBUG: Matched KMB route 10 trips after filtering:")
                print(trips_direct_matched[(trips_direct_matched['route_short_name'] == '10') & (trips_direct_matched['agency_id'] == 'KMB')][['agency_id', 'gov_route_id', 'bound', 'shape_id']].head())

            if not trips_direct_matched.empty:
                if not silent:
                    print("Validating shape directions by comparing start/end stops...")
                trips_direct_matched = _validate_and_correct_shape_directions(
                    trips_direct_matched, engine, silent
                )
            
                if not silent:
                    direct_matched_count = trips_direct_matched['shape_id'].notna().sum()
                    print(f"Direct matching: {direct_matched_count} out of {len(trips_direct_matched)} trips matched to shapes.")

        else:
            trips_direct_matched = pd.DataFrame()
        
        if not trips_without_gov_route.empty:
            if not silent:
                print(f"Using fallback matching for {len(trips_without_gov_route)} trips without gov_route_id...")
            trips_fallback_matched = _fallback_shape_matching(trips_without_gov_route, shape_info_df, engine, silent)
        else:
            trips_fallback_matched = pd.DataFrame()
        
        if not trips_direct_matched.empty and not trips_fallback_matched.empty:
            final_trips = pd.concat([trips_direct_matched, trips_fallback_matched], ignore_index=True)
        elif not trips_direct_matched.empty:
            final_trips = trips_direct_matched
        elif not trips_fallback_matched.empty:
            final_trips = trips_fallback_matched
        else:
            final_trips = trips_df.copy()
            final_trips['shape_id'] = None
        
        if not silent:
            total_matched = final_trips['shape_id'].notna().sum()
            total_trips = len(final_trips)
            print(f"Overall: {total_matched} out of {total_trips} trips matched to shapes ({total_matched/total_trips:.2%}).")
        
        return final_trips
    
    return _fallback_shape_matching(trips_df, shape_info_df, engine, silent)

def _fallback_shape_matching(trips_df, shape_info_df, engine, silent=False):
    """Fallback to complex matching for trips without gov_route_id"""
    
    gov_trips_df = pd.read_sql("SELECT trip_id, route_id, service_id FROM gov_gtfs_trips", engine)
    gov_routes_df = pd.read_sql("SELECT route_id, route_short_name, agency_id FROM gov_gtfs_routes", engine)

    gov_df = pd.merge(gov_trips_df, gov_routes_df, on='route_id')
    gov_df['service_id'] = gov_df['service_id'].astype(str)
    # Parse direction_id and convert to match our mapping: 0=outbound, 1=inbound
    parsed_direction = gov_df['trip_id'].str.split('-').str[1].astype(int)
    gov_df['direction_id'] = (parsed_direction == 2).astype(int)
    gov_df['bound'] = gov_df['direction_id'].apply(lambda x: 'I' if x == 1 else 'O')

    # create a mapping from our trip_id to the government's route_id
    # For agencies like CTB where each direction is a separate route_id in government GTFS,
    # we should match directly on gov_route_id when available
    
    # Ensure gov_route_id column exists (fill NaN for trips without it)
    if 'gov_route_id' not in trips_df.columns:
        trips_df['gov_route_id'] = None
    
    # Extract agency from trip_id for filtering (e.g., "KMB-98-O-287" -> "KMB")
    trips_df['trip_agency'] = trips_df['trip_id'].str.split('-').str[0]
    
    # Map our trip agencies to government GTFS agencies for proper filtering
    agency_mapping = {
        'KMB': ['KMB', 'LWB', 'KMB+CTB'],  # KMB can appear as KMB, LWB, or co-op
        'CTB': ['CTB', 'NWFB', 'KMB+CTB'], # CTB can appear as CTB, NWFB, or co-op  
        'GMB': ['GMB'],
        'MTRB': ['LRTFeeder'],
        'NLB': ['NLB']
    }
    
    # For trips with gov_route_id, match directly on that (most precise)
    trip_to_gov_route_map_precise = trips_df[trips_df['gov_route_id'].notna()].merge(
        gov_df,
        left_on=['gov_route_id'],
        right_on=['route_id'],
        suffixes=('', '_gov')
    )
    
    # For trips without gov_route_id, use the original service_id-based matching WITH AGENCY FILTERING
    trips_for_fallback = trips_df[trips_df['gov_route_id'].isna()].copy()
    if not trips_for_fallback.empty:
        # Add government agency filter based on trip agency
        gov_df_filtered_list = []
        for trip_agency, gov_agencies in agency_mapping.items():
            trip_subset = trips_for_fallback[trips_for_fallback['trip_agency'] == trip_agency]
            if not trip_subset.empty:
                gov_subset = gov_df[gov_df['agency_id'].isin(gov_agencies)]
                if not gov_subset.empty:
                    merged_subset = trip_subset.merge(
                        gov_subset,
                        left_on=['original_service_id', 'route_short_name', 'direction_id'],
                        right_on=['service_id', 'route_short_name', 'direction_id'],
                        suffixes=('', '_gov')
                    )
                    gov_df_filtered_list.append(merged_subset)
        
        if gov_df_filtered_list:
            trip_to_gov_route_map_fallback = pd.concat(gov_df_filtered_list, ignore_index=True)
        else:
            trip_to_gov_route_map_fallback = pd.DataFrame()
    else:
        trip_to_gov_route_map_fallback = pd.DataFrame()
    
    # Combine both approaches
    trip_to_gov_route_map = pd.concat([trip_to_gov_route_map_precise, trip_to_gov_route_map_fallback], ignore_index=True)

    # Debug: Print some information about the merge data
    if not silent:
        print(f"Debug: trip_to_gov_route_map has {len(trip_to_gov_route_map)} rows")
        if len(trip_to_gov_route_map) > 0:
            print(f"Debug: Sample trip data: route_id={trip_to_gov_route_map['route_id'].iloc[0]}, bound={trip_to_gov_route_map.get('bound', ['N/A']).iloc[0] if 'bound' in trip_to_gov_route_map.columns else 'MISSING'}")
        print(f"Debug: shape_info_df has {len(shape_info_df)} rows")
        if len(shape_info_df) > 0:
            print(f"Debug: Sample shape data: gov_route_id={shape_info_df['gov_route_id'].iloc[0]}, bound={shape_info_df['bound'].iloc[0]}")

    # Merge with shape_info_df to get the shape_id
    # Use the government route_id from the merge:
    # - For precise matches: it's in 'gov_route_id' column (our original data)
    # - For fallback matches: it's in 'route_id' column (from gov_df merge)
    # We need to use the government route_id, not our trip route_id
    
    # For precise matches, use the gov_route_id directly from our trip data
    precise_trips = trip_to_gov_route_map_precise.copy() if not trip_to_gov_route_map_precise.empty else pd.DataFrame()
    if not precise_trips.empty:
        precise_trips['gov_route_for_shapes'] = precise_trips['gov_route_id']
    
    # For fallback matches, use the route_id_gov from the merge (government route ID)
    fallback_trips = trip_to_gov_route_map_fallback.copy() if not trip_to_gov_route_map_fallback.empty else pd.DataFrame()
    if not fallback_trips.empty:
        # Debug: Check what columns are available in fallback trips
        if not silent:
            print(f"Debug: Fallback trips columns: {list(fallback_trips.columns)}")
            if len(fallback_trips) > 0:
                sample = fallback_trips.iloc[0]
                print(f"Debug: Sample fallback trip - route_id: {sample.get('route_id', 'MISSING')}, route_id_gov: {sample.get('route_id_gov', 'MISSING')}")
        
        # Use route_id_gov (government route ID) instead of route_id (our operator route ID)
        if 'route_id_gov' in fallback_trips.columns:
            fallback_trips['gov_route_for_shapes'] = fallback_trips['route_id_gov']
        else:
            # If route_id_gov doesn't exist, we need to figure out the correct column name
            if not silent:
                print("Warning: route_id_gov column not found in fallback trips!")
            # Try to find the government route ID column
            gov_route_columns = [col for col in fallback_trips.columns if 'route_id' in col and col != 'route_id']
            if gov_route_columns:
                gov_route_col = gov_route_columns[0]
                if not silent:
                    print(f"Using {gov_route_col} as government route ID column")
                fallback_trips['gov_route_for_shapes'] = fallback_trips[gov_route_col]
            else:
                # This should not happen with our enhanced matcher
                if not silent:
                    print("Error: No government route ID column found!")
                fallback_trips['gov_route_for_shapes'] = None
    
    # Combine and use the government route_id for shape matching
    if not precise_trips.empty and not fallback_trips.empty:
        combined_trips = pd.concat([precise_trips, fallback_trips], ignore_index=True)
    elif not precise_trips.empty:
        combined_trips = precise_trips
    elif not fallback_trips.empty:
        combined_trips = fallback_trips
    else:
        combined_trips = pd.DataFrame()
    
    if combined_trips.empty:
        trips_df['shape_id'] = None
        return trips_df
    
    # Ensure data types match for the merge
    # Convert to int first to remove .0, then to string for consistent matching
    try:
        combined_trips['gov_route_for_shapes'] = pd.to_numeric(combined_trips['gov_route_for_shapes'], errors='coerce').astype('Int64').astype(str)
        shape_info_df['gov_route_id'] = pd.to_numeric(shape_info_df['gov_route_id'], errors='coerce').astype('Int64').astype(str)
    except Exception as e:
        if not silent:
            print(f"Warning: Data type conversion failed: {e}")
            print(f"Sample gov_route_for_shapes values: {combined_trips['gov_route_for_shapes'].head().tolist()}")
        # Fallback: just use string conversion
        combined_trips['gov_route_for_shapes'] = combined_trips['gov_route_for_shapes'].astype(str)
        shape_info_df['gov_route_id'] = shape_info_df['gov_route_id'].astype(str)
    
    trips_with_shapes = combined_trips.merge(
        shape_info_df,
        left_on=['gov_route_for_shapes', 'bound'],
        right_on=['gov_route_id', 'bound'],
        how='left'
    )

    # Debug: Check merge results
    if not silent:
        print(f"Debug: trips_with_shapes has {len(trips_with_shapes)} rows")
        shape_matches = trips_with_shapes['shape_id'].notna().sum()
        print(f"Debug: {shape_matches} trips got shape matches")
        if len(trips_with_shapes) > 0 and shape_matches > 0:
            sample_match = trips_with_shapes[trips_with_shapes['shape_id'].notna()].iloc[0]
            print(f"Debug: Sample match: gov_route={sample_match['gov_route_for_shapes']}, bound={sample_match['bound']}, shape_id={sample_match['shape_id']}")
        elif len(trips_with_shapes) > 0:
            sample_nomatch = trips_with_shapes.iloc[0]
            print(f"Debug: Sample NO match: gov_route={sample_nomatch['gov_route_for_shapes']}, bound={sample_nomatch.get('bound', 'MISSING')}")
            print(f"Debug: Available shape route_ids: {sorted(shape_info_df['gov_route_id'].unique())[:10]}...")
            # Check if there's an exact match that should work
            test_route = sample_nomatch['gov_route_for_shapes']
            test_bound = sample_nomatch.get('bound', '')
            matching_shapes = shape_info_df[(shape_info_df['gov_route_id'] == test_route) & (shape_info_df['bound'] == test_bound)]
            print(f"Debug: Looking for gov_route={test_route}, bound={test_bound} -> found {len(matching_shapes)} shapes")

    # Deduplicate trips with shapes - keep the first shape_id for each trip_id
    trips_with_shapes_dedup = trips_with_shapes.drop_duplicates(subset=['trip_id'], keep='first')
    
    # Merge back to the original trips_df
    final_trips = trips_df.merge(trips_with_shapes_dedup[['trip_id', 'shape_id']], on='trip_id', how='left')

    if not silent:
        matched_count = final_trips['shape_id'].notna().sum()
        total_trips = len(final_trips)
        print(f"Matched {matched_count} out of {total_trips} trips to CSDI shapes ({matched_count/total_trips:.2%}).")

    return final_trips


def _validate_and_correct_shape_directions(trips_df, engine, silent=False):
    """
    Validate shape directions by comparing route start/end stops with shape start/end coordinates.
    Auto-correct inverted shapes for non-circular routes.
    """
    
    # Get unique route-bound combinations that have shapes
    if 'agency_id' not in trips_df.columns:
        # If agency_id is not available, we cannot perform the check.
        return trips_df

    routes_with_shapes = trips_df[trips_df['shape_id'].notna()][['route_short_name', 'direction_id', 'shape_id', 'agency_id']].drop_duplicates()
    
    if routes_with_shapes.empty:
        return trips_df
    
    corrections_made = 0
    
    for _, route_info in routes_with_shapes.iterrows():
        route_name = route_info['route_short_name']
        direction_id = route_info['direction_id']
        current_shape_id = route_info['shape_id']
        agency_id = route_info['agency_id']
        bound = 'I' if direction_id == 1 else 'O'
        opposite_bound = 'O' if direction_id == 1 else 'I'
        
        # Extract route ID from shape_id (e.g., "CSDI-8431-O" -> "8431")
        try:
            gov_route_id = current_shape_id.split('-')[1]
        except:
            continue
            
        # Check if this route should be corrected
        if _should_correct_shape_direction(route_name, bound, gov_route_id, agency_id, engine, silent):
            # Create corrected shape_id by swapping direction
            corrected_shape_id = f"CSDI-{gov_route_id}-{opposite_bound}"
            
            # Check if the corrected shape file exists
            import os
            corrected_shape_file = f"waypoints/{gov_route_id}-{opposite_bound}.json"
            if os.path.exists(corrected_shape_file):
                # Apply the correction
                mask = (trips_df['route_short_name'] == route_name) & (trips_df['direction_id'] == direction_id) & (trips_df['agency_id'] == agency_id)
                trips_df.loc[mask, 'shape_id'] = corrected_shape_id
                corrections_made += 1
                
                if not silent:
                    print(f"  Corrected {route_name} direction {direction_id}: {current_shape_id} -> {corrected_shape_id}")
            elif not silent:
                print(f"  Warning: Corrected shape file not found for {route_name}: {corrected_shape_file}")
    
    if not silent and corrections_made > 0:
        print(f"Applied {corrections_made} shape direction corrections.")
    
    return trips_df


def _should_correct_shape_direction(route_name, bound, gov_route_id, agency_id, engine, silent=False):
    """
    Determine if a route's shape direction should be corrected by comparing
    start/end stops with shape start/end coordinates.
    """
    
    # Get route stop sequence from our database
    try:
        params = {'route': route_name, 'bound': bound}
        if agency_id == 'KMB':
            stop_query = """
            SELECT ks.seq, ST_X(kst.geometry) as stop_lng, ST_Y(kst.geometry) as stop_lat, kst.name_en as stop_name_en
            FROM kmb_stop_sequences ks 
            JOIN kmb_routes kr ON ks.route = kr.route AND ks.bound = kr.bound AND ks.service_type = kr.service_type
            JOIN kmb_stops kst ON ks.stop = kst.stop
            WHERE ks.route = %(route)s AND ks.bound = %(bound)s
            ORDER BY ks.seq::int
            """
        elif agency_id == 'CTB':
            params['bound'] = 'outbound' if bound == 'O' else 'inbound'
            stop_query = """
            SELECT css.sequence as seq, ST_X(cs.geometry) as stop_lng, ST_Y(cs.geometry) as stop_lat, cs.name_en as stop_name_en
            FROM citybus_stop_sequences css
            JOIN citybus_routes cr ON css.unique_route_id = cr.unique_route_id
            JOIN citybus_stops cs ON css.stop_id = cs.stop
            WHERE cr.route = %(route)s AND cr.direction = %(bound)s
            ORDER BY css.sequence::int
            """
        elif agency_id == 'MTRB':
            stop_query = """
            SELECT mss.station_seqno as seq, ST_X(ms.geometry) as stop_lng, ST_Y(ms.geometry) as stop_lat, ms.name_en as stop_name_en
            FROM mtrbus_stop_sequences mss
            JOIN mtrbus_routes mr ON mss.route_id = mr.route_id
            JOIN mtrbus_stops ms ON mss.stop_id = ms.stop_id
            WHERE mr.route_id = %(route)s AND mss.direction = %(bound)s
            ORDER BY mss.station_seqno::int
            """
        else:
            # GMB and NLB are not supported yet due to lack of clear direction info
            return False

        route_stops = pd.read_sql(stop_query, engine, params=params)
        
        if route_stops.empty:
            return False
            
    except Exception as e:
        if not silent:
            print(f"    Warning: Could not get stops for {agency_id}-{route_name}-{bound}: {e}")
        return False
    
    if len(route_stops) < 2:
        return False  # Need at least 2 stops to compare
    
    # Get first and last stops
    first_stop = route_stops.iloc[0]
    last_stop = route_stops.iloc[-1]
    
    # Get shape coordinates from CSDI file
    try:
        import json
        import os
        
        shape_file = f"waypoints/{gov_route_id}-{bound}.json"
        if not os.path.exists(shape_file):
            return False
            
        with open(shape_file, 'r', encoding='utf-8') as f:
            shape_data = json.load(f)
            
        if not shape_data.get('features'):
            return False
            
        feature = shape_data['features'][0]
        geometry = feature.get('geometry', {})
        
        if geometry.get('type') != 'MultiLineString':
            return False
            
        coordinates = geometry.get('coordinates', [])
        if not coordinates or not coordinates[0]:
            return False
            
        # Get first and last coordinate points
        first_coords = coordinates[0][0]  # [longitude, latitude]
        last_coords = coordinates[-1][-1]  # [longitude, latitude]
        
        shape_start_lat, shape_start_lng = first_coords[1], first_coords[0]
        shape_end_lat, shape_end_lng = last_coords[1], last_coords[0]
        
    except Exception as e:
        if not silent:
            print(f"    Warning: Could not read shape file for {route_name}-{bound}: {e}")
        return False
    
    # Calculate distances between stops and shape endpoints
    def distance(lat1, lng1, lat2, lng2):
        return lat_long_dist(lat1, lng1, lat2, lng2)
    
    # Distance from route first stop to shape start/end
    dist_first_to_shape_start = distance(first_stop['stop_lat'], first_stop['stop_lng'], shape_start_lat, shape_start_lng)
    dist_first_to_shape_end = distance(first_stop['stop_lat'], first_stop['stop_lng'], shape_end_lat, shape_end_lng)
    
    # Distance from route last stop to shape start/end  
    dist_last_to_shape_start = distance(last_stop['stop_lat'], last_stop['stop_lng'], shape_start_lat, shape_start_lng)
    dist_last_to_shape_end = distance(last_stop['stop_lat'], last_stop['stop_lng'], shape_end_lat, shape_end_lng)
    
    # Calculate alignment scores
    correct_alignment = dist_first_to_shape_start + dist_last_to_shape_end
    inverted_alignment = dist_first_to_shape_end + dist_last_to_shape_start
    
    # If inverted alignment is significantly better, recommend correction
    threshold = 500  # 500 meters threshold
    should_correct = inverted_alignment < (correct_alignment - threshold)
    
    if not silent and agency_id == 'CTB':
        print(f"DEBUG: CTB-{route_name}-{bound}: correct_alignment={correct_alignment}, inverted_alignment={inverted_alignment}, should_correct={should_correct}")

    if not silent and should_correct:
        print(f"    Route {agency_id}-{route_name}-{bound}: Shape appears inverted")
        print(f"      Correct alignment distance: {correct_alignment:.0f}m")
        print(f"      Inverted alignment distance: {inverted_alignment:.0f}m")
        print(f"      First stop: {first_stop['stop_name_en']}")
        print(f"      Last stop: {last_stop['stop_name_en']}")
    
    return should_correct


