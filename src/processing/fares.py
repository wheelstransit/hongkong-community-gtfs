import os
import pandas as pd
from sqlalchemy.engine import Engine
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def _process_special_fares_for_trip(args):
    """Helper function to detect special fares for a single trip - must be at module level for pickling."""
    trip_id, trip_fares_data, trip_stops_map = args
    
    special_rules = []
    
    if not trip_fares_data or not trip_stops_map:
        return special_rules
    
    # Build a complete fare matrix: for each start_seq, what prices exist for different end_seq
    fare_matrix = {}
    for fare in trip_fares_data:
        start_seq = fare['start_stop_seq']
        end_seq = fare['end_stop_seq']
        price = fare['price']
        
        if start_seq not in fare_matrix:
            fare_matrix[start_seq] = {}
        fare_matrix[start_seq][end_seq] = {
            'price': price,
            'currency': fare['currency_type']
        }
    
    # Get the maximum stop sequence (last stop)
    max_stop_seq = max(trip_stops_map.keys()) if trip_stops_map else 0
    
    # Detect special fares: when a boarding stop has multiple distinct prices for different end stops
    # BUT exclude cases where it's just normal fare progression
    for start_seq, end_seq_prices in fare_matrix.items():
        # Group end sequences by price
        price_to_end_seqs = {}
        for end_seq, fare_info in end_seq_prices.items():
            price = fare_info['price']
            if price not in price_to_end_seqs:
                price_to_end_seqs[price] = []
            price_to_end_seqs[price].append(end_seq)
        
        # If there are multiple distinct prices, check if they're truly special fares
        if len(price_to_end_seqs) > 1:
            # Sort prices to identify the pattern
            sorted_prices = sorted(price_to_end_seqs.items(), key=lambda x: x[0])
            
            # Only create special fare rules for prices that DON'T extend to the last stop
            # The price that extends to the last stop is the "normal" fare
            for price, end_seqs in sorted_prices:
                max_end_seq = max(end_seqs)
                
                # Skip if this price extends to the last stop (it's normal pricing)
                if max_end_seq >= max_stop_seq:
                    continue
                
                # This is a special fare - it has a different price for a limited range
                min_end_seq = min(end_seqs)
                
                # Get the corresponding stop_ids
                if start_seq in trip_stops_map and max_end_seq in trip_stops_map:
                    boarding_stop_id = trip_stops_map[start_seq]
                    offboarding_stop_id = trip_stops_map[max_end_seq]
                    currency = end_seq_prices[max_end_seq]['currency']
                    
                    special_rules.append({
                        'special_fare_id': f"{trip_id}-{start_seq}-{min_end_seq}-{max_end_seq}-{price}",
                        'rule_type': 'trip',
                        'trip_id': trip_id,
                        'onboarding_stop_id': boarding_stop_id,
                        'offboarding_stop_id': offboarding_stop_id,
                        'price': price,
                        'currency': currency
                    })
    
    return special_rules


def _process_single_trip(args):
    """Helper function to process a single trip - must be at module level for pickling."""
    trip_id, trip_fares_data, trip_stops_data, gov_stop_count_map = args
    
    trip_fare_stages = []
    trip_mismatches = []
    
    if not trip_fares_data or not trip_stops_data:
        return trip_fare_stages, trip_mismatches

    # Get government route and direction info from first fare entry
    first_fare = trip_fares_data[0]
    gov_route_id = first_fare['gov_route_id']
    gov_dir_id = first_fare['gov_direction_id']
    
    # Check stop count mismatch using pre-computed map
    route_dir_key = f"{gov_route_id}-{gov_dir_id}"
    if route_dir_key in gov_stop_count_map:
        gov_info = gov_stop_count_map[route_dir_key]
        our_stop_count = max(stop['stop_sequence'] for stop in trip_stops_data)
        
        if gov_info['stop_count'] != our_stop_count:
            trip_mismatches.append({
                'our_trip_id': trip_id,
                'our_stop_count': our_stop_count,
                'gov_sample_trip_id': gov_info['sample_trip_id'],
                'gov_stop_count': gov_info['stop_count'],
                'gov_route_id': gov_route_id,
                'gov_direction_id': gov_dir_id
            })
    
    # Build a mapping from stop_sequence to price
    fare_by_seq = {}
    for fare in sorted(trip_fares_data, key=lambda x: x['start_stop_seq']):
        start_seq = fare['start_stop_seq']
        if start_seq not in fare_by_seq:
            fare_by_seq[start_seq] = {
                'price': fare['price'],
                'currency': fare['currency_type']
            }
    
    # Create fare stages where price changes
    last_price = -1
    for stop in trip_stops_data:
        stop_sequence = stop['stop_sequence']
        
        if stop_sequence in fare_by_seq:
            fare_info = fare_by_seq[stop_sequence]
            current_price = fare_info['price']
            
            if current_price != last_price:
                trip_fare_stages.append({
                    'trip_id': trip_id,
                    'from_stop_id': stop['stop_id'],
                    'price': current_price,
                    'currency': fare_info['currency']
                })
                last_price = current_price
    
    return trip_fare_stages, trip_mismatches


def generate_fare_stages(engine: Engine, trips_df: pd.DataFrame, stop_times_df: pd.DataFrame, silent: bool = False) -> pd.DataFrame:
    """
    Converts government GTFS fare data into fare_stages.csv format.

    The government GTFS fare_rules are defined for each stop-to-stop pair, which is highly redundant.
    This function converts it to a more efficient stage-based format. A fare stage is defined only
    at the stop where the price changes.
    
    Key mappings:
    - Gov fare_id format: [route_id]-[direction_id]-[start_stop_seq]-[end_stop_seq]
    - Our direction 0 = Gov direction 1
    - Our direction 1 = Gov direction 2
    """
    if not silent:
        print("Reading government fare data...")

    try:
        fare_attributes = pd.read_sql("SELECT fare_id, price, currency_type FROM gov_gtfs_fare_attributes", engine)
        fare_rules = pd.read_sql("SELECT fare_id, route_id FROM gov_gtfs_fare_rules", engine)
        gov_stop_times = pd.read_sql("SELECT trip_id, stop_sequence, stop_id FROM gov_gtfs_stop_times", engine)
    except Exception as e:
        if not silent:
            print(f"Could not read fare tables from database: {e}")
        return pd.DataFrame(columns=['trip_id', 'from_stop_id', 'price', 'currency'])

    # Parse fare_id to extract route, direction, start_seq, and end_seq
    # Format: [route_id]-[direction_id]-[start_stop_seq]-[end_stop_seq]
    fare_rules[['fare_route_id', 'fare_direction_id', 'start_stop_seq', 'end_stop_seq']] = \
        fare_rules['fare_id'].str.split('-', expand=True)
    
    fare_rules['fare_route_id'] = fare_rules['fare_route_id'].astype(str)
    fare_rules['fare_direction_id'] = fare_rules['fare_direction_id'].astype(int)
    fare_rules['start_stop_seq'] = fare_rules['start_stop_seq'].astype(int)
    
    fares = pd.merge(fare_rules, fare_attributes, on='fare_id')
    
    # We need to match through gov_route_id and direction_id
    if 'gov_route_id' not in trips_df.columns or 'direction_id' not in trips_df.columns:
        if not silent:
            print("Warning: trips_df does not have gov_route_id or direction_id column. Cannot match fares.")
        return pd.DataFrame(columns=['trip_id', 'from_stop_id', 'price', 'currency'])
    
    # Get trips with their government route IDs and direction
    trip_info = trips_df[['trip_id', 'gov_route_id', 'direction_id']].dropna(subset=['gov_route_id']).drop_duplicates()
    trip_info['gov_route_id'] = trip_info['gov_route_id'].astype(str)
    
    # Convert our direction to government direction (our 0 = gov 1, our 1 = gov 2)
    trip_info['gov_direction_id'] = trip_info['direction_id'].apply(lambda d: 2 if d == 1 else 1)
    
    # Match using gov_route_id and direction
    trip_fares = pd.merge(
        trip_info, 
        fares, 
        left_on=['gov_route_id', 'gov_direction_id'], 
        right_on=['fare_route_id', 'fare_direction_id'], 
        how='inner'
    )

    if trip_fares.empty:
        if not silent:
            print("No matching fares found for the provided trips.")
        return pd.DataFrame(columns=['trip_id', 'from_stop_id', 'price', 'currency'])

    # Get stop sequences for all our trips
    trip_stop_sequences = stop_times_df[['trip_id', 'stop_id', 'stop_sequence']].copy()
    trip_stop_sequences['stop_sequence'] = pd.to_numeric(trip_stop_sequences['stop_sequence'], errors='coerce')
    trip_stop_sequences.dropna(subset=['stop_sequence'], inplace=True)
    trip_stop_sequences = trip_stop_sequences.sort_values(['trip_id', 'stop_sequence'])

    # Prepare government stop times for comparison
    gov_stop_times['stop_sequence'] = pd.to_numeric(gov_stop_times['stop_sequence'], errors='coerce')
    gov_stop_times.dropna(subset=['stop_sequence'], inplace=True)
    
    # Prepare data for parallel processing - convert to dictionaries for faster access
    if not silent:
        print("Pre-processing data for parallel execution...")
    
    # Group fares by trip_id
    trip_fares_dict = {}
    for _, row in trip_fares.iterrows():
        trip_id = row['trip_id']
        if trip_id not in trip_fares_dict:
            trip_fares_dict[trip_id] = []
        trip_fares_dict[trip_id].append({
            'start_stop_seq': row['start_stop_seq'],
            'price': row['price'],
            'currency_type': row['currency_type'],
            'gov_route_id': row['gov_route_id'],
            'gov_direction_id': row['gov_direction_id']
        })
    
    # Group stops by trip_id
    trip_stops_dict = {}
    for _, row in trip_stop_sequences.iterrows():
        trip_id = row['trip_id']
        if trip_id not in trip_stops_dict:
            trip_stops_dict[trip_id] = []
        trip_stops_dict[trip_id].append({
            'stop_sequence': int(row['stop_sequence']),
            'stop_id': row['stop_id']
        })
    
    # Pre-compute government stop counts per route-direction
    if not silent:
        print("Computing government stop counts...")
    gov_stop_count_map = {}
    gov_trips_grouped = gov_stop_times.groupby(gov_stop_times['trip_id'].str.extract(r'^(\d+-\d+)-', expand=False))
    
    for route_dir, group in gov_trips_grouped:
        if pd.notna(route_dir):
            sample_trip = group['trip_id'].iloc[0]
            stop_count = group[group['trip_id'] == sample_trip]['stop_sequence'].max()
            gov_stop_count_map[route_dir] = {
                'sample_trip_id': sample_trip,
                'stop_count': stop_count
            }
    
    # Build arguments list for parallel processing
    trips_to_process = list(trip_fares_dict.keys())
    if not silent:
        print(f"Processing fares for {len(trips_to_process)} trips using {cpu_count()} cores...")
    
    args_list = [
        (trip_id, trip_fares_dict.get(trip_id, []), trip_stops_dict.get(trip_id, []), gov_stop_count_map)
        for trip_id in trips_to_process
    ]

    # Process trips in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(_process_single_trip, args_list, chunksize=50),
            total=len(args_list),
            desc="Generating Fare Stages",
            disable=silent
        ))
    
    # Combine results
    all_fare_stages = []
    stop_count_mismatches = []
    for fare_stages, mismatches in results:
        all_fare_stages.extend(fare_stages)
        stop_count_mismatches.extend(mismatches)

    # Save stop count mismatches to debug CSV
    if stop_count_mismatches:
        mismatch_df = pd.DataFrame(stop_count_mismatches)
        debug_path = 'output/unified_feed/debug_fare_stop_count_mismatches.csv'
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        mismatch_df.to_csv(debug_path, index=False)
        if not silent:
            print(f"Found {len(stop_count_mismatches)} trips with stop count mismatches. Saved to {debug_path}")
    
    if not all_fare_stages:
        if not silent:
            print("No fare stages generated.")
        return pd.DataFrame(columns=['trip_id', 'from_stop_id', 'price', 'currency'])

    return pd.DataFrame(all_fare_stages)


def generate_special_fare_rules(engine: Engine, trips_df: pd.DataFrame, stop_times_df: pd.DataFrame, silent: bool = False) -> pd.DataFrame:
    """
    Generates special_fare_rules.csv from government GTFS data.
    
    Detects trip-level special fares by finding price variations for the same boarding stop
    that differ from the standard fare progression.
    """
    if not silent:
        print("Reading government fare data for special fare detection...")

    try:
        fare_attributes = pd.read_sql("SELECT fare_id, price, currency_type FROM gov_gtfs_fare_attributes", engine)
        fare_rules = pd.read_sql("SELECT fare_id, route_id FROM gov_gtfs_fare_rules", engine)
        gov_stop_times = pd.read_sql("SELECT trip_id, stop_sequence, stop_id FROM gov_gtfs_stop_times", engine)
    except Exception as e:
        if not silent:
            print(f"Could not read fare tables from database: {e}")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])

    # Parse fare_id: [route_id]-[direction_id]-[start_stop_seq]-[end_stop_seq]
    fare_rules[['fare_route_id', 'fare_direction_id', 'start_stop_seq', 'end_stop_seq']] = \
        fare_rules['fare_id'].str.split('-', expand=True)
    
    fare_rules['fare_route_id'] = fare_rules['fare_route_id'].astype(str)
    fare_rules['fare_direction_id'] = fare_rules['fare_direction_id'].astype(int)
    fare_rules['start_stop_seq'] = fare_rules['start_stop_seq'].astype(int)
    fare_rules['end_stop_seq'] = fare_rules['end_stop_seq'].astype(int)
    
    fares = pd.merge(fare_rules, fare_attributes, on='fare_id')
    
    # Match with our trips
    if 'gov_route_id' not in trips_df.columns or 'direction_id' not in trips_df.columns:
        if not silent:
            print("Warning: trips_df does not have gov_route_id or direction_id column. Cannot match fares.")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    trip_info = trips_df[['trip_id', 'gov_route_id', 'direction_id']].dropna(subset=['gov_route_id']).drop_duplicates()
    trip_info['gov_route_id'] = trip_info['gov_route_id'].astype(str)
    trip_info['gov_direction_id'] = trip_info['direction_id'].apply(lambda d: 2 if d == 1 else 1)
    
    trip_fares = pd.merge(
        trip_info, 
        fares, 
        left_on=['gov_route_id', 'gov_direction_id'], 
        right_on=['fare_route_id', 'fare_direction_id'], 
        how='inner'
    )

    if trip_fares.empty:
        if not silent:
            print("No matching fares found.")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])

    # Get stop sequences
    trip_stop_sequences = stop_times_df[['trip_id', 'stop_id', 'stop_sequence']].copy()
    trip_stop_sequences['stop_sequence'] = pd.to_numeric(trip_stop_sequences['stop_sequence'], errors='coerce')
    trip_stop_sequences.dropna(subset=['stop_sequence'], inplace=True)

    # Prepare government stop times
    gov_stop_times['stop_sequence'] = pd.to_numeric(gov_stop_times['stop_sequence'], errors='coerce')
    gov_stop_times.dropna(subset=['stop_sequence'], inplace=True)

    # Pre-process data for parallel execution
    if not silent:
        print("Pre-processing data for parallel special fare detection...")
    
    # Group fares by trip_id into dictionaries
    trip_fares_dict = {}
    for _, row in trip_fares.iterrows():
        trip_id = row['trip_id']
        if trip_id not in trip_fares_dict:
            trip_fares_dict[trip_id] = []
        trip_fares_dict[trip_id].append({
            'start_stop_seq': row['start_stop_seq'],
            'end_stop_seq': row['end_stop_seq'],
            'price': row['price'],
            'currency_type': row['currency_type']
        })
    
    # Group stops by trip_id into dictionaries
    trip_stops_dict = {}
    for _, row in trip_stop_sequences.iterrows():
        trip_id = row['trip_id']
        if trip_id not in trip_stops_dict:
            trip_stops_dict[trip_id] = {}
        trip_stops_dict[trip_id][int(row['stop_sequence'])] = row['stop_id']
    
    trips_to_process = list(trip_fares_dict.keys())
    if not silent:
        print(f"Detecting special fares for {len(trips_to_process)} trips using {cpu_count()} cores...")
    
    # Build arguments list
    args_list = [
        (trip_id, trip_fares_dict.get(trip_id, []), trip_stops_dict.get(trip_id, {}))
        for trip_id in trips_to_process
    ]
    
    # Process trips in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(_process_special_fares_for_trip, args_list, chunksize=50),
            total=len(args_list),
            desc="Detecting Special Fares",
            disable=silent
        ))
    
    # Flatten results
    special_fare_rules = [rule for trip_rules in results for rule in trip_rules]

    if not special_fare_rules:
        if not silent:
            print("No special fare rules detected.")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])

    # Deduplicate: remove rules that are subsumed by earlier boarding stops with the same price and destination
    # Sort by trip_id, onboarding stop sequence (extract from special_fare_id), and price
    if not silent:
        print(f"Deduplicating {len(special_fare_rules)} special fare rules...")
    
    # Group by trip_id, offboarding_stop_id, and price
    from collections import defaultdict
    trip_price_dest_map = defaultdict(list)
    
    for rule in special_fare_rules:
        key = (rule['trip_id'], rule['offboarding_stop_id'], rule['price'])
        # Extract the starting sequence from special_fare_id for sorting
        parts = rule['special_fare_id'].split('-')
        # Find the start_seq (should be after trip_id components)
        try:
            # Find where the numeric sequence starts after trip components
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:
                    start_seq = int(part)
                    rule['_start_seq'] = start_seq
                    break
        except:
            rule['_start_seq'] = 999999  # fallback
        
        trip_price_dest_map[key].append(rule)
    
    # For each group, keep only the rule with the earliest boarding stop
    deduplicated_rules = []
    for key, rules in trip_price_dest_map.items():
        # Sort by starting sequence and keep the first (earliest boarding)
        rules.sort(key=lambda r: r.get('_start_seq', 999999))
        earliest_rule = rules[0]
        # Remove the temporary sorting key
        if '_start_seq' in earliest_rule:
            del earliest_rule['_start_seq']
        deduplicated_rules.append(earliest_rule)
    
    if not silent:
        removed = len(special_fare_rules) - len(deduplicated_rules)
        print(f"Removed {removed} duplicate rules. Final count: {len(deduplicated_rules)} special fare rules.")
    
    return pd.DataFrame(deduplicated_rules)


def generate_mtr_special_fare_rules(engine: Engine, silent: bool = False) -> pd.DataFrame:
    """
    Generates agency-level special fare rules for MTR heavy rail from fare CSV data.
    
    Returns DataFrame with columns: special_fare_id, rule_type, trip_id, 
    onboarding_stop_id, offboarding_stop_id, price, currency
    """
    if not silent:
        print("Generating MTR heavy rail special fare rules...")
    
    try:
        # Read MTR fares from database
        mtr_fares_df = pd.read_sql("SELECT * FROM mtr_lines_fares", engine)
        
        # Read station data to map Station ID to our stop_id format
        stations_df = pd.read_sql(
            'SELECT "Station ID" as station_id, "Station Code" as station_code FROM mtr_lines_and_stations',
            engine
        )
        stations_df = stations_df.drop_duplicates(subset=['station_id'])
        
        # Create mapping from Station ID to our stop_id (MTR-{station_code})
        station_id_to_stop_id = {}
        for _, row in stations_df.iterrows():
            station_id_to_stop_id[str(row['station_id'])] = f"MTR-{row['station_code']}"
        
    except Exception as e:
        if not silent:
            print(f"Error reading MTR fare data: {e}")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    if mtr_fares_df.empty:
        if not silent:
            print("No MTR fare data found.")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    # Process fare rules
    special_fare_rules = []
    for idx, row in mtr_fares_df.iterrows():
        src_station_id = str(row.get('SRC_STATION_ID', ''))
        dest_station_id = str(row.get('DEST_STATION_ID', ''))
        fare = row.get('OCT_ADT_FARE')
        
        # Skip if same station (zero fare)
        if src_station_id == dest_station_id:
            continue
        
        # Map to our stop_id format
        onboarding_stop_id = station_id_to_stop_id.get(src_station_id)
        offboarding_stop_id = station_id_to_stop_id.get(dest_station_id)
        
        if not onboarding_stop_id or not offboarding_stop_id:
            continue
        
        try:
            price = float(fare)
        except (ValueError, TypeError):
            continue
        
        special_fare_rules.append({
            'special_fare_id': f"MTR-{src_station_id}-{dest_station_id}",
            'rule_type': 'agency',
            'trip_id': '',  # Empty for agency-level rules
            'onboarding_stop_id': onboarding_stop_id,
            'offboarding_stop_id': offboarding_stop_id,
            'price': price,
            'currency': 'HKD'
        })
    
    if not silent:
        print(f"Generated {len(special_fare_rules)} MTR agency-level special fare rules.")
    
    return pd.DataFrame(special_fare_rules)


def generate_light_rail_special_fare_rules(engine: Engine, silent: bool = False) -> pd.DataFrame:
    """
    Generates agency-level special fare rules for Light Rail from fare CSV data.
    
    Returns DataFrame with columns: special_fare_id, rule_type, trip_id,
    onboarding_stop_id, offboarding_stop_id, price, currency
    """
    if not silent:
        print("Generating Light Rail special fare rules...")
    
    try:
        # Read Light Rail fares from database
        lr_fares_df = pd.read_sql("SELECT * FROM light_rail_fares", engine)
    except Exception as e:
        if not silent:
            print(f"Error reading Light Rail fare data: {e}")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    if lr_fares_df.empty:
        if not silent:
            print("No Light Rail fare data found.")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    # Process fare rules
    special_fare_rules = []
    for idx, row in lr_fares_df.iterrows():
        from_station_id = str(row.get('from_station_id', ''))
        to_station_id = str(row.get('to_station_id', ''))
        fare = row.get('fare_octo_adult')
        
        # Skip if same station (zero fare)
        if from_station_id == to_station_id:
            continue
        
        # Light Rail stop_id format is LR{station_id}
        onboarding_stop_id = f"LR{from_station_id}"
        offboarding_stop_id = f"LR{to_station_id}"
        
        try:
            price = float(fare)
        except (ValueError, TypeError):
            continue
        
        special_fare_rules.append({
            'special_fare_id': f"LR-{from_station_id}-{to_station_id}",
            'rule_type': 'agency',
            'trip_id': '',  # Empty for agency-level rules
            'onboarding_stop_id': onboarding_stop_id,
            'offboarding_stop_id': offboarding_stop_id,
            'price': price,
            'currency': 'HKD'
        })
    
    if not silent:
        print(f"Generated {len(special_fare_rules)} Light Rail agency-level special fare rules.")
    
    return pd.DataFrame(special_fare_rules)
