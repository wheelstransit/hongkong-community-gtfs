import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def format_timedelta(td):
    """Formats a timedelta object into an HH:MM:SS string."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def generate_stop_times_for_agency_optimized(
    agency_id: str,
    agency_trips_df: pd.DataFrame,
    agency_stoptimes_df: pd.DataFrame,
    gov_routes_df: pd.DataFrame,
    gov_trips_df: pd.DataFrame,
    gov_frequencies_df: pd.DataFrame,
    journey_time_data: dict,
    unified_to_original_map: dict,
    silent: bool = False
):
    if not silent:
        print(f"Generating stop times for {agency_id}...")

    if agency_trips_df.empty or agency_stoptimes_df.empty:
        if not silent:
            print(f"One of the dataframes for {agency_id} is empty. Skipping.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    # --- Step 1: Pre-compute Trip Patterns ---
    possible_seq_cols = ['sequence', 'seq', 'station_seqno', 'stop_sequence']
    stop_seq_col = next((col for col in possible_seq_cols if col in agency_stoptimes_df.columns), None)
    if stop_seq_col is None:
        if not silent:
            print(f"Warning: No valid sequence column found for {agency_id}. Cannot generate stop times.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    agency_stoptimes_df[stop_seq_col] = pd.to_numeric(agency_stoptimes_df[stop_seq_col], errors='coerce')
    agency_stoptimes_df.dropna(subset=[stop_seq_col], inplace=True)
    agency_stoptimes_df[stop_seq_col] = agency_stoptimes_df[stop_seq_col].astype(int)

    DEFAULT_JOURNEY_TIME_SECS = 120
    trip_patterns = {}
    
    if agency_id == 'KMB':
        pattern_key_cols = ['route', 'bound']
    elif agency_id == 'CTB':
        pattern_key_cols = ['unique_route_id']
    elif agency_id == 'GMB':
        pattern_key_cols = ['route_code', 'route_seq']
    elif agency_id == 'MTRB':
        pattern_key_cols = ['route_id', 'direction']
    elif agency_id == 'NLB':
        agency_stoptimes_df['routeId'] = agency_stoptimes_df['routeId'].astype(str)
        pattern_key_cols = ['routeId']
    else:
        pattern_key_cols = ['trip_id']

    valid_pattern_key_cols = [col for col in pattern_key_cols if col in agency_stoptimes_df.columns]
    if not valid_pattern_key_cols:
         if not silent:
            print(f"Warning: Could not determine pattern key for {agency_id}. Cannot generate stop times.")
         return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    if len(valid_pattern_key_cols) == 1:
        grouped_stops = agency_stoptimes_df.sort_values(stop_seq_col).groupby(valid_pattern_key_cols[0])
    else:
        grouped_stops = agency_stoptimes_df.sort_values(stop_seq_col).groupby(valid_pattern_key_cols)

    for name, group in tqdm(grouped_stops, desc=f"Analyzing {agency_id} trip patterns", disable=silent):
        stops = group['stop_id'].tolist()
        sequences = group[stop_seq_col].tolist()
        journey_times_sec = [0]
        for i in range(1, len(stops)):
            from_stop_id = stops[i-1]
            to_stop_id = stops[i]

            original_from_stops = unified_to_original_map.get(from_stop_id, [from_stop_id])
            original_to_stops = unified_to_original_map.get(to_stop_id, [to_stop_id])

            original_from_stops_unprefixed = [s.split('-', 1)[1] if '-' in s else s for s in original_from_stops]
            original_to_stops_unprefixed = [s.split('-', 1)[1] if '-' in s else s for s in original_to_stops]
            
            found_time = None
            for from_orig in original_from_stops_unprefixed:
                if from_orig in journey_time_data:
                    for to_orig in original_to_stops_unprefixed:
                        if to_orig in journey_time_data[from_orig]:
                            found_time = journey_time_data[from_orig][to_orig]
                            break
                if found_time is not None:
                    break
            
            try:
                time = int(found_time if found_time is not None else DEFAULT_JOURNEY_TIME_SECS)
            except (ValueError, TypeError):
                time = DEFAULT_JOURNEY_TIME_SECS
            journey_times_sec.append(time)
        
        trip_patterns[name] = {
            'stop_ids': np.array(stops),
            'sequences': np.array(sequences),
            'cumulative_offsets_sec': np.cumsum(journey_times_sec)
        }

    # --- Step 2: Generate Stop Times for a single trip ---
    all_stop_times = []

    for trip_row in tqdm(agency_trips_df.itertuples(), total=agency_trips_df.shape[0], desc=f"Generating {agency_id} stop times", disable=silent):
        if agency_id == 'KMB':
            pattern_lookup_key = (trip_row.route_short_name, trip_row.bound)
        elif agency_id == 'CTB':
            pattern_lookup_key = trip_row.unique_route_id
        elif agency_id == 'GMB':
            pattern_lookup_key = (trip_row.route_short_name, trip_row.direction_id + 1)
        elif agency_id == 'MTRB':
            direction_str = 'O' if trip_row.direction_id == 0 else 'I'
            pattern_lookup_key = (trip_row.route_short_name, direction_str)
        elif agency_id == 'NLB':
            pattern_lookup_key = str(trip_row.routeId)
        else:
            pattern_lookup_key = trip_row.trip_id

        pattern = trip_patterns.get(pattern_lookup_key)
        if not pattern:
            if agency_id in ['CTB', 'NLB'] and not silent:
                print(f"Pattern not found for key: {pattern_lookup_key}")
            continue

        # Generate arrival_time and departure_time as offsets from the start of the trip (00:00:00)
        offsets = pd.to_timedelta(pattern['cumulative_offsets_sec'], unit='s')
        
        # Format the timedelta objects into HH:MM:SS strings
        formatted_times = [format_timedelta(offset) for offset in offsets]

        for i in range(len(pattern['stop_ids'])):
            all_stop_times.append({
                'trip_id': trip_row.trip_id,
                'arrival_time': formatted_times[i],
                'departure_time': formatted_times[i],
                'stop_id': pattern['stop_ids'][i],
                'stop_sequence': pattern['sequences'][i]
            })

    if not all_stop_times:
        if not silent:
            print(f"Warning: No {agency_id} stop times were generated. Check matching logic and input data.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    # --- Step 3: Create DataFrame ---
    final_df = pd.DataFrame(all_stop_times)
    
    if not silent:
        print(f"Generated {len(final_df)} stop times for {agency_id}.")
        
    return final_df