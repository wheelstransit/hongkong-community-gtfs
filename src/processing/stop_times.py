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
        pattern_key_cols = ['routeNo', 'direction_id']
    else:
        pattern_key_cols = ['trip_id']

    valid_pattern_key_cols = [col for col in pattern_key_cols if col in agency_stoptimes_df.columns]
    if not valid_pattern_key_cols:
         if not silent:
            print(f"Warning: Could not determine pattern key for {agency_id}. Cannot generate stop times.")
         return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    grouped_stops = agency_stoptimes_df.sort_values(stop_seq_col).groupby(valid_pattern_key_cols)

    for name, group in tqdm(grouped_stops, desc=f"Analyzing {agency_id} trip patterns", disable=silent):
        stops = group['stop_id'].tolist()
        sequences = group[stop_seq_col].tolist()
        journey_times_sec = [0]
        for i in range(1, len(stops)):
            from_stop_id_unprefixed = str(stops[i-1]).split('-')[-1]
            to_stop_id_unprefixed = str(stops[i]).split('-')[-1]
            try:
                time = int(journey_time_data.get(from_stop_id_unprefixed, {}).get(to_stop_id_unprefixed, DEFAULT_JOURNEY_TIME_SECS))
            except (ValueError, TypeError):
                time = DEFAULT_JOURNEY_TIME_SECS
        
        trip_patterns[name] = {
            'stop_ids': np.array(stops),
            'sequences': np.array(sequences),
            'cumulative_offsets_sec': np.cumsum(journey_times_sec)
        }

    # --- Step 2: Vectorized Generation ---
    st_trip_ids = []
    st_arrival_times = []
    st_departure_times = []
    st_stop_ids = []
    st_stop_sequences = []

    agency_trips_with_gov_info = agency_trips_df.merge(
    gov_trips_df,
    left_on=['route_short_name', 'original_service_id'],
    right_on=['route_short_name', 'service_id'],
    how='left',
    suffixes=['_agency', '_gov']
)

    trips_with_freq = agency_trips_with_gov_info.merge(
        gov_frequencies_df,
        left_on='trip_id_gov',
        right_on='trip_id',
        how='left'
    )

    for trip_row in tqdm(trips_with_freq.itertuples(), total=trips_with_freq.shape[0], desc=f"Generating {agency_id} stop times", disable=silent):
        if agency_id == 'KMB':
            pattern_lookup_key = (trip_row.route_short_name, trip_row.bound)
        elif agency_id == 'CTB':
            direction_str = 'outbound' if trip_row.direction_id == 0 else 'inbound'
            pattern_lookup_key = f"{trip_row.route_short_name}-{direction_str}"
        elif agency_id == 'GMB':
            pattern_lookup_key = (trip_row.route_short_name, trip_row.direction_id + 1)
        elif agency_id == 'MTRB':
            direction_str = 'O' if trip_row.direction_id == 0 else 'I'
            pattern_lookup_key = (trip_row.route_short_name, direction_str)
        elif agency_id == 'NLB':
            direction_id = getattr(trip_row, 'direction_id_agency', None)
            if direction_id is None:
                direction_id = getattr(trip_row, 'direction_id', 0)
            pattern_lookup_key = (trip_row.route_short_name, direction_id)
        else:
            pattern_lookup_key = trip_row.trip_id_agency

        pattern = trip_patterns.get(pattern_lookup_key)
        if not pattern:
            continue

        start_time_str = getattr(trip_row, 'start_time', '06:00:00')
        end_time_str = getattr(trip_row, 'end_time', '23:00:00')
        headway_secs_val = getattr(trip_row, 'headway_secs', 1200)

        if pd.isna(start_time_str): start_time_str = '06:00:00'
        if pd.isna(end_time_str): end_time_str = '23:00:00'
        if pd.isna(headway_secs_val): headway_secs_val = 1200

        try:
            start_time_secs = pd.to_timedelta(start_time_str).total_seconds()
            end_time_secs = pd.to_timedelta(end_time_str).total_seconds()
            headway_secs = int(headway_secs_val)
        except (ValueError, TypeError):
            start_time_secs = pd.to_timedelta('06:00:00').total_seconds()
            end_time_secs = pd.to_timedelta('23:00:00').total_seconds()
            headway_secs = 1200

        if headway_secs == 0: continue

        all_trip_start_times_secs = np.arange(start_time_secs, end_time_secs, headway_secs)
        if all_trip_start_times_secs.size == 0:
            continue

        arrival_times_secs = pattern['cumulative_offsets_sec'][:, np.newaxis] + all_trip_start_times_secs

        num_trips_in_block = len(all_trip_start_times_secs)
        num_stops = len(pattern['stop_ids'])

        st_trip_ids.extend([trip_row.trip_id_agency] * (num_stops * num_trips_in_block))
        st_stop_ids.extend(np.tile(pattern['stop_ids'], num_trips_in_block).tolist())
        st_stop_sequences.extend(np.tile(pattern['sequences'], num_trips_in_block).tolist())

        flat_arrival_times = arrival_times_secs.flatten(order='F').astype(int)
        tds = pd.to_timedelta(flat_arrival_times, unit='s')

        components = tds.components
        hours = components.days * 24 + components.hours
        minutes = components.minutes
        seconds = components.seconds
        formatted_times = (
            hours.astype(str).str.zfill(2) + ':' +
            minutes.astype(str).str.zfill(2) + ':' +
            seconds.astype(str).str.zfill(2)
        ).tolist()

        st_arrival_times.extend(formatted_times)
        st_departure_times.extend(formatted_times)

    if not st_trip_ids:
        if not silent:
            print(f"Warning: No {agency_id} stop times were generated. Check matching logic and input data.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    # --- Step 3: Create DataFrame ---
    final_df = pd.DataFrame({
        'trip_id': st_trip_ids,
        'arrival_time': st_arrival_times,
        'departure_time': st_departure_times,
        'stop_id': st_stop_ids,
        'stop_sequence': st_stop_sequences
    })
    
    if not silent:
        print(f"Generated {len(final_df)} stop times for {agency_id}.")
        
    return final_df