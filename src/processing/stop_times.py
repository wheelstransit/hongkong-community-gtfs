import pandas as pd
import numpy as np # Import numpy for cumsum
from datetime import datetime, timedelta
from tqdm import tqdm

def format_timedelta(td):
    """Formats a timedelta object into an HH:MM:SS string."""
    # This function can handle cases where hours > 23
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def generate_stop_times_for_agency_optimized(
    agency_id: str,
    agency_trips_df: pd.DataFrame,
    agency_stoptimes_df: pd.DataFrame,
    agency_gov_routes_df: pd.DataFrame,
    agency_gov_trips_df: pd.DataFrame,
    agency_gov_frequencies_df: pd.DataFrame,
    journey_time_data: dict,
    silent: bool = False
):
    if not silent:
        print(f"Generating stop times from frequencies for {agency_id} (Optimized)...")

    # --- Step 1: Pre-computation (Mostly the same as before) ---
    gov_routes_map = agency_gov_routes_df.set_index('route_id')
    gov_trips_with_route_info = agency_gov_trips_df.merge(gov_routes_map, on='route_id')
    gov_freq_map = {}
    for _, trip in gov_trips_with_route_info.iterrows():
        key = (trip['route_short_name'], trip['direction_id'])
        frequency_info = agency_gov_frequencies_df[agency_gov_frequencies_df['trip_id'] == trip['trip_id']]
        if not frequency_info.empty:
            gov_freq_map.setdefault(key, []).extend(frequency_info.to_dict('records'))

    # --- Step 2: Pre-compute Trip Patterns and Cumulative Journey Times ---
    if not silent:
        print(f"Pre-computing trip patterns for {agency_id}...")
    possible_seq_cols = ['sequence', 'seq', 'station_seqno', 'stop_sequence']
    stop_seq_col = next((col for col in possible_seq_cols if col in agency_stoptimes_df.columns), None)

    if stop_seq_col is None:
        if not silent:
            print(f"Warning: No valid sequence column found for {agency_id}. Cannot generate stop times.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    # Ensure sequence column is numeric
    agency_stoptimes_df[stop_seq_col] = pd.to_numeric(agency_stoptimes_df[stop_seq_col], errors='coerce')
    agency_stoptimes_df.dropna(subset=[stop_seq_col], inplace=True)
    agency_stoptimes_df[stop_seq_col] = agency_stoptimes_df[stop_seq_col].astype(int)

    DEFAULT_JOURNEY_TIME_SECS = 120  # 2 minutes as a fallback
    trip_patterns = {}

    # Group by trip_id to process each unique stop pattern only once
    grouped_stops = agency_stoptimes_df.sort_values(stop_seq_col).groupby('trip_id')

    for trip_id, group in tqdm(grouped_stops, desc=f"Analyzing {agency_id} trip patterns", disable=silent):
        stops = group['stop_id'].tolist()
        sequences = group[stop_seq_col].tolist()

        # Calculate journey times between consecutive stops
        journey_times_sec = [0] # Time to first stop is 0
        for i in range(1, len(stops)):
            from_stop_id_unprefixed = stops[i-1].split('-')[-1]
            to_stop_id_unprefixed = stops[i].split('-')[-1]

            try:
                # Use .get() for safe dictionary access
                time = int(journey_time_data.get(from_stop_id_unprefixed, {}).get(to_stop_id_unprefixed, DEFAULT_JOURNEY_TIME_SECS))
            except (ValueError, TypeError):
                time = DEFAULT_JOURNEY_TIME_SECS
            journey_times_sec.append(time)

        # Calculate cumulative travel time from the start to each stop
        cumulative_offsets_sec = np.cumsum(journey_times_sec)

        trip_patterns[trip_id] = {
            'stop_ids': stops,
            'sequences': sequences,
            'cumulative_offsets_sec': cumulative_offsets_sec
        }

    # --- Step 3: Main Generation Loop (Now much faster) ---
    new_stop_times = []

    # Use itertuples() for a massive speedup over iterrows()
    for trip_row in tqdm(agency_trips_df.itertuples(), total=agency_trips_df.shape[0], desc=f"Generating {agency_id} stop times", disable=silent):

        lookup_key = (trip_row.route_short_name, trip_row.direction_id)
        frequencies = gov_freq_map.get(lookup_key)

        if not frequencies:
            continue

        # Get the pre-computed pattern for this trip
        pattern = trip_patterns.get(trip_row.trip_id)
        if not pattern:
            continue

        for frequency in frequencies:
            try:
                start_time_td = pd.to_timedelta(frequency['start_time'])
                end_time_td = pd.to_timedelta(frequency['end_time'])
                headway_secs = int(frequency['headway_secs'])
            except (ValueError, TypeError, KeyError):
                continue

            current_trip_start_td = start_time_td
            while current_trip_start_td < end_time_td:
                # Generate all stop times for this one trip in a vectorized way
                for stop_id, seq, offset_sec in zip(pattern['stop_ids'], pattern['sequences'], pattern['cumulative_offsets_sec']):
                    arrival_time = current_trip_start_td + timedelta(seconds=int(offset_sec))
                    formatted_time = format_timedelta(arrival_time)

                    new_stop_times.append({
                        'trip_id': trip_row.trip_id,
                        'arrival_time': formatted_time,
                        'departure_time': formatted_time,
                        'stop_id': stop_id,
                        'stop_sequence': seq,
                    })

                current_trip_start_td += timedelta(seconds=headway_secs)

    if not new_stop_times:
        if not silent:
            print(f"Warning: No {agency_id} stop times were generated. Check matching logic and input data.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    return pd.DataFrame(new_stop_times)
