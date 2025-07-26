import geopandas as gpd
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

def unify_stops_by_name_and_distance(stops_gdf: gpd.GeoDataFrame, name_col: str, stop_id_col: str, distance_threshold_meters: float = 2.0):
    """
    Unifies stops within a single GeoDataFrame that have the same name and are within a specified distance.
    """
    print(f"Unifying stops within a distance of {distance_threshold_meters}m...")

    projected_gdf = stops_gdf.copy().to_crs(epsg=2326)
    
    sindex = projected_gdf.sindex

    processed_indices = set()
    duplicates_map = {}
    
    for index, stop in tqdm(projected_gdf.iterrows(), total=projected_gdf.shape[0], desc="Unifying stops"):
        if index in processed_indices:
            continue

        processed_indices.add(index)
        
        possible_matches_indices = list(sindex.intersection(stop.geometry.buffer(distance_threshold_meters).bounds))
        possible_matches = projected_gdf.iloc[possible_matches_indices]

        stop_name = stops_gdf.loc[index][name_col]
        
        for match_index, match_stop in possible_matches.iterrows():
            if match_index in processed_indices:
                continue

            match_name = stops_gdf.loc[match_index][name_col]

            if stop_name == match_name:
                distance = stop.geometry.distance(match_stop.geometry)
                if distance <= distance_threshold_meters:
                    original_stop_id = stops_gdf.loc[match_index][stop_id_col]
                    canonical_stop_id = stops_gdf.loc[index][stop_id_col]
                    duplicates_map[original_stop_id] = canonical_stop_id
                    processed_indices.add(match_index)

    duplicate_indices = stops_gdf[stops_gdf[stop_id_col].isin(duplicates_map.keys())].index
    unified_stops_gdf = stops_gdf.drop(index=duplicate_indices)
    
    print(f"Unified {len(stops_gdf)} stops into {len(unified_stops_gdf)} unique locations.")
    
    return unified_stops_gdf, duplicates_map

def format_timedelta(td):
    """Formats a timedelta object into an HH:MM:SS string."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def generate_stop_times_for_agency(
    agency_id: str,
    agency_trips_df: pd.DataFrame,
    agency_stoptimes_df: pd.DataFrame,
    agency_gov_routes_df: pd.DataFrame,
    agency_gov_trips_df: pd.DataFrame,
    agency_gov_frequencies_df: pd.DataFrame,
    journey_time_data: dict
):
    print(f"Generating stop times from frequencies for {agency_id}...")

    # For faster lookups
    gov_routes_map = agency_gov_routes_df.set_index('route_id')
    
    # Create a mapping from (route_short_name, direction_id) to a list of frequency rules
    gov_trips_with_route_info = agency_gov_trips_df.merge(gov_routes_map, on='route_id')
    gov_freq_map = {}
    for _, trip in gov_trips_with_route_info.iterrows():
        key = (trip['route_short_name'], trip['direction_id'])
        frequency_info = agency_gov_frequencies_df[agency_gov_frequencies_df['trip_id'] == trip['trip_id']]
        if not frequency_info.empty:
            # A route can have multiple frequency rules (e.g., peak, off-peak)
            gov_freq_map.setdefault(key, []).extend(frequency_info.to_dict('records'))

    new_stop_times = []
    DEFAULT_JOURNEY_TIME_SECS = 120 # 2 minutes as a fallback

    stop_seq_col = 'seq' if 'seq' in agency_stoptimes_df.columns else 'sequence'
    agency_stoptimes_df[stop_seq_col] = pd.to_numeric(agency_stoptimes_df[stop_seq_col], errors='coerce')
    agency_stoptimes_df.dropna(subset=[stop_seq_col], inplace=True)
    agency_stoptimes_df[stop_seq_col] = agency_stoptimes_df[stop_seq_col].astype(int)

    for _, trip_row in tqdm(agency_trips_df.iterrows(), total=agency_trips_df.shape[0], desc=f"Generating {agency_id} stop times"):
        
        lookup_key = (trip_row['route_short_name'], trip_row['direction_id'])
        frequencies = gov_freq_map.get(lookup_key)

        if not frequencies:
            continue

        stop_sequence_df = agency_stoptimes_df[agency_stoptimes_df['trip_id'] == trip_row['trip_id']].sort_values(stop_seq_col)
        if stop_sequence_df.empty:
            continue

        # Iterate through each frequency period for the route
        for frequency in frequencies:
            try:
                start_time_str = frequency['start_time']
                end_time_str = frequency['end_time']
                headway_secs = int(frequency['headway_secs'])

                start_h, start_m, start_s = map(int, start_time_str.split(':'))
                end_h, end_m, end_s = map(int, end_time_str.split(':'))

                trip_start_time = timedelta(hours=start_h, minutes=start_m, seconds=start_s)
                service_end_time = timedelta(hours=end_h, minutes=end_m, seconds=end_s)
            except (ValueError, TypeError):
                continue

            # Generate trips for this frequency period
            while trip_start_time < service_end_time:
                current_arrival_time = trip_start_time
                
                for stop_idx, stop_row in stop_sequence_df.iterrows():
                    if stop_idx != stop_sequence_df.index[0]:
                        prev_stop_row = stop_sequence_df.loc[stop_sequence_df.index[stop_sequence_df.index.get_loc(stop_idx) - 1]]
                        
                        from_stop_id_unprefixed = prev_stop_row['stop_id'].split('-')[-1]
                        to_stop_id_unprefixed = stop_row['stop_id'].split('-')[-1]
                        
                        try:
                            journey_time = int(journey_time_data.get(from_stop_id_unprefixed, {}).get(to_stop_id_unprefixed, DEFAULT_JOURNEY_TIME_SECS))
                        except (ValueError, TypeError):
                            journey_time = DEFAULT_JOURNEY_TIME_SECS
                        
                        current_arrival_time += timedelta(seconds=journey_time)

                    new_stop_times.append({
                        'trip_id': trip_row['trip_id'],
                        'arrival_time': format_timedelta(current_arrival_time),
                        'departure_time': format_timedelta(current_arrival_time),
                        'stop_id': stop_row['stop_id'],
                        'stop_sequence': stop_row[stop_seq_col],
                    })

                trip_start_time += timedelta(seconds=headway_secs)

    if not new_stop_times:
        print(f"Warning: No {agency_id} stop times were generated. Check matching logic and input data.")
        return pd.DataFrame(columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'])

    return pd.DataFrame(new_stop_times)
