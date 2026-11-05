import pandas as pd

from .times import format_timedelta, to_hhmmss, to_seconds

COLUMNS = ["trip_id", "start_time", "end_time", "headway_secs"]
ALLOWED_GOV_AGENCIES = {
    "KMB": {"KMB", "LWB", "KMB+CTB", "KWB"},
    "CTB": {"CTB", "NWFB"},
    "GMB": {"GMB"},
    "MTRB": {"LRTFeeder"},
    "NLB": {"NLB"},
    "FERRY": {"FERRY"},
}


def trip_id_mapping(trips: pd.DataFrame, gov_trips: pd.DataFrame) -> pd.DataFrame:
    """Map government trip_ids to our trip_ids, by gov_route_id when known, else by route number within the operator's agencies."""
    mappings = []
    for prefix in ("KMB", "CTB", "GMB", "MTRB", "NLB", "FERRY"):
        ours = trips[trips["trip_id"].str.startswith(f"{prefix}-")]
        if ours.empty:
            continue
        if "gov_route_id" in ours.columns and ours["gov_route_id"].notna().any():
            left = ours[["trip_id", "original_service_id", "direction_id", "gov_route_id"]].dropna().rename(
                columns={"trip_id": "new_trip_id", "gov_route_id": "route_id"})
            m = left.merge(gov_trips[["trip_id", "service_id", "direction_id", "route_id", "gov_agency_id"]],
                           left_on=["original_service_id", "direction_id", "route_id"],
                           right_on=["service_id", "direction_id", "route_id"])
        else:
            left = ours[["trip_id", "original_service_id", "direction_id", "route_short_name"]].dropna().rename(columns={"trip_id": "new_trip_id"})
            right = gov_trips[gov_trips["gov_agency_id"].isin(ALLOWED_GOV_AGENCIES.get(prefix, {prefix}))]
            m = left.merge(right[["trip_id", "service_id", "direction_id", "route_short_name", "gov_agency_id"]],
                           left_on=["original_service_id", "direction_id", "route_short_name"],
                           right_on=["service_id", "direction_id", "route_short_name"])
        if not m.empty:
            mappings.append(m[["trip_id", "new_trip_id"]].rename(columns={"trip_id": "original_trip_id"}).drop_duplicates())
    return pd.concat(mappings, ignore_index=True).drop_duplicates()


def build_frequencies(mapping: pd.DataFrame, gov_frequencies: pd.DataFrame, gov_stop_times: pd.DataFrame,
                      ferry: pd.DataFrame, light_rail: pd.DataFrame, rail: pd.DataFrame) -> pd.DataFrame:
    freq = gov_frequencies.merge(mapping, left_on="trip_id", right_on="original_trip_id")
    freq["trip_id"] = freq["new_trip_id"]
    freq = freq.drop(columns=["new_trip_id", "original_trip_id"])
    fallback = generate_frequencies_from_schedule(mapping, freq, gov_stop_times)
    for extra in (fallback, ferry):
        if not extra.empty:
            freq = pd.concat([freq, extra], ignore_index=True)
    if not freq.empty:
        freq = freq[~freq["trip_id"].str.startswith("MTR-")]
        freq = freq[~freq["trip_id"].str.startswith("LR-")]
    if not light_rail.empty:
        freq = pd.concat([freq, light_rail[COLUMNS]], ignore_index=True)
    if not rail.empty:
        freq = pd.concat([freq, rail], ignore_index=True)
    return freq


def generate_frequencies_from_schedule(
    trip_id_mapping: pd.DataFrame,
    existing_frequencies_df: pd.DataFrame,
    gov_stop_times_df: pd.DataFrame
) -> pd.DataFrame:
    """Derive frequency entries for trips lacking gov-provided headway data."""
    SINGLE_DEPARTURE_HEADWAY_SECS = 24 * 3600  # treat single departures as once-per-day service
    required_cols = ['trip_id', 'start_time', 'end_time', 'headway_secs']
    if trip_id_mapping is None or trip_id_mapping.empty:
        return pd.DataFrame(columns=required_cols)
    if gov_stop_times_df is None or gov_stop_times_df.empty:
        return pd.DataFrame(columns=required_cols)

    existing_trips = set(existing_frequencies_df['trip_id']) if existing_frequencies_df is not None and not existing_frequencies_df.empty else set()
    all_trips = set(trip_id_mapping['new_trip_id'])
    missing_trips = sorted(all_trips - existing_trips)
    if not missing_trips:
        return pd.DataFrame(columns=required_cols)

    stop_times = gov_stop_times_df[['trip_id', 'stop_sequence', 'departure_time', 'arrival_time']].copy()
    stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce')
    stop_times.dropna(subset=['stop_sequence'], inplace=True)
    stop_times['stop_sequence'] = stop_times['stop_sequence'].astype(int)
    stop_times['time_candidate'] = stop_times['departure_time']
    blank_mask = stop_times['time_candidate'].isna() | (stop_times['time_candidate'].astype(str).str.strip() == '')
    stop_times.loc[blank_mask, 'time_candidate'] = stop_times.loc[blank_mask, 'arrival_time']
    stop_times['time_candidate'] = stop_times['time_candidate'].fillna('').astype(str).str.strip()
    stop_times = stop_times[stop_times['time_candidate'] != '']
    if stop_times.empty:
        return pd.DataFrame(columns=required_cols)

    first_departures = (
        stop_times
        .sort_values(['trip_id', 'stop_sequence'])
        .groupby('trip_id')['time_candidate']
        .first()
    )
    first_departures.index = first_departures.index.astype(str)

    fallback_records = []
    mapping_grouped = trip_id_mapping.groupby('new_trip_id')['original_trip_id'].agg(list)

    for new_trip_id in missing_trips:
        original_ids = mapping_grouped.get(new_trip_id)
        if not original_ids:
            continue
        if not isinstance(original_ids, (list, tuple, set)):
            original_ids = [original_ids]
        original_ids = [str(oid) for oid in original_ids]
        departures = first_departures.reindex(original_ids).dropna()
        if departures.empty:
            continue

        sec_values = sorted({
            sec for sec in (to_seconds(t) for t in departures.values)
            if sec is not None
        })
        if len(sec_values) < 2:
            # synthesize a once-per-day headway to keep the trip in frequencies.txt
            start_sec = sec_values[0]
            fallback_records.append({
                'trip_id': new_trip_id,
                'start_time': to_hhmmss(start_sec),
                'end_time': to_hhmmss(start_sec + SINGLE_DEPARTURE_HEADWAY_SECS),
                'headway_secs': SINGLE_DEPARTURE_HEADWAY_SECS
            })
            continue

        headway_segments = []
        for idx in range(len(sec_values) - 1):
            start_sec = sec_values[idx]
            next_sec = sec_values[idx + 1]
            headway = next_sec - start_sec
            if headway <= 0:
                continue
            headway_segments.append((start_sec, next_sec, headway))

        if not headway_segments:
            continue

        for start_sec, end_sec, headway in headway_segments:
            fallback_records.append({
                'trip_id': new_trip_id,
                'start_time': to_hhmmss(start_sec),
                'end_time': to_hhmmss(end_sec),
                'headway_secs': headway
            })

        # Extend final interval using last observed headway so last departure is included
        last_departure_sec = sec_values[-1]
        last_headway = headway_segments[-1][2]
        fallback_records.append({
            'trip_id': new_trip_id,
            'start_time': to_hhmmss(last_departure_sec),
            'end_time': to_hhmmss(last_departure_sec + last_headway),
            'headway_secs': last_headway
        })

    if not fallback_records:
        return pd.DataFrame(columns=required_cols)

    return pd.DataFrame(fallback_records)

def resolve_overlapping_frequencies(frequencies_df: pd.DataFrame) -> pd.DataFrame:
    """Merge or truncate overlapping headway windows of the same trip so no two windows overlap."""
    if frequencies_df.empty:
        return frequencies_df
    df = frequencies_df.drop_duplicates(subset=["trip_id", "start_time"], keep="first").dropna(subset=["trip_id"]).copy()
    df["start_td"] = pd.to_timedelta(df["start_time"])
    df["end_td"] = pd.to_timedelta(df["end_time"])
    df = df.sort_values(by=["trip_id", "start_td"]).reset_index(drop=True)
    trips, starts, ends, headways = df["trip_id"].tolist(), df["start_td"].tolist(), df["end_td"].tolist(), df["headway_secs"].tolist()
    rows = []
    i = 0
    while i < len(trips):
        j = i
        while j < len(trips) and trips[j] == trips[i]:
            j += 1
        current = [trips[i], starts[i], ends[i], headways[i]]
        for k in range(i + 1, j):
            nxt = [trips[k], starts[k], ends[k], headways[k]]
            if nxt[1] < current[2]:
                if nxt[3] == current[3]:
                    current[2] = max(current[2], nxt[2])
                else:
                    current[2] = nxt[1]
                    if current[1] < current[2]:
                        rows.append(current)
                    current = nxt
            elif nxt[1] == current[2] and nxt[3] == current[3]:
                current[2] = nxt[2]
            else:
                rows.append(current)
                current = nxt
        rows.append(current)
        i = j
    return pd.DataFrame({
        "trip_id": [r[0] for r in rows],
        "start_time": [format_timedelta(r[1]) for r in rows],
        "end_time": [format_timedelta(r[2]) for r in rows],
        "headway_secs": pd.Series([r[3] for r in rows], dtype=df["headway_secs"].dtype),
    })
