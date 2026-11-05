import numpy as np
import pandas as pd

from .times import format_timedelta

DEFAULT_SEGMENT_SECS = 120
COLUMNS = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
PATTERN_KEYS = {
    "KMB": ["route", "bound", "service_type"],
    "CTB": ["unique_route_id"],
    "GMB": ["route_code", "route_seq", "region"],
    "MTRB": ["route_id", "direction"],
    "NLB": ["routeId"],
}


def _unprefixed(stop_id) -> str:
    stop_id = str(stop_id)
    return stop_id.split("-", 1)[1] if "-" in stop_id else stop_id


def _trip_key(agency_id: str, trip):
    if agency_id == "KMB":
        return trip.route_short_name, trip.bound, trip.service_type
    if agency_id == "CTB":
        return trip.unique_route_id
    if agency_id == "GMB":
        return trip.route_code, trip.route_seq, trip.region
    if agency_id == "MTRB":
        return trip.route_short_name, "O" if trip.direction_id == 0 else "I"
    if agency_id == "NLB":
        return str(trip.routeId)
    return trip.trip_id


def generate_stop_times(agency_id: str, trips: pd.DataFrame, patterns: pd.DataFrame, journey_time: dict) -> pd.DataFrame:
    """Expand each trip's stop pattern into stop_times, timing segments with hk-bus-time-between-stops."""
    if trips.empty or patterns.empty:
        return pd.DataFrame(columns=COLUMNS)
    seq_col = next((c for c in ("sequence", "seq", "station_seqno", "stop_sequence") if c in patterns.columns), None)
    if seq_col is None:
        return pd.DataFrame(columns=COLUMNS)
    patterns = patterns.copy()
    patterns[seq_col] = pd.to_numeric(patterns[seq_col], errors="coerce")
    patterns = patterns.dropna(subset=[seq_col])
    patterns[seq_col] = patterns[seq_col].astype(int)
    if agency_id == "NLB":
        patterns["routeId"] = patterns["routeId"].astype(str)
    keys = [c for c in PATTERN_KEYS.get(agency_id, ["trip_id"]) if c in patterns.columns]
    if not keys:
        return pd.DataFrame(columns=COLUMNS)

    timed = {}
    for name, group in patterns.sort_values(seq_col).groupby(keys[0] if len(keys) == 1 else keys):
        stops = group["stop_id"].tolist()
        offsets = [0]
        for prev, stop in zip(stops, stops[1:]):
            seconds = (journey_time.get(_unprefixed(prev)) or {}).get(_unprefixed(stop))
            try:
                offsets.append(int(seconds if seconds is not None else DEFAULT_SEGMENT_SECS))
            except (ValueError, TypeError):
                offsets.append(DEFAULT_SEGMENT_SECS)
        times = [format_timedelta(t) for t in pd.to_timedelta(np.cumsum(offsets), unit="s")]
        timed[name] = (stops, group[seq_col].tolist(), times)

    trip_ids, arrivals, stop_ids, sequences = [], [], [], []
    for trip in trips.itertuples():
        pattern = timed.get(_trip_key(agency_id, trip))
        if not pattern:
            continue
        stops, seqs, times = pattern
        trip_ids += [trip.trip_id] * len(stops)
        stop_ids += stops
        sequences += seqs
        arrivals += times
    if not trip_ids:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame({"trip_id": trip_ids, "arrival_time": arrivals, "departure_time": arrivals,
                         "stop_id": stop_ids, "stop_sequence": sequences})
