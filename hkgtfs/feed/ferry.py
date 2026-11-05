import re
from dataclasses import dataclass

import pandas as pd

from ..db import Store
from .times import to_hhmmss, to_seconds

FERRY_TRIPS = "SELECT trip_id FROM gov_gtfs_trips WHERE route_id IN (SELECT route_id FROM gov_gtfs_routes WHERE agency_id = 'FERRY')"


@dataclass
class Ferry:
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    stops: pd.DataFrame
    frequencies: pd.DataFrame


def _template_id(trip_id: str) -> str:
    return re.sub(r"-\d{4}$", "", trip_id)


def build_ferry(db: Store) -> Ferry:
    routes = db.read("SELECT route_id, route_short_name, route_long_name, agency_id, route_type FROM gov_gtfs_routes WHERE agency_id = 'FERRY'")
    trips = db.read("SELECT route_id, service_id, trip_id FROM gov_gtfs_trips WHERE route_id IN (SELECT route_id FROM gov_gtfs_routes WHERE agency_id = 'FERRY')")
    trips["trip_id"] = "FERRY-" + trips["trip_id"].astype(str)
    trips["service_id"] = "FERRY-" + trips["service_id"].astype(str)
    trips["agency_id"] = "FERRY"
    trips["original_service_id"] = trips["service_id"]
    trips["route_short_name"] = trips["route_id"].astype(str)
    trips["direction_id"] = 0

    stop_times = db.read(f"SELECT trip_id, arrival_time, departure_time, stop_id, stop_sequence FROM gov_gtfs_stop_times WHERE trip_id IN ({FERRY_TRIPS})")
    stop_times["trip_id"] = "FERRY-" + stop_times["trip_id"].astype(str)
    stop_times["stop_id"] = "FERRY-" + stop_times["stop_id"].astype(str)

    stops = db.read(f"""SELECT DISTINCT stop_id, stop_name, stop_lat, stop_lon FROM gov_gtfs_stops
                        WHERE stop_id IN (SELECT DISTINCT stop_id FROM gov_gtfs_stop_times WHERE trip_id IN ({FERRY_TRIPS}))""")
    if stops.empty:
        stops = pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"])
    else:
        stops["stop_id"] = "FERRY-" + stops["stop_id"].astype(str)
        stops["location_type"] = 0
        stops["parent_station"] = None

    frequencies, keep = [], set()
    if not stop_times.empty and not trips.empty:
        for _, group in trips.groupby(["route_id", "service_id"]):
            first = stop_times[stop_times["trip_id"].isin(group["trip_id"].tolist()) & (stop_times["stop_sequence"] == 1)].copy()
            if first.empty:
                continue
            first["departure_seconds"] = first["departure_time"].apply(to_seconds)
            first = first.sort_values("departure_seconds").dropna(subset=["departure_seconds"])
            template = first.iloc[0]["trip_id"]
            keep.add(template)
            trip_id = _template_id(template)
            times = first["departure_seconds"].tolist()
            if len(first) < 2:
                frequencies.append({"trip_id": trip_id, "start_time": to_hhmmss(int(times[0])), "end_time": to_hhmmss(int(times[0]) + 86400), "headway_secs": 86400})
                continue
            intervals = [b - a for a, b in zip(times, times[1:])]
            block_start, current = 0, max(intervals[0], 60)
            for i in range(1, len(intervals)):
                nxt = max(intervals[i], 60)
                diff = abs(nxt - current) / current if current > 0 else 0
                if diff > 0.1 or i == len(intervals) - 1:
                    block_end = i if diff > 0.1 else i + 1
                    block = intervals[block_start:block_end]
                    headway = max(60, int(sum(block) / len(block)) if block else 1800)
                    start, end = int(times[block_start]), int(times[block_end])
                    frequencies.append({"trip_id": trip_id, "start_time": to_hhmmss(start),
                                        "end_time": to_hhmmss(end if end > start else start + headway), "headway_secs": headway})
                    block_start, current = i, max(intervals[i], 60)

    if keep:
        trips = trips[trips["trip_id"].isin(keep)].copy()
        stop_times = stop_times[stop_times["trip_id"].isin(keep)].copy()
        trips["trip_id"] = trips["trip_id"].map(_template_id)
        stop_times["trip_id"] = stop_times["trip_id"].map(_template_id)

    routes = routes.copy()
    routes["agency_id"] = "FERRY"
    return Ferry(routes, trips, stop_times, stops,
                 pd.DataFrame(frequencies) if frequencies else pd.DataFrame(columns=["trip_id", "start_time", "end_time", "headway_secs"]))
