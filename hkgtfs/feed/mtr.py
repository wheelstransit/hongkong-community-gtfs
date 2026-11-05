import logging
import re
import statistics
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from ..db import Store
from .stops import Stops
from .times import parse_headway

log = logging.getLogger(__name__)

LINE_NAMES = {
    "EAL": "East Rail Line", "TML": "Tuen Ma Line", "TWL": "Tsuen Wan Line", "KTL": "Kwun Tong Line", "ISL": "Island Line",
    "TKL": "Tseung Kwan O Line", "SIL": "South Island Line", "AEL": "Airport Express", "TCL": "Tung Chung Line",
    "DRL": "Disneyland Resort Line",
}
LINE_COLORS = {
    "EAL": "5EB7E8", "TCL": "F7943E", "TML": "923011", "AEL": "00888A", "DRL": "F173AC",
    "KTL": "00AB4E", "TWL": "ED1D24", "ISL": "007DC5", "TKL": "7D499D", "SIL": "BAC429",
}
HEADWAY_NAMES = {
    "EAL": "East Rail", "TML": "Tuen Ma", "TWL": "Tsuen Wan", "KTL": "Kwun Tong", "ISL": "Island", "TKL": "Tseung Kwan O",
    "SIL": "South Island", "AEL": "Airport Express", "TCL": "Tung Chung", "DRL": "Disneyland Resort",
}
DEFAULT_SLICES = [("05:30:00", "07:00:00", 360), ("07:00:00", "10:00:00", 180), ("10:00:00", "17:00:00", 300),
                  ("17:00:00", "20:00:00", 180), ("20:00:00", "23:00:00", 360), ("23:00:00", "25:00:00", 600)]
# TKL branches: LOHAS Park through trains are peak-only, the TIK-LHP shuttle runs off-peak, Po Lam trains all day.
TKL_PEAK_ONLY = [("07:00:00", "10:00:00", 450), ("17:00:00", "20:00:00", 450)]
TKL_SHUTTLE = [("05:30:00", "07:00:00", 720), ("10:00:00", "17:00:00", 720), ("20:00:00", "25:00:00", 720)]
TKL_TRUNK = [("05:30:00", "07:00:00", 300), ("07:00:00", "10:00:00", 210), ("10:00:00", "17:00:00", 240),
             ("17:00:00", "20:00:00", 210), ("20:00:00", "23:00:00", 300), ("23:00:00", "25:00:00", 450)]
VARIANT_SLICES = {
    "MTR-TKL-LHP-DT": TKL_PEAK_ONLY, "MTR-TKL-LHP-UT": TKL_PEAK_ONLY,
    "MTR-TKL-TKS-DT": TKL_SHUTTLE, "MTR-TKL-TKS-UT": TKL_SHUTTLE,
    "MTR-TKL-DT": TKL_TRUNK, "MTR-TKL-UT": TKL_TRUNK,
}


@dataclass
class Rail:
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame


def _seq_num(df: pd.DataFrame) -> pd.Series:
    return df["Sequence"].str.extract(r"^(\d+)").astype(int)


def _routes(stations: pd.DataFrame) -> pd.DataFrame:
    terminals = []
    for line, group in stations.groupby("Line Code"):
        directions = group["Direction"].unique()
        dt_variants = [d for d in directions if d.endswith("DT")]
        chosen = "DT"
        if "DT" not in directions and dt_variants:
            chosen = dt_variants[0]
        elif "DT" not in directions and "UT" in directions:
            chosen = "UT"
        subset = group[group["Direction"] == chosen].copy()
        if subset.empty:
            subset = group.copy()
        subset["seq_num"] = _seq_num(subset)
        subset = subset.sort_values("seq_num")
        if not subset.empty:
            terminals.append({"line_code": line, "terminal_pair": f"{subset.iloc[0]['English Name']} - {subset.iloc[-1]['English Name']}"})
    routes = pd.DataFrame(terminals)
    routes["route_id"] = "MTR-" + routes["line_code"]
    routes["agency_id"] = "MTRR"
    routes["route_short_name"] = routes["line_code"]
    routes["route_long_name"] = routes.apply(lambda r: f"{LINE_NAMES.get(r['line_code'], r['line_code'])} ({r['terminal_pair']})", axis=1)
    routes["route_type"] = 1
    routes["route_color"] = routes["line_code"].map(LINE_COLORS)
    routes["route_text_color"] = "FFFFFF"
    return routes[["route_id", "agency_id", "route_short_name", "route_long_name", "route_type", "route_color", "route_text_color"]]


def _platform_lookup(stops: Stops) -> dict:
    lookup = {}
    for code, platforms in stops.station_to_platforms.items():
        for platform in platforms:
            stop_id = stops.platform_stop_ids.get(platform["amenity_id"])
            for line_dir in platform.get("line_dirs", []):
                line, raw = line_dir.get("line_code"), line_dir.get("direction")
                direction = None
                if isinstance(raw, str):
                    raw = raw.strip().upper()
                    direction = 0 if raw.endswith("UT") else 1 if raw.endswith("DT") else None
                if line is not None and direction is not None and stop_id:
                    lookup.setdefault((code, line, direction, line_dir.get("destination_station_code")), stop_id)
                    lookup.setdefault((code, line, direction, None), stop_id)
    return lookup


def build_mtr(db: Store, journey_time: dict, stops: Stops) -> Rail:
    stations = db.read("SELECT * FROM mtr_lines_and_stations")
    edges = {(a.strip().upper(), b.strip().upper()): float(t)
             for a, dests in journey_time.items() if isinstance(dests, dict) for b, t in dests.items()}
    variants = stations[["Line Code", "Direction"]].drop_duplicates()
    trips = pd.DataFrame([{
        "route_id": f"MTR-{line}", "agency_id": "MTRR", "service_id": f"MTR-{line}-SERVICE", "trip_id": f"MTR-{line}-{variant}",
        "direction_id": 0 if variant.endswith("UT") else 1, "original_service_id": f"MTR-{line}-SERVICE", "route_short_name": line,
    } for line, variant in variants.itertuples(index=False)])

    platforms = _platform_lookup(stops)
    rail_edges = [v for (a, b), v in edges.items() if len(a) <= 4 and len(b) <= 4]
    median = statistics.median(rail_edges) if rail_edges else 120.0
    default = median if 30 <= median <= 600 else 120.0

    rows = []
    for line, variant in variants.itertuples(index=False):
        trip_id = f"MTR-{line}-{variant}"
        segment = stations[(stations["Line Code"] == line) & (stations["Direction"] == variant)].copy()
        if segment.empty:
            continue
        segment["seq_num"] = _seq_num(segment)
        segment = segment.sort_values("seq_num")
        direction = 0 if str(variant).endswith("UT") else 1
        destination = segment.iloc[-1]["Station Code"]
        elapsed, prev = 0.0, None
        for seq, code in enumerate(segment["Station Code"], start=1):
            code = str(code).strip().upper()
            if prev is not None:
                tt = edges.get((prev, code))
                if tt is None:
                    tt = edges.get((code, prev))
                if tt is None or tt <= 0 or tt > 3600:
                    tt = default
                elapsed += tt
            prev = code
            stop_id = platforms.get((code, line, direction, destination)) or platforms.get((code, line, direction, None))
            if not stop_id and (station_platforms := stops.station_to_platforms.get(code)):
                stop_id = stops.platform_stop_ids.get(station_platforms[0]["amenity_id"])
            if not stop_id:
                log.warning("no platform for station %s on %s; dropping its stop_time", code, trip_id)
                continue
            seconds = int(elapsed)
            time = f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"
            rows.append({"trip_id": trip_id, "arrival_time": time, "departure_time": time, "stop_id": stop_id, "stop_sequence": seq})
    return Rail(_routes(stations), trips, pd.DataFrame(rows))


def rail_frequencies(rail_trips: pd.DataFrame, lr_trips: pd.DataFrame, lr_frequencies: pd.DataFrame, headway: dict) -> pd.DataFrame:
    scraped = {}
    for raw_key, data in (headway or {}).items():
        if not isinstance(data, dict):
            continue
        key = raw_key.lower().replace(" line", "").strip()
        target = next((abbr for abbr, base in HEADWAY_NAMES.items()
                       if key == abbr.lower() or key == base.lower() or base.lower() in key or abbr.lower() in key), None)
        if target:
            scraped[target] = {"peak": parse_headway(data.get("weekdays", {}).get("morning_peak")),
                               "non_peak": parse_headway(data.get("weekdays", {}).get("non_peak"))}

    def slices(line):
        info = scraped.get(line, {})
        out = []
        for start, end, hw in DEFAULT_SLICES:
            if start in ("07:00:00", "17:00:00") and info.get("peak"):
                hw = info["peak"]
            if start in ("10:00:00", "20:00:00") and info.get("non_peak"):
                hw = info["non_peak"]
            out.append((start, end, int(hw)))
        return out

    rows = []
    if not rail_trips.empty:
        for line, group in rail_trips.assign(line=rail_trips["route_id"].str.replace("MTR-", "", regex=False)).groupby("line"):
            for trip in group["trip_id"].unique():
                rows += [{"trip_id": trip, "start_time": s, "end_time": e, "headway_secs": h} for s, e, h in VARIANT_SLICES.get(trip, slices(line))]
    if not lr_trips.empty:
        with_freq = set(lr_frequencies["trip_id"].dropna().astype(str)) if not lr_frequencies.empty else set()
        for line, group in lr_trips.assign(line=lr_trips["route_id"].str.replace("LR-", "", regex=False)).groupby("line"):
            for trip in [t for t in group["trip_id"].unique() if t not in with_freq]:
                rows += [{"trip_id": trip, "start_time": s, "end_time": e, "headway_secs": h} for s, e, h in slices(line)]
    return pd.DataFrame(rows)


def _norm(value):
    return None if value is None else str(value).strip().upper()


def _exit_code(exit_rec: dict):
    if not isinstance(exit_rec, dict):
        return None
    ref = exit_rec.get("ref")
    if ref is not None and str(ref).strip() != "":
        return str(ref).strip()
    if m := re.search(r"Exit\s+([A-Z][0-9]*)", str(exit_rec.get("name_en") or ""), flags=re.IGNORECASE):
        return m.group(1).upper()
    if m := re.match(r"([A-Z][0-9]*)", str(exit_rec.get("name_zh") or ""), flags=re.IGNORECASE):
        return m.group(1).upper()
    return None


def _walk_seconds(rec: dict):
    try:
        return float(rec.get("walk_time_minutes", 0)) * 60.0
    except (TypeError, ValueError):
        return None


def build_pathways(stops: Stops, crawler: dict, journeys: list, platform_journeys: list) -> pd.DataFrame:
    rows = []
    entrances = {}
    for rec in stops.mtr_entrances.to_dict("records"):
        stop_id = rec.get("stop_id", "")
        if "MTR-ENTRANCE-" in stop_id:
            station, exit_code = stop_id.split("MTR-ENTRANCE-")[1].rsplit("-", 1)
            entrances[(_norm(station), _norm(exit_code))] = stop_id
    names = dict(stops.mtr_stations[["station_code", "stop_name"]].values)
    exits = crawler.get("exits") or {}
    exit_by_amenity = {}
    for exit_rec in exits.values():
        exit_by_amenity.setdefault(exit_rec.get("amenity_id"), exit_rec)

    for rec in journeys:
        platform, exit_id = rec.get("platform_amenity_id"), rec.get("exit_amenity_id")
        if not platform or not exit_id:
            continue
        from_id, to_id, code = stops.platform_stop_ids.get(platform), None, rec.get("station_code")
        if code and (exit_rec := exit_by_amenity.get(exit_id)) is not None and (exit_code := _exit_code(exit_rec)):
            to_id = entrances.get((_norm(code), _norm(exit_code)))
            if not to_id and names.get(code):
                to_id = entrances.get((_norm(names[code]), _norm(exit_code)))
        if from_id and to_id:
            rows.append({"pathway_id": f"MTR-PATH-PEX-{platform}-{exit_id}", "from_stop_id": from_id, "to_stop_id": to_id,
                         "pathway_mode": 1, "is_bidirectional": 1, "length": rec.get("walk_distance_metres"), "traversal_time": _walk_seconds(rec)})

    for rec in platform_journeys:
        start, end = rec.get("start_platform_amenity_id"), rec.get("end_platform_amenity_id")
        from_id, to_id = stops.platform_stop_ids.get(start), stops.platform_stop_ids.get(end)
        if start and end and from_id and to_id:
            rows.append({"pathway_id": f"MTR-PATH-PP-{start}-{end}", "from_stop_id": from_id, "to_stop_id": to_id,
                         "pathway_mode": 1, "is_bidirectional": 1, "length": rec.get("walk_distance_metres"), "traversal_time": _walk_seconds(rec)})

    existing = {tuple(sorted((r["from_stop_id"], r["to_stop_id"]))) for r in rows}
    platform_ids = {code: [stops.platform_stop_ids[p["amenity_id"]] for p in platforms if stops.platform_stop_ids.get(p["amenity_id"])]
                    for code, platforms in stops.station_to_platforms.items()}
    entrance_ids = {}
    if not stops.mtr_entrances.empty:
        for code, group in stops.mtr_entrances.groupby(stops.mtr_entrances["parent_station"].str.replace("MTR-", "")):
            entrance_ids[code] = group["stop_id"].tolist()
    for code, pids in platform_ids.items():
        for pid in pids:
            for eid in entrance_ids.get(code, []):
                if tuple(sorted((pid, eid))) not in existing:
                    rows.append({"pathway_id": f"DUMMY-PEX-{pid}-{eid}", "from_stop_id": pid, "to_stop_id": eid,
                                 "pathway_mode": 1, "is_bidirectional": 1, "traversal_time": 60})
    for code, pids in platform_ids.items():
        for a, b in combinations(pids, 2):
            if tuple(sorted((a, b))) not in existing:
                rows.append({"pathway_id": f"DUMMY-PP-{a}-{b}", "from_stop_id": a, "to_stop_id": b,
                             "pathway_mode": 1, "is_bidirectional": 1, "traversal_time": 60})
    if not rows:
        return pd.DataFrame()
    pathways = pd.DataFrame(rows)
    for col in ("length", "traversal_time"):
        pathways[col] = pd.to_numeric(pathways[col], errors="coerce").round().astype("Int64")
    return pathways[["pathway_id", "from_stop_id", "to_stop_id", "pathway_mode", "is_bidirectional", "length", "traversal_time"]]
