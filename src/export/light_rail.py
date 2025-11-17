from __future__ import annotations

import asyncio
import csv
import json
import logging
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
import pandas as pd
from pyproj import Transformer

from lightrail_legacy.parse_schedule import (
    SCHEDULE_URL,
    fetch_light_rail_schedule_html,
    parse_light_rail_schedule,
)

CIRCULAR_ROUTES = {"705", "706"}
DEFAULT_FREQUENCIES = (
    {"start_time": "05:00:00", "end_time": "25:30:00", "headway_secs": 600},
)
FALLBACK_SECONDS_PER_STOP = 180
LIGHT_RAIL_ROUTES_AND_STOPS_URL = "https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv"
GEODATA_BASE_URL = "https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q="
LIGHT_RAIL_STOP_SUFFIX = "\u8f15\u9435\u7ad9"
ROUTE_COLORS = {
    "505": "DA2128",
    "507": "00A54F",
    "610": "541912",
    "614": "15C0F2",
    "614P": "F6A4AA",
    "615": "FFDC01",
    "615P": "016584",
    "705": "7BC351",
    "706": "B279B4",
    "751": "F58220",
    "761P": "6E2C91",
}
DEFAULT_ROUTE_COLOR = "FF8800"
DEFAULT_ROUTE_TEXT_COLOR = "FFFFFF"


@dataclass
class LightRailGTFSData:
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    frequencies: pd.DataFrame
    stops: pd.DataFrame


def _route_key(route_code: str, bound: str) -> str:
    return f"{route_code}_{bound}"


def _determine_bound(route_code: str, bound_value: str) -> str:
    if route_code in CIRCULAR_ROUTES:
        return "O"
    return "O" if bound_value == "1" else "I"


def _normalize_hex(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        return None
    if any(ch not in string.hexdigits for ch in cleaned):
        return None
    return cleaned.upper()


def _pick_text_color(color_hex: Optional[str]) -> str:
    normalized = _normalize_hex(color_hex)
    if not normalized:
        return DEFAULT_ROUTE_TEXT_COLOR
    r = int(normalized[0:2], 16)
    g = int(normalized[2:4], 16)
    b = int(normalized[4:6], 16)
    brightness = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "000000" if brightness >= 186 else "FFFFFF"


def _load_schedule_data(schedule_path: Optional[str], silent: bool = False) -> Dict:
    schedule_data: Dict = {}
    fetch_error: Optional[Exception] = None

    try:
        html_content = fetch_light_rail_schedule_html(url=SCHEDULE_URL)
        schedule_data = parse_light_rail_schedule(html_content)
    except Exception as exc:  # pragma: no cover - network failure fallback
        fetch_error = exc
        if not silent:
            logging.warning("Failed to download Light Rail schedule from %s: %s", SCHEDULE_URL, exc)

    if schedule_data and schedule_path:
        try:
            path = Path(schedule_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(schedule_data, f, ensure_ascii=False, indent=2)
        except OSError as exc:  # pragma: no cover - IO failure fallback
            if not silent:
                logging.warning("Unable to persist Light Rail schedule JSON to %s: %s", schedule_path, exc)

    if not schedule_data and schedule_path:
        try:
            with Path(schedule_path).open("r", encoding="utf-8") as f:
                schedule_data = json.load(f)
        except FileNotFoundError:
            if not silent:
                logging.warning("Light Rail schedule file %s not found.", schedule_path)
        except json.JSONDecodeError as exc:
            if not silent:
                logging.warning("Light Rail schedule file %s is invalid JSON: %s", schedule_path, exc)

    if not schedule_data and fetch_error and not silent:
        logging.warning("Proceeding without schedule-derived timings due to earlier failure.")

    return schedule_data


async def _fetch_route_and_stop_data(silent: bool = False) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    route_list: Dict[str, Dict] = {}
    stop_list: Dict[str, Dict] = {}
    transformer = Transformer.from_crs("epsg:2326", "epsg:4326", always_xy=True)

    timeout = httpx.Timeout(30.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(LIGHT_RAIL_ROUTES_AND_STOPS_URL)
            response.raise_for_status()
        except httpx.RequestError as exc:
            logging.error("Could not fetch Light Rail route data: %s", exc)
            return {}, {}

        csv_reader = csv.reader(response.text.splitlines())
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) != 7:
                continue
            route_code, bound, _, stop_id, stop_name_tc, stop_name_en, seq = row
            key = _route_key(route_code, bound)
            light_rail_id = f"LR{stop_id}"
            bound_direction = _determine_bound(route_code, bound)

            if key not in route_list:
                route_list[key] = {
                    "route": route_code,
                    "bound": bound_direction,
                    "orig_en": stop_name_en,
                    "orig_tc": stop_name_tc,
                    "dest_en": stop_name_en,
                    "dest_tc": stop_name_tc,
                    "stops": [light_rail_id],
                }
            else:
                suffix_en = " (Circular)" if route_code in CIRCULAR_ROUTES else ""
                suffix_tc = " (循環線)" if route_code in CIRCULAR_ROUTES else ""
                route_list[key]["dest_en"] = f"{stop_name_en}{suffix_en}".strip()
                route_list[key]["dest_tc"] = f"{stop_name_tc}{suffix_tc}".strip()
                if route_code in CIRCULAR_ROUTES and seq != "1.00" and light_rail_id == route_list[key]["stops"][0]:
                    continue
                if light_rail_id not in route_list[key]["stops"]:
                    route_list[key]["stops"].append(light_rail_id)

            if light_rail_id not in stop_list:
                stop_record = {
                    "stop_id": light_rail_id,
                    "stop_name": stop_name_en,
                    "stop_name_tc": stop_name_tc,
                    "stop_lat": None,
                    "stop_lon": None,
                }
                stop_list[light_rail_id] = stop_record
                try:
                    encoded_query = quote(f"{stop_name_tc}{LIGHT_RAIL_STOP_SUFFIX}")
                    geo_url = f"{GEODATA_BASE_URL}{encoded_query}"
                    geo_response = await client.get(geo_url, headers={"Accept": "application/json"})
                    geo_response.raise_for_status()
                    data = geo_response.json()
                    if isinstance(data, list) and data:
                        lon, lat = transformer.transform(data[0]["x"], data[0]["y"])
                        stop_record["stop_lat"] = lat
                        stop_record["stop_lon"] = lon
                    elif not silent:
                        logging.warning("No geodata result for Light Rail stop %s", stop_name_tc)
                except (httpx.RequestError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                    if not silent:
                        logging.warning("Failed to fetch geodata for Light Rail stop %s: %s", stop_name_tc, exc)

    return route_list, stop_list


def _select_schedule_entries(route_info: Dict, schedule_data: Dict) -> List[Dict]:
    schedule_for_route = schedule_data.get(route_info["route"])
    if not isinstance(schedule_for_route, dict):
        return []

    if route_info["route"] in CIRCULAR_ROUTES:
        entries = schedule_for_route.get("circular")
        return entries if isinstance(entries, list) else []

    candidates = [route_info.get("dest_en", ""), route_info.get("dest_tc", "")]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        cleaned = candidate.split("(", 1)[0].strip().lower()
        if not cleaned:
            continue
        for key, value in schedule_for_route.items():
            if isinstance(value, list) and cleaned in str(key).lower():
                return value

    for value in schedule_for_route.values():
        if isinstance(value, list):
            return value
    return []


def _compute_average_time_per_stop(route_info: Dict, schedule_data: Dict) -> float:
    stops = route_info.get("stops", [])
    if len(stops) <= 1:
        return FALLBACK_SECONDS_PER_STOP

    stop_entries = _select_schedule_entries(route_info, schedule_data)
    if not stop_entries:
        return FALLBACK_SECONDS_PER_STOP

    start_time_str = stop_entries[0].get("first_train")
    end_time_str = stop_entries[-1].get("first_train") or stop_entries[-1].get("last_train")
    if not start_time_str or not end_time_str:
        return FALLBACK_SECONDS_PER_STOP

    try:
        start_dt = datetime.strptime(start_time_str, "%H:%M")
        end_dt = datetime.strptime(end_time_str, "%H:%M")
    except ValueError:
        return FALLBACK_SECONDS_PER_STOP

    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    total_duration = (end_dt - start_dt).total_seconds()
    avg_seconds = total_duration / (len(stops) - 1)
    if avg_seconds <= 0 or avg_seconds >= 900:
        return FALLBACK_SECONDS_PER_STOP
    return avg_seconds


def _format_elapsed_time(seconds_elapsed: int) -> str:
    hours = seconds_elapsed // 3600
    minutes = (seconds_elapsed % 3600) // 60
    seconds = seconds_elapsed % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _build_routes_dataframe(route_list: Dict[str, Dict]) -> pd.DataFrame:
    records: Dict[str, Dict] = {}
    for info in route_list.values():
        route_code = info["route"]
        base_color = _normalize_hex(ROUTE_COLORS.get(route_code)) or DEFAULT_ROUTE_COLOR
        record = records.setdefault(
            route_code,
            {
                "route_id": f"LR-{route_code}",
                "agency_id": "LR",
                "route_short_name": route_code,
                "route_long_name": "",
                "route_long_name_tc": "",
                "route_type": 0,
                "route_color": base_color,
                "route_text_color": _pick_text_color(base_color),
            },
        )
        origin_en = info.get("orig_en") or info.get("orig_tc") or route_code
        destination_en = info.get("dest_en") or info.get("dest_tc") or origin_en
        origin_tc = info.get("orig_tc") or info.get("orig_en") or route_code
        destination_tc = info.get("dest_tc") or info.get("dest_en") or origin_tc

        if not record["route_long_name"]:
            if route_code in CIRCULAR_ROUTES:
                record["route_long_name"] = destination_en
            else:
                record["route_long_name"] = (
                    destination_en if origin_en == destination_en else f"{origin_en} - {destination_en}"
                )

        if not record["route_long_name_tc"]:
            if route_code in CIRCULAR_ROUTES:
                record["route_long_name_tc"] = destination_tc
            else:
                record["route_long_name_tc"] = (
                    destination_tc if origin_tc == destination_tc else f"{origin_tc} - {destination_tc}"
                )

        mapped_color = _normalize_hex(ROUTE_COLORS.get(route_code))
        if mapped_color and mapped_color != record["route_color"]:
            record["route_color"] = mapped_color
            record["route_text_color"] = _pick_text_color(mapped_color)

    columns = [
        "route_id",
        "agency_id",
        "route_short_name",
        "route_long_name",
        "route_long_name_tc",
        "route_type",
        "route_color",
        "route_text_color",
    ]
    return pd.DataFrame(records.values(), columns=columns)


def _build_trips_dataframe(route_list: Dict[str, Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for info in route_list.values():
        trip_id = f"LR-{info['route']}-{info['bound']}"
        rows.append(
            {
                "route_id": f"LR-{info['route']}",
                "agency_id": "LR",
                "route_short_name": info["route"],
                "service_id": f"LR-{info['route']}-DAILY",
                "trip_id": trip_id,
                "direction_id": 0 if info["bound"] == "O" else 1,
                "bound": info["bound"],
                "trip_headsign": info.get("dest_en") or info.get("dest_tc") or info["route"],
                "trip_headsign_tc": info.get("dest_tc") or info.get("dest_en") or info["route"],
            }
        )
    columns = [
        "route_id",
        "agency_id",
        "route_short_name",
        "service_id",
        "trip_id",
        "direction_id",
        "bound",
        "trip_headsign",
        "trip_headsign_tc",
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_stop_times_dataframe(route_list: Dict[str, Dict], schedule_data: Dict) -> pd.DataFrame:
    rows: List[Dict] = []
    for info in route_list.values():
        trip_id = f"LR-{info['route']}-{info['bound']}"
        avg_seconds = _compute_average_time_per_stop(info, schedule_data)
        for index, stop_id in enumerate(info.get("stops", [])):
            elapsed_seconds = int(round(avg_seconds * index))
            formatted_time = _format_elapsed_time(elapsed_seconds)
            rows.append(
                {
                    "trip_id": trip_id,
                    "arrival_time": formatted_time,
                    "departure_time": formatted_time,
                    "stop_id": stop_id,
                    "stop_sequence": index + 1,
                }
            )
    columns = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    return pd.DataFrame(rows, columns=columns)


def _build_frequencies_dataframe(route_list: Dict[str, Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for info in route_list.values():
        trip_id = f"LR-{info['route']}-{info['bound']}"
        for freq in DEFAULT_FREQUENCIES:
            rows.append(
                {
                    "trip_id": trip_id,
                    "start_time": freq["start_time"],
                    "end_time": freq["end_time"],
                    "headway_secs": int(freq["headway_secs"]),
                    "exact_times": 0,
                }
            )
    columns = ["trip_id", "start_time", "end_time", "headway_secs", "exact_times"]
    return pd.DataFrame(rows, columns=columns)


def _build_stops_dataframe(stop_list: Dict[str, Dict]) -> pd.DataFrame:
    if not stop_list:
        return pd.DataFrame(columns=["stop_id", "stop_name", "stop_name_tc", "stop_lat", "stop_lon"])
    rows = list(stop_list.values())
    columns = ["stop_id", "stop_name", "stop_name_tc", "stop_lat", "stop_lon"]
    return pd.DataFrame(rows, columns=columns)


def build_light_rail_gtfs_data(schedule_path: Optional[str], silent: bool = False) -> LightRailGTFSData:
    schedule_data = _load_schedule_data(schedule_path, silent=silent)
    route_list, stop_list = asyncio.run(_fetch_route_and_stop_data(silent=silent))

    if not route_list:
        empty = pd.DataFrame()
        return LightRailGTFSData(empty, empty, empty, empty, empty)

    routes_df = _build_routes_dataframe(route_list)
    trips_df = _build_trips_dataframe(route_list)
    stop_times_df = _build_stop_times_dataframe(route_list, schedule_data)
    frequencies_df = _build_frequencies_dataframe(route_list)
    stops_df = _build_stops_dataframe(stop_list)

    return LightRailGTFSData(routes_df, trips_df, stop_times_df, frequencies_df, stops_df)


__all__ = ["LightRailGTFSData", "build_light_rail_gtfs_data"]
