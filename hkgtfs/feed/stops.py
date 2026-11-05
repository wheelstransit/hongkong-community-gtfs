import logging
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

from ..db import Store
from .text import smart_title_case

log = logging.getLogger(__name__)
COLUMNS = ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station", "platform_code"]
BASE = COLUMNS[:6]


@dataclass
class Stops:
    table: pd.DataFrame
    operator: dict
    mtr_stations: pd.DataFrame
    mtr_stations_raw: gpd.GeoDataFrame
    mtr_exits: gpd.GeoDataFrame
    mtr_entrances: pd.DataFrame
    station_to_platforms: dict = field(default_factory=dict)
    platform_stop_ids: dict = field(default_factory=dict)


def _with_coords(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["stop_lat"] = gdf.geometry.y
    gdf["stop_lon"] = gdf.geometry.x
    return gdf


def _operator_stops(db: Store) -> dict:
    kmb = db.read_geo("SELECT * FROM kmb_stops")
    kmb["stop_id"] = "KMB-" + kmb["stop"].astype(str)
    kmb["stop_name"] = (kmb["name_en"]
                        .str.replace(r"\s*\([A-Za-z0-9]{5}\)", "", regex=True)
                        .str.replace(r"\s*-\s*", " - ", regex=True)
                        .str.replace(r"([^\s])(\([A-Za-z0-9]+\))", r"\1 \2", regex=True)
                        .apply(smart_title_case))
    ctb = db.read_geo("SELECT * FROM citybus_stops")
    ctb["stop_id"] = "CTB-" + ctb["stop"].astype(str)
    ctb["stop_name"] = ctb["name_en"]
    gmb = db.read_geo("SELECT * FROM gmb_stops")
    gmb["stop_id"] = "GMB-" + gmb["stop_id"].astype(str)
    gmb["stop_name"] = gmb["stop_name_en"]
    mtrb = db.read_geo("SELECT * FROM mtrbus_stops")
    mtrb["stop_id"] = "MTRB-" + mtrb["stop_id"].astype(str)
    mtrb["stop_name"] = mtrb["name_en"]
    nlb = db.read_geo("SELECT * FROM nlb_stops")
    nlb["stop_id"] = "NLB-" + nlb["stopId"].astype(str)
    nlb["stop_name"] = nlb["stopName_e"]
    return {k: _with_coords(v) for k, v in {"KMB": kmb, "CTB": ctb, "GMB": gmb, "MTRB": mtrb, "NLB": nlb}.items()}


def _platforms(crawler: dict, station_to_platforms: dict, platform_stop_ids: dict) -> pd.DataFrame:
    platforms = crawler.get("platforms") or {}
    rows = []
    for code, station in (crawler.get("stations") or {}).items():
        station_to_platforms[code] = []
        for key in station.get("platforms") or []:
            platform = platforms.get(key)
            if not platform:
                continue
            amenity_id = platform.get("amenity_id") or key
            name = platform.get("name_en") or platform.get("name_zh") or f"Platform {platform.get('ref', '')}"
            ref = str(platform.get("ref")) if platform.get("ref") is not None else None
            stop_id = f"MTR-PLATFORM-{code}-{ref}" if ref else f"MTR-PLATFORM-{code}-{amenity_id}"
            station_to_platforms[code].append({"amenity_id": amenity_id, "line_dirs": platform.get("lines_and_directions") or [],
                                               "ref": ref, "name_en": name, "stop_id": stop_id})
            platform_stop_ids[amenity_id] = stop_id
            coords = platform.get("coordinates") or {}
            rows.append({"stop_id": stop_id, "stop_name": f"Platform {ref}" if ref else name, "stop_lat": coords.get("lat"),
                         "stop_lon": coords.get("lon"), "location_type": 0, "parent_station": f"MTR-{code}", "platform_code": ref})
        location = station.get("location") or {}
        if not station_to_platforms[code] and location.get("lat") is not None and location.get("lon") is not None:
            amenity_id, stop_id = f"synthetic-{code}", f"MTR-PLATFORM-{code}-1"
            station_to_platforms[code].append({"amenity_id": amenity_id, "line_dirs": [], "ref": "1", "name_en": "Platform 1", "stop_id": stop_id})
            platform_stop_ids[amenity_id] = stop_id
            rows.append({"stop_id": stop_id, "stop_name": "Platform 1", "stop_lat": location.get("lat"), "stop_lon": location.get("lon"),
                         "location_type": 0, "parent_station": f"MTR-{code}", "platform_code": "1"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=COLUMNS)


def build_stops(db: Store, crawler: dict, ferry_stops: pd.DataFrame) -> Stops:
    operator = _operator_stops(db)

    stations_raw = db.read_geo("SELECT * FROM mtr_lines_and_stations")
    stations = (stations_raw[["Station Code", "English Name", "geometry"]].drop_duplicates(subset=["Station Code"])
                .rename(columns={"Station Code": "station_code", "English Name": "stop_name"}))
    stations["stop_id"] = "MTR-" + stations["station_code"].astype(str)
    stations = _with_coords(stations)
    stations["location_type"] = 1
    stations["parent_station"] = None

    station_to_platforms, platform_stop_ids = {}, {}
    platforms = _platforms(crawler, station_to_platforms, platform_stop_ids)

    known = set(stations["station_code"].astype(str))
    backfill = [{"station_code": code, "stop_name": s.get("name_en") or code, "stop_id": f"MTR-{code}",
                 "stop_lat": loc["lat"], "stop_lon": loc["lon"], "location_type": 1, "parent_station": None}
                for code, s in (crawler.get("stations") or {}).items()
                if code not in known and (loc := s.get("location") or {}).get("lat") is not None and loc.get("lon") is not None]
    if backfill:
        stations = pd.concat([stations, pd.DataFrame(backfill)], ignore_index=True)
        log.info("backfilled MTR stations missing from opendata: %s", [r["station_code"] for r in backfill])

    exits = db.read_geo("SELECT * FROM mtr_exits")
    if exits.empty:
        entrances = pd.DataFrame(columns=BASE)
    else:
        exits["stop_id"] = "MTR-ENTRANCE-" + exits["station_code"] + "-" + exits["exit"]
        exits["parent_station"] = exits["station_code"].map(stations.set_index("station_code")["stop_id"].to_dict())
        exits["stop_name"] = exits["exit"]
        exits = _with_coords(exits)
        exits["location_type"] = 2
        entrances = exits[BASE]

    light_rail = _with_coords(db.read_geo("SELECT * FROM light_rail_stops").rename(columns={"name_en": "stop_name"}))
    light_rail["location_type"] = 0
    light_rail["parent_station"] = None
    operator["LR"] = light_rail

    table = pd.concat([*(operator[k][["stop_id", "stop_name", "stop_lat", "stop_lon"]] for k in ("KMB", "CTB", "GMB", "MTRB", "NLB")),
                       ferry_stops[BASE], stations[BASE], platforms[COLUMNS], entrances, light_rail[BASE]], ignore_index=True)
    for col in COLUMNS:
        if col not in table.columns:
            table[col] = None
    table["location_type"] = table["location_type"].fillna(0).astype(int)
    table["stop_name"] = table["stop_name"].str.strip()
    return Stops(table[COLUMNS], operator, stations, stations_raw, exits, entrances, station_to_platforms, platform_stop_ids)
