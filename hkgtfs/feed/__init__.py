import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from ..db import Store
from .agency import build_agency, build_feed_info
from .calendar import build_calendar
from .citybus import build_citybus
from .directions import fix_inverted_directions
from .fares import generate_fare_stages, generate_light_rail_special_fare_rules, generate_mtr_special_fare_rules, generate_special_fare_rules, trip_fares
from .ferry import build_ferry
from .frequencies import build_frequencies, resolve_overlapping_frequencies, trip_id_mapping
from .gmb import build_gmb
from .headsigns import generate_trip_headsigns, generate_trip_headsigns_tc
from .kmb import build_kmb
from .light_rail import build_light_rail_gtfs_data
from .mtr import build_mtr, build_pathways, rail_frequencies
from .mtr_bus import build_mtr_bus
from .nlb import build_nlb
from .shapes import drop_misfitting_shapes, generate_shapes_from_csdi_files, match_trips_to_csdi_shapes
from .stop_times import generate_stop_times
from .stops import build_stops
from .text import fix_shouting
from .translations import build_translations

log = logging.getLogger(__name__)

AGENCY_COLORS = {"KMB": "EE171F", "CTB": "0053B9", "NLB": "8AB666", "MTRB": "AE2A42", "GMB": "34C759", "MTRR": "003DA5", "LR": "FF8800", "FERRY": "0099CC"}
ROUTE_COLUMNS = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type", "network_id", "route_color", "route_text_color"]
TRIP_COLUMNS = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
STOP_TIME_COLUMNS = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
SPECIAL_FARE_COLUMNS = ["special_fare_id", "rule_type", "trip_id", "onboarding_stop_id", "offboarding_stop_id", "price", "currency"]


def _gov_trips(db: Store) -> pd.DataFrame:
    routes = db.read("SELECT * FROM gov_gtfs_routes")
    trips = db.read("SELECT * FROM gov_gtfs_trips")
    try:
        trips["direction_id"] = (trips["trip_id"].str.split("-").str[1].astype(int) == 2).astype(int)
    except (IndexError, ValueError, TypeError):
        log.warning("could not parse direction_id from government trip_id")
        trips["direction_id"] = -1
    trips["service_id"] = trips["service_id"].astype(str)
    routes["route_short_name"] = routes["route_short_name"].astype(str)
    trips = trips.merge(routes[["route_id", "route_short_name", "agency_id", "route_long_name"]], on="route_id")
    trips["route_id"] = trips["route_id"].astype(str)
    return trips


def _routes(frames: list[pd.DataFrame], light_rail: pd.DataFrame) -> pd.DataFrame:
    routes = pd.concat(frames, ignore_index=True)
    for col in ("route_color", "route_text_color"):
        if col not in routes.columns:
            routes[col] = None
    agency_colors = routes["agency_id"].apply(lambda a: AGENCY_COLORS.get(a.split("+")[0]) if isinstance(a, str) else None)
    has = lambda col: routes[col].notna() & (routes[col].astype(str).str.strip() != "")
    routes["route_color"] = routes["route_color"].where(has("route_color"), agency_colors)
    routes["route_text_color"] = routes["route_text_color"].where(has("route_text_color"), "FFFFFF")
    if not light_rail.empty:
        for col in ("route_color", "route_text_color"):
            lookup = light_rail[["route_id", col]].dropna().set_index("route_id")[col]
            if not lookup.empty:
                routes.loc[routes["route_id"].isin(lookup.index), col] = routes["route_id"].map(lookup)
    routes["network_id"] = routes["route_id"].astype(str)
    for col in ROUTE_COLUMNS:
        if col not in routes.columns:
            routes[col] = None
    return routes[ROUTE_COLUMNS]


def _stop_times(frames: list[pd.DataFrame], stops: pd.DataFrame) -> pd.DataFrame:
    stop_times = pd.concat(frames, ignore_index=True)[STOP_TIME_COLUMNS].copy()
    stop_times.sort_values(["trip_id", "stop_sequence"], inplace=True)
    stop_times["stop_sequence"] = stop_times.groupby("trip_id").cumcount() + 1
    stop_times.drop_duplicates(subset=["trip_id", "stop_sequence"], keep="first", inplace=True)
    valid = stop_times["stop_id"].isin(set(stops["stop_id"]))
    if not valid.all():
        log.warning("dropping %d stop_times that reference unknown stops", (~valid).sum())
    return stop_times[valid]


def build_feed(db: Store, raw: dict, out: Path, waypoints: Path, version: str):
    gtfs = out / "gtfs"
    shutil.rmtree(gtfs, ignore_errors=True)
    gtfs.mkdir(parents=True)

    ferry = build_ferry(db)
    stops = build_stops(db, raw["mtr"]["crawler"], ferry.stops)
    gov_trips = _gov_trips(db)
    kmb, ctb, gmb, mtrb, nlb = build_kmb(db), build_citybus(db, gov_trips), build_gmb(db), build_mtr_bus(db), build_nlb(db)
    rail = build_mtr(db, raw["journey_time"], stops)
    lr = build_light_rail_gtfs_data(raw["mtr"]["light_rail_routes_and_stops"], raw["mtr"]["light_rail_schedule"])
    lr_routes = lr.routes if not lr.routes.empty else pd.DataFrame(columns=["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
    lr_trips = lr.trips if not lr.trips.empty else pd.DataFrame(columns=["route_id", "agency_id", "route_short_name", "service_id", "trip_id", "direction_id", "bound", "trip_headsign"])

    buses = {"KMB": kmb, "CTB": ctb, "GMB": gmb, "MTRB": mtrb, "NLB": nlb}
    bus_stop_times = []
    coords = dict(zip(stops.table["stop_id"], zip(stops.table["stop_lat"], stops.table["stop_lon"])))
    for agency, op in buses.items():
        op.trips["original_service_id"] = op.trips["original_service_id"].astype(str)
        op.trips["route_short_name"] = op.trips["route_short_name"].astype(str)
        bus_stop_times.append(generate_stop_times(agency, op.trips, op.patterns, raw["journey_time"], coords))

    routes = _routes([*(op.routes for op in buses.values()), ferry.routes, rail.routes, lr_routes], lr_routes)
    trips = pd.concat([*(op.trips for op in buses.values()), ferry.trips, rail.trips, lr_trips], ignore_index=True)
    gov_trips = gov_trips.rename(columns={"agency_id": "gov_agency_id"})
    for col in ("original_service_id", "route_short_name", "gov_route_id"):
        trips[col] = trips[col].astype(str)

    mapping = trip_id_mapping(trips, gov_trips)
    gov_stop_times = db.read("SELECT * FROM gov_gtfs_stop_times")
    frequencies = build_frequencies(mapping, db.read("SELECT * FROM gov_gtfs_frequencies"), gov_stop_times, ferry.frequencies,
                                    lr.frequencies, rail_frequencies(rail.trips, lr_trips, lr.frequencies, raw["mtr"]["headway"]))

    trips = trips.drop_duplicates(subset=["trip_id"], keep="first")
    stop_times = _stop_times([*bus_stop_times, ferry.stop_times, rail.stop_times, lr.stop_times], stops.table)
    trips = fix_inverted_directions(trips, stop_times, stops.table, mapping, gov_stop_times, db.read("SELECT * FROM gov_gtfs_stops"))

    shape_info, shape_ends = generate_shapes_from_csdi_files(gtfs / "shapes.txt", db, waypoints)
    trips = match_trips_to_csdi_shapes(trips, shape_info, db, waypoints, shape_ends)
    trips = drop_misfitting_shapes(trips, stop_times, stops.table, waypoints)
    trips = generate_trip_headsigns(db, trips)
    trips["trip_headsign_tc"] = generate_trip_headsigns_tc(db, trips, english_headsigns=trips.get("trip_headsign"))
    for col in TRIP_COLUMNS:
        if col not in trips.columns:
            trips[col] = None

    calendar, calendar_dates = build_calendar(db, trips)
    pathways = build_pathways(stops, raw["mtr"]["crawler"], raw["mtr"]["journeys"], raw["mtr"]["platform_journeys"])
    translations = build_translations(stops, kmb, ctb, nlb, rail.routes, lr_routes, trips)
    fares = trip_fares(db, trips)
    fare_stages = generate_fare_stages(fares, stop_times)
    special_fares = [df for df in (generate_special_fare_rules(fares, stop_times), generate_mtr_special_fare_rules(db),
                                   generate_light_rail_special_fare_rules(db)) if not df.empty]

    counts = stop_times["trip_id"].value_counts()
    runnable = set(counts[counts >= 2].index)
    served = set(stop_times.loc[stop_times["trip_id"].isin(runnable), "stop_id"])
    stop_table = stops.table[(stops.table["location_type"] != 0) | stops.table["stop_id"].isin(served)
                             | ~stops.table["stop_id"].str.startswith("MTR-PLATFORM-")].copy()
    if not pathways.empty:
        kept = set(stop_table["stop_id"])
        pathways = pathways[pathways["from_stop_id"].isin(kept) & pathways["to_stop_id"].isin(kept)]
    stop_table["stop_name"] = stop_table["stop_name"].map(fix_shouting)
    routes["route_long_name"] = routes["route_long_name"].map(fix_shouting)
    trips["trip_headsign"] = trips["trip_headsign"].map(fix_shouting)
    trips = trips[trips["trip_id"].isin(runnable)]
    stop_times = stop_times[stop_times["trip_id"].isin(runnable)]
    frequencies = frequencies[frequencies["trip_id"].isin(runnable)]
    translations = translations[(translations["table_name"] != "trips") | translations["record_id"].isin(runnable)]
    fare_stages = fare_stages[fare_stages["trip_id"].isin(runnable)]
    special_fares = [df[(df["rule_type"] != "trip") | df["trip_id"].isin(runnable)] for df in special_fares]

    tables = {
        "agency.txt": build_agency(),
        "feed_info.txt": build_feed_info(version, calendar),
        "stops.txt": stop_table,
        "routes.txt": routes,
        "trips.txt": trips[TRIP_COLUMNS],
        "stop_times.txt": stop_times,
        "frequencies.txt": resolve_overlapping_frequencies(frequencies)[["trip_id", "start_time", "end_time", "headway_secs"]],
        "pathways.txt": pathways,
        "calendar.txt": calendar,
        "calendar_dates.txt": calendar_dates,
        "translations.txt": translations,
        "fare_stages.csv": fare_stages,
        "special_fare_rules.csv": pd.concat(special_fares, ignore_index=True) if special_fares else pd.DataFrame(columns=SPECIAL_FARE_COLUMNS),
    }
    for name, df in tables.items():
        if not df.empty or name not in ("pathways.txt", "translations.txt", "fare_stages.csv"):
            df.to_csv(gtfs / name, index=False)
    with zipfile.ZipFile(out / "hk.gtfs.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(gtfs.iterdir()):
            zf.write(path, arcname=path.name)
