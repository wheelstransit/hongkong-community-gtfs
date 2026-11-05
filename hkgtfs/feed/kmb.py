import logging

import pandas as pd

from ..db import Store
from .matching import match_operator_routes_to_government_gtfs
from .tables import Operator
from .text import smart_title_case

log = logging.getLogger(__name__)


def _long_name(group: pd.DataFrame) -> str:
    out = group[group["bound"] == "O"]
    inb = group[group["bound"] == "I"]
    if not out.empty and not inb.empty:
        return f"{smart_title_case(out.iloc[0]['orig_en'])} - {smart_title_case(inb.iloc[0]['orig_en'])}"
    first = out.iloc[0] if not out.empty else inb.iloc[0] if not inb.empty else group.iloc[0]
    return f"{smart_title_case(first['orig_en'])} - {smart_title_case(first['dest_en'])}"


def _trip(route, bound, service_type, service_id, gov_route_id, info) -> dict:
    origin, destination = smart_title_case(info.get("orig_en", "")), smart_title_case(info.get("dest_en", ""))
    return {
        "route_id": f"KMB-{route}",
        "service_id": f"KMB-{route}-{bound}-{service_id}",
        "trip_id": f"KMB-{route}-{bound}-{service_type}-{service_id}",
        "direction_id": 1 if bound == "I" else 0,
        "bound": bound,
        "route_short_name": route,
        "route_long_name": f"{origin} - {destination}",
        "original_service_id": service_id,
        "gov_route_id": gov_route_id,
        "service_type": service_type,
        "origin_en": origin,
        "destination_en": destination,
        "agency_id": "KMB",
    }


def build_kmb(db: Store) -> Operator:
    routes = db.read("SELECT * FROM kmb_routes")
    routes["agency_id"] = "KMB"
    routes[["route", "bound", "service_type"]] = routes["unique_route_id"].str.split("_", expand=True)
    routes["direction_id"] = routes["bound"].map({"O": 0, "I": 1}).fillna(-1).astype(int)

    patterns = db.read("SELECT * FROM kmb_stop_sequences")
    patterns[["route", "bound", "service_type"]] = patterns["unique_route_id"].str.split("_", expand=True)
    patterns = patterns.dropna(subset=["service_type"]).drop_duplicates(subset=["unique_route_id", "seq"])
    patterns["stop_id"] = "KMB-" + patterns["stop"].astype(str)

    gtfs_routes = pd.DataFrame([{"route_id": f"KMB-{route}", "agency_id": "KMB", "route_short_name": route,
                                 "route_long_name": _long_name(group), "route_type": 3}
                                for route, group in routes.groupby("route")])

    matches = match_operator_routes_to_government_gtfs(db=db, operator_name="KMB")
    trips = []
    for key, route_matches in matches.items():
        route, bound, service_type = key.split("-")
        info = routes[(routes["route"] == route) & (routes["bound"] == bound) & (routes["service_type"] == service_type)]
        if not info.empty:
            trips += [_trip(route, bound, service_type, m["gov_service_id"], m["gov_route_id"], info.iloc[0]) for m in route_matches]
    if trips:
        trips = pd.DataFrame(trips)
    else:
        log.warning("KMB route matching produced no trips")
        trips = pd.DataFrame(columns=["route_id", "service_id", "trip_id", "direction_id", "bound", "route_short_name", "route_long_name",
                                      "original_service_id", "gov_route_id", "service_type", "origin_en", "destination_en"])

    dummies = [{**_trip(r["route"], r["bound"], r["service_type"], "NEVER", None, r), "is_dummy": True}
               for _, r in routes.iterrows() if f"{r['route']}-{r['bound']}-{r['service_type']}" not in matches]
    if dummies:
        trips = pd.concat([trips, pd.DataFrame(dummies)], ignore_index=True)
    if "is_dummy" not in trips.columns:
        trips["is_dummy"] = False
    trips = trips.drop_duplicates(subset=["trip_id"], keep="first")
    return Operator(gtfs_routes, trips, patterns, routes)
