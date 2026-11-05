import logging

import pandas as pd

from ..db import Store
from .matching import match_operator_routes_to_government_gtfs
from .tables import Operator

log = logging.getLogger(__name__)


def build_mtr_bus(db: Store) -> Operator:
    routes = db.read("SELECT * FROM mtrbus_routes")
    routes["agency_id"] = "MTRB"
    routes["route_type"] = 3
    routes["route_long_name"] = routes["route_name_eng"]
    routes["route_short_name"] = routes["route_id"]
    routes["route_id"] = "MTRB-" + routes["route_id"]
    gtfs_routes = routes[["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"]].drop_duplicates(subset=["route_id"]).copy()

    trips = []
    for key, route_matches in match_operator_routes_to_government_gtfs(db=db, operator_name="MTRB").items():
        route, bound, _ = key.split("-")
        info = routes[routes["route_short_name"] == route]
        if info.empty:
            continue
        name = info.iloc[0]["route_name_eng"]
        origin, destination = name.split(" to ")[:2] if isinstance(name, str) and " to " in name else ("", "")
        for match in route_matches:
            trips.append({
                "route_id": f"MTRB-{route}",
                "service_id": f"MTRB-{route}-{bound}-{match['gov_service_id']}",
                "trip_id": f"MTRB-{route}-{bound}-{match['gov_service_id']}",
                "direction_id": 0 if bound == "O" else 1,
                "route_short_name": route,
                "gov_route_id": match["gov_route_id"],
                "original_service_id": match["gov_service_id"],
                "origin_en": origin,
                "destination_en": destination,
                "agency_id": "MTRB",
            })
    if trips:
        trips = pd.DataFrame(trips)
    else:
        log.warning("MTR Bus route matching produced no trips")
        trips = pd.DataFrame(columns=["route_id", "service_id", "trip_id", "direction_id", "route_short_name",
                                      "original_service_id", "origin_en", "destination_en"])
    trips = trips.drop_duplicates(subset=["trip_id"], keep="first")

    patterns = db.read("SELECT * FROM mtrbus_stop_sequences")
    patterns["stop_id"] = "MTRB-" + patterns["stop_id"].astype(str)
    return Operator(gtfs_routes, trips, patterns, routes)
