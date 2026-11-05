import logging

import pandas as pd

from ..db import Store
from .matching import match_operator_routes_to_government_gtfs
from .tables import Operator

log = logging.getLogger(__name__)


def build_nlb(db: Store) -> Operator:
    routes = db.read("SELECT * FROM nlb_routes")
    routes[["orig_en", "dest_en"]] = routes["routeName_e"].str.split(" > ", expand=True)
    gtfs_routes = pd.DataFrame([{"route_id": f"NLB-{route}", "agency_id": "NLB", "route_short_name": route,
                                 "route_long_name": " - ".join(group["dest_en"].unique()), "route_type": 3}
                                for route, group in routes.groupby("routeNo")])

    matches = match_operator_routes_to_government_gtfs(db=db, operator_name="NLB")
    if not matches:
        log.warning("NLB route matching produced no trips")
    trips, seen = [], set()
    for key, route_matches in matches.items():
        parts = key.split("-")
        if len(parts) < 2:
            continue
        route = parts[0]
        for match in route_matches:
            bound = match.get("gov_bound")
            if bound not in ("O", "I"):
                continue
            route_id = match.get("routeid")
            if route_id is None:
                first = db.read('SELECT "routeId" FROM nlb_routes WHERE "routeNo" = ? ORDER BY "routeId" LIMIT 1', [route])
                if not first.empty:
                    route_id = first.iloc[0]["routeId"]
            trip_key = (route, route_id, match["gov_service_id"], bound)
            if trip_key in seen:
                continue
            seen.add(trip_key)
            trips.append({
                "route_id": f"NLB-{route}",
                "agency_id": "NLB",
                "service_id": f"NLB-{route}-{match['gov_service_id']}",
                "trip_id": f"NLB-{route}-{route_id}-{match['gov_service_id']}",
                "direction_id": 0 if bound == "O" else 1,
                "bound": bound,
                "route_short_name": route,
                "original_service_id": match["gov_service_id"],
                "route_long_name": match.get("gov_route_long_name", ""),
                "routeId": route_id,
                "origin_en": "",
                "destination_en": "",
                "gov_route_id": match["gov_route_id"],
            })
    trips = pd.DataFrame(trips) if trips else pd.DataFrame(columns=[
        "route_id", "service_id", "trip_id", "direction_id", "route_short_name", "original_service_id",
        "route_long_name", "routeId", "origin_en", "destination_en", "gov_route_id"])
    trips = trips.drop_duplicates(subset=["trip_id"], keep="first")

    patterns = db.read("SELECT * FROM nlb_stop_sequences")
    patterns["stop_id"] = "NLB-" + patterns["stopId"].astype(str)
    return Operator(gtfs_routes, trips, patterns, routes)
