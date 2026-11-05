import logging

import pandas as pd

from ..db import Store
from .matching import match_operator_routes_to_government_gtfs
from .tables import Operator

log = logging.getLogger(__name__)


def build_gmb(db: Store) -> Operator:
    routes = db.read("SELECT * FROM gmb_routes")
    patterns = db.read("SELECT * FROM gmb_stop_sequences")
    routes["agency_id"] = "GMB"
    routes["route_type"] = 3
    routes["route_id"] = "GMB-" + routes["region"] + "-" + routes["route_code"]
    routes["route_short_name"] = routes["region"] + "-" + routes["route_code"]
    routes["route_long_name"] = routes["region"] + " - " + routes["route_code"]
    gtfs_routes = routes[["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"]].drop_duplicates(subset=["route_id"])
    if "orig_en_primary" in routes.columns:
        ends = routes.drop_duplicates(subset=["route_id"]).set_index("route_id")[["orig_en_primary", "dest_en_primary"]]
        named = ends["orig_en_primary"].fillna("").str.strip().ne("") & ends["dest_en_primary"].fillna("").str.strip().ne("")
        names = (ends["orig_en_primary"].str.strip() + " - " + ends["dest_en_primary"].str.strip())[named]
        gtfs_routes["route_long_name"] = gtfs_routes["route_id"].map(names).fillna(gtfs_routes["route_long_name"])

    matches = match_operator_routes_to_government_gtfs(db=db, operator_name="GMB")
    trips = []
    for key, route_matches in matches.items():
        region, route_code, bound = key.split("-")[:3]
        seqs = patterns[(patterns["region"] == region) & (patterns["route_code"] == route_code)
                        & (patterns["route_seq"] == (1 if bound == "O" else 2))]["route_seq"].unique()
        for match in route_matches:
            for route_seq in seqs:
                trips.append({
                    "route_id": f"GMB-{region}-{route_code}",
                    "service_id": f"GMB-{region}-{route_code}-{match['gov_service_id']}",
                    "trip_id": f"GMB-{region}-{route_code}-{match['gov_service_id']}-{route_seq}",
                    "direction_id": int(route_seq) - 1,
                    "route_short_name": f"{region}-{route_code}",
                    "route_seq": int(route_seq),
                    "route_code": route_code,
                    "region": region,
                    "agency_id": "GMB",
                    "original_service_id": match["gov_service_id"],
                    "gov_route_id": match["gov_route_id"],
                })
    if trips:
        trips = pd.DataFrame(trips)
    else:
        log.warning("GMB route matching produced no trips, falling back to one default service per route")
        source = patterns[["route_code", "route_seq", "region"]].drop_duplicates().merge(
            routes[["route_code", "region", "route_long_name", "agency_id"]].drop_duplicates(), on=["route_code", "region"], how="left")
        source["origin_en"] = source["route_long_name"].str.split(" - ").str[0]
        source["destination_en"] = source["route_long_name"].str.split(" - ").str[1]
        source["route_seq"] = pd.to_numeric(source["route_seq"])
        source["route_id"] = "GMB-" + source["region"] + "-" + source["route_code"]
        source["direction_id"] = source["route_seq"] - 1
        source["service_id"] = "GMB_DEFAULT_SERVICE"
        source["trip_id"] = source["route_id"] + "-" + source["route_seq"].astype(str)
        trips = source[["route_id", "service_id", "trip_id", "direction_id", "route_seq", "route_code", "region", "origin_en", "destination_en", "agency_id"]].copy()
        trips["original_service_id"] = "GMB_DEFAULT_SERVICE"
        trips["route_short_name"] = trips["route_id"].str.replace("GMB-", "")
    trips = trips.drop_duplicates(subset=["trip_id"], keep="first")

    patterns["route_id"] = "GMB-" + patterns["region"] + "-" + patterns["route_code"]
    patterns["route_seq"] = pd.to_numeric(patterns["route_seq"])
    patterns = patterns.merge(trips[["route_id", "route_seq", "trip_id"]], on=["route_id", "route_seq"])
    patterns["stop_id"] = "GMB-" + patterns["stop_id"].astype(str)
    patterns = patterns.sort_values(["trip_id", "sequence", "stop_id"]).drop_duplicates(subset=["trip_id", "sequence"], keep="first")
    return Operator(gtfs_routes, trips, patterns, routes)
