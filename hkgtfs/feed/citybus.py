import pandas as pd

from ..db import Store
from .matching import match_operator_routes_to_government_gtfs
from .tables import Operator

CIRCULAR_ROUTES = """
    SELECT DISTINCT gr.route_short_name AS route
    FROM gov_gtfs_routes gr JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
    WHERE gr.agency_id = 'CTB'
    GROUP BY gr.route_short_name
    HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
"""

# Routes the government feed runs in one direction only are circular: append the inbound stops
# that are not already on the outbound leg, starting at the first one missing from it.
MERGED_SEQUENCES = """
    WITH seqs AS (
        SELECT cr.route, cr.direction, cr.unique_route_id, css.stop_id, css.sequence
        FROM citybus_routes cr JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
    ),
    outbound AS (SELECT * FROM seqs WHERE route IN ({routes}) AND direction = 'outbound'),
    inbound AS (SELECT * FROM seqs WHERE route IN ({routes}) AND direction = 'inbound')
    SELECT route, 'outbound' AS direction, unique_route_id, stop_id, sequence FROM outbound
    UNION ALL
    SELECT i.route, 'outbound', i.unique_route_id, i.stop_id,
           (SELECT MAX(sequence) FROM outbound o WHERE o.route = i.route) + ROW_NUMBER() OVER (PARTITION BY i.route ORDER BY i.sequence)
    FROM inbound i
    WHERE i.sequence >= (SELECT COALESCE(MIN(i2.sequence), 1) FROM inbound i2
                         WHERE i2.route = i.route AND NOT EXISTS (SELECT 1 FROM outbound o WHERE o.route = i2.route AND o.stop_id = i2.stop_id))
    UNION ALL
    SELECT route, direction, unique_route_id, stop_id, sequence FROM seqs WHERE route NOT IN ({routes})
    ORDER BY route, direction, sequence
"""


def _trip(route, direction, service_id, info, gov_route_id) -> dict:
    return {
        "route_id": f"CTB-{route}",
        "service_id": f"CTB-{route}-{direction}-{service_id}",
        "trip_id": f"CTB-{route}-{direction}-{service_id}",
        "direction_id": 1 if direction == "inbound" else 0,
        "bound": "I" if direction == "inbound" else "O",
        "route_short_name": route,
        "route_long_name": f"{info.get('orig_en', '')} - {info.get('dest_en', '')}",
        "original_service_id": service_id,
        "unique_route_id": info["unique_route_id"],
        "origin_en": info.get("orig_en", ""),
        "destination_en": info.get("dest_en", ""),
        "gov_route_id": gov_route_id,
        "agency_id": "CTB",
    }


def build_citybus(db: Store, gov_trips: pd.DataFrame) -> Operator:
    coop = db.read("SELECT DISTINCT route_short_name FROM gov_gtfs_routes WHERE agency_id = 'KMB+CTB'")["route_short_name"].tolist()
    routes = db.read("SELECT * FROM citybus_routes")
    routes = routes[~routes["route"].isin(coop)]
    routes["route_id"] = "CTB-" + routes["route"]
    routes["agency_id"] = "CTB"
    routes["route_short_name"] = routes["route"]
    routes["route_long_name"] = routes["orig_en"] + " - " + routes["dest_en"]
    routes["route_type"] = 3
    routes["dir"] = routes["unique_route_id"].str.split("-").str[-1]

    gtfs_routes = []
    for route, group in routes.groupby("route"):
        out = group[group["dir"] == "outbound"]
        inb = group[group["dir"] == "inbound"]
        first_out = out.iloc[0] if not out.empty else group.iloc[0]
        first_in = inb.iloc[0] if not inb.empty else group.iloc[0]
        gtfs_routes.append({"route_id": f"CTB-{route}", "agency_id": "CTB", "route_short_name": route,
                            "route_long_name": f"{first_out['orig_en']} - {first_in['orig_en']}", "route_type": 3})

    matches = match_operator_routes_to_government_gtfs(db=db, operator_name="CTB")
    trips = []
    for route, gov_route_id in matches.items():
        candidates = routes[routes["route"] == route]
        for direction in ("inbound", "outbound"):
            info = candidates[candidates["dir"] == direction]
            if info.empty:
                continue
            for service_id in gov_trips[gov_trips["route_id"] == gov_route_id]["service_id"].unique():
                trips.append(_trip(route, direction, service_id, info.iloc[0], gov_route_id))
    trips = pd.DataFrame(trips) if trips else pd.DataFrame(columns=[
        "route_id", "service_id", "trip_id", "direction_id", "bound", "route_short_name", "route_long_name",
        "original_service_id", "unique_route_id", "origin_en", "destination_en", "gov_route_id", "agency_id"])
    trips = trips.drop_duplicates(subset=["trip_id"], keep="first")

    dummies = [{**_trip(r["route"], r["dir"], "NEVER", r, None), "is_dummy": True}
               for _, r in routes.iterrows() if r["route"] not in matches]
    if dummies:
        trips = pd.concat([trips, pd.DataFrame(dummies)], ignore_index=True)
    if "is_dummy" not in trips.columns:
        trips["is_dummy"] = False

    circular = db.read(CIRCULAR_ROUTES)["route"].tolist()
    if circular:
        patterns = db.read(MERGED_SEQUENCES.format(routes=", ".join(f"'{r}'" for r in circular)))
    else:
        patterns = db.read("SELECT * FROM citybus_stop_sequences")
    patterns["stop_id"] = "CTB-" + patterns["stop_id"].astype(str)
    return Operator(pd.DataFrame(gtfs_routes), trips, patterns, routes)
