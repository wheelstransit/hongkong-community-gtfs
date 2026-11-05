import pandas as pd
import os
import json
import shutil
import tempfile
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ProcessPoolExecutor

from pyproj import Transformer
from shapely import MultiLineString, Point
import logging

log = logging.getLogger(__name__)

def lat_long_dist(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    return c * r * 1000

def _process_shape_file(args):
    """Write one shape's rows to part_path; return its metadata and (start, end) coordinates for direction checks."""
    path, part_path = args
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return base, None, None
    if not data.get("features"):
        return base, None, None
    feature = data["features"][0]
    geometry = feature.get("geometry", {})
    coords = geometry.get("coordinates", [])
    endpoints = None
    if geometry.get("type") == "MultiLineString" and coords and coords[0]:
        try:
            endpoints = (coords[0][0][1], coords[0][0][0], coords[-1][-1][1], coords[-1][-1][0])
        except (IndexError, TypeError):
            pass
    try:
        _, bound = base.split("-")
    except ValueError:
        return base, None, endpoints
    points = [p for line in coords for p in line] if feature["geometry"]["type"] == "MultiLineString" else coords
    shape_id = f"CSDI-{base}"
    lines, dist, prev = [], 0, None
    for i, (lon, lat) in enumerate(points, start=1):
        if prev is not None:
            dist += lat_long_dist(prev[0], prev[1], lat, lon)
        lines.append(f"{shape_id},{lat},{lon},{i},{dist}\n")
        prev = (lat, lon)
    with open(part_path, "w") as part:
        part.writelines(lines)
    return base, {"shape_id": shape_id, "gov_route_id": feature["properties"].get("ROUTE_ID"), "bound": bound}, endpoints


def generate_shapes_from_csdi_files(output_path, db, waypoints_dir):
    """Write shapes.txt from hkbus/route-waypoints; returns shape metadata and per-file endpoints."""
    if not os.path.exists(waypoints_dir):
        return [], {}
    names = sorted(f for f in os.listdir(waypoints_dir) if f.endswith(".json") and f != "0versions.json")
    with tempfile.TemporaryDirectory(dir=os.path.dirname(output_path)) as parts:
        jobs = [(os.path.join(waypoints_dir, f), os.path.join(parts, f"{i:05}")) for i, f in enumerate(names)]
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(_process_shape_file, jobs, chunksize=32))
        with open(output_path, "wb") as out:
            out.write(b"shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n")
            for (_, part), (_, meta, _) in zip(jobs, results):
                if meta:
                    with open(part, "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 20)
    shape_info = pd.DataFrame([meta for _, meta, _ in results if meta])
    routes = db.read(f"SELECT route_id, agency_id FROM gov_gtfs_routes WHERE route_id IN ({','.join(shape_info['gov_route_id'].astype(str))})")
    routes["route_id"] = routes["route_id"].astype(str)
    shape_info["gov_route_id"] = shape_info["gov_route_id"].astype(str)
    shape_info = shape_info.merge(routes, left_on="gov_route_id", right_on="route_id", how="left").drop(columns=["route_id"])
    return shape_info.to_dict("records"), {base: ends for base, _, ends in results}


SHAPE_AGENCIES = {
    "KMB": ["KMB", "LWB", "KMB+CTB", "KWB"],
    "CTB": ["CTB", "NWFB", "KMB+CTB"],
    "GMB": ["GMB"],
    "MTRB": ["LRTFeeder"],
    "NLB": ["NLB"],
}


def match_trips_to_csdi_shapes(trips_df, shape_info, db, waypoints_dir, endpoints):
    if not shape_info:
        trips_df["shape_id"] = None
        return trips_df
    shapes = pd.DataFrame(shape_info)
    shapes["gov_route_id"] = pd.to_numeric(shapes["gov_route_id"], errors="coerce").astype("Int64").astype(str)
    trips = trips_df.copy()
    trips["bound"] = trips["direction_id"].apply(lambda x: "I" if x == 1 else "O")
    trips["gov_route_id"] = pd.to_numeric(trips["gov_route_id"], errors="coerce").astype("Int64").astype(str)
    matched = trips.merge(shapes[["gov_route_id", "bound", "shape_id", "agency_id"]], on=["gov_route_id", "bound"], how="left", suffixes=("", "_shape"))
    keep = matched.apply(lambda r: pd.isna(r["agency_id_shape"]) or r["agency_id_shape"] in SHAPE_AGENCIES.get(r["agency_id"], [r["agency_id"]]), axis=1)
    matched = matched[keep]
    if matched.empty:
        trips = trips_df.copy()
        trips["shape_id"] = None
        return trips
    return _validate_and_correct_shape_directions(matched, db, waypoints_dir, endpoints)


ROUTE_STOPS = {
    "KMB": """SELECT ST_X(kst.geometry) AS stop_lng, ST_Y(kst.geometry) AS stop_lat
              FROM kmb_stop_sequences ks
              JOIN kmb_routes kr ON ks.route = kr.route AND ks.bound = kr.bound AND ks.service_type = kr.service_type
              JOIN kmb_stops kst ON ks.stop = kst.stop
              WHERE ks.route = $route AND ks.bound = $bound ORDER BY ks.seq::int""",
    "CTB": """SELECT ST_X(cs.geometry) AS stop_lng, ST_Y(cs.geometry) AS stop_lat
              FROM citybus_stop_sequences css
              JOIN citybus_routes cr ON css.unique_route_id = cr.unique_route_id
              JOIN citybus_stops cs ON css.stop_id = cs.stop
              WHERE cr.route = $route AND cr.direction = $bound ORDER BY css.sequence::int""",
    "MTRB": """SELECT ST_X(ms.geometry) AS stop_lng, ST_Y(ms.geometry) AS stop_lat
               FROM mtrbus_stop_sequences mss
               JOIN mtrbus_routes mr ON mss.route_id = mr.route_id
               JOIN mtrbus_stops ms ON mss.stop_id = ms.stop_id
               WHERE mr.route_id = $route AND mss.direction = $bound ORDER BY mss.station_seqno::int""",
}


def _validate_and_correct_shape_directions(trips_df, db, waypoints_dir, endpoints):
    """Swap a trip's shape for the opposite-bound one when the route's first/last stops line up with it reversed."""
    if "agency_id" not in trips_df.columns:
        return trips_df
    candidates = trips_df[trips_df["shape_id"].notna()][["route_short_name", "direction_id", "shape_id", "agency_id"]].drop_duplicates()
    for route, direction_id, shape_id, agency_id in candidates.itertuples(index=False):
        bound, opposite = ("I", "O") if direction_id == 1 else ("O", "I")
        try:
            gov_route_id = shape_id.split("-")[1]
        except (AttributeError, IndexError):
            continue
        if not _is_inverted(route, bound, agency_id, db, endpoints.get(f"{gov_route_id}-{bound}")):
            continue
        if os.path.exists(os.path.join(waypoints_dir, f"{gov_route_id}-{opposite}.json")):
            mask = (trips_df["route_short_name"] == route) & (trips_df["direction_id"] == direction_id) & (trips_df["agency_id"] == agency_id)
            trips_df.loc[mask, "shape_id"] = f"CSDI-{gov_route_id}-{opposite}"
        else:
            log.warning("no opposite-direction shape for %s: %s-%s", route, gov_route_id, opposite)
    return trips_df


def _is_inverted(route, bound, agency_id, db, shape_ends) -> bool:
    if agency_id not in ROUTE_STOPS or shape_ends is None:
        return False
    params = {"route": route, "bound": ("outbound" if bound == "O" else "inbound") if agency_id == "CTB" else bound}
    stops = db.read(ROUTE_STOPS[agency_id], params)
    if len(stops) < 2:
        return False
    first, last = stops.iloc[0], stops.iloc[-1]
    start_lat, start_lng, end_lat, end_lng = shape_ends
    aligned = lat_long_dist(first["stop_lat"], first["stop_lng"], start_lat, start_lng) + lat_long_dist(last["stop_lat"], last["stop_lng"], end_lat, end_lng)
    inverted = lat_long_dist(first["stop_lat"], first["stop_lng"], end_lat, end_lng) + lat_long_dist(last["stop_lat"], last["stop_lng"], start_lat, start_lng)
    return inverted < aligned - 500


MAX_STOP_OFFSET_M = 100
_to_hk1980 = Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)


def _misfits(args):
    path, trips = args
    with open(path, encoding="utf-8") as f:
        geometry = json.load(f)["features"][0]["geometry"]
    lines = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
    shape = MultiLineString([[_to_hk1980.transform(lon, lat) for lon, lat in line] for line in lines if len(line) > 1])
    return [trip for trip, stops in trips.items()
            if any(shape.distance(Point(_to_hk1980.transform(lon, lat))) > MAX_STOP_OFFSET_M for lat, lon in stops)]


def drop_misfitting_shapes(trips_df, stop_times, stops, waypoints_dir):
    """Unset CSDI shapes that pass more than 100 m from one of the trip's stops, so pfaedle map-matches those trips."""
    coords = stops.set_index("stop_id")[["stop_lat", "stop_lon"]]
    with_shape = trips_df.loc[trips_df["shape_id"].notna(), ["trip_id", "shape_id"]]
    points = stop_times[stop_times["trip_id"].isin(with_shape["trip_id"])].join(coords, on="stop_id")
    by_trip = {t: list(zip(g["stop_lat"], g["stop_lon"])) for t, g in points.groupby("trip_id")}
    jobs = {}
    for trip, shape_id in with_shape.itertuples(index=False):
        jobs.setdefault(shape_id, {})[trip] = by_trip.get(trip, [])
    args = [(os.path.join(waypoints_dir, shape_id.removeprefix("CSDI-") + ".json"), t) for shape_id, t in jobs.items()]
    with ProcessPoolExecutor() as pool:
        bad = {trip for result in pool.map(_misfits, args, chunksize=16) for trip in result}
    trips_df.loc[trips_df["trip_id"].isin(bad), "shape_id"] = None
    log.info("%d trips had a shape too far from their stops; left for pfaedle", len(bad))
    return trips_df
