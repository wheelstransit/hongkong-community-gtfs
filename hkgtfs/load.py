import geopandas as gpd
import pandas as pd

from .db import Store
from .mtr_bus_stops import unify_mtrbus_stops


def _points(df: pd.DataFrame, lon: str, lat: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df.drop(columns=[lon, lat]),
                            geometry=gpd.points_from_xy(df[lon].astype(float), df[lat].astype(float)), crs="EPSG:4326")


def load_all(db: Store, raw: dict):
    load_kmb(db, raw["kmb"])
    load_gmb(db, raw["gmb"])
    load_mtr_bus(db, raw["mtr_bus"])
    load_citybus(db, raw["citybus"])
    load_nlb(db, raw["nlb"])
    load_mtr(db, raw["mtr"])
    load_gov_gtfs(db, raw["gov_gtfs"])
    load_journey_time(db, raw["journey_time"])
    db.write("td_routes", raw["td_routes"])


def load_kmb(db: Store, raw: dict):
    routes = pd.DataFrame(raw["routes"])
    routes["unique_route_id"] = routes["route"] + "_" + routes["bound"] + "_" + routes["service_type"]
    db.write("kmb_routes", routes)
    db.write("kmb_stops", _points(pd.DataFrame(raw["stops"]), "long", "lat"))
    route_stops = pd.DataFrame(raw["route_stops"])
    route_stops["unique_route_id"] = route_stops["route"] + "_" + route_stops["bound"] + "_" + route_stops["service_type"]
    db.write("kmb_stop_sequences", route_stops)


def load_gmb(db: Store, raw: dict):
    routes = pd.DataFrame([{"region": region, "route_code": code, "unique_route_id": f"{region}_{code}"}
                           for region, codes in raw["routes"].items() for code in codes])
    dirs = pd.DataFrame(raw["route_directions"])
    if not dirs.empty:
        dirs = dirs[[c for c in ["region", "route_code", "route_id", "route_seq", "orig_en", "dest_en", "orig_tc", "dest_tc", "is_circular"]
                     if c in dirs.columns]].sort_values(["route_id", "route_seq"])
        rows = []
        for route_id, grp in dirs.groupby("route_id"):
            grp = grp.reset_index(drop=True)
            primary = grp.iloc[0]
            alt = grp.iloc[1] if len(grp) > 1 else None
            rows.append({
                "route_id": route_id,
                **{f"{k}_primary": primary.get(k, "") for k in ("orig_en", "dest_en", "orig_tc", "dest_tc")},
                **{f"{k}_alt": alt.get(k, "") if alt is not None else "" for k in ("orig_en", "dest_en", "orig_tc", "dest_tc")},
                "is_circular_any": bool(grp["is_circular"].any()),
            })
        agg = pd.DataFrame(rows).merge(dirs.drop_duplicates("route_id").set_index("route_id")[["region", "route_code"]],
                                       left_on="route_id", right_index=True, how="left")
        routes = routes.merge(agg.drop(columns="route_id"), on=["region", "route_code"], how="left")
    db.write("gmb_routes", routes)

    route_stops = pd.DataFrame(raw["route_stops"])
    names = route_stops[["stop_id", "stop_name_en", "stop_name_tc", "stop_name_sc"]].drop_duplicates("stop_id").set_index("stop_id")
    stops = pd.DataFrame(raw["stops"]).join(names, on="stop_id")
    stops["lat"] = stops["coordinates"].apply(lambda x: x["wgs84"]["latitude"] if x and "wgs84" in x else None)
    stops["long"] = stops["coordinates"].apply(lambda x: x["wgs84"]["longitude"] if x and "wgs84" in x else None)
    db.write("gmb_stops", _points(stops.drop(columns=["coordinates"]), "long", "lat"))

    normal = (route_stops["variant_description_en"].fillna("").str.lower().str.contains("normal")
              | route_stops["variant_description_tc"].fillna("").str.contains("正常")
              | route_stops["variant_description_sc"].fillna("").str.contains("正常"))
    has_normal = normal.groupby(route_stops["route_id"]).transform("any")
    normals, fallback = route_stops[normal & has_normal], route_stops[~has_normal]
    route_stops = (normals if not normals.empty else fallback if not fallback.empty else route_stops).copy()

    key = ["region", "route_code", "route_seq"]
    best = (route_stops.groupby(key + ["route_id"], as_index=False).agg(stop_count=("sequence", "count"))
            .sort_values(key + ["stop_count"], ascending=[True, True, True, False]).drop_duplicates(subset=key))
    route_stops = route_stops.merge(best[key + ["route_id"]], on=key + ["route_id"])
    route_stops = (route_stops.sort_values(["route_id", "route_seq", "sequence", "stop_id"])
                   .drop_duplicates(subset=["route_id", "route_seq", "sequence", "stop_id"])
                   .drop_duplicates(subset=["route_id", "route_seq", "sequence"]))
    db.write("gmb_stop_sequences", route_stops[["route_id", "route_seq", "region", "route_code", "stop_id", "sequence",
                                                "stop_name_en", "stop_name_tc", "stop_name_sc"]])


def load_mtr_bus(db: Store, raw: dict):
    routes = pd.DataFrame(raw["routes"])
    routes.columns = routes.columns.str.lower()
    routes["unique_route_id"] = routes["route_id"]
    db.write("mtrbus_routes", routes)

    raw_stops = [{**s, "stop_id": str(s.get("stop_id", ""))} for s in raw["stops"]]
    unified, orig_to_unified, _ = unify_mtrbus_stops(raw_stops)
    db.write("mtrbus_stops", _points(pd.DataFrame(unified), "long", "lat"))

    route_stops = pd.DataFrame(raw["route_stops"])
    route_stops.columns = route_stops.columns.str.lower()
    route_stops = route_stops.rename(columns={"station_id": "stop_id"})
    route_stops["stop_id"] = route_stops["stop_id"].astype(str).map(orig_to_unified).fillna(route_stops["stop_id"])
    db.write("mtrbus_stop_sequences", route_stops)


def load_citybus(db: Store, raw: dict):
    info = pd.DataFrame(raw["routes"])
    circular = {r["route"] for _, r in info.iterrows()
                if any("circular" in str(r.get(c, "")).lower() for c in ("dest_en", "dest_tc", "dest_sc"))}

    regular, by_route = [], {}
    for seq in raw["route_sequences"]:
        if seq["route_id"] in circular:
            by_route.setdefault(seq["route_id"], {})[seq["direction"]] = seq
        else:
            regular.append(seq)
    merged = []
    for route_id, dirs in by_route.items():
        if "inbound" in dirs and "outbound" in dirs:
            out, inb = dirs["outbound"]["stop_ids"], dirs["inbound"]["stop_ids"]
            overlap = 0
            for i in range(1, min(len(inb), len(out)) + 1):
                if out[-i] != inb[i - 1]:
                    break
                overlap = i
            merged.append({"route_id": route_id, "direction": "outbound", "stop_ids": out + inb[overlap:]})
        else:
            merged.append({**(dirs.get("outbound") or dirs["inbound"]), "direction": "outbound"})
    sequences = regular + merged

    seq_df = pd.DataFrame([{"route_id": s["route_id"], "direction": s["direction"], "stop_id": stop_id, "sequence": i}
                           for s in sequences for i, stop_id in enumerate(s["stop_ids"], 1)])
    seq_df["unique_route_id"] = seq_df["route_id"] + "-" + seq_df["direction"]
    db.write("citybus_stop_sequences", seq_df)

    routes = info.merge(pd.DataFrame([{"route_id": s["route_id"], "direction": s["direction"]} for s in sequences]).drop_duplicates(),
                        left_on="route", right_on="route_id")
    for route_id in circular:
        mask = routes["route"] == route_id
        if mask.any():
            for col in ("dest_en", "dest_tc", "dest_sc"):
                if col in routes.columns:
                    dest = str(routes.loc[mask, col].iloc[0])
                    if "circular" not in dest.lower() and dest != "nan":
                        routes.loc[mask, col] = dest + " (Circular)"
    inbound = (routes["direction"] == "inbound") & ~routes["route"].isin(circular)
    orig, dest = ["orig_en", "orig_tc", "orig_sc"], ["dest_en", "dest_tc", "dest_sc"]
    if inbound.any():
        routes.loc[inbound, orig + dest] = routes.loc[inbound, dest + orig].values
    routes["unique_route_id"] = routes["route"] + "-" + routes["direction"]
    db.write("citybus_routes", routes)
    db.write("citybus_stops", _points(pd.DataFrame(raw["stop_details"]), "long", "lat"))


def load_nlb(db: Store, raw: dict):
    routes = pd.DataFrame(raw["routes"])
    routes["unique_route_id"] = routes["routeId"]
    db.write("nlb_routes", routes)
    db.write("nlb_stops", _points(pd.DataFrame(raw["stops"]), "longitude", "latitude"))
    db.write("nlb_stop_sequences", pd.DataFrame(raw["route_stops"]))


def load_mtr(db: Store, raw: dict):
    stations = pd.DataFrame(raw["lines_and_stations"]).dropna(subset=["longitude", "latitude"])
    tkl = stations[stations["Line Code"] == "TKL"]
    if not tkl.empty and not (tkl["Direction"] == "LHP-DT").any():
        rows = {r["Station Code"]: r for _, r in tkl.iterrows()}
        lhp_dt = ["LHP", "TKO", "TIK", "YAT", "QUB", "NOP"]
        synthetic = []
        for direction, codes in (("LHP-DT", lhp_dt), ("LHP-UT", lhp_dt[::-1])):
            for i, code in enumerate(codes, start=1):
                if code not in rows:
                    break
                synthetic.append({**rows[code], "Direction": direction, "Sequence": f"{i}.00"})
        if synthetic:
            stations = pd.concat([stations, pd.DataFrame(synthetic)], ignore_index=True)
    db.write("mtr_lines_and_stations", _points(stations, "longitude", "latitude"))
    db.write("mtr_lines_fares", pd.DataFrame(raw["lines_fares"]))
    db.write("light_rail_routes_and_stops", pd.DataFrame(raw["light_rail_routes_and_stops"]))
    db.write("light_rail_fares", pd.DataFrame(raw["light_rail_fares"]))
    db.write("mtr_exits", _points(pd.DataFrame(raw["exits"]), "lon", "lat"))
    db.write("light_rail_stops", _points(pd.DataFrame(raw["light_rail_stops"].values()), "lon", "lat"))


def load_gov_gtfs(db: Store, raw: dict):
    for name, df in raw.items():
        if name == "stops":
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.stop_lon, df.stop_lat), crs="EPSG:4326")
        db.write(f"gov_gtfs_{name}", df)


def load_journey_time(db: Store, raw: dict):
    db.write("journey_time_data", pd.DataFrame(
        [{"from_stop_id": a, "to_stop_id": b, "travel_time_seconds": t}
         for a, dests in raw.items() if isinstance(dests, dict) for b, t in dests.items()]))
