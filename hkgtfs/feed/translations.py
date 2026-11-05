import pandas as pd

from .stops import Stops
from .tables import Operator

LANG = "zh-Hant"


def _row(table, field, record_id, text):
    return {"table_name": table.removesuffix(".txt"), "field_name": field, "language": LANG, "record_id": record_id, "translation": text}


def _stop_names(stops: Stops) -> list:
    rows = []
    kmb = stops.operator["KMB"]
    kmb_tc = (kmb["name_tc"].str.replace(r"\s*\([A-Za-z0-9]{5}\)", "", regex=True)
              .str.replace(r"\s*-\s*", " - ", regex=True)
              .str.replace(r"([^\s])(\([A-Za-z0-9]+\))", r"\1 \2", regex=True))
    rows += [_row("stops.txt", "stop_name", s, n) for s, n in zip(kmb["stop_id"], kmb_tc) if pd.notna(n) and str(n).strip()]
    for agency, col in (("CTB", "name_tc"), ("GMB", "stop_name_tc"), ("NLB", "stopName_c")):
        rows += [_row("stops.txt", "stop_name", r.stop_id, getattr(r, col)) for r in stops.operator[agency][["stop_id", col]].dropna().itertuples()]
    stations = (stops.mtr_stations_raw[["Station Code", "Chinese Name"]].rename(columns={"Station Code": "code", "Chinese Name": "name"})
                .dropna(subset=["code", "name"]).drop_duplicates(subset=["code"]))
    rows += [_row("stops.txt", "stop_name", f"MTR-{r.code}", str(r.name).strip()) for r in stations.itertuples(index=False) if str(r.name).strip()]
    rows += [_row("stops.txt", "stop_name", r.stop_id, f"{r.station_name_zh} 出口 {r.exit}")
             for r in stops.mtr_exits[["stop_id", "station_name_zh", "exit"]].dropna(subset=["station_name_zh"]).itertuples()]
    rows += [_row("stops.txt", "stop_name", r.stop_id, str(r.name_zh).strip())
             for r in stops.operator["MTRB"][["stop_id", "name_zh"]].dropna(subset=["name_zh"]).itertuples() if str(r.name_zh).strip()]
    rows += [_row("stops.txt", "stop_name", r.stop_id, r.name_tc) for r in stops.operator["LR"][["stop_id", "name_tc"]].dropna().itertuples()]
    return rows


def _route_names(kmb: Operator, ctb: Operator, nlb: Operator, mtr_routes: pd.DataFrame, mtr_stations: pd.DataFrame, lr_routes: pd.DataFrame) -> list:
    rows = []
    for route, grp in kmb.source_routes.groupby("route"):
        out, inb = grp[grp["bound"] == "O"], grp[grp["bound"] == "I"]
        first_out = out.iloc[0] if not out.empty else grp.iloc[0]
        if not inb.empty:
            name = f"{first_out.get('orig_tc', '')} - {inb.iloc[0].get('orig_tc', '')}"
        else:
            name = f"{first_out.get('orig_tc', '')} - {first_out.get('dest_tc', '')}"
        rows.append(_row("routes.txt", "route_long_name", f"KMB-{route}", name))
    for route, grp in ctb.source_routes.groupby("route"):
        out, inb = grp[grp["dir"] == "outbound"], grp[grp["dir"] == "inbound"]
        first_out = out.iloc[0] if not out.empty else grp.iloc[0]
        first_in = inb.iloc[0] if not inb.empty else first_out
        rows.append(_row("routes.txt", "route_long_name", f"CTB-{route}", f"{first_out.get('orig_tc', '')} - {first_in.get('orig_tc', '')}"))
    rows += [_row("routes.txt", "route_long_name", f"NLB-{r.routeNo}", r.routeName_c.replace(" > ", " - "))
             for r in nlb.source_routes[["routeNo", "routeName_c"]].dropna().itertuples()]
    try:
        names = {}
        for line, grp in mtr_stations.groupby("Line Code"):
            ordered = grp.sort_values("Sequence", key=lambda s: s.astype(int) if s.dtype == object else s)
            if not ordered.empty:
                names[line] = f"{ordered.iloc[0]['Chinese Name']} - {ordered.iloc[-1]['Chinese Name'] if len(ordered) > 1 else ordered.iloc[0]['Chinese Name']}"
        rows += [_row("routes.txt", "route_long_name", r.route_id, names.get(r.route_short_name, r.route_long_name)) for r in mtr_routes.itertuples()]
    except Exception:
        pass
    if "route_long_name_tc" in lr_routes.columns:
        rows += [_row("routes.txt", "route_long_name", r.route_id, str(r.route_long_name_tc).strip())
                 for r in lr_routes[["route_id", "route_long_name_tc"]].dropna().itertuples(index=False) if str(r.route_long_name_tc).strip()]
    return rows


def build_translations(stops: Stops, kmb: Operator, ctb: Operator, nlb: Operator, mtr_routes: pd.DataFrame,
                       lr_routes: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    rows = _stop_names(stops) + _route_names(kmb, ctb, nlb, mtr_routes, stops.mtr_stations_raw, lr_routes)
    tc = trips[["trip_id", "trip_headsign_tc"]].dropna(subset=["trip_id"])
    tc = tc[tc["trip_headsign_tc"].astype(str).str.strip() != ""]
    rows += [_row("trips.txt", "trip_headsign", r.trip_id, r.trip_headsign_tc) for r in tc.itertuples(index=False)]
    translations = pd.DataFrame(rows).drop_duplicates(subset=["table_name", "field_name", "language", "record_id"])
    translations["translation"] = translations["translation"].astype(str).str.strip()
    return translations
