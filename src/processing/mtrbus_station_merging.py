import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict, List, Any
from geopy.distance import geodesic

def unify_mtrbus_stops(
    raw_stops: List[dict],
    stop_id_key: str = "stop_id",
    lat_key: str = "lat",
    lon_key: str = "long",
    name_key: str = "name_en",
    precision: int = 7,
    distance_threshold_m: float = 5.0
) -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:
    #Unify MTR Bus stops that have identical coordinates, or have the same name and are within 5 meters.

    coord_to_stopids = defaultdict(list)
    coord_to_stopdata = defaultdict(list)
    stopid_to_coord = {}

    for stop in raw_stops:
        try:
            lat = round(float(stop[lat_key]), precision)
            lon = round(float(stop[lon_key]), precision)
        except (KeyError, ValueError, TypeError):
            continue

        coord = (lat, lon)
        coord_to_stopids[coord].append(stop[stop_id_key])
        coord_to_stopdata[coord].append(stop)
        stopid_to_coord[stop[stop_id_key]] = (float(stop[lat_key]), float(stop[lon_key]))

    unified_stops = []
    orig_to_unified = {}
    unified_to_orig = {}
    coord_representative_stopid = {}

    for coord, stopids in coord_to_stopids.items():
        rep_stop = coord_to_stopdata[coord][0].copy()
        unified_stop_id = stopids[0]
        rep_stop[stop_id_key] = unified_stop_id
        rep_stop[lat_key] = coord[0]
        rep_stop[lon_key] = coord[1]
        unified_stops.append(rep_stop)
        for orig_id in stopids:
            orig_to_unified[orig_id] = unified_stop_id
        unified_to_orig[unified_stop_id] = stopids
        coord_representative_stopid[coord] = unified_stop_id

    stopid_to_stop = {stop[stop_id_key]: stop for stop in unified_stops}
    stopid_to_group = {stop[stop_id_key]: stop[stop_id_key] for stop in unified_stops}

    name_to_stopids = defaultdict(list)
    for stop in unified_stops:
        name = stop.get(name_key)
        if name:
            name_to_stopids[name].append(stop[stop_id_key])

    merged = set()
    for name, stopids in name_to_stopids.items():
        n = len(stopids)
        for i in range(n):
            id1 = stopids[i]
            if id1 in merged:
                continue
            coord1 = (
                float(stopid_to_stop[id1][lat_key]),
                float(stopid_to_stop[id1][lon_key])
            )
            group_leader = id1
            group_members = [id1]
            for j in range(i + 1, n):
                id2 = stopids[j]
                if id2 in merged:
                    continue
                coord2 = (
                    float(stopid_to_stop[id2][lat_key]),
                    float(stopid_to_stop[id2][lon_key])
                )
                dist = geodesic(coord1, coord2).meters
                if dist <= distance_threshold_m:
                    merged.add(id2)
                    group_members.append(id2)
                    stopid_to_group[id2] = group_leader
            if len(group_members) > 1:
                for orig_id in group_members:
                    orig_to_unified[orig_id] = group_leader
                if group_leader in unified_to_orig:
                    unified_to_orig[group_leader].extend([mid for mid in group_members if mid not in unified_to_orig[group_leader]])
                else:
                    unified_to_orig[group_leader] = group_members

    final_group_leaders = set(orig_to_unified.values())
    final_unified_stops = []
    seen = set()
    for stop in unified_stops:
        leader = orig_to_unified[stop[stop_id_key]]
        if leader not in seen:
            rep_stop = stopid_to_stop[leader].copy()
            final_unified_stops.append(rep_stop)
            seen.add(leader)

    for stop in raw_stops:
        orig_id = stop[stop_id_key]
        coord = (round(float(stop[lat_key]), precision), round(float(stop[lon_key]), precision))
        coord_leader = coord_representative_stopid.get(coord, orig_id)
        group_leader = orig_to_unified.get(coord_leader, coord_leader)
        orig_to_unified[orig_id] = group_leader
        if group_leader in unified_to_orig:
            if orig_id not in unified_to_orig[group_leader]:
                unified_to_orig[group_leader].append(orig_id)
        else:
            unified_to_orig[group_leader] = [orig_id]

    return final_unified_stops, orig_to_unified, unified_to_orig
