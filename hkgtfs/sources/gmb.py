import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ..http import session

BASE_URL = "https://data.etagmb.gov.hk"
log = logging.getLogger(__name__)
_pause_until = 0.0
_pause_lock = threading.Lock()


def _get(path: str):
    global _pause_until
    for attempt in range(12):
        time.sleep(max(0.0, _pause_until - time.time()))
        response = session.get(BASE_URL + path, timeout=30)
        if response.status_code == 200:
            return response.json()["data"]
        if response.status_code == 404:
            return None
        if response.status_code not in (403, 429) and response.status_code < 500:
            response.raise_for_status()
        with _pause_lock:
            wait = min(30 * 2 ** attempt, 300)
            if _pause_until < time.time() + wait:
                log.info("gmb: HTTP %s, backing off %ss", response.status_code, wait)
                _pause_until = time.time() + wait
    raise RuntimeError(f"gmb: giving up on {path}")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _refresh(store: dict, kind: str, keys, updated: dict, path, desc: str):
    entries = store.setdefault(kind, {})
    stale = [k for k in keys if k not in entries or (k in updated and _parse(updated[k]) > _parse(entries[k]["fetched"]))]
    if not stale:
        return
    now = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(4) as pool:
        for key, data in tqdm(zip(stale, pool.map(lambda k: _get(path(k)), stale)), total=len(stale), desc=desc, leave=False):
            entries[key] = {"fetched": now, "data": data}


def fetch(store_path: Path):
    store = json.loads(store_path.read_text()) if store_path.exists() else {}
    regions = _get("/route")["routes"]
    route_updated = {str(r["route_id"]): r["last_update_date"] for r in _get("/last-update/route")}
    route_stop_updated = {f"{r['route_id']}/{r['route_seq']}": r["last_update_date"] for r in _get("/last-update/route-stop")}
    stop_updated = {str(r["stop_id"]): r["last_update_date"] for r in _get("/last-update/stop")}

    pairs = [f"{region}/{code}" for region, codes in regions.items() for code in codes]
    cached_routes = store.get("routes", {})
    pair_updated = {}
    known_ids = set()
    for pair, entry in cached_routes.items():
        ids = [str(v["route_id"]) for v in entry["data"] or []]
        known_ids.update(ids)
        dates = [route_updated[i] for i in ids if i in route_updated]
        if dates:
            pair_updated[pair] = max(dates, key=_parse)
        if any(i not in route_updated for i in ids):
            pair_updated[pair] = datetime.now(timezone.utc).isoformat()
    for route_id in set(route_updated) - known_ids:
        variant = (_get(f"/route/{route_id}") or [{}])[0]
        if variant.get("region"):
            pair_updated[f"{variant['region']}/{variant['route_code']}"] = datetime.now(timezone.utc).isoformat()
    _refresh(store, "routes", pairs, pair_updated, lambda p: f"/route/{p}", "gmb routes")

    route_stops, route_directions, stop_ids = [], [], set()
    needed = []
    for pair in pairs:
        region, route_code = pair.split("/", 1)
        for variant, direction in _selected_directions(store["routes"][pair]["data"]):
            needed.append((region, route_code, variant, direction))
    _refresh(store, "route_stops", [f"{v['route_id']}/{d['route_seq']}" for _, _, v, d in needed],
             route_stop_updated, lambda k: f"/route-stop/{k}", "gmb route-stops")

    duplicates = 0
    for region, route_code, variant, direction in needed:
        route_id, route_seq = variant["route_id"], direction["route_seq"]
        desc = {lang: (variant.get(f"description_{lang}") or "").strip() for lang in ("en", "tc", "sc")}
        names = {f"{k}_{lang}": (direction.get(f"{k}_{lang}") or "").strip() for k in ("orig", "dest") for lang in ("en", "tc", "sc")}
        circular = any(t and any(tok in t.lower() for tok in ("circular", "循環", "环线", "循環線", "循環線)"))
                       for t in (names["dest_en"].lower(), names["dest_tc"], names["dest_sc"]))
        route_directions.append({"route_id": route_id, "route_seq": route_seq, "region": region, "route_code": route_code,
                                 **names, "is_circular": circular, "variant_description_en": desc["en"]})
        data = store["route_stops"][f"{route_id}/{route_seq}"]["data"]
        if not data or "route_stops" not in data:
            continue
        seen, chosen = set(), {}
        for stop in data["route_stops"]:
            stop_id, stop_seq = stop.get("stop_id"), stop.get("stop_seq")
            if not stop_id or stop_seq is None:
                continue
            full_key, seq_key = (route_id, route_seq, stop_seq, stop_id), (route_id, route_seq, stop_seq)
            if full_key in seen or (seq_key in chosen and chosen[seq_key] != stop_id):
                duplicates += 1
                continue
            seen.add(full_key)
            chosen.setdefault(seq_key, stop_id)
            stop_ids.add(stop_id)
            route_stops.append({
                "route_id": route_id, "route_seq": route_seq, "region": region, "route_code": route_code,
                "stop_id": stop_id, "sequence": stop_seq,
                "stop_name_en": stop.get("name_en"), "stop_name_tc": stop.get("name_tc"), "stop_name_sc": stop.get("name_sc"),
                "variant_description_en": desc["en"], "variant_description_tc": desc["tc"], "variant_description_sc": desc["sc"],
            })

    stop_keys = sorted(str(s) for s in stop_ids)
    _refresh(store, "stops", stop_keys, stop_updated, lambda s: f"/stop/{s}", "gmb stops")
    store_path.write_text(json.dumps(store, ensure_ascii=False))

    stops = []
    for stop_id in sorted(stop_ids):
        data = store["stops"][str(stop_id)]["data"]
        if data:
            stops.append({**data, "stop_id": stop_id})
    return {"routes": regions, "stops": stops, "route_stops": route_stops, "route_directions": route_directions}


def _selected_directions(variants):
    variants = [v for v in variants or [] if v.get("route_id")]
    normal = [v for v in variants if "normal" in (v.get("description_en") or "").strip().lower()
              or "正常" in (v.get("description_tc") or "") or "正常" in (v.get("description_sc") or "")]
    for variant in normal or variants:
        for direction in variant.get("directions", []):
            if direction.get("route_seq"):
                yield variant, direction
