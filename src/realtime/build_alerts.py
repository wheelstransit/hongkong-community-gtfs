from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List
from datetime import datetime
from dateutil import parser as dateparser

from google.transit import gtfs_realtime_pb2

from .alert_models import Event, EventType


EVENTS_DIR = os.environ.get("RT_EVENTS_DIR", os.path.join("data", "rt", "events"))
OUT_DIR = os.environ.get("RT_OUT_DIR", os.path.join("data", "rt", "build"))
OUT_FILE = os.path.join(OUT_DIR, "alerts.pb")


CAUSE_MAP = {name: gtfs_realtime_pb2.Alert.Cause.Value(name) for name in [
    "UNKNOWN_CAUSE",
    "OTHER_CAUSE",
    "TECHNICAL_PROBLEM",
    "STRIKE",
    "DEMONSTRATION",
    "ACCIDENT",
    "HOLIDAY",
    "WEATHER",
    "MAINTENANCE",
    "CONSTRUCTION",
    "POLICE_ACTIVITY",
    "MEDICAL_EMERGENCY",
]}

DEFAULT_EFFECT = {
    EventType.stop_moved: "STOP_MOVED",
    EventType.stop_closed: "NO_SERVICE",
    EventType.detour: "DETOUR",
    EventType.modified_service: "MODIFIED_SERVICE",
    EventType.network_notice: "UNKNOWN_EFFECT",
    EventType.accessibility_notice: "ACCESSIBILITY_ISSUE" if hasattr(gtfs_realtime_pb2.Alert.Effect, "ACCESSIBILITY_ISSUE") else "OTHER_EFFECT",
}


def effect_value(name: str) -> int:
    enum = gtfs_realtime_pb2.Alert.Effect
    try:
        return enum.Value(name)
    except ValueError:
        return enum.Value("OTHER_EFFECT")


def to_unix(ts: str | None) -> int | None:
    if not ts:
        return None
    dt = dateparser.parse(ts)
    return int(dt.timestamp())


def add_translations(msg_field, translations: Dict[str, str]):
    for lang, text in (translations or {}).items():
        tr = msg_field.translation.add()
        tr.text = text
        if lang:
            tr.language = lang


def load_events() -> List[Event]:
    events: List[Event] = []
    if not os.path.isdir(EVENTS_DIR):
        return events
    # Walk recursively so editors can group files in subfolders.
    json_paths: List[str] = []
    for root, _dirs, files in os.walk(EVENTS_DIR):
        for name in files:
            if name.lower().endswith(".json"):
                json_paths.append(os.path.join(root, name))
    for path in sorted(json_paths):
        # Skip known non-event JSON files
        base = os.path.basename(path).lower()
        if base == "schema.json":
            continue
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Heuristic: skip JSON Schema-like files
        if isinstance(data, dict) and ("$schema" in data or "$defs" in data) and "id" not in data:
            continue
        if isinstance(data, dict):
            data = [data]
        for item in data:
            events.append(Event.model_validate(item))
    # Deterministic order: start asc, then id
    def sort_key(e: Event):
        start = to_unix(e.start) or 0
        return (start, e.id)

    events.sort(key=sort_key)
    return events


def build_feed(events: List[Event]) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.incrementality = gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    feed.header.timestamp = int(time.time())

    for ev in events:
        entity = feed.entity.add()
        entity.id = ev.id
        alert = entity.alert

        # Active window
        ap = alert.active_period.add()
        if ev.start:
            ap.start = to_unix(ev.start) or 0
        if ev.end:
            ap.end = to_unix(ev.end) or 0

        # Cause/effect
        if ev.cause:
            alert.cause = CAUSE_MAP.get(ev.cause, gtfs_realtime_pb2.Alert.Cause.Value("OTHER_CAUSE"))
        eff_name = ev.effect or DEFAULT_EFFECT.get(ev.type, "OTHER_EFFECT")
        alert.effect = effect_value(eff_name)

        # Texts
        add_translations(alert.header_text, ev.header)
        add_translations(alert.description_text, ev.description)
        if ev.url:
            add_translations(alert.url, ev.url)

        # Scope
        if not (ev.route_ids or ev.stop_ids or ev.trip_ids):
            alert.informed_entity.add()  # network-wide
        else:
            for rid in ev.route_ids:
                sel = alert.informed_entity.add()
                sel.route_id = rid
            for sid in ev.stop_ids:
                sel = alert.informed_entity.add()
                sel.stop_id = sid
            for tid in ev.trip_ids:
                sel = alert.informed_entity.add()
                sel.trip.trip_id = tid

        # Helpful hint for moved/closed stops
        if ev.replacement_stop_id and ev.type in {EventType.stop_moved, EventType.stop_closed}:
            # If no english text provided, inject a minimal hint
            if not ev.description.get("en"):
                add_translations(alert.description_text, {"en": f"Use replacement stop {ev.replacement_stop_id}."})

    return feed


def main() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    events = load_events()
    feed = build_feed(events)
    with open(OUT_FILE, "wb") as fh:
        fh.write(feed.SerializeToString())
    print(f"Wrote {OUT_FILE} with {len(feed.entity)} alert(s).")
    return OUT_FILE


if __name__ == "__main__":
    main()
