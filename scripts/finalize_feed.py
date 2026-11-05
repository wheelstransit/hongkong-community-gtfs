"""Clean a pfaedle output folder: readable shape ids, no repeated shape points or stop distances, no dangling translations."""
import csv
import re
import sys
from pathlib import Path

RECORD_FILES = {"stops": ("stops.txt", "stop_id"), "routes": ("routes.txt", "route_id"), "trips": ("trips.txt", "trip_id"),
                "agency": ("agency.txt", "agency_id"), "pathways": ("pathways.txt", "pathway_id")}


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rename_shapes(folder: Path):
    """pfaedle names new shapes shp_N_M; name them after the first trip using them."""
    fields, trips = read(folder / "trips.txt")
    mapping = {}
    for trip in trips:
        if re.match(r"^shp_\d+_\d+$", trip.get("shape_id", "").strip()):
            mapping.setdefault(trip["shape_id"].strip(), trip["trip_id"].strip())
    if not mapping:
        return
    for row in trips:
        row["shape_id"] = mapping.get(row["shape_id"], row["shape_id"])
    write(folder / "trips.txt", fields, trips)
    fields, shapes = read(folder / "shapes.txt")
    for row in shapes:
        row["shape_id"] = mapping.get(row["shape_id"], row["shape_id"])
    write(folder / "shapes.txt", fields, shapes)


def blank_repeated_distances(folder: Path):
    """Consecutive stops at the same point on a shape get equal distances, which GTFS forbids; leave the later one blank."""
    fields, rows = read(folder / "stop_times.txt")
    if "shape_dist_traveled" not in fields:
        return
    by_trip = {}
    for row in rows:
        by_trip.setdefault(row["trip_id"], []).append(row)
    for trip_rows in by_trip.values():
        prev = None
        for row in sorted(trip_rows, key=lambda r: int(r["stop_sequence"])):
            if row["shape_dist_traveled"] == "":
                continue
            dist = float(row["shape_dist_traveled"])
            if prev is not None and dist <= prev:
                row["shape_dist_traveled"] = ""
            else:
                prev = dist
    write(folder / "stop_times.txt", fields, rows)


def drop_repeated_shape_points(folder: Path):
    fields, rows = read(folder / "shapes.txt")
    by_shape = {}
    for row in rows:
        by_shape.setdefault(row["shape_id"], []).append(row)
    kept = []
    for points in by_shape.values():
        prev = None
        for row in sorted(points, key=lambda r: int(r["shape_pt_sequence"])):
            point = (row["shape_pt_lat"], row["shape_pt_lon"])
            if point != prev:
                kept.append(row)
            prev = point
    write(folder / "shapes.txt", fields, kept)


def drop_dangling_translations(folder: Path):
    if not (folder / "translations.txt").exists():
        return
    ids = {}
    for table, (name, key) in RECORD_FILES.items():
        if (folder / name).exists():
            ids[table] = {row[key] for row in read(folder / name)[1]}
    fields, rows = read(folder / "translations.txt")
    write(folder / "translations.txt", fields,
          [r for r in rows if not r.get("record_id") or r["record_id"] in ids.get(r["table_name"], ())])


folder = Path(sys.argv[1])
if (folder / "shapes.txt").exists():
    rename_shapes(folder)
    drop_repeated_shape_points(folder)
blank_repeated_distances(folder)
drop_dangling_translations(folder)
