"""Rename pfaedle's generated shp_N_M shape ids to the id of the first trip using them."""
import csv
import re
import sys
from pathlib import Path


def rewrite(path: Path, mapping: dict):
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["shape_id"] = mapping.get(row["shape_id"], row["shape_id"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


folder = Path(sys.argv[1])
trips, shapes = folder / "trips.txt", folder / "shapes.txt"
if trips.exists() and shapes.exists():
    mapping = {}
    with trips.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if re.match(r"^shp_\d+_\d+$", row.get("shape_id", "").strip()):
                mapping.setdefault(row["shape_id"].strip(), row["trip_id"].strip())
    if mapping:
        rewrite(trips, mapping)
        rewrite(shapes, mapping)
