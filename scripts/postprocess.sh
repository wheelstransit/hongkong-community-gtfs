#!/usr/bin/env bash
# Tidy output/hk.gtfs.zip with gtfstidy, map-match shapes with pfaedle and package the published feeds.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cache="$root/.cache"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$cache" "$work/osm" "$work/gtfs" "$work/original" "$work/standard" "$work/drop-shapes"

unzip -q "$root/output/hk.gtfs.zip" -d "$work/original"
gtfstidy -OscRCSeD "$root/output/hk.gtfs.zip" -o "$work/gtfs/input.gtfs.zip"
unzip -q "$work/gtfs/input.gtfs.zip" -d "$work/tidy"

pbf="$cache/hong-kong-latest.osm.pbf"
[ -f "$pbf" ] || curl -sSfL -o "$pbf" https://download.geofabrik.de/asia/china/hong-kong-latest.osm.pbf
osmium cat "$pbf" -o "$work/osm/hk.osm.bz2"

for variant in standard drop-shapes; do
    flags=""; [ "$variant" = drop-shapes ] && flags="-D"
    docker run -i --rm --user "$(id -u):$(id -g)" \
        -v "$work/osm:/osm" -v "$work/gtfs:/gtfs" -v "$work/$variant:/gtfs-out" \
        ghcr.io/ad-freiburg/pfaedle:latest $flags -x /osm/hk.osm.bz2 -i /gtfs/input.gtfs.zip
    python3 "$root/scripts/rename_pfaedle_shapes.py" "$work/$variant"
    for f in translations.txt feed_info.txt; do
        [ -f "$work/$variant/$f" ] || cp "$work/tidy/$f" "$work/$variant/"
    done
done

rm -f "$root"/hk.gtfs.zip "$root"/hk-drop-shapes.gtfs.zip "$root"/includefares-hk.gtfs.zip
(cd "$work/standard" && zip -qr "$root/hk.gtfs.zip" ./*.txt)
(cd "$work/drop-shapes" && zip -qr "$root/hk-drop-shapes.gtfs.zip" ./*.txt)
cp "$root/hk.gtfs.zip" "$root/includefares-hk.gtfs.zip"
(cd "$work/original" && zip -qr "$root/includefares-hk.gtfs.zip" ./*.csv)
