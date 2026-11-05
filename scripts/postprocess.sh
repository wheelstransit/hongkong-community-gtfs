#!/usr/bin/env bash
# Tidy output/hk.gtfs.zip with gtfstidy, map-match shapes with pfaedle and package the published feeds.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cache="$root/.cache"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$cache" "$work/osm" "$work/gtfs" "$work/original" "$work/standard" "$work/drop-shapes"

unzip -q "$root/output/hk.gtfs.zip" -d "$work/original"
pbf="$cache/hong-kong-latest.osm.pbf"
[ -f "$pbf" ] || curl -sSfL -o "$pbf" https://download.geofabrik.de/asia/china/hong-kong-latest.osm.pbf
osmium cat "$pbf" -o "$work/osm/hk.osm.bz2" &
osm=$!
gtfstidy -OscRCSeD "$root/output/hk.gtfs.zip" -o "$work/gtfs/input.gtfs.zip"
wait "$osm"

pfaedle_variant() {
    local variant="$1"
    shift
    if command -v pfaedle >/dev/null; then
        pfaedle "$@" -x "$work/osm/hk.osm.bz2" -i "$work/gtfs/input.gtfs.zip" -o "$work/$variant"
    else
        docker run -i --rm --user "$(id -u):$(id -g)" \
            -v "$work/osm:/osm" -v "$work/gtfs:/gtfs" -v "$work/$variant:/gtfs-out" \
            ghcr.io/ad-freiburg/pfaedle:latest "$@" -x /osm/hk.osm.bz2 -i /gtfs/input.gtfs.zip
    fi
    for f in translations.txt feed_info.txt; do
        [ -f "$work/$variant/$f" ] || cp "$work/original/$f" "$work/$variant/"
    done
    python3 "$root/scripts/finalize_feed.py" "$work/$variant"
}
pfaedle_variant standard &
standard=$!
pfaedle_variant drop-shapes -D
wait "$standard"

rm -f "$root"/hk.gtfs.zip "$root"/hk-drop-shapes.gtfs.zip "$root"/includefares-hk.gtfs.zip
(cd "$work/standard" && zip -qr "$root/hk.gtfs.zip" ./*.txt)
(cd "$work/drop-shapes" && zip -qr "$root/hk-drop-shapes.gtfs.zip" ./*.txt)
cp "$root/hk.gtfs.zip" "$root/includefares-hk.gtfs.zip"
(cd "$work/original" && zip -qr "$root/includefares-hk.gtfs.zip" ./*.csv)
