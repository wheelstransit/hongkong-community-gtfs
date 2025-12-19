#!/bin/bash

# Clean GTFS script - processes GTFS feed and generates shapes using pfaedle
# Outputs clean hk.gtfs.zip in repo root directory

set -e  # Exit on any error

REPO_ROOT="$(pwd)"
INPUT_GTFS="$REPO_ROOT/output/hk.gtfs.zip"
OUTPUT_GTFS="$REPO_ROOT/hk.gtfs.zip"
OUTPUT_GTFS_WITH_FARES="$REPO_ROOT/includefares-hk.gtfs.zip"
OUTPUT_GTFS_DROP_SHAPES="$REPO_ROOT/hk-drop-shapes.gtfs.zip"

OSM_FILE=""
OSM_PARENT_DIR="$(realpath "$REPO_ROOT")"
OSM_PBF_PATH="$OSM_PARENT_DIR/hong-kong-latest.osm.pbf"
OSM_PBF_ALT1="$OSM_PARENT_DIR/hk.osm.pbf"
OSM_BZ2_ALT="$OSM_PARENT_DIR/hk.osm.bz2"
TEMP_DIR="/tmp/gtfs_processing"
PFAEDLE_OUTPUT_STANDARD="$TEMP_DIR/output_standard"
PFAEDLE_OUTPUT_DROP_SHAPES="$TEMP_DIR/output_drop_shapes"

echo "Starting GTFS cleaning and shape generation process..."

ensure_osmium() {
    if command -v osmium >/dev/null 2>&1; then
        return 0
    fi
    echo "'osmium' not found. Attempting to install 'osmium-tool'..."
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Error: apt-get is not available. Please install 'osmium-tool' manually."
        exit 1
    fi
    if [ "$EUID" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo -n"
        else
            echo "Error: Need root privileges to install 'osmium-tool' and 'sudo' is not available."
            echo "Please run this script as root or install 'osmium-tool' manually."
            exit 1
        fi
    fi
    $SUDO apt-get update -y
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y osmium-tool
    if ! command -v osmium >/dev/null 2>&1; then
        echo "Error: Failed to install 'osmium-tool'."
        exit 1
    fi
}

run_pfaedle_variant() {
    # $1 = output directory, $2 = additional pfaedle flags (optional)
    local output_dir="$1"
    local extra_flags="$2"

    mkdir -p "$output_dir"

    echo "Running pfaedle with flags: '${extra_flags}' (output: $output_dir)"
    docker run -i --rm \
        --user "$(id -u):$(id -g)" \
        --volume "$TEMP_DIR/osm:/osm" \
        --volume "$TEMP_DIR/gtfs:/gtfs" \
        --volume "$output_dir:/gtfs-out" \
        ghcr.io/ad-freiburg/pfaedle:latest \
        ${extra_flags} -x "/osm/$OSM_DOCKER_FILE" -i /gtfs/input.gtfs.zip
}

rename_shapes_in_dir() {
    # $1 = directory containing trips.txt/shapes.txt
    local target_dir="$1"

    if [ ! -f "$target_dir/trips.txt" ] || [ ! -f "$target_dir/shapes.txt" ]; then
        echo "No trips.txt or shapes.txt in $target_dir, skipping shape ID renaming"
        return 0
    fi

    (cd "$target_dir" && python3 - << 'EOF'
import csv
import re
from collections import OrderedDict
from pathlib import Path

trips_path = Path('trips.txt')
shapes_path = Path('shapes.txt')

shape_to_first_trip = OrderedDict()

print("Reading trips.txt to build shape ID mapping...")
with trips_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        shape_id = row.get('shape_id', '').strip()
        trip_id = row.get('trip_id', '').strip()
        if shape_id and re.match(r'^shp_\d+_\d+$', shape_id):
            shape_to_first_trip.setdefault(shape_id, trip_id)

print(f"Found {len(shape_to_first_trip)} pfaedle shapes to rename")
if not shape_to_first_trip:
    raise SystemExit(0)

trips_data = []
with trips_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row['shape_id'] in shape_to_first_trip:
            row['shape_id'] = shape_to_first_trip[row['shape_id']]
        trips_data.append(row)

with trips_path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(trips_data)

shapes_data = []
with shapes_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row['shape_id'] in shape_to_first_trip:
            row['shape_id'] = shape_to_first_trip[row['shape_id']]
        shapes_data.append(row)

with shapes_path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(shapes_data)

print("Shape ID renaming completed successfully")
EOF
    )
}

# Check if input GTFS exists
if [ ! -f "$INPUT_GTFS" ]; then
    echo "Error: Input GTFS file not found at $INPUT_GTFS"
    exit 1
fi


if [ -f "$OSM_PBF_PATH" ]; then
    OSM_FILE="$OSM_PBF_PATH"
elif [ -f "$OSM_PBF_ALT1" ]; then
    OSM_FILE="$OSM_PBF_ALT1"
elif [ -f "$OSM_BZ2_ALT" ]; then
    OSM_FILE="$OSM_BZ2_ALT"
else
    echo "No local OSM file found in repository directory. Downloading hong-kong-latest.osm.pbf..."
    mkdir -p "$OSM_PARENT_DIR"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$OSM_PBF_PATH" "https://download.geofabrik.de/asia/china/hong-kong-latest.osm.pbf"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$OSM_PBF_PATH" "https://download.geofabrik.de/asia/china/hong-kong-latest.osm.pbf"
    else
        exit 1
    fi
    OSM_FILE="$OSM_PBF_PATH"
fi


# Create temporary directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Step 1: Extract translations.txt and fare files from original GTFS (if they exist)
echo "Step 1: Extracting translations.txt and fare files from original GTFS..."
mkdir -p "$TEMP_DIR/original"
unzip -q "$INPUT_GTFS" -d "$TEMP_DIR/original"
if [ -f "$TEMP_DIR/original/translations.txt" ]; then
    echo "Found translations.txt in original GTFS, will preserve it"
    PRESERVE_TRANSLATIONS=true
else
    echo "No translations.txt found in original GTFS"
    PRESERVE_TRANSLATIONS=false
fi

# Check for non-standard fare files
PRESERVE_FARE_STAGES=false
PRESERVE_SPECIAL_FARE_RULES=false
if [ -f "$TEMP_DIR/original/fare_stages.csv" ]; then
    echo "Found fare_stages.csv in original GTFS, will preserve it"
    PRESERVE_FARE_STAGES=true
fi
if [ -f "$TEMP_DIR/original/special_fare_rules.csv" ]; then
    echo "Found special_fare_rules.csv in original GTFS, will preserve it"
    PRESERVE_SPECIAL_FARE_RULES=true
fi

# Step 2: Clean GTFS using gtfstidy
echo "Step 2: Cleaning GTFS with gtfstidy..."
echo "Running: gtfstidy -OscRCSeD $INPUT_GTFS -o $TEMP_DIR/cleaned.gtfs.zip"

if ! gtfstidy -OscRCSeD "$INPUT_GTFS" -o "$TEMP_DIR/cleaned.gtfs.zip"; then
    echo "Error: gtfstidy failed"
    exit 1
fi

echo "GTFS cleaning completed"

# Step 3: Generate shapes using pfaedle (standard and drop-shapes variants)
echo "Step 3: Generating shapes with pfaedle..."

# Pull the latest pfaedle Docker image
echo "Pulling pfaedle Docker image..."
docker pull ghcr.io/ad-freiburg/pfaedle:latest

# Create directories for Docker volumes
mkdir -p "$TEMP_DIR/osm"
mkdir -p "$TEMP_DIR/gtfs"

# Copy files to temporary locations for Docker
echo "Preparing files for pfaedle..."
if [[ "$OSM_FILE" == *.pbf ]]; then
    echo "Converting PBF to OSM XML (.bz2) with osmium..."
    ensure_osmium
    osmium cat "$(realpath "$OSM_FILE")" -o "$TEMP_DIR/osm/hk.osm.bz2"
    OSM_DOCKER_FILE="hk.osm.bz2"
else
    if [[ "$OSM_FILE" == *.bz2 ]]; then
        cp "$(realpath "$OSM_FILE")" "$TEMP_DIR/osm/hk.osm.bz2"
        OSM_DOCKER_FILE="hk.osm.bz2"
    else
        cp "$(realpath "$OSM_FILE")" "$TEMP_DIR/osm/hk.osm"
        OSM_DOCKER_FILE="hk.osm"
    fi
fi
cp "$TEMP_DIR/cleaned.gtfs.zip" "$TEMP_DIR/gtfs/input.gtfs.zip"

echo "Running pfaedle (standard)..."
run_pfaedle_variant "$PFAEDLE_OUTPUT_STANDARD" ""

echo "Running pfaedle (-D drop shapes)..."
run_pfaedle_variant "$PFAEDLE_OUTPUT_DROP_SHAPES" "-D"

# Check if pfaedle output exists (pfaedle outputs individual files, not a zip)
for output_dir in "$PFAEDLE_OUTPUT_STANDARD" "$PFAEDLE_OUTPUT_DROP_SHAPES"; do
    if [ ! -f "$output_dir/agency.txt" ]; then
        echo "Error: pfaedle did not produce expected output in $output_dir"
        exit 1
    fi
done

# Step 4: Rename pfaedle shape IDs to use first trip ID
echo "Step 4: Renaming pfaedle shape IDs to use first trip ID..."
rename_shapes_in_dir "$PFAEDLE_OUTPUT_STANDARD"
rename_shapes_in_dir "$PFAEDLE_OUTPUT_DROP_SHAPES"

# Step 5: Add back translations.txt if it existed in original and create final zips
echo "Step 5: Creating final GTFS zip files..."

package_standard_zip() {
    local source_dir="$1"
    local zip_target="$2"
    local include_fares="$3"

    if [ "$PRESERVE_TRANSLATIONS" = true ]; then
        echo "Adding back translations.txt from original GTFS to $source_dir..."
        cp "$TEMP_DIR/original/translations.txt" "$source_dir"
    fi

    (cd "$source_dir" && zip -r "$zip_target" *.txt)

    if [ "$include_fares" = true ]; then
        if compgen -G "$source_dir/"'*.csv' >/dev/null; then
            (cd "$source_dir" && zip -r "$zip_target" *.csv)
        else
            echo "No fare CSV files to include for $zip_target"
        fi
    fi
}

echo "Creating standard GTFS zip: $OUTPUT_GTFS"
package_standard_zip "$PFAEDLE_OUTPUT_STANDARD" "$OUTPUT_GTFS" false

echo "Creating drop-shapes GTFS zip: $OUTPUT_GTFS_DROP_SHAPES"
package_standard_zip "$PFAEDLE_OUTPUT_DROP_SHAPES" "$OUTPUT_GTFS_DROP_SHAPES" false

echo "Creating includefares GTFS zip: $OUTPUT_GTFS_WITH_FARES"
INCLUDE_FARES=false
if [ "$PRESERVE_FARE_STAGES" = true ]; then
    echo "Adding fare_stages.csv..."
    cp "$TEMP_DIR/original/fare_stages.csv" "$PFAEDLE_OUTPUT_STANDARD"
    INCLUDE_FARES=true
fi

if [ "$PRESERVE_SPECIAL_FARE_RULES" = true ]; then
    echo "Adding special_fare_rules.csv..."
    cp "$TEMP_DIR/original/special_fare_rules.csv" "$PFAEDLE_OUTPUT_STANDARD"
    INCLUDE_FARES=true
fi
package_standard_zip "$PFAEDLE_OUTPUT_STANDARD" "$OUTPUT_GTFS_WITH_FARES" "$INCLUDE_FARES"

cd "$REPO_ROOT"

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Success! Clean GTFS files generated:"
echo "  Standard GTFS: $OUTPUT_GTFS"
echo "  With fares GTFS: $OUTPUT_GTFS_WITH_FARES"
echo "  Drop-shapes GTFS: $OUTPUT_GTFS_DROP_SHAPES"
echo "Process completed successfully."
