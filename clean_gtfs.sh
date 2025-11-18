#!/bin/bash

# Clean GTFS script - processes GTFS feed and generates shapes using pfaedle
# Outputs clean hk.gtfs.zip in repo root directory

set -e  # Exit on any error

REPO_ROOT="$(pwd)"
INPUT_GTFS="$REPO_ROOT/output/hk.gtfs.zip"
OUTPUT_GTFS="$REPO_ROOT/hk.gtfs.zip"

OSM_FILE=""
OSM_PARENT_DIR="$(realpath "$REPO_ROOT")"
OSM_PBF_PATH="$OSM_PARENT_DIR/hong-kong-latest.osm.pbf"
OSM_PBF_ALT1="$OSM_PARENT_DIR/hk.osm.pbf"
OSM_BZ2_ALT="$OSM_PARENT_DIR/hk.osm.bz2"
TEMP_DIR="/tmp/gtfs_processing"

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

# Step 1: Extract translations.txt from original GTFS (if it exists)
echo "Step 1: Extracting translations.txt from original GTFS..."
mkdir -p "$TEMP_DIR/original"
unzip -q "$INPUT_GTFS" -d "$TEMP_DIR/original"
if [ -f "$TEMP_DIR/original/translations.txt" ]; then
    echo "Found translations.txt in original GTFS, will preserve it"
    PRESERVE_TRANSLATIONS=true
else
    echo "No translations.txt found in original GTFS"
    PRESERVE_TRANSLATIONS=false
fi

# Step 2: Clean GTFS using gtfstidy
echo "Step 2: Cleaning GTFS with gtfstidy..."
echo "Running: gtfstidy -OscRCSeD $INPUT_GTFS -o $TEMP_DIR/cleaned.gtfs.zip"

if ! gtfstidy -OscRCSeD "$INPUT_GTFS" -o "$TEMP_DIR/cleaned.gtfs.zip"; then
    echo "Error: gtfstidy failed"
    exit 1
fi

echo "GTFS cleaning completed"

# Step 3: Generate shapes using pfaedle
echo "Step 3: Generating shapes with pfaedle..."

# Pull the latest pfaedle Docker image
echo "Pulling pfaedle Docker image..."
docker pull ghcr.io/ad-freiburg/pfaedle:latest

# Create directories for Docker volumes
mkdir -p "$TEMP_DIR/osm"
mkdir -p "$TEMP_DIR/gtfs"
mkdir -p "$TEMP_DIR/output"

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

# Run pfaedle with Docker
echo "Running pfaedle to generate shapes..."
docker run -i --rm \
    --volume "$TEMP_DIR/osm:/osm" \
    --volume "$TEMP_DIR/gtfs:/gtfs" \
    --volume "$TEMP_DIR/output:/gtfs-out" \
    ghcr.io/ad-freiburg/pfaedle:latest \
    -x "/osm/$OSM_DOCKER_FILE" -i /gtfs/input.gtfs.zip

# Check if pfaedle output exists (pfaedle outputs individual files, not a zip)
if [ ! -f "$TEMP_DIR/output/agency.txt" ]; then
    echo "Error: pfaedle did not produce expected output"
    exit 1
fi

# Step 4: Rename pfaedle shape IDs to use first trip ID
echo "Step 4: Renaming pfaedle shape IDs to use first trip ID..."
cd "$TEMP_DIR/output"

# Create a Python script to rename shape IDs
cat > rename_shapes.py << 'EOF'
import csv
import re
from collections import defaultdict, OrderedDict

def rename_shape_ids():
    # Read trips.txt to build shape_id -> first_trip_id mapping
    shape_to_first_trip = OrderedDict()
    
    print("Reading trips.txt to build shape ID mapping...")
    with open('trips.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape_id = row.get('shape_id', '').strip()
            trip_id = row.get('trip_id', '').strip()
            
            # Only process pfaedle-generated shape IDs (pattern: shp_x_x)
            if shape_id and re.match(r'^shp_\d+_\d+$', shape_id):
                if shape_id not in shape_to_first_trip:
                    shape_to_first_trip[shape_id] = trip_id
                    print(f"Mapping shape {shape_id} -> {trip_id}")
    
    print(f"Found {len(shape_to_first_trip)} pfaedle shapes to rename")
    
    if not shape_to_first_trip:
        print("No pfaedle-generated shape IDs found, skipping rename")
        return
    
    # Update trips.txt with new shape IDs
    print("Updating trips.txt with new shape IDs...")
    trips_data = []
    with open('trips.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['shape_id'] in shape_to_first_trip:
                row['shape_id'] = shape_to_first_trip[row['shape_id']]
            trips_data.append(row)
    
    with open('trips.txt', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trips_data)
    
    # Update shapes.txt with new shape IDs
    print("Updating shapes.txt with new shape IDs...")
    shapes_data = []
    with open('shapes.txt', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['shape_id'] in shape_to_first_trip:
                row['shape_id'] = shape_to_first_trip[row['shape_id']]
            shapes_data.append(row)
    
    with open('shapes.txt', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shapes_data)
    
    print("Shape ID renaming completed successfully")

if __name__ == "__main__":
    rename_shape_ids()
EOF

# Run the shape ID renaming script
python3 rename_shapes.py

# Clean up the script
rm rename_shapes.py

# Step 5: Add back translations.txt if it existed in original and create final zip
echo "Step 5: Creating final GTFS zip file..."

# Add back translations.txt if it was in the original GTFS
if [ "$PRESERVE_TRANSLATIONS" = true ]; then
    echo "Adding back translations.txt from original GTFS..."
    cp "$TEMP_DIR/original/translations.txt" .
fi

zip -r "$OUTPUT_GTFS" *.txt
cd "$REPO_ROOT"

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Success! Clean GTFS with shapes generated at: $OUTPUT_GTFS"
echo "Process completed successfully."
