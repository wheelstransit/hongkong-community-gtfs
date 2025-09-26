import datetime
from zoneinfo import ZoneInfo
import requests
import json
import re
import os
import zipfile
import io
import glob
import shutil
import fiona
from tempfile import TemporaryDirectory
import logging
from shapely.geometry import shape, mapping, LineString
from shapely.ops import linemerge
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def _ensure_clean_directory(path: str):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        target = os.path.join(path, name)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        except Exception as e:
            logging.warning(f"Could not remove {target}: {e}")

def store_version(key: str, version: str):
    """Store version information for a dataset."""
    logging.info(f"{key} version: {version}")
    try:
        with open('waypoints/0versions.json', 'r') as f:
            version_dict = json.load(f)
    except BaseException:
        version_dict = {}
    version_dict[key] = version
    version_dict = dict(sorted(version_dict.items()))
    with open('waypoints/0versions.json', 'w', encoding='UTF-8') as f:
        json.dump(version_dict, f, indent=4)

def check_waypoints_exist():
    """Check if waypoints directory exists and has data."""
    if not os.path.exists("waypoints"):
        return False

    waypoint_files = glob.glob("waypoints/*.json")
    if not waypoint_files:
        return False

    if not os.path.exists("waypoints/0versions.json"):
        return False

    logging.info(f"Found existing waypoints directory with {len(waypoint_files)} files")
    return True

def _process_csdi_dataset(csdi_dataset, silent=False):
    """Process a single CSDI dataset in a memory-efficient way."""
    if not silent:
        logging.info("Processing CSDI dataset: " + json.dumps(csdi_dataset))

    r = requests.get(
        "https://portal.csdi.gov.hk/geoportal/rest/metadata/item/" +
        csdi_dataset["id"])
    r.raise_for_status()
    src_id = json.loads(r.content)['_source']['fileid'].replace('-', '')

    r = requests.get(
        "https://static.csdi.gov.hk/csdi-webpage/download/" + src_id + "/fgdb")
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    version = min([f.date_time for f in z.infolist()])
    version = datetime.datetime(
        *version, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    store_version(csdi_dataset["name"], version.isoformat())
    gdb_name = next(s[0:s.index('/')]
                    for s in z.namelist() if s != "__MACOSX")

    with TemporaryDirectory() as tmpdir:
        z.extractall(tmpdir)
        gdb_path = os.path.join(tmpdir, gdb_name)
        
        with fiona.open(gdb_path, 'r') as source:
            grouped_features = {}
            for feature in tqdm(source, desc=f"Reading {csdi_dataset['name']} features", unit="feature", disable=silent):
                props = feature['properties']
                key = (props['ROUTE_ID'], props['ROUTE_SEQ'])
                if key not in grouped_features:
                    grouped_features[key] = []
                grouped_features[key].append(feature)

            for key, features in tqdm(grouped_features.items(), desc=f"Processing {csdi_dataset['name']} routes", unit="route", disable=silent):
                all_coords = []
                for feature in features:
                    geom = shape(feature['geometry'])
                    if geom.geom_type == 'MultiLineString':
                        for line in geom.geoms:
                            all_coords.extend(list(line.coords))
                    else:
                        all_coords.extend(list(geom.coords))

                properties = features[0]['properties']

                new_feature = {
                    'type': 'Feature',
                    'properties': dict(properties),
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': all_coords
                    }
                }
                _store_waypoint_feature(new_feature)

    return True

def _store_waypoint_feature(feature):
    """Store a single waypoint feature as a JSON file."""
    properties = feature["properties"]
    direction = "O" if properties["ROUTE_SEQ"] == 1 else "I"
    filename = f"{properties['ROUTE_ID']}-{direction}.json"

    geojson_data = {
        "features": [feature],
        "type": "FeatureCollection"
    }

    with open(f"waypoints/{filename}", "w", encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, separators=(",", ":"))

def _copy_static_waypoint_files(silent=False):
    if not silent:
        logging.info("Copying static waypoint data")

    static_dirs = ['./mtr', './lrt', './ferry']
    files_copied = 0

    for static_dir in static_dirs:
        pattern = os.path.join(static_dir, '*.json')
        files = glob.glob(pattern)

        for file_path in files:
            try:
                shutil.copy(file_path, "waypoints")
                files_copied += 1
            except Exception as e:
                logging.warning(f"Could not copy {file_path}: {e}")

    if not silent:
        logging.info(f"Copied {files_copied} static waypoint files")

    return files_copied

def fetch_csdi_waypoints_data(force_ingest=False, silent=False):
    if not force_ingest and check_waypoints_exist():
        if not silent:
            logging.info("Waypoints already exist. Use --force-ingest to re-fetch data.")
        return True

    if not silent:
        logging.info("Fetching waypoints data from CSDI and static sources...")

    _ensure_clean_directory("waypoints")

    csdi_datasets = [
        {"name": "bus", "id": "td_rcd_1638844988873_41214"},
        {"name": "gmb", "id": "td_rcd_1697082463580_57453"}
    ]

    for csdi_dataset in csdi_datasets:
        try:
            _process_csdi_dataset(csdi_dataset, silent)
        except Exception as e:
            logging.error(f"Error processing CSDI dataset {csdi_dataset['name']}: {e}", exc_info=True)
            return False

    _copy_static_waypoint_files(silent)

    if not silent:
        logging.info("Waypoints data ingestion completed successfully")

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch waypoints data from CSDI and static sources')
    parser.add_argument('--force-ingest', action='store_true',
                       help='Force re-ingestion even if waypoints already exist')
    parser.add_argument('--silent', action='store_true',
                       help='Suppress non-error output')

    args = parser.parse_args()

    success = fetch_csdi_waypoints_data(
        force_ingest=args.force_ingest,
        silent=args.silent
    )

    if success:
        print("Waypoints data ingestion completed successfully")
    else:
        print("Waypoints data ingestion failed")
        exit(1)