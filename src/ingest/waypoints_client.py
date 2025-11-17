import requests
import json
import os
import zipfile
import io
import glob
import shutil
from tempfile import TemporaryDirectory
import logging

logging.basicConfig(level=logging.INFO)

WAYPOINTS_REPO_ZIP_URL = "https://codeload.github.com/hkbus/route-waypoints/zip/refs/heads/gh-pages"
WAYPOINTS_REPO_ROOT = "route-waypoints-gh-pages"

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

def _download_prebuilt_waypoints(destination: str, silent: bool = False) -> bool:
    """Download pre-generated waypoints from the public repository."""
    if not silent:
        logging.info("Downloading pre-generated waypoints (gh-pages branch)")

    try:
        response = requests.get(WAYPOINTS_REPO_ZIP_URL, timeout=120)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network failures should bubble up
        logging.error(f"Failed to download waypoints archive: {exc}")
        return False

    with TemporaryDirectory() as tmpdir:
        try:
            zip_path = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(tmpdir)
        except Exception as exc:
            logging.error(f"Failed to extract waypoints archive: {exc}")
            return False

        extracted_root = os.path.join(tmpdir, WAYPOINTS_REPO_ROOT)
        if not os.path.isdir(extracted_root):
            logging.error(f"Expected directory '{WAYPOINTS_REPO_ROOT}' not found in archive")
            return False

        files_copied = 0
        for root, _, files in os.walk(extracted_root):
            for filename in files:
                if not filename.lower().endswith('.json'):
                    continue
                source_path = os.path.join(root, filename)
                destination_path = os.path.join(destination, filename)
                try:
                    shutil.copy2(source_path, destination_path)
                    files_copied += 1
                except Exception as exc:
                    logging.warning(f"Could not copy {source_path}: {exc}")

        if files_copied == 0:
            logging.error("No waypoint JSON files were copied from the archive")
            return False

    if not silent:
        logging.info(f"Copied {files_copied} waypoint files from pre-generated dataset")

    version_path = os.path.join(destination, "0versions.json")
    if not os.path.exists(version_path):
        logging.warning("Waypoints archive did not contain 0versions.json; downstream consumers may not have version info")

    return True

def fetch_csdi_waypoints_data(force_ingest=False, silent=False):
    """Sync waypoints by downloading the pre-generated dataset."""
    if not force_ingest and check_waypoints_exist():
        if not silent:
            logging.info("Waypoints already exist. Use --force-ingest to re-fetch data.")
        return True

    _ensure_clean_directory("waypoints")

    if not _download_prebuilt_waypoints("waypoints", silent=silent):
        logging.error("Failed to refresh waypoints from pre-generated dataset")
        return False

    if not silent:
        logging.info("Waypoints data refresh completed successfully")

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch pre-generated waypoints data (gh-pages branch)')
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