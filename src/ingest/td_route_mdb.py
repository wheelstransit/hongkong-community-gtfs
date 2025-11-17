import os
import subprocess
import tempfile
from typing import Dict, Optional
import pandas as pd

TD_ROUTE_MDB_URL = "https://static.data.gov.hk/td/routes-and-fares/ROUTE_BUS.mdb"

def _ensure_download(dest_dir: str, silent: bool = False) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    mdb_path = os.path.join(dest_dir, "ROUTE_BUS.mdb")
    if os.path.exists(mdb_path) and os.path.getsize(mdb_path) > 0:
        if not silent:
            print(f"ROUTE_BUS.mdb already present at {mdb_path}")
        return mdb_path
    if not silent:
        print("Downloading ROUTE_BUS.mdb from TD…")
    import requests
    resp = requests.get(TD_ROUTE_MDB_URL, timeout=60)
    resp.raise_for_status()
    with open(mdb_path, "wb") as f:
        f.write(resp.content)
    if not silent:
        print(f"Saved to {mdb_path} ({len(resp.content)} bytes)")
    return mdb_path

def _export_table_to_csv(mdb_path: str, table: str, silent: bool = False) -> pd.DataFrame:
    """Export a table using mdbtools (mdb-export) into a DataFrame."""
    # Check mdb-export availability
    try:
        subprocess.run(["mdb-export", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError("mdb-export (mdbtools) is required but not installed in the container.") from e

    with tempfile.TemporaryDirectory() as tmpd:
        csv_path = os.path.join(tmpd, f"{table}.csv")
        if not silent:
            print(f"Exporting {table} to CSV via mdb-export…")
        proc = subprocess.run([
            "mdb-export", "-q", "\"", mdb_path, table
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(csv_path, "wb") as f:
            f.write(proc.stdout)
        df = pd.read_csv(csv_path)
        return df

def get_ctb_route_id_map(cache_dir: str = ".cache", silent: bool = False) -> Dict[str, str]:
    """
    Return a mapping of Citybus route number (string) -> first ROUTE_ID (string) from TD ROUTE MDB.
    
    Logic:
    1. First, try to find routes with SPECIAL_TYPE=0 (standard routes)
    2. For routes that don't have any SPECIAL_TYPE=0 variant, include the first available route
       regardless of its SPECIAL_TYPE value
    """
    mdb_path = _ensure_download(cache_dir, silent)
    route_df = _export_table_to_csv(mdb_path, "ROUTE", silent)

    # Normalize column names
    route_df.columns = [c.strip().upper() for c in route_df.columns]
    required = {"ROUTE_ID", "COMPANY_CODE", "ROUTE_NAMEC", "SPECIAL_TYPE"}
    missing = required - set(route_df.columns)
    if missing:
        raise RuntimeError(f"ROUTE table missing columns: {missing}")

    # Filter CTB routes
    ctb_all = route_df[route_df["COMPANY_CODE"] == "CTB"]
    
    # First, get routes with SPECIAL_TYPE=0 (standard routes)
    ctb_standard = ctb_all[ctb_all["SPECIAL_TYPE"].astype(int) == 0]
    
    # Get all route numbers that have a standard variant
    standard_route_numbers = set(ctb_standard["ROUTE_NAMEC"].unique())
    
    # Get routes that don't have any standard variant
    ctb_non_standard = ctb_all[~ctb_all["ROUTE_NAMEC"].isin(standard_route_numbers)]
    
    # For non-standard routes, get the first occurrence per route number
    ctb_non_standard_sorted = ctb_non_standard.sort_values(["ROUTE_NAMEC", "ROUTE_ID"], kind="stable")
    first_non_standard_per_route = ctb_non_standard_sorted.groupby("ROUTE_NAMEC", as_index=False).first()
    
    # For standard routes, get the first occurrence per route number
    ctb_standard_sorted = ctb_standard.sort_values(["ROUTE_NAMEC", "ROUTE_ID"], kind="stable")
    first_standard_per_route = ctb_standard_sorted.groupby("ROUTE_NAMEC", as_index=False).first()
    
    # Combine both sets
    combined_routes = pd.concat([first_standard_per_route, first_non_standard_per_route], ignore_index=True)

    # Build map
    mapping: Dict[str, str] = {}
    for _, row in combined_routes.iterrows():
        route_num = str(row["ROUTE_NAMEC"]).strip()
        route_id = str(row["ROUTE_ID"]).strip()
        mapping[route_num] = route_id

    if not silent:
        standard_count = len(first_standard_per_route)
        non_standard_count = len(first_non_standard_per_route)
        print(f"CTB route_id mapping from TD MDB: {len(mapping)} routes ({standard_count} standard, {non_standard_count} fallback)")
    return mapping

def lookup_ctb_route_id(route_number: str, cache_dir: str = ".cache", silent: bool = False) -> Optional[str]:
    mapping = get_ctb_route_id_map(cache_dir=cache_dir, silent=silent)
    return mapping.get(str(route_number).strip())

if __name__ == "__main__":
    import sys
    silent = False
    if len(sys.argv) > 1:
        for route in sys.argv[1:]:
            rid = lookup_ctb_route_id(route, silent=silent)
            print(f"CTB {route} -> ROUTE_ID: {rid}")
    else:
        m = get_ctb_route_id_map(silent=silent)
        sample = list(m.items())[:10]
        print(f"Loaded {len(m)} CTB mappings. Sample: {sample}")
