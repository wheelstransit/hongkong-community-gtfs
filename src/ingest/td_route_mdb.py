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
    Return a mapping of Citybus route number (string) -> first ROUTE_ID (string) from TD ROUTE MDB
    filtered by COMPANY_CODE='CTB' and SPECIAL_TYPE=0.
    """
    mdb_path = _ensure_download(cache_dir, silent)
    route_df = _export_table_to_csv(mdb_path, "ROUTE", silent)

    # Normalize column names
    route_df.columns = [c.strip().upper() for c in route_df.columns]
    required = {"ROUTE_ID", "COMPANY_CODE", "ROUTE_NAMEC", "SPECIAL_TYPE"}
    missing = required - set(route_df.columns)
    if missing:
        raise RuntimeError(f"ROUTE table missing columns: {missing}")

    # Filter CTB, SPECIAL_TYPE=0 and group by route name; pick first ROUTE_ID
    ctb = route_df[(route_df["COMPANY_CODE"] == "CTB") & (route_df["SPECIAL_TYPE"].astype(int) == 0)]

    # ROUTE_NAMEC holds the route number as string per requirement
    # Keep first occurrence per route number
    ctb_sorted = ctb.sort_values(["ROUTE_NAMEC", "ROUTE_ID"], kind="stable")
    first_per_route = ctb_sorted.groupby("ROUTE_NAMEC", as_index=False).first()

    # Build map
    mapping: Dict[str, str] = {}
    for _, row in first_per_route.iterrows():
        route_num = str(row["ROUTE_NAMEC"]).strip()
        route_id = str(row["ROUTE_ID"]).strip()
        mapping[route_num] = route_id

    if not silent:
        print(f"CTB route_id mapping from TD MDB: {len(mapping)} routes")
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
