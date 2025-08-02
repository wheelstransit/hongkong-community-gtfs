import geopandas as gpd
import pandas as pd
import requests
import zipfile
import os
import shutil

FGDB_URL = "https://static.csdi.gov.hk/csdi-webpage/download/7faa97a82780505c9673c4ba128fbfed/fgdb"
TEMP_DIR = "/tmp/bus_routes_data"
LAYER_NAME = "FB_ROUTE_LINE"

def fetch_bus_routes_data(silent=False):
    if not silent:
        print("Fetching bus routes data from CSDI FGDB source...")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    try:
        zip_path = os.path.join(TEMP_DIR, "fgdb.zip")

        if not silent:
            print(f"Downloading data from {FGDB_URL}...")
        with requests.get(FGDB_URL, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        if not silent:
            print("Download complete.")

        if not silent:
            print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        if not silent:
            print("Extraction complete.")

        gdb_path = None
        for item in os.listdir(TEMP_DIR):
            if item.endswith(".gdb"):
                gdb_path = os.path.join(TEMP_DIR, item)
                break

        if not gdb_path:
            raise FileNotFoundError("Could not find .gdb directory in the extracted files.")

        if not silent:
            print(f"Reading layer '{LAYER_NAME}' from {gdb_path}...")
        gdf = gpd.read_file(gdb_path, layer=LAYER_NAME)
        if not silent:
            print(f"Successfully fetched {len(gdf)} bus route records.")

        geometry_col_name = gdf.geometry.name
        gdf[geometry_col_name] = gdf[geometry_col_name].apply(lambda geom: geom.wkt)
        df = pd.DataFrame(gdf)

        return df.to_dict('records')

    except Exception as e:
        if not silent:
            print(f"Error fetching bus routes data: {e}")
        return None
    finally:
        # 4. Clean up
        if os.path.exists(TEMP_DIR):
            if not silent:
                print(f"Cleaning up temporary directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)

if __name__ == '__main__':
    print("testing :)")

    bus_routes_data = fetch_bus_routes_data()
    if bus_routes_data:
        print("\nSample bus routes data:")
        print(bus_routes_data[0])
    else:
        print("Failed to fetch bus routes data.")

    print("-" * 20)
    print("done testing :)")
