
import pandas as pd

TRAMWAY_ROUTES_URLS = {
    "en": "https://static.data.gov.hk/tramways/datasets/main_routes/tramways_main_routes_en.csv",
    "tc": "https://static.data.gov.hk/tramways/datasets/main_routes/tramways_main_routes_tc.csv",
    "sc": "https://static.data.gov.hk/tramways/datasets/main_routes/tramways_main_routes_sc.csv",
}

TRAMWAY_STOPS_URLS = {
    "en": "https://static.data.gov.hk/tramways/datasets/tram_stops/summary_tram_stops_en.csv",
    "tc": "https://static.data.gov.hk/tramways/datasets/tram_stops/summary_tram_stops_tc.csv",
    "sc": "https://static.data.gov.hk/tramways/datasets/tram_stops/summary_tram_stops_sc.csv",
}


class TramwayClient:
    def __init__(self):
        pass

    def _clean_columns(self, df):
        # Remove BOM and strip whitespace from all columns
        df.columns = [col.encode('utf-8').decode('utf-8-sig').strip() for col in df.columns]
        return df

    def fetch_tram_routes(self):
        dfs = {}
        for lang, url in TRAMWAY_ROUTES_URLS.items():
            df = pd.read_csv(url)
            df = self._clean_columns(df)
            print(f"[{lang}] columns after cleaning: {df.columns.tolist()}")
            dfs[lang] = df

        # Robust column mapping with correct Chinese column names
        rename_map = {
            "en": {"Route ID": "route_id", "Route start": "start_en", "Route end": "end_en"},
            "tc": {"路線號碼": "route_id", "路線起點": "start_tc", "路線終點": "end_tc"},
            "sc": {"路线号码": "route_id", "路线起点": "start_sc", "路线终点": "end_sc"},
        }
        for lang in dfs:
            # Only rename columns that exist
            rename_dict = {k: v for k, v in rename_map[lang].items() if k in dfs[lang].columns}
            dfs[lang] = dfs[lang].rename(columns=rename_dict)

        # Check columns
        for lang in ["en", "tc", "sc"]:
            for col in rename_map[lang].values():
                if col not in dfs[lang].columns:
                    raise KeyError(f"Column '{col}' missing in {lang} dataframe after renaming. Columns: {dfs[lang].columns.tolist()}")

        merged = dfs["en"][["route_id", "start_en", "end_en"]].copy()
        merged = merged.merge(dfs["tc"][["route_id", "start_tc", "end_tc"]], on="route_id", how="left")
        merged = merged.merge(dfs["sc"][["route_id", "start_sc", "end_sc"]], on="route_id", how="left")
        return merged.to_dict(orient="records")

    def fetch_tram_stops(self):
        dfs = {}
        for lang, url in TRAMWAY_STOPS_URLS.items():
            df = pd.read_csv(url)
            df = self._clean_columns(df)
            print(f"[{lang}] columns after cleaning: {df.columns.tolist()}")
            dfs[lang] = df

        rename_map = {
            "en": {"Stops Code": "stop_code", "Traveling Direction": "direction_en", "Stops Name": "stop_name_en"},
            "tc": {"車站代號": "stop_code", "行駛方向": "direction_tc", "車站名稱": "stop_name_tc"},
            "sc": {"车站代号": "stop_code", "行驶方向": "direction_sc", "车站名称": "stop_name_sc"},
        }
        for lang in dfs:
            rename_dict = {k: v for k, v in rename_map[lang].items() if k in dfs[lang].columns}
            dfs[lang] = dfs[lang].rename(columns=rename_dict)

        for lang in ["en", "tc", "sc"]:
            for col in rename_map[lang].values():
                if col not in dfs[lang].columns:
                    raise KeyError(f"Column '{col}' missing in {lang} dataframe after renaming. Columns: {dfs[lang].columns.tolist()}")

        merged = dfs["en"][["stop_code", "direction_en", "stop_name_en"]].copy()
        merged = merged.merge(dfs["tc"][["stop_code", "direction_tc", "stop_name_tc"]], on="stop_code", how="left")
        merged = merged.merge(dfs["sc"][["stop_code", "direction_sc", "stop_name_sc"]], on="stop_code", how="left")
        return merged.to_dict(orient="records")


# Example usage:
if __name__ == "__main__":
    client = TramwayClient()
    routes = client.fetch_tram_routes()
    stops = client.fetch_tram_stops()
    print(f"Fetched {len(routes)} tramway routes (merged multilingual).")
    print(f"Fetched {len(stops)} tramway stops (merged multilingual).")
    print("Sample route:", routes[0] if routes else "None")
    print("Sample stop:", stops[0] if stops else "None")
