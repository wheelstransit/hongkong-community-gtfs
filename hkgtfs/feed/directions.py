import numpy as np
import pandas as pd


def _haversine(a_lat, a_lon, b_lat, b_lon):
    valid = (~pd.isna(a_lat)) & (~pd.isna(a_lon)) & (~pd.isna(b_lat)) & (~pd.isna(b_lon))
    out = np.full_like(a_lat, fill_value=np.inf, dtype="float64")
    if valid.any():
        dlat = np.radians(b_lat[valid] - a_lat[valid])
        dlon = np.radians(b_lon[valid] - a_lon[valid])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(a_lat[valid])) * np.cos(np.radians(b_lat[valid])) * np.sin(dlon / 2) ** 2
        out[valid] = 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return out


def _endpoints(stop_times: pd.DataFrame) -> pd.DataFrame:
    ordered = stop_times.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id")["stop_id"]
    return pd.DataFrame({"first_stop_id": ordered.first(), "last_stop_id": ordered.last()}).reset_index()


def _swap_bound(value):
    if not isinstance(value, str):
        return value
    low = value.lower()
    if low in ("o", "outbound"):
        return "I" if value.isupper() else ("inbound" if low == "outbound" else "I")
    if low in ("i", "inbound"):
        return "O" if value.isupper() else ("outbound" if low == "inbound" else "O")
    return value


def fix_inverted_directions(trips: pd.DataFrame, stop_times: pd.DataFrame, stops: pd.DataFrame, mapping: pd.DataFrame,
                            gov_stop_times: pd.DataFrame, gov_stops: pd.DataFrame) -> pd.DataFrame:
    """Flip direction_id/bound of trips whose endpoints match their government trip only when reversed."""
    if mapping.empty:
        return trips
    coords = stops[["stop_id", "stop_lat", "stop_lon"]]
    ours = (_endpoints(stop_times)
            .merge(coords.add_prefix("first_"), on="first_stop_id", how="left")
            .merge(coords.add_prefix("last_"), on="last_stop_id", how="left"))
    gov_ids = mapping["original_trip_id"].dropna().unique()
    gov = _endpoints(gov_stop_times.loc[gov_stop_times["trip_id"].isin(gov_ids), ["trip_id", "stop_id", "stop_sequence"]])
    gov = gov.rename(columns={"first_stop_id": "gov_first_stop_id", "last_stop_id": "gov_last_stop_id"})
    gov_coords = gov_stops[["stop_id", "stop_lat", "stop_lon"]]
    gov = (gov.merge(gov_coords.add_prefix("gov_first_"), on="gov_first_stop_id", how="left")
              .merge(gov_coords.add_prefix("gov_last_"), on="gov_last_stop_id", how="left"))
    combo = (mapping.merge(ours, left_on="new_trip_id", right_on="trip_id")
                    .merge(gov, left_on="original_trip_id", right_on="trip_id", suffixes=("", "_gov")))
    if combo.empty:
        return trips

    col = lambda name: combo[name].to_numpy()
    aligned = (_haversine(col("first_stop_lat"), col("first_stop_lon"), col("gov_first_stop_lat"), col("gov_first_stop_lon"))
               + _haversine(col("last_stop_lat"), col("last_stop_lon"), col("gov_last_stop_lat"), col("gov_last_stop_lon")))
    reversed_ = (_haversine(col("first_stop_lat"), col("first_stop_lon"), col("gov_last_stop_lat"), col("gov_last_stop_lon"))
                 + _haversine(col("last_stop_lat"), col("last_stop_lon"), col("gov_first_stop_lat"), col("gov_first_stop_lon")))
    candidates = reversed_ + 200 < aligned
    reversed_m = reversed_[candidates].astype(int)
    invert = combo.loc[candidates, "new_trip_id"].values[(reversed_m >= 0) & (reversed_m < 500)].tolist()
    if not invert:
        return trips
    mask = trips["trip_id"].isin(invert)
    if "direction_id" in trips.columns:
        trips.loc[mask, "direction_id"] = trips.loc[mask, "direction_id"].apply(lambda d: 1 - int(d) if pd.notna(d) and str(d).isdigit() else d)
    for col_name in ("bound", "dir"):
        if col_name in trips.columns:
            trips.loc[mask, col_name] = trips.loc[mask, col_name].apply(_swap_bound)
    return trips
