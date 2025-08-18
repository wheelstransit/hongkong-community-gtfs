# UNUSED


import geopandas as gpd
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

def unify_stops_by_name_and_distance(stops_gdf: gpd.GeoDataFrame, name_col: str, stop_id_col: str, distance_threshold_meters: float = 2.0, silent: bool = False):
    """
    Unifies stops within a single GeoDataFrame that have the same name and are within a specified distance.
    """
    if not silent:
        print(f"Unifying stops within a distance of {distance_threshold_meters}m...")

    projected_gdf = stops_gdf.copy().to_crs(epsg=2326)
    
    sindex = projected_gdf.sindex

    processed_indices = set()
    duplicates_map = {}
    
    for index, stop in tqdm(projected_gdf.iterrows(), total=projected_gdf.shape[0], desc="Unifying stops", disable=silent):
        if index in processed_indices:
            continue

        processed_indices.add(index)
        
        possible_matches_indices = list(sindex.intersection(stop.geometry.buffer(distance_threshold_meters).bounds))
        possible_matches = projected_gdf.iloc[possible_matches_indices]

        stop_name = stops_gdf.loc[index][name_col]
        
        for match_index, match_stop in possible_matches.iterrows():
            if match_index in processed_indices:
                continue

            match_name = stops_gdf.loc[match_index][name_col]

            if stop_name == match_name:
                distance = stop.geometry.distance(match_stop.geometry)
                if distance <= distance_threshold_meters:
                    original_stop_id = stops_gdf.loc[match_index][stop_id_col]
                    canonical_stop_id = stops_gdf.loc[index][stop_id_col]
                    duplicates_map[original_stop_id] = canonical_stop_id
                    processed_indices.add(match_index)

    duplicate_indices = stops_gdf[stops_gdf[stop_id_col].isin(duplicates_map.keys())].index
    unified_stops_gdf = stops_gdf.drop(index=duplicate_indices)
    
    if not silent:
        print(f"Unified {len(stops_gdf)} stops into {len(unified_stops_gdf)} unique locations.")
    
    return unified_stops_gdf, duplicates_map
