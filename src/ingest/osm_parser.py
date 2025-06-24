import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import pickle
import os

class OSMRoadNetwork:
    def __init__(self, place_name="Hong Kong", cache_dir="data/osm_cache"):
        self.place_name = place_name
        self.cache_dir = cache_dir
        self.graph = None
        self.edges_gdf = None
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_road_network(self, network_type='drive'):
        cache_file = os.path.join(self.cache_dir, f"hk_roads_{network_type}.pkl")

        if os.path.exists(cache_file):
            print("Loading cached OSM road network...")
            with open(cache_file, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            print("Downloading OSM road network...")
            self.graph = ox.graph_from_place(
                self.place_name,
                network_type=network_type,
                simplify=True,
                retain_all=False
            )

            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph, f)

        self.edges_gdf = ox.graph_to_gdfs(self.graph, nodes=False, edges=True)
        return self.graph, self.edges_gdf

    def get_road_edges_in_bbox(self, bbox):
        if self.edges_gdf is None:
            raise ValueError("Road network not loaded. Call fetch_road_network() first.")

        minx, miny, maxx, maxy = bbox
        mask = (
            (self.edges_gdf.geometry.bounds['minx'] <= maxx) &
            (self.edges_gdf.geometry.bounds['maxx'] >= minx) &
            (self.edges_gdf.geometry.bounds['miny'] <= maxy) &
            (self.edges_gdf.geometry.bounds['maxy'] >= miny)
        )
        return self.edges_gdf[mask]
