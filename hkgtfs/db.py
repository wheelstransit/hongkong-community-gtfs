import duckdb
import geopandas as gpd
import pandas as pd


class Store:
    def __init__(self):
        self.con = duckdb.connect(config={"threads": 1})
        self.con.execute("CREATE MACRO ST_X(g) AS g.x")
        self.con.execute("CREATE MACRO ST_Y(g) AS g.y")

    def write(self, name: str, df: pd.DataFrame):
        if isinstance(df, gpd.GeoDataFrame):
            df = pd.DataFrame(df).assign(geometry=[{"x": p.x, "y": p.y} for p in df.geometry])
        self.con.register("_df", df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _df')
        self.con.unregister("_df")

    def read(self, sql: str, params=None) -> pd.DataFrame:
        return self.con.execute(sql, params).df()

    def read_geo(self, sql: str) -> gpd.GeoDataFrame:
        df = self.read(f"SELECT * EXCLUDE (geometry), geometry.x AS __x, geometry.y AS __y FROM ({sql})")
        geometry = gpd.points_from_xy(df.pop("__x"), df.pop("__y"))
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
