from dataclasses import dataclass

import pandas as pd


@dataclass
class Operator:
    routes: pd.DataFrame
    trips: pd.DataFrame
    patterns: pd.DataFrame
    source_routes: pd.DataFrame
