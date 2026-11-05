import io
import subprocess
import tempfile

import pandas as pd

from ..http import get

URL = "https://static.data.gov.hk/td/routes-and-fares/ROUTE_BUS.mdb"


def fetch():
    with tempfile.NamedTemporaryFile(suffix=".mdb") as f:
        f.write(get(URL, timeout=300).content)
        f.flush()
        csv = subprocess.run(["mdb-export", "-q", '"', f.name, "ROUTE"], check=True, capture_output=True).stdout
    return pd.read_csv(io.BytesIO(csv))
