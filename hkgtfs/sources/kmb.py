from ..http import get_json

BASE_URL = "https://data.etabus.gov.hk/v1/transport/kmb"


def fetch():
    return {name: get_json(f"{BASE_URL}/{path}")["data"] for name, path in
            (("routes", "route"), ("stops", "stop"), ("route_stops", "route-stop"))}
