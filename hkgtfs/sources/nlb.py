from concurrent.futures import ThreadPoolExecutor

from ..http import get_json

BASE_URL = "https://rt.data.gov.hk/v2/transport/nlb"


def fetch():
    routes = get_json(f"{BASE_URL}/route.php?action=list")["routes"]
    with ThreadPoolExecutor(20) as pool:
        results = pool.map(lambda r: get_json(f"{BASE_URL}/stop.php?action=list&routeId={r['routeId']}").get("stops"), routes)
        stops, route_stops = {}, []
        for route, route_stop_list in zip(routes, results):
            for seq, stop in enumerate(route_stop_list or [], start=1):
                stop_id = stop.get("stopId")
                if not stop_id:
                    continue
                stops.setdefault(stop_id, stop)
                route_stops.append({"routeId": route["routeId"], "routeNo": route["routeNo"], "stopId": stop_id, "sequence": seq})
    return {"routes": routes, "stops": list(stops.values()), "route_stops": route_stops}
