from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ..http import get_json

BASE_URL = "https://rt.data.gov.hk/v2/transport/citybus"


def fetch():
    routes = get_json(f"{BASE_URL}/route/CTB")["data"]
    tasks = [(r["route"], d) for r in routes if r.get("route") for d in ("inbound", "outbound")]
    with ThreadPoolExecutor(20) as pool:
        results = list(tqdm(pool.map(lambda t: get_json(f"{BASE_URL}/route-stop/CTB/{t[0]}/{t[1]}")["data"], tasks),
                            total=len(tasks), desc="citybus route-stops", leave=False))
    sequences = []
    stop_ids = set()
    for (route_id, direction), stops in zip(tasks, results):
        if stops:
            ids = [s["stop"] for s in stops if s and "stop" in s]
            sequences.append({"route_id": route_id, "direction": direction, "stop_ids": ids})
            stop_ids.update(ids)
    stop_ids = sorted(stop_ids)
    with ThreadPoolExecutor(20) as pool:
        details = list(tqdm(pool.map(lambda s: get_json(f"{BASE_URL}/stop/{s}")["data"], stop_ids),
                            total=len(stop_ids), desc="citybus stops", leave=False))
    return {"routes": routes, "stop_details": [d for d in details if d], "route_sequences": sequences}
