from ..http import get_csv


def fetch():
    routes = get_csv("https://opendata.mtr.com.hk/data/mtr_bus_routes.csv", unescape=True)
    route_stops = get_csv("https://opendata.mtr.com.hk/data/mtr_bus_stops.csv", unescape=True)
    stops = {}
    for row in route_stops:
        station_id = row.get("STATION_ID")
        if station_id and station_id not in stops:
            stops[station_id] = {
                "stop_id": station_id,
                "name_en": row.get("STATION_NAME_ENG"),
                "name_zh": row.get("STATION_NAME_CHI"),
                "lat": row.get("STATION_LATITUDE"),
                "long": row.get("STATION_LONGITUDE"),
            }
    return {"routes": routes, "stops": list(stops.values()), "route_stops": route_stops}
