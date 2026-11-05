import re

from bs4 import BeautifulSoup

from ..http import get, get_csv, get_json
from . import light_rail

OPENDATA = "https://opendata.mtr.com.hk/data"
CRAWLER = "https://raw.githubusercontent.com/wheelstransit/mtr-platform-exits-crawler/refs/heads/main/data/output"
HEADWAY_URL = "https://www.mtr.com.hk/en/customer/services/train_service_index.html"


def fetch():
    crawler = get_json(f"{CRAWLER}/mtr_data_complete.json")
    stations_csv = get_csv(f"{OPENDATA}/mtr_lines_and_stations.csv")
    lr_routes_and_stops = get_csv(f"{OPENDATA}/light_rail_routes_and_stops.csv")
    return {
        "crawler": crawler,
        "journeys": get_json(f"{CRAWLER}/journeys.json", timeout=100),
        "platform_journeys": get_json(f"{CRAWLER}/platform_journeys.json"),
        "lines_and_stations": lines_and_stations(stations_csv, crawler),
        "lines_fares": get_csv(f"{OPENDATA}/mtr_lines_fares.csv"),
        "light_rail_routes_and_stops": lr_routes_and_stops,
        "light_rail_fares": get_csv(f"{OPENDATA}/light_rail_fares.csv"),
        "light_rail_stops": light_rail.fetch_stops(lr_routes_and_stops),
        "light_rail_schedule": light_rail.fetch_schedule(),
        "headway": headway(),
        "exits": exits(stations_csv, crawler),
    }


def lines_and_stations(rows, crawler):
    stations = [row for row in rows if row.get("Line Code") and row.get("Station ID")]
    known = {
        code: {"latitude": float(loc["lat"]), "longitude": float(loc["lon"])}
        for code, rec in (crawler.get("stations") or {}).items()
        if (loc := rec.get("location") or {}).get("lat") is not None and loc.get("lon") is not None
    }
    locations = {}
    for row in {row["Station ID"]: row for row in stations}.values():
        location = known.get(row["Station Code"])
        if location is None:
            try:
                lat_lon = light_rail.geocode(f"港鐵{row['Chinese Name']}站")
            except Exception:
                lat_lon = None
            location = lat_lon and {"latitude": lat_lon[0], "longitude": lat_lon[1]}
        if location:
            locations[row["Station ID"]] = location
    for row in stations:
        row.update(locations.get(row["Station ID"]) or {"latitude": None, "longitude": None})
    return stations


def headway():
    table = BeautifulSoup(get(HEADWAY_URL).content, "html.parser").find("table", class_="table-d table-7col table-topline pulldownTable")
    if not table or not (header_row := table.find("tr", class_="th-bottom")):
        return {}
    keys = [th.get_text(strip=True).replace("\n", "").replace("\r", "").replace("(based on minutes)", "").replace("*", "").strip()
            for th in header_row.find_all("th")]
    data = {}
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != len(keys):
            continue
        values = {keys[j]: cells[j].get_text(strip=True) for j in range(1, len(keys))}
        data[cells[0].get_text(strip=True).rstrip("~#")] = {
            "weekdays": {"morning_peak": values.get(keys[1], ""), "evening_peak": values.get(keys[2], ""), "non_peak": values.get(keys[3], "")},
            "saturdays": values.get(keys[4], ""),
            "sundays_and_holidays": values.get(keys[5], ""),
        }
    return data


def exits(stations_csv, crawler):
    stops = {row["Station ID"]: {"station_code": row["Station Code"], "name_tc": row["Chinese Name"], "name_en": row["English Name"]}
             for row in stations_csv}
    for row in get_csv(f"{OPENDATA}/barrier_free_facilities.csv"):
        values = list(row.values())
        if len(values) > 3 and values[2] == "Y" and values[3] and values[0] in stops:
            for exit_code in re.findall(r"[A-Z][0-9]*", values[3]):
                stops[values[0]][exit_code.strip()] = True
    venue_to_code = {s["venue_id"]: code for code, s in (crawler.get("stations") or {}).items() if s.get("venue_id")}
    code_to_stop = {stop["station_code"]: stop for stop in stops.values()}
    results = {}
    for exit_rec in (crawler.get("exits") or {}).values():
        code = venue_to_code.get(exit_rec.get("venue_id"))
        stop = code_to_stop.get(code)
        coords = exit_rec.get("coordinates") or {}
        exit_code = (exit_rec.get("name_en") or "").strip().replace("Exit", "").strip()
        if not stop or not exit_code or coords.get("lat") is None or coords.get("lon") is None:
            continue
        results[stop["name_tc"] + exit_code] = {
            "station_code": code, "station_name_en": stop["name_en"], "station_name_zh": stop["name_tc"],
            "exit": exit_code, "lat": coords["lat"], "lon": coords["lon"], "barrier_free": exit_code in stop,
        }
    return list(results.values())
