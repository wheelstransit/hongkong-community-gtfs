import json
from pathlib import Path

from bs4 import BeautifulSoup
from pyproj import Transformer

from ..http import get, get_json

ROUTES_AND_STOPS_URL = "https://opendata.mtr.com.hk/data/light_rail_routes_and_stops.csv"
SCHEDULE_URL = "https://www.mtr.com.hk/en/customer/services/schedule_index.html"
HKBUS_STOPS_URL = "https://raw.githubusercontent.com/hkbus/hk-bus-crawling/gh-pages/routeFareList.min.json"
BUNDLED_SCHEDULE = Path(__file__).resolve().parents[1] / "data" / "light_rail_schedule.json"
GEOCODER_URL = "https://www.map.gov.hk/gs/api/v1.0.0/locationSearch?q="
_hk1980 = Transformer.from_crs("epsg:2326", "epsg:4326")


def geocode(query: str):
    results = get_json(GEOCODER_URL + query, headers={"Accept": "application/json", "User-Agent": ""}, timeout=30)
    if not results:
        return None
    return _hk1980.transform(results[0]["y"], results[0]["x"])


def parse_light_rail_schedule(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')
    schedule_data = {}

    for schedule_box in soup.find_all('div', class_='scheduleBox'):
        route_anchor = schedule_box.find('a', class_='list_btn')
        if not route_anchor:
            continue

        route_title = route_anchor.get('title', '').strip()
        if not route_title:
            continue
        schedule_data[route_title] = {}

        table = schedule_box.find('table', class_='table-a')
        if not table:
            continue

        headers = [th.get_text(separator=' ', strip=True) for th in table.find_all('th')]
        tbody = table.find('tbody')
        if not tbody:
            continue

        rows = tbody.find_all('tr')
        if not rows:
            continue

        # Bi-directional routes have "To" in the second header column.
        if len(headers) > 1 and "To" in headers[1]:
            header_row = table.find('tr')
            if not header_row:
                continue
            th_elements = header_row.find_all('th')
            if len(th_elements) < 3:
                continue
            dest_spans = th_elements[1].find_all('span', class_='sdleHead')
            if len(dest_spans) < 2:
                continue

            direction1_name = dest_spans[0].get_text(strip=True)
            direction2_name = dest_spans[1].get_text(strip=True)
            schedule_data[route_title][direction1_name] = []
            schedule_data[route_title][direction2_name] = []

            for row in rows[1:]:
                cells = row.find_all('td')
                if len(cells) < 5:
                    continue
                stop_name = cells[0].get_text(strip=True)

                first_train_d1 = cells[1].get_text(strip=True)
                last_train_d1 = cells[3].get_text(strip=True)
                if first_train_d1 and first_train_d1 != '-':
                    schedule_data[route_title][direction1_name].append({
                        "stop": stop_name,
                        "first_train": first_train_d1,
                        "last_train": last_train_d1
                    })

                first_train_d2 = cells[2].get_text(strip=True)
                last_train_d2 = cells[4].get_text(strip=True)
                if first_train_d2 and first_train_d2 != '-':
                    schedule_data[route_title][direction2_name].append({
                        "stop": stop_name,
                        "first_train": first_train_d2,
                        "last_train": last_train_d2
                    })
        else:
            # Circular routes: one set of times per stop.
            schedule_data[route_title]["circular"] = []
            for row in rows[1:]:
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue
                stop_name = cells[0].get_text(strip=True)
                first_train = cells[1].get_text(strip=True)
                last_train = cells[2].get_text(strip=True)
                if first_train and first_train != '-':
                    schedule_data[route_title]["circular"].append({
                        "stop": stop_name,
                        "first_train": first_train,
                        "last_train": last_train
                    })

    return schedule_data


def fetch_schedule():
    try:
        schedule = parse_light_rail_schedule(get(SCHEDULE_URL, timeout=30).text)
    except Exception:
        schedule = {}
    return schedule or json.loads(BUNDLED_SCHEDULE.read_text(encoding="utf-8"))


def fetch_stops(routes_and_stops: list[dict]):
    stops, hkbus = {}, None
    for row in routes_and_stops:
        stop_id = "LR" + row["Stop ID"]
        if stop_id in stops:
            continue
        name_tc, name_en = row["Chinese Name"], row["English Name"]
        try:
            location = geocode(f"{name_tc}輕鐵站")
        except Exception:
            location = None
        if location is None:
            if hkbus is None:
                hkbus = get_json(HKBUS_STOPS_URL).get("stopList", {})
            loc = (hkbus.get(stop_id) or {}).get("location") or {}
            if loc.get("lat") is None or loc.get("lng") is None:
                continue
            location = float(loc["lat"]), float(loc["lng"])
        stops[stop_id] = {"stop_id": stop_id, "name_en": name_en, "name_tc": name_tc, "lat": location[0], "lon": location[1]}
    return stops
