import json
from pathlib import Path
from typing import Dict

import httpx
from bs4 import BeautifulSoup

SCHEDULE_URL = "https://www.mtr.com.hk/en/customer/services/schedule_index.html"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("lightrailschedule.json")


def parse_light_rail_schedule(html_content: str) -> Dict:
    """Parse the MTR Light Rail schedule HTML and extract first/last train times."""
    soup = BeautifulSoup(html_content, 'html.parser')
    schedule_data: Dict = {}

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


def fetch_light_rail_schedule_html(url: str = SCHEDULE_URL, timeout: float = 30.0) -> str:
    """Download the Light Rail schedule HTML from the given URL."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        response.encoding = response.encoding or 'utf-8'
        return response.text


def generate_schedule_json(output_path: Path = DEFAULT_OUTPUT_PATH, url: str = SCHEDULE_URL) -> Dict:
    """Fetch, parse, and persist the Light Rail schedule JSON."""
    html_content = fetch_light_rail_schedule_html(url=url)
    parsed_schedule = parse_light_rail_schedule(html_content)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(parsed_schedule, f, ensure_ascii=False, indent=4)

    return parsed_schedule


if __name__ == '__main__':
    try:
        generate_schedule_json()
    except httpx.HTTPError as exc:
        print(f"Error downloading Light Rail schedule: {exc}")
    else:
        print(f"lightrailschedule.json has been created successfully at {DEFAULT_OUTPUT_PATH}.")