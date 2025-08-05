import requests
from bs4 import BeautifulSoup
import json
import logging

logging.basicConfig(level=logging.INFO)

def scrape_train_frequency(silent=False):
    url = "https://www.mtr.com.hk/en/customer/services/train_service_index.html"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if not silent:
            logging.error(f"error fetching url: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    if not silent:
        logging.info("html content fetched. parsing table...")

    table = soup.find('table', class_='table-d table-7col table-topline pulldownTable')
    if not table:
        if not silent:
            logging.error("could not find the train services table with the specified class.")
        return []

    header_row = table.find('tr', class_='th-bottom')
    if not header_row:
        if not silent:
            logging.error("could not find the header row in the table.")
        return []
    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]

    cleaned_headers = []
    for h in headers:
        header_text = h.replace(' ', '').replace('', '').replace('(based on minutes)', '').replace('*', '').strip()
        cleaned_headers.append(header_text)

    structured_data = {}

    rows = table.find_all('tr')
    if not silent:
        logging.info(f"found {len(rows)} rows in the table.")

    header_keys = [h.replace('\n', '').replace('\r', '').replace('(based on minutes)', '').replace('*', '').strip() for h in headers]

    for i, row in enumerate(rows):
        cells = row.find_all('td')
        if len(cells) == len(header_keys):
            line_name = cells[0].get_text(strip=True).replace(' ', ' ').rstrip('~#')
            frequencies = {
                header_keys[j]: cells[j].get_text(strip=True).replace(' ', ' ')
                for j in range(1, len(header_keys))
            }
            if not silent and i < 5:
                logging.info(f"row {i}: line: {line_name}, frequencies: {frequencies}")
            structured_data[line_name] = {
                'weekdays': {
                    'morning_peak': frequencies.get(header_keys[1], ''),
                    'evening_peak': frequencies.get(header_keys[2], ''),
                    'non_peak': frequencies.get(header_keys[3], '')
                },
                'saturdays': frequencies.get(header_keys[4], ''),
                'sundays_and_holidays': frequencies.get(header_keys[5], '')
            }

    return structured_data

if __name__ == "__main__":
    frequency_data = scrape_train_frequency()
    if frequency_data:
        print("successfully scraped mtr train frequency data:")
        print(json.dumps(frequency_data, indent=2))
    else:
        print("could not scrape the data.")
