import requests
from bs4 import BeautifulSoup
import json

def scrape_train_frequency():
    """
    Scrapes the Average Train Frequency data from the MTR website.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              train line and its service frequency at different times.
              Returns an empty list if scraping fails.
    """
    url = "https://www.mtr.com.hk/en/customer/services/train_service_index.html"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', class_='table-d table-7col table-topline pulldownTable')
    if not table:
        print("Could not find the train services table.")
        return []

    # Extract headers from the <thead> section for accuracy
    header_row = table.find('tr', class_='th-bottom')
    if not header_row:
        print("Could not find the header row in the table.")
        return []
    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]

    # Clean up header names to be used as dictionary keys
    cleaned_headers = []
    for h in headers:
        # Splits the header text and removes "(minutes)" etc.
        header_text = h.split("(minutes)")[0].strip().replace('\r\n', ' / ')
        cleaned_headers.append(header_text)

    structured_data = {}

    # Find all table body rows
    rows = table.find_all('tr')[1:]  # Skip header row
    for row in rows:
        cells = row.find_all('td')
        if len(cells) == len(cleaned_headers):
            line_name = cells[0].get_text(strip=True).replace('\u00a0', ' ').rstrip('~#')
            frequencies = {
                cleaned_headers[i]: cells[i].get_text(strip=True).replace('\u00a0', ' ')
                for i in range(1, len(cleaned_headers))
            }
            structured_data[line_name] = {
                'weekdays': {
                    'morning_peak': frequencies.get('WeekdaysMorning Peak Hours', ''),
                    'evening_peak': frequencies.get('WeekdaysEvening Peak Hours', ''),
                    'non_peak': frequencies.get('WeekdaysNon-peak Hours', '')
                },
                'saturdays': frequencies.get('Saturdays', ''),
                'sundays_and_holidays': frequencies.get('Sundays and Public Holidays', '')
            }

    return structured_data

if __name__ == "__main__":
    frequency_data = scrape_train_frequency()
    if frequency_data:
        print("Successfully scraped MTR train frequency data:")
        # Pretty print the JSON data for readability
        print(frequency_data)
    else:
        print("Could not scrape the data.")
