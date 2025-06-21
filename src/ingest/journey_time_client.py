import requests
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/HK-Bus-ETA/hk-bus-time-between-stops/pages"

def fetch_all_journey_time_data():
    endpoint = f"{BASE_URL}/times/all.json"
    print("Fetching all journey time data...")

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data is not None:
            print("Successfully fetched all journey time data.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching journey time data: {e}")
        return None
    except ValueError:
        print("Error: Invalid JSON response.")
        return None

def fetch_hourly_journey_time_data(weekday, hour):
    hour_str = f"{hour:02d}"
    endpoint = f"{BASE_URL}/times_hourly/{weekday}/{hour_str}/all.json"
    print(f"Fetching journey time data for weekday {weekday}, hour {hour_str}...")

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data is not None:
            print(f"Successfully fetched journey time data for weekday {weekday}, hour {hour_str}.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching journey time data for weekday {weekday}, hour {hour_str}: {e}")
        return None
    except ValueError:
        print("Error: Invalid JSON response.")
        return None

def fetch_all_hourly_journey_time_data():
    all_data = {}
    print("Fetching all hourly journey time data...")

    for weekday in range(7):
        weekday_data = {}
        for hour in range(24):
            data = fetch_hourly_journey_time_data(weekday, hour)
            if data is not None:
                weekday_data[hour] = data
        all_data[weekday] = weekday_data

    print("Finished fetching all hourly journey time data.")
    return all_data


if __name__ == '__main__':
    print("testing :)")

    all_data = fetch_all_journey_time_data()
    if all_data:
        print("Sample all journey time data structure fetched successfully.")
    print("-" * 20)

    hourly_data = fetch_hourly_journey_time_data(0, 8)
    if hourly_data:
        print("Sample hourly journey time data for Monday 8AM fetched successfully.")
    print("-" * 20)

    print("done testing :)")
