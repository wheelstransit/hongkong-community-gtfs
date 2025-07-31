import requests
import pandas as pd
import concurrent.futures
from tqdm import tqdm

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

    try:
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching journey time data for weekday {weekday}, hour {hour_str}: {e}")
        return None
    except ValueError:
        print("Error: Invalid JSON response.")
        return None

def worker_fetch_hourly_journey_time_data(task):
    weekday, hour = task
    return weekday, hour, fetch_hourly_journey_time_data(weekday, hour)

def fetch_all_hourly_journey_time_data_threaded(max_workers=20, silent=False):
    if not silent:
        print(f"Fetching all hourly journey time data with up to {max_workers} threads...")

    # Create tasks for all weekday-hour combinations
    tasks = []
    for weekday in range(7):
        for hour in range(24):
            tasks.append((weekday, hour))

    all_data = {}

    # Initialize the nested dictionary structure
    for weekday in range(7):
        all_data[weekday] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(worker_fetch_hourly_journey_time_data, tasks)

        results_iterator = tqdm(results_iterator, total=len(tasks), desc="Fetching hourly journey time data")

        for weekday, hour, data in results_iterator:
            if data is not None:
                all_data[weekday][hour] = data

    # Count successful fetches
    total_successful = sum(len(weekday_data) for weekday_data in all_data.values())
    print(f"\nSuccessfully fetched data for {total_successful} out of {len(tasks)} weekday-hour combinations.")
    return all_data

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

    print("Testing threaded hourly journey time data fetching (first 5 combinations)...")
    all_hourly_data = fetch_all_hourly_journey_time_data_threaded(max_workers=5)
    if all_hourly_data:
        print("Threaded hourly journey time data fetched successfully.")
        # Check how many weekdays and hours we got data for
        for weekday in range(min(2, 7)):  # Just check first 2 weekdays
            hour_count = len(all_hourly_data.get(weekday, {}))
            print(f"Weekday {weekday}: {hour_count} hours of data")
    print("-" * 20)

    print("done testing :)")
