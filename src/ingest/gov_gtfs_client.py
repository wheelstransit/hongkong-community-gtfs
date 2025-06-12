import pandas as pd

FREQUENCIES_URL = "https://static.data.gov.hk/td/pt-headway-en/frequencies.txt"
FARE_ATTRIBUTES_URL = "https://static.data.gov.hk/td/pt-headway-en/fare_attributes.txt"
FARE_RULES_URL = "https://static.data.gov.hk/td/pt-headway-en/fare_rules.txt"
CALENDAR_URL = "https://static.data.gov.hk/td/pt-headway-en/calendar.txt"
TRIPS_URL = "https://static.data.gov.hk/td/pt-headway-en/trips.txt"
ROUTES_URL = "https://static.data.gov.hk/td/pt-headway-en/routes.txt"

def fetch_frequencies_data():
    """Fetches the frequencies (headway) data from frequencies.txt."""
    print("Fetching frequencies data from Gov GTFS source...")
    try:
        df = pd.read_csv(FREQUENCIES_URL)
        print(f"Successfully fetched {len(df)} frequency records.")
        return df.to_dict('records')
    except Exception as e:
        print(f"Error fetching frequencies data: {e}")
        return None

def fetch_trips_data():
    """Fetches trips data which contains the link between trip_id and route_id."""
    print("Fetching trips data from Gov GTFS source...")
    try:
        df = pd.read_csv(TRIPS_URL)
        print(f"Successfully fetched {len(df)} trip records.")
        return df.to_dict('records')
    except Exception as e:
        print(f"Error fetching trips data: {e}")
        return None

def fetch_routes_data():
    """Fetches routes data which contains the link between route_id and agency_id."""
    print("Fetching routes data from Gov GTFS source...")
    try:
        df = pd.read_csv(ROUTES_URL)
        print(f"Successfully fetched {len(df)} route records.")
        return df.to_dict('records')
    except Exception as e:
        print(f"Error fetching routes data: {e}")
        return None

def fetch_fare_data():
    """Fetches both fare_attributes and fare_rules data."""
    print("Fetching fare data from Gov GTFS source...")
    try:
        fare_attributes_df = pd.read_csv(FARE_ATTRIBUTES_URL)
        print(f"Successfully fetched {len(fare_attributes_df)} fare attribute records.")

        fare_rules_df = pd.read_csv(FARE_RULES_URL)
        print(f"Successfully fetched {len(fare_rules_df)} fare rule records.")

        return {
            'attributes': fare_attributes_df.to_dict('records'),
            'rules': fare_rules_df.to_dict('records')
        }
    except Exception as e:
        print(f"Error fetching fare data: {e}")
        return None

def fetch_calendar_data():
    """Fetches the calendar data."""
    print("Fetching calendar data from Gov GTFS source...")
    try:
        df = pd.read_csv(CALENDAR_URL)
        print(f"Successfully fetched {len(df)} calendar records.")
        return df.to_dict('records')
    except Exception as e:
        print(f"Error fetching calendar data: {e}")
        return None

if __name__ == '__main__':
    print("testing :)")

    headway_data = fetch_frequencies_data()
    if headway_data:
        print("Sample headway data:")
        print(headway_data[0])
    print("-" * 20)

    trips_data = fetch_trips_data()
    if trips_data:
        print("Sample trips data:")
        print(trips_data[0])
    print("-" * 20)

    routes_data = fetch_routes_data()
    if routes_data:
        print("Sample routes data:")
        print(routes_data[0])
    print("-" * 20)

    fare_data = fetch_fare_data()
    if fare_data:
        print("Sample fare attributes data:")
        print(fare_data['attributes'][0] if fare_data['attributes'] else "No fare attributes")
        print("Sample fare rules data:")
        print(fare_data['rules'][0] if fare_data['rules'] else "No fare rules")
    print("-" * 20)

    calendar_data = fetch_calendar_data()
    if calendar_data:
        print("Sample calendar data:")
        print(calendar_data[0])
    print("-" * 20)

    print("done testing :)")