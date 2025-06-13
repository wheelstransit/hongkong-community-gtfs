from src.common.database import get_db_engine
from src.ingest import kmb_client, citybus_client, gov_gtfs_client
from src.processing.load_raw_data import process_and_load_kmb_data, process_and_load_gov_data

def main():
    print("--- Starting Data Pipeline ---")
    engine = get_db_engine()

    # INGEST
    print("\n--- Starting Ingest Phase ---")
    raw_kmb_routes = kmb_client.fetch_all_routes()
    raw_kmb_stops = kmb_client.fetch_all_stops()
    raw_kmb_route_stops = kmb_client.fetch_all_route_stops()
    raw_gov_frequencies = gov_gtfs_client.fetch_frequencies_data()
    raw_gov_trips = gov_gtfs_client.fetch_trips_data()
    raw_gov_routes = gov_gtfs_client.fetch_routes_data()
    raw_gov_calendar = gov_gtfs_client.fetch_calendar_data()
    raw_gov_fares = gov_gtfs_client.fetch_fare_data()
    print("--- Ingest Phase Complete ---\n")

    # PROCESS
    print("--- Starting Process & Load (Staging) Phase ---")
    process_and_load_kmb_data(
        raw_routes=raw_kmb_routes,
        raw_stops=raw_kmb_stops,
        raw_route_stops=raw_kmb_route_stops,
        engine=engine
    )

    print("--- Process & Load (Staging) Phase Complete ---\n")
    

if __name__ == "__main__":
    main()