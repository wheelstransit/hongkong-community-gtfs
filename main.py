from src.common.database import get_db_engine
from src.ingest import kmb_client, citybus_client, gov_gtfs_client,gov_csdi_client,gmb_client, mtrbus_client, nlb_client, journey_time_client
from src.processing.load_raw_data import (
    process_and_load_kmb_data,
    process_and_load_gmb_data,
    process_and_load_mtrbus_data,
    process_and_load_citybus_data,
    process_and_load_nlb_data
)

def main():
    print("Starting")
    engine = get_db_engine()

    # INGEST
    # KMB
    raw_kmb_routes = kmb_client.fetch_all_routes()
    raw_kmb_stops = kmb_client.fetch_all_stops()
    raw_kmb_route_stops = kmb_client.fetch_all_route_stops()

    # Government GTFS
    raw_gov_frequencies = gov_gtfs_client.fetch_frequencies_data()
    raw_gov_trips = gov_gtfs_client.fetch_trips_data()
    raw_gov_routes = gov_gtfs_client.fetch_routes_data()
    raw_gov_calendar = gov_gtfs_client.fetch_calendar_data()
    raw_gov_fares = gov_gtfs_client.fetch_fare_data()

    # GMB
    gmb_client_instance = gmb_client.GMBClient()
    raw_gmb_routes = gmb_client_instance.get_all_routes()
    raw_gmb_stops, raw_gmb_route_stops = gmb_client_instance.get_all_stops_and_route_stops()

    # MTR Bus
    raw_mtrbus_routes = mtrbus_client.fetch_all_routes()
    raw_mtrbus_stops = mtrbus_client.fetch_all_stops()
    raw_mtrbus_route_stops = mtrbus_client.fetch_all_route_stops()
    raw_mtrbus_fares = mtrbus_client.fetch_all_fares()

    # Citybus
    raw_citybus_routes = citybus_client.fetch_all_routes()
    raw_citybus_stop_ids = citybus_client.fetch_all_stops_threaded()
    raw_citybus_stop_details = [citybus_client.fetch_stop_details(stop_id) for stop_id in raw_citybus_stop_ids] if raw_citybus_stop_ids else []
    raw_citybus_route_stops = []  # This would need to be fetched separately if needed

    # NLB
    raw_nlb_routes = nlb_client.fetch_all_routes()
    raw_nlb_stops, raw_nlb_route_stops = nlb_client.fetch_all_stops_and_route_stops_threaded()

    # CSDI
    raw_CSDI_data = gov_csdi_client.fetch_bus_routes_data()

    # Journey Time
    raw_journey_time_data = journey_time_client.fetch_all_journey_time_data()
    raw_hourly_journey_time_data = journey_time_client.fetch_all_hourly_journey_time_data()

    # PROCESS
    print("processing data")

    process_and_load_kmb_data(
        raw_routes=raw_kmb_routes,
        raw_stops=raw_kmb_stops,
        raw_route_stops=raw_kmb_route_stops,
        engine=engine
    )

    process_and_load_gmb_data(
        raw_routes=raw_gmb_routes,
        raw_stops=raw_gmb_stops,
        raw_route_stops=raw_gmb_route_stops,
        engine=engine
    )

    process_and_load_mtrbus_data(
        raw_routes=raw_mtrbus_routes,
        raw_stops=raw_mtrbus_stops,
        raw_route_stops=raw_mtrbus_route_stops,
        raw_fares=raw_mtrbus_fares,
        engine=engine
    )

    process_and_load_citybus_data(
        raw_routes=raw_citybus_routes,
        raw_stops=raw_citybus_route_stops,
        raw_stop_details=raw_citybus_stop_details,
        engine=engine
    )

    process_and_load_nlb_data(
        raw_routes=raw_nlb_routes,
        raw_stops=raw_nlb_stops,
        raw_route_stops=raw_nlb_route_stops,
        engine=engine
    )

    print("processing complete")


if __name__ == "__main__":
    main()
