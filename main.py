from src.common.database import get_db_engine
from src.ingest import kmb_client, citybus_client, gov_gtfs_client,gov_csdi_client,gmb_client, mtrbus_client, nlb_client, journey_time_client, osm_parser
from src.processing.load_raw_data import (
    process_and_load_kmb_data,
    process_and_load_gmb_data,
    process_and_load_mtrbus_data,
    process_and_load_citybus_data,
    process_and_load_nlb_data,
    process_and_load_gov_gtfs_data,
    process_and_load_csdi_data,
    process_and_load_journey_time_data
)

def main():
    print("Starting")
    engine = get_db_engine()

    # INGEST
    # KMB
    raw_kmb_routes = kmb_client.fetch_all_routes()
    raw_kmb_stops = kmb_client.fetch_all_stops()
    raw_kmb_route_stops = kmb_client.fetch_all_route_stops()
    print(f"KMB data - Routes: {len(raw_kmb_routes) if raw_kmb_routes else 0}, Stops: {len(raw_kmb_stops) if raw_kmb_stops else 0}, Route-stops: {len(raw_kmb_route_stops) if raw_kmb_route_stops else 0}")

    # Government GTFS
    raw_gov_frequencies = gov_gtfs_client.fetch_frequencies_data()
    raw_gov_trips = gov_gtfs_client.fetch_trips_data()
    raw_gov_routes = gov_gtfs_client.fetch_routes_data()
    raw_gov_calendar = gov_gtfs_client.fetch_calendar_data()
    raw_gov_fares = gov_gtfs_client.fetch_fare_data()
    print(f"Gov GTFS data - Frequencies: {len(raw_gov_frequencies) if raw_gov_frequencies else 0}, Trips: {len(raw_gov_trips) if raw_gov_trips else 0}, Routes: {len(raw_gov_routes) if raw_gov_routes else 0}, Calendar: {len(raw_gov_calendar) if raw_gov_calendar else 0}, Fares: {type(raw_gov_fares)} with {len(raw_gov_fares) if raw_gov_fares else 0} keys")

    # GMB
    gmb_client_instance = gmb_client.GMBClient()
    raw_gmb_routes = gmb_client_instance.get_all_routes()
    raw_gmb_stops, raw_gmb_route_stops = gmb_client_instance.get_all_stops_and_route_stops()
    print(f"GMB data - Routes: {len(raw_gmb_routes) if raw_gmb_routes else 0}, Stops: {len(raw_gmb_stops) if raw_gmb_stops else 0}, Route-stops: {len(raw_gmb_route_stops) if raw_gmb_route_stops else 0}")

    # MTR Bus
    raw_mtrbus_routes = mtrbus_client.fetch_all_routes()
    raw_mtrbus_stops = mtrbus_client.fetch_all_stops()
    raw_mtrbus_route_stops = mtrbus_client.fetch_all_route_stops()
    raw_mtrbus_fares = mtrbus_client.fetch_all_fares()
    print(f"MTR Bus data - Routes: {len(raw_mtrbus_routes) if raw_mtrbus_routes else 0}, Stops: {len(raw_mtrbus_stops) if raw_mtrbus_stops else 0}, Route-stops: {len(raw_mtrbus_route_stops) if raw_mtrbus_route_stops else 0}, Fares: {len(raw_mtrbus_fares) if raw_mtrbus_fares else 0}")

    # Citybus
    raw_citybus_routes = citybus_client.fetch_all_routes()
    raw_citybus_stop_ids = citybus_client.fetch_all_stops_threaded()
    raw_citybus_stop_details = citybus_client.fetch_all_stop_details_threaded(raw_citybus_stop_ids) if raw_citybus_stop_ids else []
    raw_citybus_route_stops = []  # This would need to be fetched separately if needed
    print(f"Citybus data - Routes: {len(raw_citybus_routes) if raw_citybus_routes else 0}, Stop IDs: {len(raw_citybus_stop_ids) if raw_citybus_stop_ids else 0}, Stop details: {len(raw_citybus_stop_details) if raw_citybus_stop_details else 0}, Route-stops: {len(raw_citybus_route_stops)}")

    # NLB
    raw_nlb_routes = nlb_client.fetch_all_routes()
    raw_nlb_stops, raw_nlb_route_stops = nlb_client.fetch_all_stops_and_route_stops_threaded()
    print(f"NLB data - Routes: {len(raw_nlb_routes) if raw_nlb_routes else 0}, Stops: {len(raw_nlb_stops) if raw_nlb_stops else 0}, Route-stops: {len(raw_nlb_route_stops) if raw_nlb_route_stops else 0}")

    # CSDI
    raw_CSDI_data = gov_csdi_client.fetch_bus_routes_data()
    print(f"CSDI data - Records: {len(raw_CSDI_data) if raw_CSDI_data else 0}")

    # Journey Time
    raw_journey_time_data = journey_time_client.fetch_all_journey_time_data()
    raw_hourly_journey_time_data = journey_time_client.fetch_all_hourly_journey_time_data_threaded()
    print(f"Journey Time data - Basic: {len(raw_journey_time_data) if raw_journey_time_data else 0} records, Hourly: {len(raw_hourly_journey_time_data) if raw_hourly_journey_time_data else 0} records")

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

    process_and_load_gov_gtfs_data(
        raw_frequencies=raw_gov_frequencies,
        raw_trips=raw_gov_trips,
        raw_routes=raw_gov_routes,
        raw_calendar=raw_gov_calendar,
        raw_fares=raw_gov_fares,
        engine=engine
    )

    process_and_load_csdi_data(
        raw_csdi_data=raw_CSDI_data,
        engine=engine
    )

    process_and_load_journey_time_data(
        raw_journey_time_data=raw_journey_time_data,
        raw_hourly_journey_time_data=raw_hourly_journey_time_data,
        engine=engine
    )

    print("processing complete")


if __name__ == "__main__":
    main()
