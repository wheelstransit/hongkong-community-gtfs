import argparse
import os
import pickle
from src.common.database import get_db_engine
from src.ingest import kmb_client, citybus_client, gov_gtfs_client,gov_csdi_client,gmb_client, mtrbus_client, nlb_client, journey_time_client, osm_parser, mtr_rails_client
from src.processing.load_raw_data import (
    process_and_load_kmb_data,
    process_and_load_gmb_data,
    process_and_load_mtrbus_data,
    process_and_load_citybus_data,
    process_and_load_nlb_data,
    process_and_load_gov_gtfs_data,
    process_and_load_csdi_data,
    process_and_load_journey_time_data,
    process_and_load_osm_data,
    process_and_load_mtr_rails_data
)
from src.export.export_gtfs import export_unified_feed

CACHE_DIR = ".cache"

def fetch_or_load_from_cache(cache_key, fetch_func, force_ingest=False, *args, **kwargs):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if not force_ingest and os.path.exists(cache_file):
        print(f"Loading {cache_key} from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Fetching {cache_key} from source...")
    data = fetch_func(*args, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data

def main():
    parser = argparse.ArgumentParser(description="Hong Kong Community GTFS Data Pipeline")
    parser.add_argument('--force-ingest', action='store_true', help='Force re-ingestion of data, ignoring cache.')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode, suppressing progress bars.')
    args = parser.parse_args()

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not args.silent:
        print("Starting")
    engine = get_db_engine()

    # INGEST
    # KMB
    raw_kmb_routes = fetch_or_load_from_cache("kmb_routes", kmb_client.fetch_all_routes, args.force_ingest, silent=args.silent)
    raw_kmb_stops = fetch_or_load_from_cache("kmb_stops", kmb_client.fetch_all_stops, args.force_ingest, silent=args.silent)
    raw_kmb_route_stops = fetch_or_load_from_cache("kmb_route_stops", kmb_client.fetch_all_route_stops, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"KMB data - Routes: {len(raw_kmb_routes) if raw_kmb_routes else 0}, Stops: {len(raw_kmb_stops) if raw_kmb_stops else 0}, Route-stops: {len(raw_kmb_route_stops) if raw_kmb_route_stops else 0}")

    # Government GTFS
    raw_gov_frequencies = fetch_or_load_from_cache("gov_frequencies", gov_gtfs_client.fetch_frequencies_data, args.force_ingest)
    raw_gov_trips = fetch_or_load_from_cache("gov_trips", gov_gtfs_client.fetch_trips_data, args.force_ingest)
    raw_gov_routes = fetch_or_load_from_cache("gov_routes", gov_gtfs_client.fetch_routes_data, args.force_ingest)
    raw_gov_calendar = fetch_or_load_from_cache("gov_calendar", gov_gtfs_client.fetch_calendar_data, args.force_ingest)
    raw_gov_fares = fetch_or_load_from_cache("gov_fares", gov_gtfs_client.fetch_fare_data, args.force_ingest)
    print(f"Gov GTFS data - Frequencies: {len(raw_gov_frequencies) if raw_gov_frequencies else 0}, Trips: {len(raw_gov_trips) if raw_gov_trips else 0}, Routes: {len(raw_gov_routes) if raw_gov_routes else 0}, Calendar: {len(raw_gov_calendar) if raw_gov_calendar else 0}, Fares: {type(raw_gov_fares)} with {len(raw_gov_fares) if raw_gov_fares else 0} keys")

    # GMB
    gmb_client_instance = gmb_client.GMBClient()
    raw_gmb_routes = fetch_or_load_from_cache("gmb_routes", gmb_client_instance.get_all_routes, args.force_ingest)
    gmb_stops_data = fetch_or_load_from_cache("gmb_stops_and_route_stops", gmb_client_instance.get_all_stops_and_route_stops, args.force_ingest)
    raw_gmb_stops, raw_gmb_route_stops = gmb_stops_data if gmb_stops_data else ([], [])
    print(f"GMB data - Routes: {len(raw_gmb_routes) if raw_gmb_routes else 0}, Stops: {len(raw_gmb_stops) if raw_gmb_stops else 0}, Route-stops: {len(raw_gmb_route_stops) if raw_gmb_route_stops else 0}")

    # MTR Bus
    raw_mtrbus_routes = fetch_or_load_from_cache("mtrbus_routes", mtrbus_client.fetch_all_routes, args.force_ingest)
    raw_mtrbus_stops = fetch_or_load_from_cache("mtrbus_stops", mtrbus_client.fetch_all_stops, args.force_ingest)
    raw_mtrbus_route_stops = fetch_or_load_from_cache("mtrbus_route_stops", mtrbus_client.fetch_all_route_stops, args.force_ingest)
    raw_mtrbus_fares = fetch_or_load_from_cache("mtrbus_fares", mtrbus_client.fetch_all_fares, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"MTR Bus data - Routes: {len(raw_mtrbus_routes) if raw_mtrbus_routes else 0}, Stops: {len(raw_mtrbus_stops) if raw_mtrbus_stops else 0}, Route-stops: {len(raw_mtrbus_route_stops) if raw_mtrbus_route_stops else 0}, Fares: {len(raw_mtrbus_fares) if raw_mtrbus_fares else 0}")

    raw_mtr_lines_and_stations = fetch_or_load_from_cache("mtr_lines_and_stations", mtr_rails_client.fetch_mtr_lines_and_stations, args.force_ingest)
    raw_mtr_lines_fares = fetch_or_load_from_cache("mtr_lines_fares", mtr_rails_client.fetch_mtr_lines_fares, args.force_ingest)
    raw_light_rail_routes_and_stops = fetch_or_load_from_cache("light_rail_routes_and_stops", mtr_rails_client.fetch_light_rail_routes_and_stops, args.force_ingest)
    raw_light_rail_fares = fetch_or_load_from_cache("light_rail_fares", mtr_rails_client.fetch_light_rail_fares, args.force_ingest)
    raw_airport_express_fares = fetch_or_load_from_cache("airport_express_fares", mtr_rails_client.fetch_airport_express_fares, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"MTR Rail data - Lines/Stations: {len(raw_mtr_lines_and_stations) if raw_mtr_lines_and_stations else 0}, Fares: {len(raw_mtr_lines_fares) if raw_mtr_lines_fares else 0}")
        print(f"Light Rail data - Routes/Stops: {len(raw_light_rail_routes_and_stops) if raw_light_rail_routes_and_stops else 0}, Fares: {len(raw_light_rail_fares) if raw_light_rail_fares else 0}")
        print(f"Airport Express Fares: {len(raw_airport_express_fares) if raw_airport_express_fares else 0}")



    # Citybus
    raw_citybus_routes = fetch_or_load_from_cache("citybus_routes", citybus_client.fetch_all_routes, args.force_ingest)
    citybus_stops_data = fetch_or_load_from_cache("citybus_stops_and_sequences", citybus_client.fetch_all_stops_threaded, args.force_ingest, raw_citybus_routes)
    raw_citybus_stop_id, raw_citybus_route_sequences = citybus_stops_data if citybus_stops_data else ([], [])
    raw_citybus_stop_details = fetch_or_load_from_cache("citybus_stop_details", citybus_client.fetch_all_stop_details_threaded, args.force_ingest, raw_citybus_stop_id)

    # NLB
    raw_nlb_routes = fetch_or_load_from_cache("nlb_routes", nlb_client.fetch_all_routes, args.force_ingest)
    nlb_stops_data = fetch_or_load_from_cache("nlb_stops_and_route_stops", nlb_client.fetch_all_stops_and_route_stops_threaded, args.force_ingest, raw_nlb_routes)
    raw_nlb_stops, raw_nlb_route_stops = nlb_stops_data if nlb_stops_data else ([], [])
    print(f"NLB data - Routes: {len(raw_nlb_routes) if raw_nlb_routes else 0}, Stops: {len(raw_nlb_stops) if raw_nlb_stops else 0}, Route-stops: {len(raw_nlb_route_stops) if raw_nlb_route_stops else 0}")

    # CSDI
    raw_CSDI_data = fetch_or_load_from_cache("csdi_bus_routes", gov_csdi_client.fetch_bus_routes_data, args.force_ingest)
    print(f"CSDI data - Records: {len(raw_CSDI_data) if raw_CSDI_data else 0}")

    # Journey Time
    raw_journey_time_data = fetch_or_load_from_cache("journey_time", journey_time_client.fetch_all_journey_time_data, args.force_ingest)
    raw_hourly_journey_time_data = fetch_or_load_from_cache("hourly_journey_time", journey_time_client.fetch_all_hourly_journey_time_data_threaded, args.force_ingest)
    print(f"Journey Time data - Basic: {len(raw_journey_time_data) if raw_journey_time_data else 0} records, Hourly: {len(raw_hourly_journey_time_data) if raw_hourly_journey_time_data else 0} records")

    # OSM
    raw_osm_routes = fetch_or_load_from_cache("osm_routes", osm_parser.fetch_osm_routes, args.force_ingest)
    print(f"OSM data - Elements: {len(raw_osm_routes.get('elements', [])) if raw_osm_routes else 0}")

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
        raw_stop_details=raw_citybus_stop_details,
        raw_route_sequences=raw_citybus_route_sequences,
        engine=engine
    )

    process_and_load_nlb_data(
        raw_routes=raw_nlb_routes,
        raw_stops=raw_nlb_stops,
        raw_route_stops=raw_nlb_route_stops,
        engine=engine
    )

    process_and_load_mtr_rails_data(
        raw_mtr_lines_and_stations=raw_mtr_lines_and_stations,
        raw_mtr_lines_fares=raw_mtr_lines_fares,
        raw_light_rail_routes_and_stops=raw_light_rail_routes_and_stops,
        raw_light_rail_fares=raw_light_rail_fares,
        raw_airport_express_fares=raw_airport_express_fares,
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

    process_and_load_osm_data(
        raw_osm_data=raw_osm_routes,
        engine=engine,
        silent=args.silent
    )

    export_unified_feed(
        engine=engine,
        output_dir=os.path.join("output", "gtfs"),
        journey_time_data=raw_journey_time_data
    )

    if not args.silent:
        print("processing complete")


if __name__ == "__main__":
    main()