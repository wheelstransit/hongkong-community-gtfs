import argparse
import os
import pickle
import asyncio
import inspect
import logging
from src.common.database import get_db_engine
from src.ingest import (
    kmb_client, citybus_client, gov_gtfs_client, gov_csdi_client, gmb_client,
    mtrbus_client, nlb_client, journey_time_client, mtr_rails_client,
    mtr_headway, mtr_exit_client, light_rail_stops_fetcher
)
from src.processing.load_raw_data import (
    process_and_load_kmb_data,
    process_and_load_gmb_data,
    process_and_load_mtrbus_data,
    process_and_load_citybus_data,
    process_and_load_nlb_data,
    process_and_load_gov_gtfs_data,
    process_and_load_csdi_data,
    process_and_load_journey_time_data,
    process_and_load_mtr_exits_data,
    process_and_load_mtr_rails_data,
    process_and_load_light_rail_stops_data
)
from src.export.export_gtfs import export_unified_feed

cache_dir = ".cache"

def fetch_or_load_from_cache(cache_key, fetch_func, force_ingest=False, force_ingest_osm=False, *args, **kwargs):
    if force_ingest_osm and 'osm' in cache_key:
        force_ingest = True

    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    if not force_ingest and os.path.exists(cache_file):
        print(f"loading {cache_key} from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"fetching {cache_key} from source...")
    if inspect.iscoroutinefunction(fetch_func):
        data = asyncio.run(fetch_func(*args, **kwargs))
    else:
        data = fetch_func(*args, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data

def main():
    logging.getLogger('httpx').setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="hong kong community gtfs data pipeline")
    parser.add_argument('--force-ingest', action='store_true', help='force re-ingestion of data, ignoring cache')
    parser.add_argument('--silent', action='store_true', help='run in silent mode, suppressing progress bars')
    parser.add_argument('--force-ingest-osm', action='store_true', help='force re-ingestion of osm data, ignoring cache')
    args = parser.parse_args()

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not args.silent:
        print("starting")
    engine = get_db_engine()

    # ingest
    raw_kmb_routes = fetch_or_load_from_cache("kmb_routes", kmb_client.fetch_all_routes, args.force_ingest, silent=args.silent)
    raw_kmb_stops = fetch_or_load_from_cache("kmb_stops", kmb_client.fetch_all_stops, args.force_ingest, silent=args.silent)
    raw_kmb_route_stops = fetch_or_load_from_cache("kmb_route_stops", kmb_client.fetch_all_route_stops, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"kmb data - routes: {len(raw_kmb_routes) if raw_kmb_routes else 0}, stops: {len(raw_kmb_stops) if raw_kmb_stops else 0}, route-stops: {len(raw_kmb_route_stops) if raw_kmb_route_stops else 0}")

    # gov gtfs
    raw_gov_frequencies = fetch_or_load_from_cache("gov_frequencies", gov_gtfs_client.fetch_frequencies_data, args.force_ingest, silent=args.silent)
    raw_gov_trips = fetch_or_load_from_cache("gov_trips", gov_gtfs_client.fetch_trips_data, args.force_ingest, silent=args.silent)
    raw_gov_routes = fetch_or_load_from_cache("gov_routes", gov_gtfs_client.fetch_routes_data, args.force_ingest, silent=args.silent)
    raw_gov_calendar = fetch_or_load_from_cache("gov_calendar", gov_gtfs_client.fetch_calendar_data, args.force_ingest, silent=args.silent)
    raw_gov_calendar_dates = fetch_or_load_from_cache("gov_calendar_dates", gov_gtfs_client.fetch_calendar_dates_data, args.force_ingest, silent=args.silent)
    raw_gov_fares = fetch_or_load_from_cache("gov_fares", gov_gtfs_client.fetch_fare_data, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"gov gtfs data - frequencies: {len(raw_gov_frequencies) if raw_gov_frequencies else 0}, trips: {len(raw_gov_trips) if raw_gov_trips else 0}, routes: {len(raw_gov_routes) if raw_gov_routes else 0}, calendar: {len(raw_gov_calendar) if raw_gov_calendar else 0}, fares: {type(raw_gov_fares)} with {len(raw_gov_fares) if raw_gov_fares else 0} keys")

    # gmb
    gmb_client_instance = gmb_client.GMBClient(silent=args.silent)
    raw_gmb_routes = fetch_or_load_from_cache("gmb_routes", gmb_client_instance.get_all_routes, args.force_ingest)
    gmb_stops_data = fetch_or_load_from_cache("gmb_stops_and_route_stops", gmb_client_instance.get_all_stops_and_route_stops, args.force_ingest, silent=args.silent)
    raw_gmb_stops, raw_gmb_route_stops = gmb_stops_data if gmb_stops_data else ([], [])
    if not args.silent:
        print(f"gmb data - routes: {len(raw_gmb_routes) if raw_gmb_routes else 0}, stops: {len(raw_gmb_stops) if raw_gmb_stops else 0}, route-stops: {len(raw_gmb_route_stops) if raw_gmb_route_stops else 0}")

    # mtr bus
    raw_mtrbus_routes = fetch_or_load_from_cache("mtrbus_routes", mtrbus_client.fetch_all_routes, args.force_ingest, silent=args.silent)
    raw_mtrbus_stops = fetch_or_load_from_cache("mtrbus_stops", mtrbus_client.fetch_all_stops, args.force_ingest, silent=args.silent)
    raw_mtrbus_route_stops = fetch_or_load_from_cache("mtrbus_route_stops", mtrbus_client.fetch_all_route_stops, args.force_ingest, silent=args.silent)
    raw_mtrbus_fares = fetch_or_load_from_cache("mtrbus_fares", mtrbus_client.fetch_all_fares, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"mtr bus data - routes: {len(raw_mtrbus_routes) if raw_mtrbus_routes else 0}, stops: {len(raw_mtrbus_stops) if raw_mtrbus_stops else 0}, route-stops: {len(raw_mtrbus_route_stops) if raw_mtrbus_route_stops else 0}, fares: {len(raw_mtrbus_fares) if raw_mtrbus_fares else 0}")

    raw_mtr_lines_and_stations = fetch_or_load_from_cache("mtr_lines_and_stations", mtr_rails_client.fetch_mtr_lines_and_stations_with_locations, args.force_ingest, silent=args.silent)
    raw_mtr_lines_fares = fetch_or_load_from_cache("mtr_lines_fares", mtr_rails_client.fetch_mtr_lines_fares, args.force_ingest, silent=args.silent)
    raw_light_rail_routes_and_stops = fetch_or_load_from_cache("light_rail_routes_and_stops", mtr_rails_client.fetch_light_rail_routes_and_stops, args.force_ingest, silent=args.silent)
    raw_light_rail_fares = fetch_or_load_from_cache("light_rail_fares", mtr_rails_client.fetch_light_rail_fares, args.force_ingest, silent=args.silent)
    raw_airport_express_fares = fetch_or_load_from_cache("airport_express_fares", mtr_rails_client.fetch_airport_express_fares, args.force_ingest, silent=args.silent)
    raw_mtr_headway = fetch_or_load_from_cache("mtr_headway", mtr_headway.scrape_train_frequency, args.force_ingest, silent=args.silent)
    raw_mtr_exits = fetch_or_load_from_cache("mtr_exits", mtr_exit_client.fetch_mtr_exits, args.force_ingest, silent=args.silent)
    raw_light_rail_stops = fetch_or_load_from_cache("light_rail_stops", light_rail_stops_fetcher.fetch_light_rail_stops, args.force_ingest, silent=args.silent)

    # citybus
    raw_citybus_routes = fetch_or_load_from_cache("citybus_routes", citybus_client.fetch_all_routes, args.force_ingest, silent=args.silent)
    citybus_stops_data = fetch_or_load_from_cache("citybus_stops_and_sequences", citybus_client.fetch_all_stops_threaded, args.force_ingest, all_routes=raw_citybus_routes, silent=args.silent)
    raw_citybus_stop_id, raw_citybus_route_sequences = citybus_stops_data if citybus_stops_data else ([], [])
    raw_citybus_stop_details = fetch_or_load_from_cache("citybus_stop_details", citybus_client.fetch_all_stop_details_threaded, args.force_ingest, list_of_route_stops=raw_citybus_stop_id, silent=args.silent)

    # nlb
    raw_nlb_routes = fetch_or_load_from_cache("nlb_routes", nlb_client.fetch_all_routes, args.force_ingest, silent=args.silent)
    nlb_stops_data = fetch_or_load_from_cache("nlb_stops_and_route_stops", nlb_client.fetch_all_stops_and_route_stops_threaded, args.force_ingest, routes=raw_nlb_routes, silent=args.silent)
    raw_nlb_stops, raw_nlb_route_stops = nlb_stops_data if nlb_stops_data else ([], [])
    if not args.silent:
        print(f"nlb data - routes: {len(raw_nlb_routes) if raw_nlb_routes else 0}, stops: {len(raw_nlb_stops) if raw_nlb_stops else 0}, route-stops: {len(raw_nlb_route_stops) if raw_nlb_route_stops else 0}")

    # journey time
    raw_journey_time_data = fetch_or_load_from_cache("journey_time", journey_time_client.fetch_all_journey_time_data, args.force_ingest, silent=args.silent)
    raw_hourly_journey_time_data = fetch_or_load_from_cache("hourly_journey_time", journey_time_client.fetch_all_hourly_journey_time_data_threaded, args.force_ingest, silent=args.silent)
    if not args.silent:
        print(f"journey time data - basic: {len(raw_journey_time_data) if raw_journey_time_data else 0} records, hourly: {len(raw_hourly_journey_time_data) if raw_hourly_journey_time_data else 0} records")

    # process
    if not args.silent:
        print("processing data")

    process_and_load_kmb_data(
        raw_routes=raw_kmb_routes,
        raw_stops=raw_kmb_stops,
        raw_route_stops=raw_kmb_route_stops,
        engine=engine,
        silent=args.silent
    )

    process_and_load_gmb_data(
        raw_routes=raw_gmb_routes,
        raw_stops=raw_gmb_stops,
        raw_route_stops=raw_gmb_route_stops,
        engine=engine,
        silent=args.silent
    )

    process_and_load_mtrbus_data(
        raw_routes=raw_mtrbus_routes,
        raw_stops=raw_mtrbus_stops,
        raw_route_stops=raw_mtrbus_route_stops,
        raw_fares=raw_mtrbus_fares,
        engine=engine,
        silent=args.silent
    )

    process_and_load_citybus_data(
        raw_routes=raw_citybus_routes,
        raw_stop_details=raw_citybus_stop_details,
        raw_route_sequences=raw_citybus_route_sequences,
        engine=engine,
        silent=args.silent
    )

    process_and_load_nlb_data(
        raw_routes=raw_nlb_routes,
        raw_stops=raw_nlb_stops,
        raw_route_stops=raw_nlb_route_stops,
        engine=engine,
        silent=args.silent
    )

    process_and_load_mtr_rails_data(
        raw_mtr_lines_and_stations=raw_mtr_lines_and_stations,
        raw_mtr_lines_fares=raw_mtr_lines_fares,
        raw_light_rail_routes_and_stops=raw_light_rail_routes_and_stops,
        raw_light_rail_fares=raw_light_rail_fares,
        raw_airport_express_fares=raw_airport_express_fares,
        engine=engine,
        silent=args.silent
    )

    process_and_load_gov_gtfs_data(
        raw_frequencies=raw_gov_frequencies,
        raw_trips=raw_gov_trips,
        raw_routes=raw_gov_routes,
        raw_calendar=raw_gov_calendar,
        raw_calendar_dates=raw_gov_calendar_dates,
        raw_fares=raw_gov_fares,
        engine=engine,
        silent=args.silent
    )

    process_and_load_journey_time_data(
        raw_journey_time_data=raw_journey_time_data,
        raw_hourly_journey_time_data=raw_hourly_journey_time_data,
        engine=engine,
        silent=args.silent
    )

    process_and_load_mtr_exits_data(
        raw_mtr_exits_data=raw_mtr_exits,
        engine=engine,
        silent=args.silent
    )

    process_and_load_light_rail_stops_data(
        raw_light_rail_stops=raw_light_rail_stops,
        engine=engine,
        silent=args.silent
    )

    export_unified_feed(
        engine=engine,
        output_dir=os.path.join("output", "gtfs"),
        journey_time_data=raw_journey_time_data,
        mtr_headway_data=raw_mtr_headway,
        silent=args.silent
    )

    if not args.silent:
        print("processing complete")


if __name__ == "__main__":
    main()
