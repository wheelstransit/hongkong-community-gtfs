import pandas as pd

from ..db import Store
import logging

log = logging.getLogger(__name__)

def _process_special_fares_for_trip(args):
    """Special fare rules for one trip from its stop-pair fare matrix."""
    trip_id, trip_fares_data, trip_stops_map = args
    
    special_rules = []
    
    if not trip_fares_data or not trip_stops_map:
        return special_rules
    
    fare_matrix = {}
    for fare in trip_fares_data:
        start_seq = fare['start_stop_seq']
        end_seq = fare['end_stop_seq']
        price = fare['price']
        
        if start_seq not in fare_matrix:
            fare_matrix[start_seq] = {}
        fare_matrix[start_seq][end_seq] = {
            'price': price,
            'currency': fare['currency_type']
        }
    
    max_stop_seq = max(trip_stops_map.keys()) if trip_stops_map else 0
    
    for start_seq, end_seq_prices in fare_matrix.items():
        price_to_end_seqs = {}
        for end_seq, fare_info in end_seq_prices.items():
            price = fare_info['price']
            if price not in price_to_end_seqs:
                price_to_end_seqs[price] = []
            price_to_end_seqs[price].append(end_seq)
        
        if len(price_to_end_seqs) > 1:
            sorted_prices = sorted(price_to_end_seqs.items(), key=lambda x: x[0])
            
            for price, end_seqs in sorted_prices:
                max_end_seq = max(end_seqs)
                
                if max_end_seq >= max_stop_seq:
                    continue
                
                min_end_seq = min(end_seqs)
                
                if start_seq in trip_stops_map and max_end_seq in trip_stops_map:
                    boarding_stop_id = trip_stops_map[start_seq]
                    offboarding_stop_id = trip_stops_map[max_end_seq]
                    currency = end_seq_prices[max_end_seq]['currency']
                    
                    special_rules.append({
                        'special_fare_id': f"{trip_id}-{start_seq}-{min_end_seq}-{max_end_seq}-{price}",
                        'rule_type': 'trip',
                        'trip_id': trip_id,
                        'onboarding_stop_id': boarding_stop_id,
                        'offboarding_stop_id': offboarding_stop_id,
                        'price': price,
                        'currency': currency
                    })
    
    return special_rules

FARE_STAGE_COLUMNS = ["trip_id", "from_stop_id", "price", "currency"]
SPECIAL_COLUMNS = ["special_fare_id", "rule_type", "trip_id", "onboarding_stop_id", "offboarding_stop_id", "price", "currency"]

def trip_fares(db: Store, trips_df: pd.DataFrame) -> pd.DataFrame:
    """Government per stop-pair fares (fare_id is route-direction-start_seq-end_seq) joined onto our trips."""
    fare_rules = db.read("SELECT fare_id FROM gov_gtfs_fare_rules")
    fare_rules[["fare_route_id", "fare_direction_id", "start_stop_seq", "end_stop_seq"]] = fare_rules["fare_id"].str.split("-", expand=True)
    fare_rules["fare_route_id"] = fare_rules["fare_route_id"].astype(str)
    fare_rules["fare_direction_id"] = fare_rules["fare_direction_id"].astype(int)
    fare_rules["start_stop_seq"] = fare_rules["start_stop_seq"].astype(int)
    fare_rules["end_stop_seq"] = fare_rules["end_stop_seq"].astype(int)
    fares = fare_rules.merge(db.read("SELECT fare_id, price, currency_type FROM gov_gtfs_fare_attributes"), on="fare_id")
    trip_info = trips_df[["trip_id", "gov_route_id", "direction_id"]].dropna(subset=["gov_route_id"]).drop_duplicates()
    trip_info["gov_route_id"] = trip_info["gov_route_id"].astype(str)
    trip_info["gov_direction_id"] = trip_info["direction_id"].apply(lambda d: 2 if d == 1 else 1)
    return trip_info.merge(fares, left_on=["gov_route_id", "gov_direction_id"], right_on=["fare_route_id", "fare_direction_id"])

def _stop_sequences(stop_times_df: pd.DataFrame) -> pd.DataFrame:
    stops = stop_times_df[["trip_id", "stop_id", "stop_sequence"]].copy()
    stops["stop_sequence"] = pd.to_numeric(stops["stop_sequence"], errors="coerce")
    return stops.dropna(subset=["stop_sequence"])

def generate_fare_stages(trip_fares: pd.DataFrame, stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """One row per stop where a trip's boarding fare changes."""
    if trip_fares.empty:
        return pd.DataFrame(columns=FARE_STAGE_COLUMNS)
    trip_order = pd.Series(range(trip_fares["trip_id"].nunique()), index=trip_fares["trip_id"].unique())
    first_fare = (trip_fares.sort_values("start_stop_seq", kind="stable")
                  .drop_duplicates(["trip_id", "start_stop_seq"])[["trip_id", "start_stop_seq", "price", "currency_type"]])
    stops = _stop_sequences(stop_times_df).sort_values(["trip_id", "stop_sequence"])
    stops = stops[stops["trip_id"].isin(trip_order.index)]
    stops["start_stop_seq"] = stops["stop_sequence"].astype(int)
    staged = stops.merge(first_fare, on=["trip_id", "start_stop_seq"], how="inner", sort=False)
    staged = staged.sort_values(["trip_id", "stop_sequence"], kind="stable")
    previous = staged.groupby("trip_id")["price"].shift(fill_value=-1)
    staged = staged[staged["price"] != previous]
    staged = staged.assign(_order=staged["trip_id"].map(trip_order)).sort_values(["_order", "stop_sequence"], kind="stable")
    if staged.empty:
        return pd.DataFrame(columns=FARE_STAGE_COLUMNS)
    return pd.DataFrame({"trip_id": staged["trip_id"].values, "from_stop_id": staged["stop_id"].values,
                         "price": staged["price"].values, "currency": staged["currency_type"].values})

def generate_special_fare_rules(trip_fares: pd.DataFrame, stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """Trip-level fares for boarding stops whose cheaper fares only cover part of the remaining route."""
    if trip_fares.empty:
        return pd.DataFrame(columns=SPECIAL_COLUMNS)
    fares_by_trip = {}
    for trip_id, start, end, price, currency in zip(trip_fares["trip_id"].tolist(), trip_fares["start_stop_seq"].tolist(),
                                                    trip_fares["end_stop_seq"].tolist(), trip_fares["price"].tolist(),
                                                    trip_fares["currency_type"].tolist()):
        fares_by_trip.setdefault(trip_id, []).append({"start_stop_seq": start, "end_stop_seq": end, "price": price, "currency_type": currency})
    stops = _stop_sequences(stop_times_df)
    stops = stops[stops["trip_id"].isin(fares_by_trip)]
    stops_by_trip = {}
    for trip_id, seq, stop_id in zip(stops["trip_id"].tolist(), stops["stop_sequence"].tolist(), stops["stop_id"].tolist()):
        stops_by_trip.setdefault(trip_id, {})[int(seq)] = stop_id
    results = [_process_special_fares_for_trip((trip_id, fares, stops_by_trip.get(trip_id, {}))) for trip_id, fares in fares_by_trip.items()]

    rules = [rule for trip_rules in results for rule in trip_rules]
    if not rules:
        return pd.DataFrame(columns=SPECIAL_COLUMNS)
    # Keep one rule per (trip, alighting stop, price). The sort key is the first all-digit segment of the
    # rule id after its first segment, which is what earlier feeds used.
    groups = {}
    for rule in rules:
        parts = rule["special_fare_id"].split("-")
        start = next((int(part) for i, part in enumerate(parts) if part.isdigit() and i > 0), 999999)
        groups.setdefault((rule["trip_id"], rule["offboarding_stop_id"], rule["price"]), []).append((start, rule))
    return pd.DataFrame([min(group, key=lambda item: item[0])[1] for group in groups.values()])

def generate_mtr_special_fare_rules(db: Store) -> pd.DataFrame:
    """Agency-wide MTR heavy rail station-to-station fares."""
    
    try:
        mtr_fares_df = db.read("SELECT * FROM mtr_lines_fares")
        
        stations_df = db.read(
            'SELECT "Station ID" as station_id, "Station Code" as station_code FROM mtr_lines_and_stations')
        stations_df = stations_df.drop_duplicates(subset=['station_id'])
        
        station_id_to_stop_id = {}
        for _, row in stations_df.iterrows():
            station_id_to_stop_id[str(row['station_id'])] = f"MTR-{row['station_code']}"
        
    except Exception as e:
        log.warning(f"Error reading MTR fare data: {e}")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    if mtr_fares_df.empty:
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    special_fare_rules = []
    for idx, row in mtr_fares_df.iterrows():
        src_station_id = str(row.get('SRC_STATION_ID', ''))
        dest_station_id = str(row.get('DEST_STATION_ID', ''))
        fare = row.get('OCT_ADT_FARE')
        
        if src_station_id == dest_station_id:
            continue
        
        onboarding_stop_id = station_id_to_stop_id.get(src_station_id)
        offboarding_stop_id = station_id_to_stop_id.get(dest_station_id)
        
        if not onboarding_stop_id or not offboarding_stop_id:
            continue
        
        try:
            price = float(fare)
        except (ValueError, TypeError):
            continue
        
        special_fare_rules.append({
            'special_fare_id': f"MTR-{src_station_id}-{dest_station_id}",
            'rule_type': 'agency',
            'trip_id': '',
            'onboarding_stop_id': onboarding_stop_id,
            'offboarding_stop_id': offboarding_stop_id,
            'price': price,
            'currency': 'HKD'
        })
    
    return pd.DataFrame(special_fare_rules)

def generate_light_rail_special_fare_rules(db: Store) -> pd.DataFrame:
    """Agency-wide Light Rail stop-to-stop fares."""
    
    try:
        lr_fares_df = db.read("SELECT * FROM light_rail_fares")
    except Exception as e:
        log.warning(f"Error reading Light Rail fare data: {e}")
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    if lr_fares_df.empty:
        return pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
    
    special_fare_rules = []
    for idx, row in lr_fares_df.iterrows():
        from_station_id = str(row.get('from_station_id', ''))
        to_station_id = str(row.get('to_station_id', ''))
        fare = row.get('fare_octo_adult')
        
        if from_station_id == to_station_id:
            continue
        
        onboarding_stop_id = f"LR{from_station_id}"
        offboarding_stop_id = f"LR{to_station_id}"
        
        try:
            price = float(fare)
        except (ValueError, TypeError):
            continue
        
        special_fare_rules.append({
            'special_fare_id': f"LR-{from_station_id}-{to_station_id}",
            'rule_type': 'agency',
            'trip_id': '',
            'onboarding_stop_id': onboarding_stop_id,
            'offboarding_stop_id': offboarding_stop_id,
            'price': price,
            'currency': 'HKD'
        })
    
    return pd.DataFrame(special_fare_rules)
