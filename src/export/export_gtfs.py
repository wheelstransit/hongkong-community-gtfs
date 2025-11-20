import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine
import os
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from src.processing.stop_unification import unify_stops_by_name_and_distance
from src.processing.stop_times import generate_stop_times_for_agency_optimized as generate_stop_times_for_agency
from src.processing.shapes import generate_shapes_from_csdi_files, match_trips_to_csdi_shapes
from src.processing.utils import get_direction, smart_title_case
from src.processing.gtfs_route_matcher import match_operator_routes_to_government_gtfs, match_operator_routes_with_coop_fallback
from src.processing.fares import (
    generate_fare_stages,
    generate_special_fare_rules,
    generate_mtr_special_fare_rules,
    generate_light_rail_special_fare_rules
)
from src.export.light_rail import build_light_rail_gtfs_data
from datetime import timedelta
import re
from typing import Union, Optional, Tuple
import math
import time, json
import requests

class PhaseTimer:
    """Lightweight context manager for phase timing (non-invasive)."""
    def __init__(self, name: str, collector: list, silent: bool=False):
        self.name = name
        self.collector = collector
        self.silent = silent

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = time.time() - self.start
        self.collector.append({'phase': self.name, 'seconds': dur})
        if not self.silent:
            print(f"[TIMER] {self.name}: {dur:.2f}s")

def format_timedelta(td): #timedelta object turning into HH:MM:SS
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def parse_headway_to_avg_secs(headway_str: str) -> Union[int, None]:
    #shitty function to parse headway strings like '2.1', '3.6-5', '2.5 / 4' into average seconds because MTR doesn't have detailed headway data :sneeze:
    if not headway_str or headway_str == '-':
        return None

    # remove any explanatory text in brackets or after special characters
    headway_str = re.sub(r'\s*\(.*\)$', '', headway_str).strip()
    headway_str = re.sub(r'\s*#.*$', '', headway_str).strip()
    headway_str = re.sub(r'\s*~.*$', '', headway_str).strip()

    try:
        if '-' in headway_str:
            low, high = map(float, headway_str.split('-'))
            avg_mins = (low + high) / 2
        elif '/' in headway_str:
            parts = [float(p.strip()) for p in headway_str.split('/')]
            avg_mins = sum(parts) / len(parts)
        else:
            avg_mins = float(headway_str)

        return int(avg_mins * 60)
    except (ValueError, TypeError):
        return None


def _hhmmss_to_seconds(timestr: str) -> Optional[int]:
    if timestr is None:
        return None
    try:
        parts = str(timestr).strip().split(':')
        if len(parts) != 3:
            return None
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError):
        return None


def _seconds_to_hhmmss(total_seconds: int) -> str:
    return format_timedelta(timedelta(seconds=int(total_seconds)))


def _load_mtr_terminal_maps(engine: Engine) -> Tuple[dict, dict]:
    """Load a mapping of (line_code, direction_variant) -> terminal station names (EN, TC).

    Returns two dicts: (en_map, tc_map)
    """
    try:
        query = '''
            SELECT "Line Code" AS line_code,
                   "Direction" AS direction,
                   "English Name" AS station_name_en,
                   "Chinese Name" AS station_name_tc,
                   "Sequence" AS sequence
            FROM mtr_lines_and_stations
        '''
        df = pd.read_sql(query, engine)
    except Exception:
        return {}, {}

    if df.empty:
        return {}, {}

    # coerce sequence to numeric and drop bad rows
    df['seq_num'] = pd.to_numeric(df['sequence'], errors='coerce')
    df = df.dropna(subset=['seq_num'])
    if df.empty:
        return {}, {}

    df = df.dropna(subset=['station_name_en'])
    if df.empty:
        return {}, {}

    df = df.sort_values('seq_num')

    terminals = (
        df.groupby(['line_code', 'direction'], as_index=False)
        .agg(last_en=('station_name_en', lambda s: s.iloc[-1]),
             last_tc=('station_name_tc', lambda s: s.iloc[-1]))
    )

    en_map: dict = {}
    tc_map: dict = {}
    for rec in terminals.itertuples(index=False):
        dir_raw = str(rec.direction or '').strip()
        if not dir_raw:
            continue

        en_value = str(rec.last_en).strip() if pd.notna(rec.last_en) else ''
        tc_value = str(rec.last_tc).strip() if pd.notna(rec.last_tc) else ''

        # store normalized keys and some fallbacks
        if en_value and en_value.lower() != 'nan':
            en_map[(rec.line_code, dir_raw)] = en_value
        if tc_value and tc_value.lower() != 'nan':
            tc_map[(rec.line_code, dir_raw)] = tc_value

        # add variants (upper/lower and suffix after dash)
        fallback_keys = {(rec.line_code, dir_raw.upper()), (rec.line_code, dir_raw.lower())}
        if '-' in dir_raw:
            suffix = dir_raw.split('-', 1)[-1]
            fallback_keys.update({(rec.line_code, suffix), (rec.line_code, suffix.upper()), (rec.line_code, suffix.lower())})
        if dir_raw.upper().endswith('UT'):
            fallback_keys.update({(rec.line_code, 'UT'), (rec.line_code, 'ut')})
        if dir_raw.upper().endswith('DT'):
            fallback_keys.update({(rec.line_code, 'DT'), (rec.line_code, 'dt')})

        for key in fallback_keys:
            if en_value and en_value.lower() != 'nan':
                en_map.setdefault(key, en_value)
            if tc_value and tc_value.lower() != 'nan':
                tc_map.setdefault(key, tc_value)

    return en_map, tc_map


def _resolve_mtr_terminal(line_code: str, variant: str, direction_id: Optional[Union[int, float, str]], lookup: dict) -> Optional[str]:
    """Resolve the terminal station name from a lookup using a variety of candidate keys."""
    if not lookup or not line_code:
        return None

    variant_str = str(variant or '').strip()
    candidates = []
    if variant_str:
        candidates.extend([(line_code, variant_str), (line_code, variant_str.upper()), (line_code, variant_str.lower())])
        if '-' in variant_str:
            suffix = variant_str.split('-', 1)[-1]
            candidates.extend([(line_code, suffix), (line_code, suffix.upper()), (line_code, suffix.lower())])

    # direction-based fallback
    dir_value = None
    if direction_id is not None and direction_id != '':
        try:
            dir_float = float(direction_id)
            if not math.isnan(dir_float):
                dir_value = int(dir_float)
        except (TypeError, ValueError):
            try:
                dir_value = int(str(direction_id))
            except (TypeError, ValueError):
                dir_value = None

    if dir_value in (0, 1):
        fallback_variant = 'UT' if dir_value == 0 else 'DT'
        candidates.extend([(line_code, fallback_variant), (line_code, fallback_variant.lower())])

    for key in candidates:
        value = lookup.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str and value_str.lower() != 'nan':
            return value_str

    return None


def _expand_stop_area_records(records):
    rows = []
    for rec in records:
        area_id = rec.get('area_id')
        stop_ids = rec.get('stop_ids') or []
        if not area_id or not stop_ids:
            continue
        for stop_id in stop_ids:
            rows.append({'area_id': area_id, 'stop_id': stop_id})
    return rows


def _build_stop_areas_df(stop_group: pd.DataFrame, enable_parallel: bool = True) -> pd.DataFrame:
    if stop_group.empty:
        return pd.DataFrame(columns=['area_id', 'stop_id'])

    records = stop_group[['area_id', 'stop_ids']].to_dict('records')
    if not records:
        return pd.DataFrame(columns=['area_id', 'stop_id'])

    should_parallelize = enable_parallel and len(records) > 500
    if should_parallelize:
        max_workers = min((os.cpu_count() or 1), 16)
        if max_workers > 1:
            chunk = max(1, math.ceil(len(records) / max_workers))
            chunks = [records[i:i + chunk] for i in range(0, len(records), chunk)]
            rows = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for future in executor.map(_expand_stop_area_records, chunks):
                    rows.extend(future)
            return pd.DataFrame(rows, columns=['area_id', 'stop_id']) if rows else pd.DataFrame(columns=['area_id', 'stop_id'])

    rows = _expand_stop_area_records(records)
    return pd.DataFrame(rows, columns=['area_id', 'stop_id']) if rows else pd.DataFrame(columns=['area_id', 'stop_id'])


def _haversine_vec_parallel(a_lat, a_lon, b_lat, b_lon):
    """Vectorized haversine distance calculation for parallel processing."""
    import numpy as np
    import pandas as pd
    
    mask_valid = (~pd.isna(a_lat)) & (~pd.isna(a_lon)) & (~pd.isna(b_lat)) & (~pd.isna(b_lon))
    res = np.full_like(a_lat, fill_value=np.inf, dtype='float64')
    if mask_valid.any():
        R = 6371000.0
        dlat = np.radians(b_lat[mask_valid] - a_lat[mask_valid])
        dlon = np.radians(b_lon[mask_valid] - a_lon[mask_valid])
        lat1 = np.radians(a_lat[mask_valid])
        lat2 = np.radians(b_lat[mask_valid])
        aa = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        res_valid = R * 2 * np.arctan2(np.sqrt(aa), np.sqrt(1-aa))
        res[mask_valid] = res_valid
    return res


def _compute_direction_distances_chunk(chunk_data):
    """Compute haversine distances for a chunk of trip data - picklable top-level function."""
    f_lat, f_lon, l_lat, l_lon, gf_lat, gf_lon, gl_lat, gl_lon = chunk_data
    
    aligned = _haversine_vec_parallel(f_lat, f_lon, gf_lat, gf_lon) + _haversine_vec_parallel(l_lat, l_lon, gl_lat, gl_lon)
    reversed_sum = _haversine_vec_parallel(f_lat, f_lon, gl_lat, gl_lon) + _haversine_vec_parallel(l_lat, l_lon, gf_lat, gf_lon)
    
    return aligned, reversed_sum


def generate_frequencies_from_schedule(
    trip_id_mapping: pd.DataFrame,
    existing_frequencies_df: pd.DataFrame,
    gov_stop_times_df: pd.DataFrame,
    silent: bool = False
) -> pd.DataFrame:
    """Derive frequency entries for trips lacking gov-provided headway data."""
    SINGLE_DEPARTURE_HEADWAY_SECS = 24 * 3600  # treat single departures as once-per-day service
    required_cols = ['trip_id', 'start_time', 'end_time', 'headway_secs']
    if trip_id_mapping is None or trip_id_mapping.empty:
        return pd.DataFrame(columns=required_cols)
    if gov_stop_times_df is None or gov_stop_times_df.empty:
        return pd.DataFrame(columns=required_cols)

    existing_trips = set(existing_frequencies_df['trip_id']) if existing_frequencies_df is not None and not existing_frequencies_df.empty else set()
    all_trips = set(trip_id_mapping['new_trip_id'])
    missing_trips = sorted(all_trips - existing_trips)
    if not missing_trips:
        return pd.DataFrame(columns=required_cols)

    stop_times = gov_stop_times_df[['trip_id', 'stop_sequence', 'departure_time', 'arrival_time']].copy()
    stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce')
    stop_times.dropna(subset=['stop_sequence'], inplace=True)
    stop_times['stop_sequence'] = stop_times['stop_sequence'].astype(int)
    stop_times['time_candidate'] = stop_times['departure_time']
    blank_mask = stop_times['time_candidate'].isna() | (stop_times['time_candidate'].astype(str).str.strip() == '')
    stop_times.loc[blank_mask, 'time_candidate'] = stop_times.loc[blank_mask, 'arrival_time']
    stop_times['time_candidate'] = stop_times['time_candidate'].fillna('').astype(str).str.strip()
    stop_times = stop_times[stop_times['time_candidate'] != '']
    if stop_times.empty:
        return pd.DataFrame(columns=required_cols)

    first_departures = (
        stop_times
        .sort_values(['trip_id', 'stop_sequence'])
        .groupby('trip_id')['time_candidate']
        .first()
    )
    first_departures.index = first_departures.index.astype(str)

    fallback_records = []
    mapping_grouped = trip_id_mapping.groupby('new_trip_id')['original_trip_id'].agg(list)

    for new_trip_id in missing_trips:
        original_ids = mapping_grouped.get(new_trip_id)
        if not original_ids:
            continue
        if not isinstance(original_ids, (list, tuple, set)):
            original_ids = [original_ids]
        original_ids = [str(oid) for oid in original_ids]
        departures = first_departures.reindex(original_ids).dropna()
        if departures.empty:
            continue

        sec_values = sorted({
            sec for sec in (_hhmmss_to_seconds(t) for t in departures.values)
            if sec is not None
        })
        if len(sec_values) < 2:
            # synthesize a once-per-day headway to keep the trip in frequencies.txt
            start_sec = sec_values[0]
            fallback_records.append({
                'trip_id': new_trip_id,
                'start_time': _seconds_to_hhmmss(start_sec),
                'end_time': _seconds_to_hhmmss(start_sec + SINGLE_DEPARTURE_HEADWAY_SECS),
                'headway_secs': SINGLE_DEPARTURE_HEADWAY_SECS
            })
            if not silent:
                print(f"Generated synthetic 24h headway for single-departure trip {new_trip_id}.")
            continue

        headway_segments = []
        for idx in range(len(sec_values) - 1):
            start_sec = sec_values[idx]
            next_sec = sec_values[idx + 1]
            headway = next_sec - start_sec
            if headway <= 0:
                continue
            headway_segments.append((start_sec, next_sec, headway))

        if not headway_segments:
            continue

        for start_sec, end_sec, headway in headway_segments:
            fallback_records.append({
                'trip_id': new_trip_id,
                'start_time': _seconds_to_hhmmss(start_sec),
                'end_time': _seconds_to_hhmmss(end_sec),
                'headway_secs': headway
            })

        # Extend final interval using last observed headway so last departure is included
        last_departure_sec = sec_values[-1]
        last_headway = headway_segments[-1][2]
        fallback_records.append({
            'trip_id': new_trip_id,
            'start_time': _seconds_to_hhmmss(last_departure_sec),
            'end_time': _seconds_to_hhmmss(last_departure_sec + last_headway),
            'headway_secs': last_headway
        })

    if not fallback_records:
        if not silent:
            print("No fallback frequency records generated from schedules.")
        return pd.DataFrame(columns=required_cols)

    return pd.DataFrame(fallback_records)

def resolve_overlapping_frequencies(frequencies_df: pd.DataFrame) -> pd.DataFrame:
    # resolves overlapping frequency intervals for the same trip_id
    if frequencies_df.empty:
        return frequencies_df

    # thank you claude
    # drop exact duplicates on the composite key that causes validation errors
    frequencies_df = frequencies_df.drop_duplicates(subset=['trip_id', 'start_time'], keep='first').copy()

    # convert time strings to timedelta for comparison
    frequencies_df['start_time_td'] = pd.to_timedelta(frequencies_df['start_time'])
    frequencies_df['end_time_td'] = pd.to_timedelta(frequencies_df['end_time'])

    # sort by trip_id and start_time
    frequencies_df = frequencies_df.sort_values(by=['trip_id', 'start_time_td']).reset_index(drop=True)

    resolved_frequencies = []
    for trip_id, group in frequencies_df.groupby('trip_id'):
        if len(group) <= 1:
            resolved_frequencies.append(group)
            continue

        merged_group = []
        current_entry = group.iloc[0].to_dict()

        for i in range(1, len(group)):
            next_entry = group.iloc[i].to_dict()

            # check for overlap
            if next_entry['start_time_td'] < current_entry['end_time_td']:
                # if headway is the same, we can merge by extending the end_time.
                if next_entry['headway_secs'] == current_entry['headway_secs']:
                    current_entry['end_time_td'] = max(current_entry['end_time_td'], next_entry['end_time_td'])
                # if headway is different, we must truncate the current entry to avoid overlap.
                else:
                    current_entry['end_time_td'] = next_entry['start_time_td']
                    # add the truncated current entry to the list, if it's still a valid interval
                    if current_entry['start_time_td'] < current_entry['end_time_td']:
                        merged_group.append(current_entry)
                    current_entry = next_entry
            # check for continuous intervals with same headway to merge
            elif next_entry['start_time_td'] == current_entry['end_time_td'] and next_entry['headway_secs'] == current_entry['headway_secs']:
                 current_entry['end_time_td'] = next_entry['end_time_td']
            # no overlap, so we finalize current_entry
            else:
                merged_group.append(current_entry)
                current_entry = next_entry

        # add the very last entry ^-^
        merged_group.append(current_entry)
        if merged_group:
            resolved_frequencies.append(pd.DataFrame(merged_group))

    if not resolved_frequencies:
        return pd.DataFrame(columns=frequencies_df.columns)

    final_df = pd.concat(resolved_frequencies, ignore_index=True)
    # convert timedelta back to string format HH:MM:SS
    final_df['start_time'] = final_df['start_time_td'].apply(lambda td: format_timedelta(td))
    final_df['end_time'] = final_df['end_time_td'].apply(lambda td: format_timedelta(td))

    return final_df.drop(columns=['start_time_td', 'end_time_td'])

# -------------------------------------------------------------
# Trip Headsign Generation (English)
# -------------------------------------------------------------
def generate_trip_headsigns(engine: Engine, trips_df: pd.DataFrame, silent: bool=False) -> pd.DataFrame:
    """Populate trip_headsign for each trip across agencies (English only).

    Rules (current heuristic):
      KMB: use direction-specific dest_en from kmb_routes (matched on route, bound, service_type)
      CTB: use dest_en from citybus_routes for that direction (bound O = outbound, I = inbound)
      NLB: parse routeName_e (Origin > Destination) => bound O dest part, bound I origin part
      MTRB: use last stop name for (route_id, direction) from mtrbus_stop_sequences
      GMB: fallback to route_short_name (insufficient structured data for direction) or parsed if available later
    """
    if trips_df.empty:
        trips_df['trip_headsign'] = ''
        return trips_df

    # Ensure columns for later joins exist
    work_df = trips_df.copy()
    if 'trip_headsign' not in work_df.columns:
        work_df['trip_headsign'] = ''

    # Derive agency_id from route_id if missing (e.g., final trips assembly omitted it)
    if 'agency_id' not in work_df.columns and 'route_id' in work_df.columns:
        work_df['agency_id'] = work_df['route_id'].astype(str).apply(lambda rid: rid.split('-', 1)[0] if isinstance(rid, str) and '-' in rid else str(rid))

    # Derive route_short_name consistently if absent. Patterns:
    #  KMB-<route> / CTB-<route> / NLB-<routeNo> => two segments
    #  GMB-<region>-<code> => three segments we want region-code
    #  MTRB-<id>, others fallback to second segment if exists
    if 'route_short_name' not in work_df.columns and 'route_id' in work_df.columns:
        def _derive_short(rid: str):
            if not isinstance(rid, str):
                return None
            parts = rid.split('-')
            if len(parts) == 3 and parts[0] == 'GMB':
                return f"{parts[1]}-{parts[2]}"  # region-code
            if len(parts) >= 2:
                return parts[1]
            return rid

        work_df['route_short_name'] = work_df['route_id'].apply(_derive_short)

    # NOTE: Previous implementation used merges then wrote using merge frame indices, causing
    # misalignment (RangeIndex) and cross-agency contamination. Replaced with explicit mapping
    # dictionaries keyed by original row index to guarantee correctness.

    # KMB
    try:
        kmb_mask = work_df['agency_id'] == 'KMB'
        if kmb_mask.any():
            kmb_subset = work_df.loc[kmb_mask]
            if 'service_type' not in kmb_subset.columns:
                service_type_series = kmb_subset['trip_id'].str.split('-').str[3]
            else:
                service_type_series = kmb_subset['service_type']
            kmb_routes = pd.read_sql("SELECT unique_route_id, dest_en FROM kmb_routes", engine)
            parts = kmb_routes['unique_route_id'].str.split('_', expand=True)
            kmb_routes['route'] = parts[0]; kmb_routes['bound'] = parts[1]; kmb_routes['service_type'] = parts[2]
            # Apply title case to destinations
            kmb_routes['dest_en'] = kmb_routes['dest_en'].apply(smart_title_case)
            kmb_lookup = kmb_routes.drop_duplicates(['route','bound','service_type']).set_index(['route','bound','service_type'])['dest_en'].to_dict()
            dests = []
            for idx, row in kmb_subset.iterrows():
                key = (row.get('route_short_name'), row.get('bound'), service_type_series.loc[idx])
                dests.append(kmb_lookup.get(key, ''))
            work_df.loc[kmb_subset.index, 'trip_headsign'] = dests
    except Exception as e:
        if not silent:
            print(f"Headsign warning (KMB map): {e}")

    # CTB
    try:
        ctb_mask = work_df['agency_id'] == 'CTB'
        if ctb_mask.any():
            ctb_subset = work_df.loc[ctb_mask]
            city_routes = pd.read_sql("SELECT route, direction, dest_en FROM citybus_routes", engine)
            city_lookup = city_routes.drop_duplicates(['route','direction']).set_index(['route','direction'])['dest_en'].to_dict()
            mapped = []
            for idx, row in ctb_subset.iterrows():
                direction_str = {'O':'outbound','I':'inbound'}.get(row.get('bound'),'outbound')
                mapped.append(city_lookup.get((row.get('route_short_name'), direction_str), work_df.at[idx,'trip_headsign']))
            work_df.loc[ctb_subset.index, 'trip_headsign'] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign warning (CTB map): {e}")

    # NLB
    try:
        nlb_mask = work_df['agency_id'] == 'NLB'
        if nlb_mask.any():
            nlb_subset = work_df.loc[nlb_mask]
            nlb_routes = pd.read_sql('SELECT "routeNo" as route_no, "routeName_e" as route_name FROM nlb_routes', engine)
            nlb_routes['route_short_name'] = nlb_routes['route_no'].astype(str)
            split_df = nlb_routes['route_name'].str.replace('\u00a0',' ').str.replace('  ',' ').str.split(' > ', n=1, expand=True)
            nlb_routes['origin_en'] = split_df[0]
            nlb_routes['destination_en'] = split_df[1].fillna(split_df[0])
            nlb_lookup = (
                nlb_routes
                .groupby('route_short_name', as_index=True)[['origin_en','destination_en']]
                .first()
                .to_dict('index')
            )
            mapped = []
            for idx, row in nlb_subset.iterrows():
                rec = nlb_lookup.get(str(row.get('route_short_name')))
                if not rec:
                    mapped.append(work_df.at[idx,'trip_headsign']); continue
                mapped.append(rec['destination_en'] if row.get('bound')=='O' else rec['origin_en'])
            work_df.loc[nlb_subset.index, 'trip_headsign'] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign warning (NLB map): {e}")

    try:
        mtrb_mask = work_df['agency_id'] == 'MTRB'
        if mtrb_mask.any():
            mtr_subset = work_df.loc[mtrb_mask]
            mtr_routes_df = pd.read_sql("SELECT route_id, route_name_eng FROM mtrbus_routes", engine)
            mtr_routes_df['route_short_name'] = mtr_routes_df['route_id'].astype(str)
            split_df = mtr_routes_df['route_name_eng'].str.split(' to ', n=1, expand=True)
            mtr_routes_df['origin_guess'] = split_df[0]
            mtr_routes_df['dest_guess'] = split_df[1].fillna(split_df[0])
            lookup = mtr_routes_df.set_index('route_short_name')[['origin_guess','dest_guess']].to_dict('index')
            mapped = []
            for idx, row in mtr_subset.iterrows():
                rec = lookup.get(row.get('route_short_name'))
                if not rec:
                    mapped.append(work_df.at[idx,'trip_headsign']); continue
                bound = row.get('bound') if 'bound' in row else ({0:'O',1:'I'}.get(row.get('direction_id'), 'O'))
                mapped.append(rec['dest_guess'] if bound=='O' else rec['origin_guess'])
            work_df.loc[mtr_subset.index, 'trip_headsign'] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign warning (MTRB map): {e}")

    try:
        gmb_mask = work_df['agency_id'] == 'GMB'
        if gmb_mask.any():
            gmb_routes = pd.read_sql('SELECT region, route_code, orig_en_primary, dest_en_primary, is_circular_any FROM gmb_routes', engine)
            gmb_routes['route_short_name'] = gmb_routes['region'].astype(str)+'-'+gmb_routes['route_code'].astype(str)
            gmb_routes = gmb_routes.drop_duplicates('route_short_name', keep='first')
            # Outbound -> destination primary, Inbound -> origin primary
            gmb_routes['headsign_outbound'] = gmb_routes.apply(lambda r: r['dest_en_primary'] if r.get('dest_en_primary') else r['route_short_name'], axis=1)
            gmb_routes['headsign_inbound'] = gmb_routes.apply(lambda r: r['orig_en_primary'] if r.get('orig_en_primary') else r['route_short_name'], axis=1)
            gmb_lookup = gmb_routes.set_index('route_short_name')[['headsign_outbound','headsign_inbound']].to_dict('index')
            gmb_subset = work_df.loc[gmb_mask]
            mapped = []
            for idx, row in gmb_subset.iterrows():
                rsn = row.get('route_short_name')
                if not rsn and isinstance(row.get('route_id'), str) and row.get('route_id').count('-')>=2:
                    parts = row.get('route_id').split('-',2)
                    if len(parts)==3:
                        rsn = f"{parts[1]}-{parts[2]}"
                rec = gmb_lookup.get(rsn)
                if not rec:
                    mapped.append(work_df.at[idx,'trip_headsign']); continue
                bound = row.get('bound') if 'bound' in row else ({0:'O',1:'I'}.get(row.get('direction_id'), 'O'))
                mapped.append(rec['headsign_outbound'] if bound=='O' else rec['headsign_inbound'])
            work_df.loc[gmb_subset.index, 'trip_headsign'] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign warning (GMB map): {e}")

    # MTR Rail (English)
    try:
        mtrr_mask = work_df['agency_id'] == 'MTRR'
        if mtrr_mask.any():
            terminal_map_en, _ = _load_mtr_terminal_maps(engine)
            if terminal_map_en:
                subset = work_df.loc[mtrr_mask]
                resolved = []
                for idx, row in subset.iterrows():
                    trip_id = str(row.get('trip_id', '') or '')
                    line_code = None
                    variant = ''
                    if trip_id:
                        parts = trip_id.split('-')
                        if len(parts) >= 3:
                            line_code = parts[1]
                            variant = '-'.join(parts[2:])
                    dest = _resolve_mtr_terminal(line_code, variant, row.get('direction_id'), terminal_map_en)
                    if dest:
                        resolved.append(dest)
                    else:
                        fallback_val = work_df.at[idx, 'trip_headsign']
                        fallback_val = '' if pd.isna(fallback_val) else str(fallback_val)
                        resolved.append(fallback_val)
                work_df.loc[subset.index, 'trip_headsign'] = resolved
    except Exception as e:
        if not silent:
            print(f"Headsign warning (MTR Rail map): {e}")

    # Final cleanup: ensure string type and fill blanks with route_short_name if still empty
    if 'route_short_name' in work_df.columns:
        empty_mask = (work_df['trip_headsign'].isna()) | (work_df['trip_headsign'].astype(str).str.strip()=='')
        work_df.loc[empty_mask, 'trip_headsign'] = work_df.loc[empty_mask, 'route_short_name'].astype(str)
    else:
        # Derive a temporary route_short_name to fill empties later
        work_df['__tmp_route_short_name'] = work_df['route_id'].astype(str).apply(lambda rid: rid.split('-', 1)[1] if isinstance(rid, str) and '-' in rid else str(rid))
        empty_mask = (work_df['trip_headsign'].isna()) | (work_df['trip_headsign'].astype(str).str.strip()=='')
        work_df.loc[empty_mask, 'trip_headsign'] = work_df.loc[empty_mask, '__tmp_route_short_name'].astype(str)

    # Second-pass normalization for NLB and GMB if still equal to short name (indicates mapping failure)
    try:
        # NLB normalization
        nlb_sec_mask = work_df['agency_id']=='NLB'
        if nlb_sec_mask.any():
            # ensure normalized short name (strip agency prefix)
            if 'route_short_name' not in work_df.columns:
                work_df.loc[nlb_sec_mask,'__norm_rsn'] = work_df.loc[nlb_sec_mask,'route_id'].astype(str).apply(lambda rid: rid.split('-', 1)[1] if isinstance(rid, str) and '-' in rid else str(rid))
                rsn_series = work_df['__norm_rsn']
            else:
                rsn_series = work_df['route_short_name']
            # Fetch route names again if needed
            nlb_routes = pd.read_sql('SELECT "routeNo" as route_no, "routeName_e" as route_name FROM nlb_routes', engine)
            nlb_routes['route_no'] = nlb_routes['route_no'].astype(str)
            split_df = nlb_routes['route_name'].str.split(' > ', n=1, expand=True)
            nlb_routes['origin_en'] = split_df[0]
            nlb_routes['destination_en'] = split_df[1].fillna(split_df[0])
            nlb_lookup = nlb_routes.set_index('route_no')[['origin_en','destination_en']].to_dict('index')
            updated = []
            for idx in work_df.loc[nlb_sec_mask].index:
                current = work_df.at[idx,'trip_headsign']
                rsn = str(rsn_series.loc[idx])
                rec = nlb_lookup.get(rsn)
                if rec and (current == rsn or not current):
                    bound = work_df.at[idx,'bound'] if 'bound' in work_df.columns else ({0:'O',1:'I'}.get(work_df.at[idx,'direction_id'], 'O'))
                    updated.append((idx, rec['destination_en'] if bound=='O' else rec['origin_en']))
            for idx,val in updated:
                work_df.at[idx,'trip_headsign'] = val
    except Exception as e:
        if not silent:
            print(f"Headsign second-pass warning (NLB): {e}")

    try:
        # GMB second pass (if headsign still short name)
        gmb_sec_mask = work_df['agency_id']=='GMB'
        if gmb_sec_mask.any():
            gmb_routes = pd.read_sql('SELECT region, route_code, orig_en_primary, dest_en_primary FROM gmb_routes', engine)
            gmb_routes['rsn'] = gmb_routes['region'].astype(str)+'-'+gmb_routes['route_code'].astype(str)
            gmb_routes = gmb_routes.drop_duplicates('rsn', keep='first')
            gmb_lookup = gmb_routes.set_index('rsn')[['orig_en_primary','dest_en_primary']].to_dict('index')
            for idx in work_df.loc[gmb_sec_mask].index:
                rsn = work_df.at[idx,'route_short_name'] if 'route_short_name' in work_df.columns else work_df.at[idx,'route_id'].split('-',1)[1]
                current = work_df.at[idx,'trip_headsign']
                rec = gmb_lookup.get(rsn)
                if rec and (current==rsn or not current):
                    bound = work_df.at[idx,'bound'] if 'bound' in work_df.columns else ({0:'O',1:'I'}.get(work_df.at[idx,'direction_id'],'O'))
                    chosen = rec['dest_en_primary'] if bound=='O' else rec['orig_en_primary']
                    if chosen:
                        work_df.at[idx,'trip_headsign'] = chosen
    except Exception as e:
        if not silent:
            print(f"Headsign second-pass warning (GMB): {e}")
    work_df['trip_headsign'] = work_df['trip_headsign'].fillna('')
    work_df['trip_headsign'] = work_df['trip_headsign'].astype(str)
    work_df['trip_headsign'] = work_df['trip_headsign'].replace({'nan': '', 'None': ''})
    return work_df

def generate_trip_headsigns_tc(engine: Engine, trips_df: pd.DataFrame, english_headsigns: Optional[pd.Series] = None, silent: bool=False) -> pd.Series:
    """Generate Traditional Chinese trip headsigns aligned with trips_df index."""
    if trips_df.empty:
        return pd.Series(index=trips_df.index, dtype=str)

    work_df = trips_df.copy()
    if 'agency_id' not in work_df.columns and 'route_id' in work_df.columns:
        work_df['agency_id'] = work_df['route_id'].astype(str).apply(lambda rid: rid.split('-', 1)[0] if isinstance(rid, str) and '-' in rid else str(rid))

    if 'route_short_name' not in work_df.columns and 'route_id' in work_df.columns:
        def _derive_short(rid: str):
            if not isinstance(rid, str):
                return None
            parts = rid.split('-')
            if len(parts) == 3 and parts[0] == 'GMB':
                return f"{parts[1]}-{parts[2]}"
            if len(parts) >= 2:
                return parts[1]
            return rid

        work_df['route_short_name'] = work_df['route_id'].apply(_derive_short)

    headsigns = pd.Series('', index=work_df.index, dtype='object')

    if 'trip_headsign_tc' in work_df.columns:
        prefilled = work_df['trip_headsign_tc'].fillna('').astype(str)
        for idx, value in prefilled.items():
            stripped = value.strip()
            if stripped:
                headsigns.at[idx] = stripped

    # KMB
    try:
        kmb_mask = work_df['agency_id'] == 'KMB'
        if kmb_mask.any():
            kmb_subset = work_df.loc[kmb_mask]
            if 'service_type' not in kmb_subset.columns:
                service_type_series = kmb_subset['trip_id'].str.split('-').str[3]
            else:
                service_type_series = kmb_subset['service_type']
            kmb_routes = pd.read_sql("SELECT unique_route_id, dest_tc FROM kmb_routes", engine)
            parts = kmb_routes['unique_route_id'].str.split('_', expand=True)
            kmb_routes['route'] = parts[0]; kmb_routes['bound'] = parts[1]; kmb_routes['service_type'] = parts[2]
            kmb_lookup = (
                kmb_routes
                .drop_duplicates(['route','bound','service_type'])
                .set_index(['route','bound','service_type'])['dest_tc']
                .to_dict()
            )
            mapped = []
            for idx, row in kmb_subset.iterrows():
                key = (row.get('route_short_name'), row.get('bound'), service_type_series.loc[idx])
                mapped.append(kmb_lookup.get(key, ''))
            headsigns.loc[kmb_subset.index] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign TC warning (KMB map): {e}")

    # CTB
    try:
        ctb_mask = work_df['agency_id'] == 'CTB'
        if ctb_mask.any():
            ctb_subset = work_df.loc[ctb_mask]
            city_routes = pd.read_sql("SELECT route, direction, dest_tc FROM citybus_routes", engine)
            city_lookup = city_routes.drop_duplicates(['route','direction']).set_index(['route','direction'])['dest_tc'].to_dict()
            mapped = []
            for idx, row in ctb_subset.iterrows():
                direction_str = {'O':'outbound','I':'inbound'}.get(row.get('bound'),'outbound')
                mapped.append(city_lookup.get((row.get('route_short_name'), direction_str), headsigns.at[idx]))
            headsigns.loc[ctb_subset.index] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign TC warning (CTB map): {e}")

    # NLB
    try:
        nlb_mask = work_df['agency_id'] == 'NLB'
        if nlb_mask.any():
            nlb_subset = work_df.loc[nlb_mask]
            nlb_routes = pd.read_sql('SELECT "routeNo" as route_no, "routeName_c" as route_name FROM nlb_routes', engine)
            nlb_routes['route_short_name'] = nlb_routes['route_no'].astype(str)
            clean = nlb_routes['route_name'].astype(str).str.replace('\u00a0',' ', regex=False)
            clean = clean.str.replace('  > ', ' > ', regex=False)
            split_df = clean.str.split(' > ', n=1, expand=True)
            nlb_routes['origin_tc'] = split_df[0]
            nlb_routes['destination_tc'] = split_df[1].fillna(split_df[0])
            nlb_lookup = (
                nlb_routes
                .groupby('route_short_name', as_index=True)[['origin_tc','destination_tc']]
                .first()
                .to_dict('index')
            )
            mapped = []
            for idx, row in nlb_subset.iterrows():
                rec = nlb_lookup.get(str(row.get('route_short_name')))
                if not rec:
                    mapped.append(headsigns.at[idx]); continue
                mapped.append(rec['destination_tc'] if row.get('bound')=='O' else rec['origin_tc'])
            headsigns.loc[nlb_subset.index] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign TC warning (NLB map): {e}")

    # MTR Bus
    try:
        mtrb_mask = work_df['agency_id'] == 'MTRB'
        if mtrb_mask.any():
            mtr_subset = work_df.loc[mtrb_mask]
            mtr_routes_df = pd.read_sql("SELECT route_id, route_name_chi FROM mtrbus_routes", engine)
            mtr_routes_df['route_short_name'] = mtr_routes_df['route_id'].astype(str)

            def split_cn(text):
                if not isinstance(text, str):
                    return ('', '')
                text = text.strip()
                if not text:
                    return ('', '')
                separators = ['至', '往', '到', '->', '－', '-', ' — ', '—', ' – ']
                for sep in separators:
                    if sep in text:
                        parts = text.split(sep, 1)
                        return (parts[0].strip(), parts[1].strip())
                return (text, text)

            parsed = mtr_routes_df['route_name_chi'].apply(split_cn)
            mtr_routes_df['origin_guess'] = parsed.apply(lambda x: x[0])
            mtr_routes_df['dest_guess'] = parsed.apply(lambda x: x[1])
            lookup = mtr_routes_df.set_index('route_short_name')[['origin_guess','dest_guess']].to_dict('index')
            mapped = []
            for idx, row in mtr_subset.iterrows():
                rec = lookup.get(row.get('route_short_name'))
                if not rec:
                    mapped.append(headsigns.at[idx]); continue
                bound = row.get('bound') if 'bound' in row else ({0:'O',1:'I'}.get(row.get('direction_id'), 'O'))
                mapped.append(rec['dest_guess'] if bound=='O' else rec['origin_guess'])
            headsigns.loc[mtr_subset.index] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign TC warning (MTRB map): {e}")

    # GMB
    try:
        gmb_mask = work_df['agency_id'] == 'GMB'
        if gmb_mask.any():
            gmb_routes = pd.read_sql('SELECT region, route_code, orig_tc_primary, dest_tc_primary FROM gmb_routes', engine)
            gmb_routes['route_short_name'] = gmb_routes['region'].astype(str)+'-'+gmb_routes['route_code'].astype(str)
            gmb_routes = gmb_routes.drop_duplicates('route_short_name', keep='first')
            gmb_routes['headsign_outbound'] = gmb_routes.apply(lambda r: r['dest_tc_primary'] if r.get('dest_tc_primary') else r['route_short_name'], axis=1)
            gmb_routes['headsign_inbound'] = gmb_routes.apply(lambda r: r['orig_tc_primary'] if r.get('orig_tc_primary') else r['route_short_name'], axis=1)
            gmb_lookup = gmb_routes.set_index('route_short_name')[['headsign_outbound','headsign_inbound']].to_dict('index')
            gmb_subset = work_df.loc[gmb_mask]
            mapped = []
            for idx, row in gmb_subset.iterrows():
                rsn = row.get('route_short_name')
                if not rsn and isinstance(row.get('route_id'), str) and row.get('route_id').count('-')>=2:
                    parts = row.get('route_id').split('-',2)
                    if len(parts)==3:
                        rsn = f"{parts[1]}-{parts[2]}"
                rec = gmb_lookup.get(rsn)
                if not rec:
                    mapped.append(headsigns.at[idx]); continue
                bound = row.get('bound') if 'bound' in row else ({0:'O',1:'I'}.get(row.get('direction_id'), 'O'))
                mapped.append(rec['headsign_outbound'] if bound=='O' else rec['headsign_inbound'])
            headsigns.loc[gmb_subset.index] = mapped
    except Exception as e:
        if not silent:
            print(f"Headsign TC warning (GMB map): {e}")

    # MTR Rail (TC): use terminal name Chinese mapping
    try:
        mtrr_mask = work_df['agency_id'] == 'MTRR'
        if mtrr_mask.any():
            _, terminal_map_tc = _load_mtr_terminal_maps(engine)
            if terminal_map_tc:
                subset = work_df.loc[mtrr_mask]
                resolved = []
                for idx, row in subset.iterrows():
                    trip_id = str(row.get('trip_id', '') or '')
                    line_code = None
                    variant = ''
                    if trip_id:
                        parts = trip_id.split('-')
                        if len(parts) >= 3:
                            line_code = parts[1]
                            variant = '-'.join(parts[2:])
                    dest = _resolve_mtr_terminal(line_code, variant, row.get('direction_id'), terminal_map_tc)
                    if dest:
                        resolved.append(dest)
                    else:
                        resolved.append(str(headsigns.at[idx]))
                headsigns.loc[subset.index] = resolved
    except Exception as e:
        if not silent:
            print(f"Headsign TC warning (MTR Rail map): {e}")

    # Fill empties using fallbacks
    headsigns = headsigns.fillna('').astype(str)
    empty_mask = headsigns.str.strip() == ''
    if english_headsigns is not None:
        fallback = english_headsigns.reindex(headsigns.index).astype(str)
        headsigns.loc[empty_mask] = fallback.loc[empty_mask]
        empty_mask = headsigns.str.strip() == ''

    if empty_mask.any():
        if 'route_short_name' in work_df.columns:
            headsigns.loc[empty_mask] = work_df.loc[empty_mask, 'route_short_name'].astype(str)
        else:
            headsigns.loc[empty_mask] = work_df.loc[empty_mask, 'route_id'].astype(str)

    # Second-pass normalization for NLB and GMB similar to English
    try:
        nlb_sec_mask = work_df['agency_id']=='NLB'
        if nlb_sec_mask.any():
            if 'route_short_name' not in work_df.columns:
                work_df.loc[nlb_sec_mask,'__norm_rsn'] = work_df.loc[nlb_sec_mask,'route_id'].astype(str).apply(lambda rid: rid.split('-', 1)[1] if isinstance(rid, str) and '-' in rid else str(rid))
                rsn_series = work_df['__norm_rsn']
            else:
                rsn_series = work_df['route_short_name']
            nlb_routes = pd.read_sql('SELECT "routeNo" as route_no, "routeName_c" as route_name FROM nlb_routes', engine)
            nlb_routes['route_no'] = nlb_routes['route_no'].astype(str)
            clean = nlb_routes['route_name'].astype(str).str.replace('\u00a0',' ', regex=False)
            clean = clean.str.replace('  > ', ' > ', regex=False)
            split_df = clean.str.split(' > ', n=1, expand=True)
            nlb_routes['origin_tc'] = split_df[0]
            nlb_routes['destination_tc'] = split_df[1].fillna(split_df[0])
            nlb_lookup = nlb_routes.set_index('route_no')[['origin_tc','destination_tc']].to_dict('index')
            for idx in work_df.loc[nlb_sec_mask].index:
                current = headsigns.at[idx]
                rsn = str(rsn_series.loc[idx])
                rec = nlb_lookup.get(rsn)
                if rec and (current == rsn or not str(current).strip()):
                    bound = work_df.at[idx,'bound'] if 'bound' in work_df.columns else ({0:'O',1:'I'}.get(work_df.at[idx,'direction_id'], 'O'))
                    headsigns.at[idx] = rec['destination_tc'] if bound=='O' else rec['origin_tc']
    except Exception as e:
        if not silent:
            print(f"Headsign TC second-pass warning (NLB): {e}")

    try:
        gmb_sec_mask = work_df['agency_id']=='GMB'
        if gmb_sec_mask.any():
            gmb_routes = pd.read_sql('SELECT region, route_code, orig_tc_primary, dest_tc_primary FROM gmb_routes', engine)
            gmb_routes['rsn'] = gmb_routes['region'].astype(str)+'-'+gmb_routes['route_code'].astype(str)
            gmb_routes = gmb_routes.drop_duplicates('rsn', keep='first')
            gmb_lookup = gmb_routes.set_index('rsn')[['orig_tc_primary','dest_tc_primary']].to_dict('index')
            for idx in work_df.loc[gmb_sec_mask].index:
                if 'route_short_name' in work_df.columns:
                    rsn = work_df.at[idx,'route_short_name']
                else:
                    rid = work_df.at[idx,'route_id']
                    rsn = rid.split('-',1)[1] if isinstance(rid, str) and '-' in rid else rid
                current = headsigns.at[idx]
                rec = gmb_lookup.get(rsn)
                if rec and (current == rsn or not str(current).strip()):
                    bound = work_df.at[idx,'bound'] if 'bound' in work_df.columns else ({0:'O',1:'I'}.get(work_df.at[idx,'direction_id'], 'O'))
                    chosen = rec['dest_tc_primary'] if bound=='O' else rec['orig_tc_primary']
                    if chosen:
                        headsigns.at[idx] = chosen
    except Exception as e:
        if not silent:
            print(f"Headsign TC second-pass warning (GMB): {e}")

    return headsigns.fillna('').astype(str)

def export_unified_feed(engine: Engine, output_dir: str, journey_time_data: dict, mtr_headway_data: dict, osm_data: dict, silent: bool = False, no_regenerate_shapes: bool = False):
    phase_timings = []
    print("==========================================")
    print("ENTERING export_unified_feed")
    print("==========================================")
    if not silent:
        print("--- Starting Unified GTFS Export Process ---")

    # MTR platform/exit integration state
    mtr_meta: dict = {}
    station_to_platforms: dict = {}
    platform_amenity_to_stop_id: dict = {}

    final_output_dir = os.path.join(output_dir, "gtfs")
    os.makedirs(final_output_dir, exist_ok=True)

    if not silent:
        print("Building agency.txt...")
    agencies = [
        {'agency_id': 'KMB', 'agency_name': 'Kowloon Motor Bus', 'agency_url': 'https://kmb.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'CTB', 'agency_name': 'Citybus', 'agency_url': 'https://www.citybus.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'MTRB', 'agency_name': 'MTR Bus', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'GMB', 'agency_name': 'Green Minibus', 'agency_url': 'https://td.gov.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'NLB', 'agency_name': 'New Lantao Bus', 'agency_url': 'https://www.nlb.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'FERRY', 'agency_name': 'Ferry Services', 'agency_url': 'https://td.gov.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'MTRR', 'agency_name': 'MTR Rail', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'LR', 'agency_name': 'Light Rail', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'AE', 'agency_name': 'Airport Express', 'agency_url': 'https://www.mtr.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'PT', 'agency_name': 'Peak Tram', 'agency_url': 'https://thepeak.com.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'TRAM', 'agency_name': 'Tramways', 'agency_url': 'https://www.hktramways.com', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
        {'agency_id': 'FERRY', 'agency_name': 'Ferry Services', 'agency_url': 'https://www.td.gov.hk', 'agency_timezone': 'Asia/Hong_Kong', 'agency_lang': 'en'},
    ]
    # we have zh-hant translations, check the end
    agency_df = pd.DataFrame(agencies)
    agency_df.to_csv(os.path.join(final_output_dir, 'agency.txt'), index=False)

    # -- Ferry Routes (load before stops) --
    if not silent:
        print("Processing Ferry routes from government GTFS...")
    
    # Get all ferry routes from government GTFS
    ferry_routes_df = pd.read_sql("""
        SELECT route_id, route_short_name, route_long_name, agency_id, route_type 
        FROM gov_gtfs_routes 
        WHERE agency_id = 'FERRY'
    """, engine)
    
    # Get all ferry trips
    ferry_trips_df = pd.read_sql("""
        SELECT gt.route_id, gt.service_id, gt.trip_id
        FROM gov_gtfs_trips gt
        WHERE gt.route_id IN (SELECT route_id FROM gov_gtfs_routes WHERE agency_id = 'FERRY')
    """, engine)
    
    # Prefix trip_id and service_id with FERRY- to avoid conflicts
    ferry_trips_df['trip_id'] = 'FERRY-' + ferry_trips_df['trip_id'].astype(str)
    ferry_trips_df['service_id'] = 'FERRY-' + ferry_trips_df['service_id'].astype(str)
    
    # Add required columns for downstream processing
    ferry_trips_df['agency_id'] = 'FERRY'
    ferry_trips_df['original_service_id'] = ferry_trips_df['service_id']
    ferry_trips_df['route_short_name'] = ferry_trips_df['route_id'].astype(str)
    ferry_trips_df['direction_id'] = 0  # Default direction
    
    # Get ferry stop times
    ferry_stoptimes_df = pd.read_sql("""
        SELECT gst.trip_id, gst.arrival_time, gst.departure_time, gst.stop_id, 
               gst.stop_sequence
        FROM gov_gtfs_stop_times gst
        WHERE gst.trip_id IN (
            SELECT trip_id FROM gov_gtfs_trips WHERE route_id IN (
                SELECT route_id FROM gov_gtfs_routes WHERE agency_id = 'FERRY'
            )
        )
    """, engine)
    
    # Prefix stop_id and trip_id with FERRY-
    ferry_stoptimes_df['trip_id'] = 'FERRY-' + ferry_stoptimes_df['trip_id'].astype(str)
    ferry_stoptimes_df['stop_id'] = 'FERRY-' + ferry_stoptimes_df['stop_id'].astype(str)
    
    # Rename stop_sequence to match expected column name
    if 'stop_sequence' in ferry_stoptimes_df.columns:
        ferry_stoptimes_df.rename(columns={'stop_sequence': 'stop_sequence'}, inplace=True)
    
    # Get ferry stops
    ferry_stops_df = pd.read_sql("""
        SELECT DISTINCT gs.stop_id, gs.stop_name, gs.stop_lat, gs.stop_lon
        FROM gov_gtfs_stops gs
        WHERE gs.stop_id IN (
            SELECT DISTINCT gst.stop_id FROM gov_gtfs_stop_times gst
            WHERE gst.trip_id IN (
                SELECT trip_id FROM gov_gtfs_trips WHERE route_id IN (
                    SELECT route_id FROM gov_gtfs_routes WHERE agency_id = 'FERRY'
                )
            )
        )
    """, engine)
    
    # Prefix stop_id with FERRY-
    if not ferry_stops_df.empty:
        ferry_stops_df['stop_id'] = 'FERRY-' + ferry_stops_df['stop_id'].astype(str)
        ferry_stops_df['location_type'] = 0
        ferry_stops_df['parent_station'] = None
    else:
        # Create empty dataframe with required columns if no ferry stops
        ferry_stops_df = pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station'])
    
    ferry_frequencies_list = []
    ferry_trips_to_keep = set()
    
    if not ferry_stoptimes_df.empty and not ferry_trips_df.empty:
        import re
        for (route_id, service_id), group_trips in ferry_trips_df.groupby(['route_id', 'service_id']):
            trip_ids = group_trips['trip_id'].tolist()
            
            first_stops = ferry_stoptimes_df[
                (ferry_stoptimes_df['trip_id'].isin(trip_ids)) & 
                (ferry_stoptimes_df['stop_sequence'] == 1)
            ].copy()
            
            if first_stops.empty:
                continue
            
            first_stops['departure_seconds'] = first_stops['departure_time'].apply(_hhmmss_to_seconds)
            first_stops = first_stops.sort_values('departure_seconds').dropna(subset=['departure_seconds'])
            
            if len(first_stops) < 2:
                # Single trip - create frequency with 86400 second headway (once per day)
                template_trip_id = first_stops.iloc[0]['trip_id']
                clean_trip_id = re.sub(r'-\d{4}$', '', template_trip_id)
                start_time = _seconds_to_hhmmss(int(first_stops['departure_seconds'].iloc[0]))
                
                ferry_frequencies_list.append({
                    'trip_id': clean_trip_id,
                    'start_time': start_time,
                    'end_time': start_time,
                    'headway_secs': 86400
                })
                ferry_trips_to_keep.add(template_trip_id)
                continue
            
            # Calculate intervals between consecutive trips
            intervals = []
            times = first_stops['departure_seconds'].tolist()
            for i in range(len(times) - 1):
                intervals.append(times[i + 1] - times[i])
            
            template_trip_id = first_stops.iloc[0]['trip_id']
            clean_trip_id = re.sub(r'-\d{4}$', '', template_trip_id)
            ferry_trips_to_keep.add(template_trip_id)
            
            if not intervals:
                continue
                
            block_start_idx = 0
            current_headway = max(intervals[0], 60)
            
            for i in range(1, len(intervals)):
                next_headway = max(intervals[i], 60)
                headway_diff = abs(next_headway - current_headway) / current_headway if current_headway > 0 else 0
                
                if headway_diff > 0.1 or i == len(intervals) - 1:
                    block_end_idx = i if headway_diff > 0.1 else i + 1
                    
                    start_time = _seconds_to_hhmmss(int(times[block_start_idx]))
                    end_time = _seconds_to_hhmmss(int(times[block_end_idx]))
                    
                    block_intervals = intervals[block_start_idx:block_end_idx]
                    avg_headway = int(sum(block_intervals) / len(block_intervals)) if block_intervals else 1800
                    
                    ferry_frequencies_list.append({
                        'trip_id': clean_trip_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'headway_secs': max(60, avg_headway)
                    })
                    
                    block_start_idx = i
                    current_headway = max(intervals[i], 60)
    
    if ferry_trips_to_keep:
        ferry_trips_df = ferry_trips_df[ferry_trips_df['trip_id'].isin(ferry_trips_to_keep)].copy()
        ferry_stoptimes_df = ferry_stoptimes_df[ferry_stoptimes_df['trip_id'].isin(ferry_trips_to_keep)].copy()
        
        import re
        trip_id_map = {old_id: re.sub(r'-\d{4}$', '', old_id) for old_id in ferry_trips_df['trip_id'].unique()}
        ferry_trips_df['trip_id'] = ferry_trips_df['trip_id'].map(trip_id_map)
        ferry_stoptimes_df['trip_id'] = ferry_stoptimes_df['trip_id'].map(lambda x: trip_id_map.get(x, x))
        
        if not silent:
            print(f"Generated {len(ferry_frequencies_list)} frequency patterns for ferry routes")
    
    ferry_frequencies_df = pd.DataFrame(ferry_frequencies_list) if ferry_frequencies_list else pd.DataFrame(columns=['trip_id', 'start_time', 'end_time', 'headway_secs'])
    
    # Prepare final ferry routes
    final_ferry_routes = ferry_routes_df.copy()
    final_ferry_routes['agency_id'] = 'FERRY'
    
    if not silent:
        print(f"Loaded {len(ferry_routes_df)} ferry routes, {len(ferry_trips_df)} trips, {len(ferry_stops_df)} stops")

    if not silent:
        print("Building stops.txt...")

    # KMB
    kmb_stops_gdf = gpd.read_postgis("SELECT * FROM kmb_stops", engine, geom_col='geometry')
    kmb_stops_gdf['stop_id'] = 'KMB-' + kmb_stops_gdf['stop'].astype(str)
    kmb_stops_gdf['stop_name'] = (
        kmb_stops_gdf['name_en']
        .str.replace(r'\s*\([A-Za-z0-9]{5}\)', '', regex=True)
        .str.replace(r'\s*-\s*', ' - ', regex=True)
        .str.replace(r'([^\s])(\([A-Za-z0-9]+\))', r'\1 \2', regex=True)
        .apply(smart_title_case)
    )
    kmb_stops_gdf, kmb_duplicates_map = (kmb_stops_gdf, {})
    kmb_stops_gdf['stop_lat'] = kmb_stops_gdf.geometry.y
    kmb_stops_gdf['stop_lon'] = kmb_stops_gdf.geometry.x
    kmb_stops_final = kmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # Citybus
    ctb_stops_gdf = gpd.read_postgis("SELECT * FROM citybus_stops", engine, geom_col='geometry')
    ctb_stops_gdf['stop_id'] = 'CTB-' + ctb_stops_gdf['stop'].astype(str)
    ctb_stops_gdf['stop_name'] = ctb_stops_gdf['name_en']
    ctb_duplicates_map = {}
    ctb_stops_gdf['stop_lat'] = ctb_stops_gdf.geometry.y
    ctb_stops_gdf['stop_lon'] = ctb_stops_gdf.geometry.x
    ctb_stops_final = ctb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # GMB
    gmb_stops_gdf = gpd.read_postgis("SELECT * FROM gmb_stops", engine, geom_col='geometry')
    gmb_stops_gdf['stop_id'] = 'GMB-' + gmb_stops_gdf['stop_id'].astype(str)
    gmb_stops_gdf['stop_name'] = gmb_stops_gdf['stop_name_en']
    gmb_stops_gdf, gmb_duplicates_map = (gmb_stops_gdf, {})
    gmb_stops_gdf['stop_lat'] = gmb_stops_gdf.geometry.y
    gmb_stops_gdf['stop_lon'] = gmb_stops_gdf.geometry.x
    gmb_stops_final = gmb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    #TODO: Fix GMB trips for some reason it only includes the first one it finds on the government GTFS

    # MTR Bus
    mtrbus_stops_gdf = gpd.read_postgis("SELECT * FROM mtrbus_stops", engine, geom_col='geometry')
    mtrbus_stops_gdf['stop_id'] = 'MTRB-' + mtrbus_stops_gdf['stop_id'].astype(str)
    mtrbus_stops_gdf['stop_name'] = mtrbus_stops_gdf['name_en']
    mtrbus_stops_gdf, mtrbus_duplicates_map = (mtrbus_stops_gdf, {})
    mtrbus_stops_gdf['stop_lat'] = mtrbus_stops_gdf.geometry.y
    mtrbus_stops_gdf['stop_lon'] = mtrbus_stops_gdf.geometry.x
    mtrbus_stops_final = mtrbus_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # NLB
    nlb_stops_gdf = gpd.read_postgis("SELECT * FROM nlb_stops", engine, geom_col='geometry')
    nlb_stops_gdf['stop_id'] = 'NLB-' + nlb_stops_gdf['stopId'].astype(str)
    nlb_stops_gdf['stop_name'] = nlb_stops_gdf['stopName_e']
    nlb_stops_gdf, nlb_duplicates_map = (nlb_stops_gdf, {})
    nlb_stops_gdf['stop_lat'] = nlb_stops_gdf.geometry.y
    nlb_stops_gdf['stop_lon'] = nlb_stops_gdf.geometry.x
    nlb_stops_final = nlb_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

    # MTR Rails
    mtr_stations_gdf = gpd.read_postgis("SELECT * FROM mtr_lines_and_stations", engine, geom_col='geometry')
    # use Station Code (e.g. WHA, HOM) so journey_time_data can map
    mtr_stations_df = mtr_stations_gdf[['Station Code', 'English Name', 'geometry']].drop_duplicates(subset=['Station Code'])
    mtr_stations_df.rename(columns={'Station Code': 'station_code', 'English Name': 'stop_name'}, inplace=True)
    mtr_stations_df['stop_id'] = 'MTR-' + mtr_stations_df['station_code'].astype(str)
    mtr_stations_df['stop_lat'] = mtr_stations_df.geometry.y
    mtr_stations_df['stop_lon'] = mtr_stations_df.geometry.x
    mtr_stations_df['location_type'] = 1  # Station
    mtr_stations_df['parent_station'] = None

    # Load real platforms and exits from external dataset
    real_platforms_df = pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station', 'platform_code'])
    try:
        if not silent:
            print("Fetching real MTR platform/exit data...")
        data_url = "https://raw.githubusercontent.com/wheelstransit/mtr-platform-exits-crawler/refs/heads/main/data/output/mtr_data_complete.json"
        resp = requests.get(data_url, timeout=20)
        resp.raise_for_status()
        mtr_meta = resp.json()

        stations_obj = mtr_meta.get('stations') or {}
        platforms_obj = mtr_meta.get('platforms') or {}
        platform_rows = []
        for scode, srec in stations_obj.items():
            platform_keys = srec.get('platforms') or []
            station_to_platforms[scode] = []
            for pkey in platform_keys:
                prec = platforms_obj.get(pkey)
                if not prec:
                    continue

                amenity_id = prec.get('amenity_id') or pkey
                plat_name = prec.get('name_en') or prec.get('name_zh') or f"Platform {prec.get('ref','')}"
                platform_code = str(prec.get('ref')) if prec.get('ref') is not None else None

                station_to_platforms[scode].append({
                    'amenity_id': amenity_id,
                    'line_dirs': prec.get('lines_and_directions') or [],
                    'ref': platform_code,
                    'name_en': plat_name
                })

                # New compatibility stop_id format: MTR-PLATFORM-[station_code]-[ref]
                stop_id_val = f"MTR-PLATFORM-{scode}-{platform_code}" if platform_code else f"MTR-PLATFORM-{scode}-{amenity_id}"
                platform_amenity_to_stop_id[amenity_id] = stop_id_val

                coords = prec.get('coordinates') or {}
                platform_rows.append({
                    'stop_id': stop_id_val,
                    'stop_name': (f"Platform {platform_code}" if platform_code else plat_name),
                    'stop_lat': coords.get('lat'),
                    'stop_lon': coords.get('lon'),
                    'location_type': 0,
                    'parent_station': f"MTR-{scode}",
                    'platform_code': platform_code
                })
        if platform_rows:
            real_platforms_df = pd.DataFrame(platform_rows)
            if not silent:
                print(f"Loaded {len(real_platforms_df)} real MTR platforms.")

    except Exception as e:
        if not silent:
            print(f"Warning: could not load real MTR platforms/exits. MTR platform data will be missing. Error: {e}")

    # MTR Entrances (from DB)
    try:
        mtr_exits_gdf = gpd.read_postgis("SELECT * FROM mtr_exits", engine, geom_col='geometry')
        if not mtr_exits_gdf.empty:
            if 'station_code' in mtr_exits_gdf.columns:
                mtr_exits_gdf['stop_id'] = 'MTR-ENTRANCE-' + mtr_exits_gdf['station_code'] + '-' + mtr_exits_gdf['exit']
                station_code_to_id_map = mtr_stations_df.set_index('station_code')['stop_id'].to_dict()
                mtr_exits_gdf['parent_station'] = mtr_exits_gdf['station_code'].map(station_code_to_id_map)
            else:
                mtr_exits_gdf['stop_id'] = 'MTR-ENTRANCE-' + mtr_exits_gdf['station_name_en'] + '-' + mtr_exits_gdf['exit']
                station_name_to_id_map = mtr_stations_df.set_index('stop_name')['stop_id'].to_dict()
                mtr_exits_gdf['parent_station'] = mtr_exits_gdf['station_name_en'].map(station_name_to_id_map)
            mtr_exits_gdf['stop_name'] = mtr_exits_gdf['exit']
            mtr_exits_gdf['stop_lat'] = mtr_exits_gdf.geometry.y
            mtr_exits_gdf['stop_lon'] = mtr_exits_gdf.geometry.x
            mtr_exits_gdf['location_type'] = 2

            mtr_entrances_df = mtr_exits_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']]
        else:
            mtr_entrances_df = pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station'])
    except Exception:
        mtr_entrances_df = pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station'])

    # Light Rail
    lr_stops_gdf = gpd.read_postgis("SELECT * FROM light_rail_stops", engine, geom_col='geometry')
    lr_stops_gdf.rename(columns={'name_en': 'stop_name'}, inplace=True)
    lr_stops_gdf['stop_lat'] = lr_stops_gdf.geometry.y
    lr_stops_gdf['stop_lon'] = lr_stops_gdf.geometry.x
    lr_stops_gdf['location_type'] = 0
    lr_stops_gdf['parent_station'] = None

    # Combine all agencies
    all_stops_df = pd.concat([
        kmb_stops_final,
        ctb_stops_final,
        gmb_stops_final,
        mtrbus_stops_final,
        nlb_stops_final,
        ferry_stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']],
        mtr_stations_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']],        # include platform_code for platforms so it reaches stops.txt
        real_platforms_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station', 'platform_code']],
        mtr_entrances_df,
        lr_stops_gdf[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station']]
    ], ignore_index=True)

    gtfs_stops_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'location_type', 'parent_station', 'platform_code']
    for col in gtfs_stops_cols:
        if col not in all_stops_df.columns:
            all_stops_df[col] = None

    # Ensure location_type is an integer, defaulting missing values to 0 (stop)
    all_stops_df['location_type'] = all_stops_df['location_type'].fillna(0).astype(int)

    all_stops_output_df = all_stops_df[gtfs_stops_cols]
    all_stops_output_df.to_csv(os.path.join(final_output_dir, 'stops.txt'), index=False)

    if not silent:
        print(f"Generated stops.txt with {len(all_stops_output_df)} total stops.")

    # --- 3. Build `routes.txt`, `trips.txt`, `stop_times.txt` ---
    gov_routes_df = pd.read_sql("SELECT * FROM gov_gtfs_routes", engine)
    gov_trips_df = pd.read_sql("SELECT * FROM gov_gtfs_trips", engine)
    gov_frequencies_df = pd.read_sql("SELECT * FROM gov_gtfs_frequencies", engine)
    
    try:
        gov_stops_df = pd.read_sql("SELECT * FROM gov_gtfs_stops", engine)
        gov_stop_times_df = pd.read_sql("SELECT * FROM gov_gtfs_stop_times", engine)
        if not silent:
            print(f"Loaded government stops ({len(gov_stops_df)}) and stop_times ({len(gov_stop_times_df)}) for stop-sequence matching.")
    except Exception as e:
        if not silent:
            print(f"Warning: Could not load government stops/stop_times data: {e}")
        gov_stops_df = pd.DataFrame()
        gov_stop_times_df = pd.DataFrame()
    
    try:
        parsed_direction = gov_trips_df['trip_id'].str.split('-').str[1].astype(int)
        # Government uses: 1=outbound, 2=inbound
        # We want: 0=outbound, 1=inbound (CTB and KMB both use O/I mapping)
        gov_trips_df['direction_id'] = (parsed_direction == 2).astype(int)
        if not silent:
            print("Successfully parsed 'direction_id' from government trip_id.")
    except (IndexError, ValueError, TypeError):
        if not silent:
            print("Warning: Could not parse 'direction_id' from government trip_id.")
        gov_trips_df['direction_id'] = -1

    # standardize data types before merge
    gov_trips_df['service_id'] = gov_trips_df['service_id'].astype(str)
    gov_routes_df['route_short_name'] = gov_routes_df['route_short_name'].astype(str)

    gov_trips_with_route_info = gov_trips_df.merge(
        gov_routes_df[['route_id', 'route_short_name', 'agency_id', 'route_long_name']], on='route_id'
    )

    # -- KMB --
    if not silent:
        print("Processing KMB routes, trips, and stop_times...")
    kmb_routes_df = pd.read_sql("SELECT * FROM kmb_routes", engine)
    kmb_routes_df['agency_id'] = 'KMB'
    kmb_routes_df[['route', 'bound', 'service_type']] = kmb_routes_df['unique_route_id'].str.split('_', expand=True)
    kmb_routes_df['direction_id'] = kmb_routes_df['bound'].map({'O': 0, 'I': 1}).fillna(-1).astype(int)

    kmb_stoptimes_df = pd.read_sql("SELECT * FROM kmb_stop_sequences", engine)
    kmb_stoptimes_df[['route', 'bound', 'service_type']] = kmb_stoptimes_df['unique_route_id'].str.split('_', expand=True)
    kmb_stoptimes_df.dropna(subset=['service_type'], inplace=True)
    kmb_stoptimes_df = kmb_stoptimes_df.drop_duplicates(subset=['unique_route_id', 'seq'])
    kmb_stoptimes_df['stop_id'] = 'KMB-' + kmb_stoptimes_df['stop'].astype(str)
    kmb_stoptimes_df['stop_id'] = kmb_stoptimes_df['stop_id'].replace(kmb_duplicates_map)

    final_kmb_routes_list = []
    for route_num, group in kmb_routes_df.groupby('route'):
        first_outbound = group[group['bound'] == 'O'].iloc[0] if not group[group['bound'] == 'O'].empty else None
        first_inbound = group[group['bound'] == 'I'].iloc[0] if not group[group['bound'] == 'I'].empty else None

        if first_outbound is not None and first_inbound is not None:
            route_long_name = f"{smart_title_case(first_outbound['orig_en'])} - {smart_title_case(first_inbound['orig_en'])}"
        elif first_outbound is not None:
            route_long_name = f"{smart_title_case(first_outbound['orig_en'])} - {smart_title_case(first_outbound['dest_en'])}"
        elif first_inbound is not None:
            route_long_name = f"{smart_title_case(first_inbound['orig_en'])} - {smart_title_case(first_inbound['dest_en'])}"
        else:
            route_long_name = f"{smart_title_case(group.iloc[0]['orig_en'])} - {smart_title_case(group.iloc[0]['dest_en'])}" if not group.empty else ""

        final_kmb_routes_list.append({
            'route_id': f"KMB-{route_num}",
            'agency_id': 'KMB',
            'route_short_name': route_num,
            'route_long_name': route_long_name,
            'route_type': 3
        })
    final_kmb_routes = pd.DataFrame(final_kmb_routes_list)

    # Use enhanced matching for KMB routes
    if not silent:
        print("Using enhanced stop-count-based matching for KMB routes...")
    
    with PhaseTimer('KMB route matching', phase_timings, silent):
        kmb_route_matches = match_operator_routes_to_government_gtfs(
            engine=engine,
            operator_name="KMB",
            debug=not silent
        )
    
    kmb_trips_list = []
    if kmb_route_matches:
        for route_key, route_matches in kmb_route_matches.items():
            route_short_name, bound, service_type = route_key.split('-')
            direction_id = 1 if bound == 'I' else 0
            
            # Find the corresponding database route info
            matching_db_routes = kmb_routes_df[
                (kmb_routes_df['route'] == route_short_name) & 
                (kmb_routes_df['bound'] == bound) &
                (kmb_routes_df['service_type'] == service_type)
            ]
            
            if not matching_db_routes.empty:
                route_info = matching_db_routes.iloc[0]
                for match in route_matches:
                    kmb_trips_list.append({
                        'route_id': f"KMB-{route_short_name}",
                        'service_id': f"KMB-{route_short_name}-{bound}-{match['gov_service_id']}",
                        'trip_id': f"KMB-{route_short_name}-{bound}-{service_type}-{match['gov_service_id']}",
                        'direction_id': direction_id,
                        'bound': bound,
                        'route_short_name': route_short_name,
                        'route_long_name': f"{smart_title_case(route_info.get('orig_en', ''))} - {smart_title_case(route_info.get('dest_en', ''))}",
                        'original_service_id': match['gov_service_id'],
                        'gov_route_id': match['gov_route_id'],  # Add this for proper shape matching
                        'service_type': service_type,
                        'origin_en': smart_title_case(route_info.get('orig_en', '')),
                        'destination_en': smart_title_case(route_info.get('dest_en', '')),
                        'agency_id': 'KMB'
                    })
    
    # Create DataFrame with required columns even if empty
    if kmb_trips_list:
        kmb_trips_df = pd.DataFrame(kmb_trips_list)
    else:
        if not silent:
            print("Warning: Enhanced KMB matching failed, creating empty DataFrame")
        kmb_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'bound', 'route_short_name', 
            'route_long_name', 'original_service_id', 'gov_route_id', 'service_type', 'origin_en', 'destination_en'
        ])
    
    if not silent:
        print("Creating dummy trips for unmatched KMB routes...")
    
    matched_route_keys = set()
    if kmb_route_matches:
        matched_route_keys = set(kmb_route_matches.keys())
    
    unmatched_trips_list = []
    for _, route_info in kmb_routes_df.iterrows():
        route_short_name = route_info['route']
        bound = route_info['bound']
        service_type = route_info['service_type']
        route_key = f"{route_short_name}-{bound}-{service_type}"
        
        if route_key not in matched_route_keys:
            direction_id = 1 if bound == 'I' else 0
            dummy_service_id = 'NEVER'
            
            unmatched_trips_list.append({
                'route_id': f"KMB-{route_short_name}",
                'service_id': f"KMB-{route_short_name}-{bound}-{dummy_service_id}",
                'trip_id': f"KMB-{route_short_name}-{bound}-{service_type}-{dummy_service_id}",
                'direction_id': direction_id,
                'bound': bound,
                'route_short_name': route_short_name,
                'route_long_name': f"{smart_title_case(route_info.get('orig_en', ''))} - {smart_title_case(route_info.get('dest_en', ''))}",
                'original_service_id': dummy_service_id,
                'gov_route_id': None,
                'service_type': service_type,
                'origin_en': smart_title_case(route_info.get('orig_en', '')),
                'destination_en': smart_title_case(route_info.get('dest_en', '')),
                'agency_id': 'KMB',
                'is_dummy': True
            })
    
    if unmatched_trips_list:
        unmatched_trips_df = pd.DataFrame(unmatched_trips_list)
        kmb_trips_df = pd.concat([kmb_trips_df, unmatched_trips_df], ignore_index=True)
        if not silent:
            print(f"Created {len(unmatched_trips_list)} dummy trip(s) for unmatched KMB routes")
    
    if 'original_service_id' not in kmb_trips_df.columns:
        kmb_trips_df['original_service_id'] = 'DEFAULT'
    if 'route_short_name' not in kmb_trips_df.columns:
        kmb_trips_df['route_short_name'] = ''
    if 'is_dummy' not in kmb_trips_df.columns:
        kmb_trips_df['is_dummy'] = False
    kmb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)

    # -- Citybus --
    if not silent:
        print("Processing Citybus routes, trips, and stop_times...")

    # Get co-op routes to exclude from Citybus processing
    co_op_routes_df = pd.read_sql("SELECT DISTINCT route_short_name FROM gov_gtfs_routes WHERE agency_id = 'KMB+CTB'", engine)
    co_op_routes_to_exclude = co_op_routes_df['route_short_name'].tolist()

    ctb_routes_df = pd.read_sql("SELECT * FROM citybus_routes", engine)

    # Exclude co-op routes
    ctb_routes_df = ctb_routes_df[~ctb_routes_df['route'].isin(co_op_routes_to_exclude)]

    ctb_routes_df['route_id'] = 'CTB-' + ctb_routes_df['route']
    ctb_routes_df['agency_id'] = 'CTB'
    ctb_routes_df['route_short_name'] = ctb_routes_df['route']
    ctb_routes_df['route_long_name'] = ctb_routes_df['orig_en'] + ' - ' + ctb_routes_df['dest_en']
    ctb_routes_df['route_type'] = 3
    ctb_routes_df['dir'] = ctb_routes_df['unique_route_id'].str.split('-').str[-1]
    final_ctb_routes_list = []
    for route_num, group in ctb_routes_df.groupby('route'):
        first_outbound = group[group['dir'] == 'outbound'].iloc[0] if not group[group['dir'] == 'outbound'].empty else group.iloc[0]
        first_inbound = group[group['dir'] == 'inbound'].iloc[0] if not group[group['dir'] == 'inbound'].empty else group.iloc[0]
        final_ctb_routes_list.append({
            'route_id': f"CTB-{route_num}",
            'agency_id': 'CTB',
            'route_short_name': route_num,
            'route_long_name': f"{first_outbound['orig_en']} - {first_inbound['orig_en']}",
            'route_type': 3
        })
    final_ctb_routes = pd.DataFrame(final_ctb_routes_list)

    # Match CTB routes using TD ROUTE MDB mapping with co-op route handling
    if not silent:
        print("Matching CTB routes via TD ROUTE MDB (with co-op handling)...")
    
    with PhaseTimer('CTB route matching', phase_timings, silent):
        ctb_route_matches = match_operator_routes_to_government_gtfs(
            engine=engine,
            operator_name="CTB",
            debug=not silent
        )
    if not silent:
        print(f"Found stop-count matches for {len(ctb_route_matches)} CTB routes.")

    ctb_trips_list = []
    
    # Use enhanced matching results to create trips
    if ctb_route_matches:
        for route_short_name, gov_route_id in ctb_route_matches.items():
            # Find the corresponding database route info
            matching_db_routes = ctb_routes_df[ctb_routes_df['route'] == route_short_name]
            
            if not matching_db_routes.empty:
                # Create trips for both directions
                for direction in ['inbound', 'outbound']:
                    direction_id = 1 if direction == 'inbound' else 0
                    bound = 'I' if direction == 'inbound' else 'O'
                    
                    route = matching_db_routes[matching_db_routes['dir'] == direction]
                    if not route.empty:
                        route = route.iloc[0]
                        route_long_name = f"{route['orig_en']} - {route['dest_en']}"
                        
                        # Get all service ids for the government route
                        gov_trips_with_route_info['route_id'] = gov_trips_with_route_info['route_id'].astype(str)
                        gov_trips = gov_trips_with_route_info[gov_trips_with_route_info['route_id'] == gov_route_id]
                        service_ids = gov_trips['service_id'].unique()

                        for service_id in service_ids:
                            ctb_trips_list.append({
                                'route_id': f"CTB-{route_short_name}",
                                'service_id': f"CTB-{route_short_name}-{direction}-{service_id}",
                                'trip_id': f"CTB-{route_short_name}-{direction}-{service_id}",
                                'direction_id': direction_id,
                                'bound': bound,
                                'route_short_name': route_short_name,
                                'route_long_name': route_long_name,
                                'original_service_id': service_id,
                                'unique_route_id': route['unique_route_id'],
                                'origin_en': route['orig_en'],
                                'destination_en': route['dest_en'],
                                'gov_route_id': gov_route_id,
                                'agency_id': 'CTB'
                            })
    
    ctb_trips_df = pd.DataFrame(ctb_trips_list)
    if ctb_trips_df.empty:
        ctb_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'bound', 'route_short_name', 
            'route_long_name', 'original_service_id', 'unique_route_id', 'origin_en', 'destination_en', 
            'gov_route_id', 'agency_id'
        ])
    # Remove duplicate trips that might be created from multiple citybus route entries
    ctb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)
    
    # Create dummy trips for unmatched routes (for GTFS-RT activation)
    if not silent:
        print("Creating dummy trips for unmatched Citybus routes...")
    
    # Get all matched route short names
    matched_route_short_names = set()
    if ctb_route_matches:
        matched_route_short_names = set(ctb_route_matches.keys())
    
    # Find all unmatched routes from the database
    unmatched_ctb_trips_list = []
    for _, route_info in ctb_routes_df.iterrows():
        route_short_name = route_info['route']
        direction = route_info['dir']
        
        # If this route wasn't matched, create a dummy trip
        if route_short_name not in matched_route_short_names:
            direction_id = 1 if direction == 'inbound' else 0
            bound = 'I' if direction == 'inbound' else 'O'
            dummy_service_id = 'NEVER'  # Special service ID that never runs
            
            unmatched_ctb_trips_list.append({
                'route_id': f"CTB-{route_short_name}",
                'service_id': f"CTB-{route_short_name}-{direction}-{dummy_service_id}",
                'trip_id': f"CTB-{route_short_name}-{direction}-{dummy_service_id}",
                'direction_id': direction_id,
                'bound': bound,
                'route_short_name': route_short_name,
                'route_long_name': f"{route_info.get('orig_en', '')} - {route_info.get('dest_en', '')}",
                'original_service_id': dummy_service_id,
                'unique_route_id': route_info['unique_route_id'],
                'origin_en': route_info.get('orig_en', ''),
                'destination_en': route_info.get('dest_en', ''),
                'gov_route_id': None,  # No government route match
                'agency_id': 'CTB',
                'is_dummy': True  # Mark as dummy for downstream processing
            })
    
    if unmatched_ctb_trips_list:
        unmatched_ctb_trips_df = pd.DataFrame(unmatched_ctb_trips_list)
        ctb_trips_df = pd.concat([ctb_trips_df, unmatched_ctb_trips_df], ignore_index=True)
        if not silent:
            print(f"Created {len(unmatched_ctb_trips_list)} dummy trip(s) for unmatched Citybus routes")
    
    # Ensure is_dummy column exists
    if 'is_dummy' not in ctb_trips_df.columns:
        ctb_trips_df['is_dummy'] = False
    
    # Load CTB stop sequences with special handling for circular routes
    if not silent:
        print("Loading CTB stop sequences with circular route handling...")
    
    # First, detect circular routes using the same logic as our matcher
    circular_routes_query = """
        SELECT DISTINCT gr.route_short_name as route
        FROM gov_gtfs_routes gr
        JOIN gov_gtfs_trips gt ON gr.route_id = gt.route_id
        WHERE gr.agency_id = 'CTB'
        GROUP BY gr.route_short_name
        HAVING COUNT(DISTINCT CASE WHEN SPLIT_PART(gt.trip_id, '-', 2) = '2' THEN 1 END) = 0
    """
    circular_routes_df = pd.read_sql(circular_routes_query, engine)
    circular_routes = circular_routes_df['route'].tolist() if not circular_routes_df.empty else []
    
    if not silent:
        print(f"Detected {len(circular_routes)} circular CTB routes: {circular_routes}")
        print(f"Is 22M circular? {'22M' in circular_routes}")
    
    # Debug: Check what stop sequences exist for 22M
    debug_22m_query = """
        SELECT 
            cr.route,
            cr.direction,
            COUNT(css.sequence) as stop_count
        FROM citybus_routes cr
        JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
        WHERE cr.route = '22M'
        GROUP BY cr.route, cr.direction
        ORDER BY cr.route, cr.direction
    """
    debug_22m_df = pd.read_sql(debug_22m_query, engine)
    if not silent:
        print("22M stop counts by direction:")
        for _, row in debug_22m_df.iterrows():
            print(f"  {row['route']} {row['direction']}: {row['stop_count']} stops")
    
    # Load stop sequences with circular route merging
    if circular_routes:
        # For circular routes, merge outbound and inbound sequences
        circular_routes_list = "','".join(circular_routes)
        ctb_stop_sequences_query = f"""
            WITH outbound_sequences AS (
                -- Get outbound sequences for circular routes
                SELECT 
                    cr.route,
                    cr.unique_route_id,
                    css.stop_id,
                    css.sequence,
                    'outbound' as original_direction
                FROM citybus_routes cr
                JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
                WHERE cr.route IN ('{circular_routes_list}') 
                    AND cr.direction = 'outbound'
            ),
            inbound_sequences AS (
                -- Get inbound sequences for circular routes
                SELECT 
                    cr.route,
                    cr.unique_route_id,
                    css.stop_id,
                    css.sequence,
                    'inbound' as original_direction
                FROM citybus_routes cr
                JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
                WHERE cr.route IN ('{circular_routes_list}') 
                    AND cr.direction = 'inbound'
            ),
            circular_merged AS (
                -- Create proper circular route: skip inbound stops until we find non-overlapping pattern
                SELECT 
                    route,
                    'outbound' as direction,
                    unique_route_id,
                    stop_id,
                    sequence as merged_sequence,
                    'from_outbound' as source
                FROM outbound_sequences
                
                UNION ALL
                
                -- Add inbound stops starting from the first stop that doesn't exist in outbound
                -- This creates proper circular: [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,b,a]
                SELECT 
                    ins.route,
                    'outbound' as direction,
                    ins.unique_route_id,
                    ins.stop_id,
                    (SELECT MAX(sequence) FROM outbound_sequences os WHERE os.route = ins.route) + 
                    ROW_NUMBER() OVER (PARTITION BY ins.route ORDER BY ins.sequence) as merged_sequence,
                    'from_inbound' as source
                FROM inbound_sequences ins
                WHERE ins.sequence >= (
                    -- Find the first inbound sequence where the stop doesn't exist in outbound
                    SELECT COALESCE(MIN(i2.sequence), 1)
                    FROM inbound_sequences i2
                    WHERE i2.route = ins.route
                    AND NOT EXISTS (
                        SELECT 1 FROM outbound_sequences obs
                        WHERE obs.route = i2.route AND obs.stop_id = i2.stop_id
                    )
                )
            ),
            regular_routes AS (
                -- For regular routes, keep normal outbound/inbound separation
                SELECT 
                    cr.route,
                    cr.direction,
                    cr.unique_route_id,
                    css.stop_id,
                    css.sequence as merged_sequence,
                    'regular' as source
                FROM citybus_routes cr
                JOIN citybus_stop_sequences css ON cr.unique_route_id = css.unique_route_id
                WHERE cr.route NOT IN ('{circular_routes_list}')
            )
            SELECT 
                route,
                direction,
                unique_route_id,
                stop_id,
                merged_sequence as sequence,
                source
            FROM circular_merged
            
            UNION ALL
            
            SELECT 
                route,
                direction,
                unique_route_id,
                stop_id,
                merged_sequence as sequence,
                source
            FROM regular_routes
            
            ORDER BY route, direction, sequence
        """
        ctb_stoptimes_df = pd.read_sql(ctb_stop_sequences_query, engine)
        
        if not silent:
            route_22m_stops = ctb_stoptimes_df[ctb_stoptimes_df['route'] == '22M']
            if not route_22m_stops.empty:
                print(f"22M after intelligent circular merging: {len(route_22m_stops)} stops")
                print(f"22M direction: {route_22m_stops['direction'].unique()}")
                print(f"22M sequence range: {route_22m_stops['sequence'].min()} to {route_22m_stops['sequence'].max()}")
                
                # Show merge source breakdown
                if 'source' in route_22m_stops.columns:
                    source_counts = route_22m_stops['source'].value_counts()
                    print(f"22M merge sources: {dict(source_counts)}")
                
                # Show first few and last few stops to verify merging
                sorted_stops = route_22m_stops.sort_values('sequence')
                print("First 5 stops:", sorted_stops.head(5)['sequence'].tolist())
                print("Last 5 stops:", sorted_stops.tail(5)['sequence'].tolist())
                
                # Show total unique stops vs government GTFS expectation
                print(f"22M total stops: {len(route_22m_stops)} (expected ~39 from gov GTFS)")
            else:
                print("No 22M stops found after merging!")
                
            # Also check if we have any circular routes processed
            circular_processed = ctb_stoptimes_df[ctb_stoptimes_df['route'].isin(circular_routes)]
            print(f"Total stops for all circular routes: {len(circular_processed)}")
            
            # Sample a few circular routes to check merging effectiveness
            sample_routes = circular_routes[:5] if len(circular_routes) >= 5 else circular_routes
            for route in sample_routes:
                route_stops = ctb_stoptimes_df[ctb_stoptimes_df['route'] == route]
                if len(route_stops) > 0:
                    if 'source' in route_stops.columns:
                        source_counts = route_stops['source'].value_counts()
                        print(f"  {route}: {len(route_stops)} stops {dict(source_counts)}")
                    else:
                        print(f"  {route}: {len(route_stops)} stops")
    else:
        if not silent:
            print("No circular routes detected, using normal loading...")
        # No circular routes detected, use normal loading
        ctb_stoptimes_df = pd.read_sql("SELECT * FROM citybus_stop_sequences", engine)
    ctb_stoptimes_df['stop_id'] = 'CTB-' + ctb_stoptimes_df['stop_id'].astype(str)
    ctb_stoptimes_df['stop_id'] = ctb_stoptimes_df['stop_id'].replace(ctb_duplicates_map)

    # -- GMB --
    if not silent:
        print("Processing GMB routes, trips, and stop_times...")

    gmb_routes_base_df = pd.read_sql("SELECT * FROM gmb_routes", engine)
    gmb_stoptimes_df = pd.read_sql("SELECT * FROM gmb_stop_sequences", engine)

    gmb_routes_base_df['agency_id'] = 'GMB'
    gmb_routes_base_df['route_type'] = 3
    gmb_routes_base_df['route_id'] = 'GMB-' + gmb_routes_base_df['region'] + '-' + gmb_routes_base_df['route_code']
    gmb_routes_base_df['route_short_name'] = gmb_routes_base_df['region'] + '-' + gmb_routes_base_df['route_code']
    gmb_routes_base_df['route_long_name'] = gmb_routes_base_df['region'] + ' - ' + gmb_routes_base_df['route_code']
    final_gmb_routes = gmb_routes_base_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].copy()

    # Use enhanced matching for GMB routes
    if not silent:
        print("Using enhanced location-based matching for GMB routes...")
    
    gmb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="GMB",
        debug=not silent
    )
    
    gmb_trips_list = []
    if gmb_route_matches:
        for route_key, route_matches in gmb_route_matches.items():
            # route_key format: "HKI-1-O-1" (outbound) or "HKI-1-I-1" (inbound)
            route_parts = route_key.split('-')
            region = route_parts[0]
            route_code = route_parts[1]
            bound = route_parts[2]  # 'O' for outbound, 'I' for inbound
            # Determine which route_seq corresponds to this direction
            if bound == 'O':
                target_route_seq = 1  # Outbound
            else:
                target_route_seq = 2  # Inbound
            actual_route_seqs = gmb_stoptimes_df[
                (gmb_stoptimes_df['region'] == region) &
                (gmb_stoptimes_df['route_code'] == route_code) &
                (gmb_stoptimes_df['route_seq'] == target_route_seq)
            ]['route_seq'].unique()
            for match in route_matches:
                for route_seq in actual_route_seqs:
                    gmb_trips_list.append({
                        'route_id': f"GMB-{region}-{route_code}",
                        'service_id': f"GMB-{region}-{route_code}-{match['gov_service_id']}",
                        'trip_id': f"GMB-{region}-{route_code}-{match['gov_service_id']}-{route_seq}",
                        'direction_id': int(route_seq) - 1,
                        'route_short_name': f"{region}-{route_code}",
                        'route_seq': int(route_seq),
                        'route_code': route_code,
                        'region': region,  # ADDED region for disambiguation
                        'agency_id': 'GMB',
                        'original_service_id': match['gov_service_id'],
                        'gov_route_id': match['gov_route_id']
                    })
    
    # Create DataFrame with required columns even if empty
    if gmb_trips_list:
        gmb_trips_df = pd.DataFrame(gmb_trips_list)
    else:
        if not silent:
            print("Warning: Enhanced GMB matching failed, falling back to default logic")
        # Fallback to old logic
        gmb_trips_source = gmb_stoptimes_df[['route_code', 'route_seq', 'region']].drop_duplicates().copy()
        gmb_trips_source = pd.merge(gmb_trips_source, gmb_routes_base_df[['route_code', 'region', 'route_long_name', 'agency_id']].drop_duplicates(), on=['route_code', 'region'], how='left')
        gmb_trips_source['orig_en'] = gmb_trips_source['route_long_name'].str.split(' - ').str[0]
        gmb_trips_source['dest_en'] = gmb_trips_source['route_long_name'].str.split(' - ').str[1]
        gmb_trips_source['route_seq'] = pd.to_numeric(gmb_trips_source['route_seq'])
        gmb_trips_source['route_id'] = 'GMB-' + gmb_trips_source['region'] + '-' + gmb_trips_source['route_code']
        gmb_trips_source['direction_id'] = gmb_trips_source['route_seq'] - 1
        gmb_trips_source['service_id'] = 'GMB_DEFAULT_SERVICE'
        gmb_trips_source['trip_id'] = gmb_trips_source['route_id'] + '-' + gmb_trips_source['route_seq'].astype(str)
        gmb_trips_df = gmb_trips_source[['route_id', 'service_id', 'trip_id', 'direction_id', 'route_seq', 'route_code', 'orig_en', 'dest_en', 'agency_id']].copy()
        gmb_trips_df.rename(columns={'orig_en': 'origin_en', 'dest_en': 'destination_en'}, inplace=True)
        gmb_trips_df['original_service_id'] = 'GMB_DEFAULT_SERVICE'
        gmb_trips_df['route_short_name'] = gmb_trips_df['route_id'].str.replace('GMB-', '')

    gmb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)

    gmb_stoptimes_df['route_id'] = 'GMB-' + gmb_stoptimes_df['region'] + '-' + gmb_stoptimes_df['route_code']
    gmb_stoptimes_df['route_seq'] = pd.to_numeric(gmb_stoptimes_df['route_seq'])
    gmb_stoptimes_df = gmb_stoptimes_df.merge(
        gmb_trips_df[['route_id', 'route_seq', 'trip_id']],
        on=['route_id', 'route_seq']
    )
    gmb_stoptimes_df['stop_id'] = 'GMB-' + gmb_stoptimes_df['stop_id'].astype(str)

    # Defensive dedupe for GMB stop_times
    if 'sequence' in gmb_stoptimes_df.columns:
        _before = len(gmb_stoptimes_df)
        gmb_stoptimes_df = (
            gmb_stoptimes_df
            .sort_values(['trip_id', 'sequence', 'stop_id'])
            .drop_duplicates(subset=['trip_id', 'sequence'], keep='first')
        )
        if not silent:
            print(f"GMB stop_times deduped (export): {_before}->{len(gmb_stoptimes_df)}")
    else:
        if not silent:
            print("Warning: 'sequence' column missing in gmb_stoptimes_df; skipping GMB dedupe")

    # -- MTR Bus --
    if not silent:
        print("Processing MTR Bus routes, trips, and stop_times...")
    mtrbus_routes_df = pd.read_sql("SELECT * FROM mtrbus_routes", engine)
    mtrbus_routes_df['agency_id'] = 'MTRB'
    mtrbus_routes_df['route_type'] = 3
    mtrbus_routes_df['route_long_name'] = mtrbus_routes_df['route_name_eng']
    mtrbus_routes_df['route_short_name'] = mtrbus_routes_df['route_id']
    mtrbus_routes_df['route_id'] = 'MTRB-' + mtrbus_routes_df['route_id']
    final_mtrbus_routes = mtrbus_routes_df[['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']].drop_duplicates(subset=['route_id']).copy()
    # Use enhanced matching for MTRB routes  
    if not silent:
        print("Using enhanced stop-count-based matching for MTRB routes...")
    
    mtrb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="MTRB", 
        debug=not silent
    )
    
    mtrbus_trips_list = []
    if mtrb_route_matches:
        for route_key, route_matches in mtrb_route_matches.items():
            route_short_name, bound, service_type = route_key.split('-')
            direction_id = 0 if bound == 'O' else 1
            
            # Find the corresponding database route info
            matching_db_routes = mtrbus_routes_df[mtrbus_routes_df['route_short_name'] == route_short_name]
            
            if not matching_db_routes.empty:
                route_info = matching_db_routes.iloc[0]
                
                # Extract origin and destination from route name
                origin_en = ''
                destination_en = ''
                if isinstance(route_info['route_name_eng'], str) and ' to ' in route_info['route_name_eng']:
                    parts = route_info['route_name_eng'].split(' to ')
                    origin_en = parts[0]
                    destination_en = parts[1]
                
                for match in route_matches:
                    mtrbus_trips_list.append({
                        'route_id': f"MTRB-{route_short_name}",
                        'service_id': f"MTRB-{route_short_name}-{bound}-{match['gov_service_id']}",
                        'trip_id': f"MTRB-{route_short_name}-{bound}-{match['gov_service_id']}",
                        'direction_id': direction_id,
                        'route_short_name': route_short_name,
                        'gov_route_id': match['gov_route_id'],
                        'original_service_id': match['gov_service_id'],
                        'origin_en': origin_en,
                        'destination_en': destination_en,
                        'agency_id': 'MTRB'
                    })
    
    # Create DataFrame with required columns even if empty
    if mtrbus_trips_list:
        mtrbus_trips_df = pd.DataFrame(mtrbus_trips_list)
    else:
        if not silent:
            print("Warning: Enhanced MTRB matching failed, creating empty DataFrame")
        mtrbus_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'route_short_name', 
            'original_service_id', 'origin_en', 'destination_en'
        ])
    mtrbus_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)
    
    # Ensure required columns exist for downstream processing
    if 'original_service_id' not in mtrbus_trips_df.columns:
        mtrbus_trips_df['original_service_id'] = 'DEFAULT'
    if 'route_short_name' not in mtrbus_trips_df.columns:
        mtrbus_trips_df['route_short_name'] = ''
    mtrbus_stoptimes_df = pd.read_sql("SELECT * FROM mtrbus_stop_sequences", engine)
    mtrbus_stoptimes_df['trip_id'] = 'MTRB-' + mtrbus_stoptimes_df['route_id'] + '-' + mtrbus_stoptimes_df['direction']
    mtrbus_stoptimes_df['stop_id'] = 'MTRB-' + mtrbus_stoptimes_df['stop_id'].astype(str)
    mtrbus_stoptimes_df['stop_id'] = mtrbus_stoptimes_df['stop_id'].replace(mtrbus_duplicates_map)

    # -- NLB --
    if not silent:
        print("Processing NLB routes, trips, and stop_times...")
    nlb_routes_df = pd.read_sql("SELECT * FROM nlb_routes", engine)
    nlb_routes_df[['orig_en', 'dest_en']] = nlb_routes_df['routeName_e'].str.split(' > ', expand=True)
    final_nlb_routes_list = []
    for route_no, group in nlb_routes_df.groupby('routeNo'):
        all_dests = group['dest_en'].unique()
        long_name = ' - '.join(all_dests)
        final_nlb_routes_list.append({
            'route_id': f'NLB-{route_no}',
            'agency_id': 'NLB',
            'route_short_name': route_no,
            'route_long_name': long_name,
            'route_type': 3
        })
    final_nlb_routes = pd.DataFrame(final_nlb_routes_list)
    nlb_routes_df['direction_id'] = nlb_routes_df.apply(lambda row: get_direction(row['routeName_e'], row['orig_en']), axis=1)

    # Use enhanced matching for NLB routes
    if not silent:
        print("Using enhanced stop-count-based matching for NLB routes...")
    
    nlb_route_matches = match_operator_routes_to_government_gtfs(
        engine=engine,
        operator_name="NLB",
        debug=not silent
    )
    if not silent:
        print(f"NLB matching produced {len(nlb_route_matches)} route keys")
        if len(nlb_route_matches)==0:
            # diagnostics
            sample_nlb = pd.read_sql("SELECT routeId, \"routeNo\", \"routeName_e\" FROM nlb_routes ORDER BY routeId LIMIT 10", engine)
            print("Sample NLB routes (first 10):")
            print(sample_nlb.to_string(index=False))
            print("Creating fallback NLB trips (one per directionId heuristic)...")
    
    nlb_trips_list = []
    if nlb_route_matches:
        seen_trip_keys = set()
        for route_key, route_matches in nlb_route_matches.items():
            # route_key like "1-O-1" or "1-I-1"; we ignore service_type portion for NLB direction mapping
            parts = route_key.split('-')
            if len(parts) < 2:
                continue
            route_short_name = parts[0]
            for match in route_matches:
                gov_bound = match.get('gov_bound')
                if gov_bound not in ('O','I'):
                    continue
                direction_id = 0 if gov_bound == 'O' else 1
                routeid = match.get('routeid')
                if routeid is None:
                    try:
                        q = f"SELECT \"routeId\" FROM nlb_routes WHERE \"routeNo\"='{route_short_name}' ORDER BY \"routeId\" LIMIT 1"
                        routeid_df = pd.read_sql(q, engine)
                        if not routeid_df.empty:
                            routeid = routeid_df.iloc[0]['routeId']
                    except Exception:
                        pass
                # Use combination to prevent duplicates
                trip_key = (route_short_name, routeid, match['gov_service_id'], gov_bound)
                if trip_key in seen_trip_keys:
                    continue
                seen_trip_keys.add(trip_key)
                trip_id = f"NLB-{route_short_name}-{routeid}-{match['gov_service_id']}"
                nlb_trips_list.append({
                    'route_id': f"NLB-{route_short_name}",
                    'agency_id': 'NLB',
                    'service_id': f"NLB-{route_short_name}-{match['gov_service_id']}",
                    'trip_id': trip_id,
                    'direction_id': direction_id,
                    'bound': gov_bound,
                    'route_short_name': route_short_name,
                    'original_service_id': match['gov_service_id'],
                    'route_long_name': match.get('gov_route_long_name',''),
                    'routeId': routeid,
                    'origin_en': '',
                    'destination_en': '',
                    'gov_route_id': match['gov_route_id']
                })
    else:
        # Fallback removed: we require matches now for NLB
        if not silent:
            print("Warning: No NLB matches found; no NLB trips will be exported")
    
    nlb_trips_df = pd.DataFrame(nlb_trips_list)
    if nlb_trips_df.empty:
        nlb_trips_df = pd.DataFrame(columns=[
            'route_id', 'service_id', 'trip_id', 'direction_id', 'route_short_name', 
            'original_service_id', 'route_long_name', 'routeId', 'origin_en', 'destination_en', 'gov_route_id'
        ])
    nlb_trips_df.drop_duplicates(subset=['trip_id'], keep='first', inplace=True)
    
    # Ensure required columns exist for downstream processing  
    if 'original_service_id' not in nlb_trips_df.columns:
        nlb_trips_df['original_service_id'] = 'DEFAULT'
    if 'route_short_name' not in nlb_trips_df.columns:
        nlb_trips_df['route_short_name'] = ''
    nlb_stoptimes_df = pd.read_sql("SELECT * FROM nlb_stop_sequences", engine)
    nlb_stoptimes_df['trip_id'] = 'NLB-' + nlb_stoptimes_df['routeNo'] + '-' + nlb_stoptimes_df['routeId'].astype(str)
    nlb_stoptimes_df['stop_id'] = 'NLB-' + nlb_stoptimes_df['stopId'].astype(str)
    nlb_stoptimes_df['stop_id'] = nlb_stoptimes_df['stop_id'].replace(nlb_duplicates_map)

    if not silent:
        print(f"Loaded {len(final_ferry_routes)} ferry routes, {len(ferry_trips_df)} trips, {len(ferry_stops_df)} stops, {len(ferry_stoptimes_df)} stop_times")

    # -- MTR Rail --
    if not silent:
        print("==========================================")
        print("Processing MTR Rail routes, trips, stop_times...")
        print("==========================================")
    mtr_lines_and_stations_df = pd.read_sql("SELECT * FROM mtr_lines_and_stations", engine)

    # Prepare journey time lookup (from Station Code to Station Code)
    jt_lookup = {}
    try:
        if isinstance(journey_time_data, (list, tuple)):
            jt_df = pd.DataFrame(journey_time_data)
        elif isinstance(journey_time_data, dict):
            # Attempt to coerce dict-of-lists or list-of-dicts
            if {'from_stop_id','to_stop_id','travel_time_seconds'}.issubset(set(journey_time_data.keys())):
                jt_df = pd.DataFrame(journey_time_data)
            else:
                jt_df = pd.DataFrame(list(journey_time_data))
        else:
            jt_df = pd.DataFrame()
        if not jt_df.empty and {'from_stop_id','to_stop_id','travel_time_seconds'}.issubset(jt_df.columns):
            # Keep only plausible rail codes (alphanumeric <=4 chars) to reduce noise
            jt_df['from_stop_id'] = jt_df['from_stop_id'].astype(str).str.strip().str.upper()
            jt_df['to_stop_id'] = jt_df['to_stop_id'].astype(str).str.strip().str.upper()
            jt_df = jt_df[jt_df['from_stop_id'].str.len() <= 4]
            jt_df = jt_df[jt_df['to_stop_id'].str.len() <= 4]
            jt_lookup = {(r.from_stop_id, r.to_stop_id): float(r.travel_time_seconds) for r in jt_df.itertuples()}
            if not silent:
                print(f"Loaded {len(jt_lookup)} journey-time edges for potential MTR timing (from function arg).")
    except Exception as e:
        if not silent:
            print(f"Journey time integration skipped (error building lookup): {e}")
        jt_lookup = {}

    # Fallback: if no edges from function arg, try database table `journey_time_data`
    if not jt_lookup:
        try:
            db_jt = pd.read_sql("SELECT from_stop_id, to_stop_id, travel_time_seconds FROM journey_time_data", engine)
            if not db_jt.empty:
                db_jt['from_stop_id'] = db_jt['from_stop_id'].astype(str).str.strip().str.upper()
                db_jt['to_stop_id'] = db_jt['to_stop_id'].astype(str).str.strip().str.upper()
                jt_lookup = {(r.from_stop_id, r.to_stop_id): float(r.travel_time_seconds) for r in db_jt.itertuples()}
                if not silent:
                    print(f"Loaded {len(jt_lookup)} journey-time edges from database table journey_time_data.")
        except Exception as e:
            if not silent:
                print(f"Journey time DB fallback failed: {e}")

    # Build route metadata (one GTFS route per line code)
    line_code_to_name = {
        'EAL': 'East Rail Line',
        'TML': 'Tuen Ma Line',
        'TWL': 'Tsuen Wan Line',
        'KTL': 'Kwun Tong Line',
        'ISL': 'Island Line',
        'TKL': 'Tseung Kwan O Line',
        'SIL': 'South Island Line',
        'AEL': 'Airport Express',
        'TCL': 'Tung Chung Line',
        'DRL': 'Disneyland Resort Line'
    }
    # Derive terminal pairs from base DT direction (or UT if DT missing)
    terminals = []
    for lc, group in mtr_lines_and_stations_df.groupby('Line Code'):
        direction_order_candidates = [d for d in group['Direction'].unique() if d.endswith('DT')]
        chosen_dir = 'DT'
        if 'DT' not in group['Direction'].unique() and direction_order_candidates:
            chosen_dir = direction_order_candidates[0]
        elif 'DT' not in group['Direction'].unique() and 'UT' in group['Direction'].unique():
            chosen_dir = 'UT'
        seq_subset = group[group['Direction'] == chosen_dir].copy()
        if seq_subset.empty:
            seq_subset = group.copy()
        seq_subset['seq_num'] = seq_subset['Sequence'].str.extract(r'^(\d+)').astype(int)
        seq_subset = seq_subset.sort_values('seq_num')
        if not seq_subset.empty:
            first_station = seq_subset.iloc[0]['English Name']
            last_station = seq_subset.iloc[-1]['English Name']
            terminals.append({'line_code': lc, 'terminal_pair': f"{first_station} - {last_station}"})
    terminals_df = pd.DataFrame(terminals)
    mtr_routes_df = terminals_df.copy()
    mtr_routes_df['route_id'] = 'MTR-' + mtr_routes_df['line_code']
    mtr_routes_df['agency_id'] = 'MTRR'
    mtr_routes_df['route_short_name'] = mtr_routes_df['line_code']
    mtr_routes_df['route_long_name'] = mtr_routes_df.apply(lambda r: f"{line_code_to_name.get(r['line_code'], r['line_code'])} ({r['terminal_pair']})", axis=1)
    mtr_routes_df['route_type'] = 1
    mtr_routes_df = mtr_routes_df[['route_id','agency_id','route_short_name','route_long_name','route_type']]

    direction_variants = mtr_lines_and_stations_df[['Line Code','Direction']].drop_duplicates()
    mtr_trips_list = []
    for _, row in direction_variants.iterrows():
        line_code = row['Line Code']
        variant = row['Direction']
        direction_id = 0 if variant.endswith('UT') else 1
        trip_id = f"MTR-{line_code}-{variant}"
        mtr_trips_list.append({
            'route_id': f"MTR-{line_code}",
            'agency_id': 'MTRR',
            'service_id': f"MTR-{line_code}-SERVICE",
            'trip_id': trip_id,
            'direction_id': direction_id,
            'original_service_id': f"MTR-{line_code}-SERVICE",
            'route_short_name': line_code
        })
    mtr_trips_df = pd.DataFrame(mtr_trips_list)

    # Precompute platform selection mapping for faster lookup
    platform_lookup = {}
    try:
        if station_to_platforms:
            for scode, plist in station_to_platforms.items():
                for p in plist:
                    for ld in p.get('line_dirs', []):
                        lc = ld.get('line_code')
                        d_raw = ld.get('direction')
                        # Map direction strings to direction_id: UT-like -> 0, DT-like -> 1
                        dir_id = None
                        if isinstance(d_raw, str):
                            d_raw_up = d_raw.strip().upper()
                            if d_raw_up.endswith('UT'):
                                dir_id = 0
                            elif d_raw_up.endswith('DT'):
                                dir_id = 1
                        dest = ld.get('destination_station_code')
                        key1 = (scode, lc, dir_id, dest)
                        key2 = (scode, lc, dir_id, None)
                        stop_id_val = platform_amenity_to_stop_id.get(p['amenity_id'])
                        if lc is not None and dir_id is not None and stop_id_val:
                            if key1 not in platform_lookup:
                                platform_lookup[key1] = stop_id_val
                            if key2 not in platform_lookup:
                                platform_lookup[key2] = stop_id_val
    except Exception as e:
        if not silent:
            print(f"Platform lookup build warning: {e}")

    # Stop times using journey_time_data where possible
    mtr_stoptimes_rows = []
    # Compute a robust default from journey_time_data (median) else fallback 120s
    mtr_edge_times = [v for (a,b), v in jt_lookup.items() if len(a)<=4 and len(b)<=4]
    import statistics
    try:
        median_edge = statistics.median(mtr_edge_times) if mtr_edge_times else 120.0
    except statistics.StatisticsError:
        median_edge = 120.0
    DEFAULT_SEGMENT = median_edge if 30 <= median_edge <= 600 else 120.0
    if not silent:
        print(f"MTR default segment time set to {int(DEFAULT_SEGMENT)}s (median of journey_time_data)" )
    for _, row in direction_variants.iterrows():
        line_code = row['Line Code']
        variant = row['Direction']
        trip_id = f"MTR-{line_code}-{variant}"
        seg_df = mtr_lines_and_stations_df[(mtr_lines_and_stations_df['Line Code']==line_code) & (mtr_lines_and_stations_df['Direction']==variant)].copy()
        if seg_df.empty:
            continue
        seg_df['seq_num'] = seg_df['Sequence'].str.extract(r'^(\d+)').astype(int)
        seg_df = seg_df.sort_values('seq_num')
        # current direction id and terminal station code for lookup
        dir_id_cur = 0 if str(variant).endswith('UT') else 1
        try:
            dest_code = seg_df.iloc[-1]['Station Code']
        except Exception:
            dest_code = None
        cumulative = 0.0
        prev_code = None
        for idx, r in enumerate(seg_df.itertuples(index=False), start=1):
            # safe station_code access
            if hasattr(r, 'Station_Code'):
                station_code = getattr(r, 'Station_Code')
            else:
                station_code = seg_df.iloc[idx-1]['Station Code']
            station_code = str(station_code).strip().upper()
            if prev_code is not None:
                prev_code_norm = str(prev_code).strip().upper()
                tt = jt_lookup.get((prev_code_norm, station_code))
                if tt is None:
                    # try reverse (assume symmetric)
                    rev = jt_lookup.get((station_code, prev_code_norm))
                    tt = rev
                if tt is None:
                    tt = DEFAULT_SEGMENT
                # sanity clamp
                if tt <= 0:
                    tt = DEFAULT_SEGMENT
                # accept long segments (Airport Express etc.). Only cap at 3600s to avoid outliers
                if tt > 3600:
                    tt = DEFAULT_SEGMENT
                cumulative += tt
            else:
                cumulative = 0.0
            hh = int(cumulative)//3600; mm=(int(cumulative)%3600)//60; ss=int(cumulative)%60
            t_str = f"{hh:02}:{mm:02}:{ss:02}"
            # Resolve platform stop_id using lookup; if missing, choose first platform for the station
            stop_id_val = None
            key1 = (station_code, line_code, dir_id_cur, dest_code)
            key2 = (station_code, line_code, dir_id_cur, None)
            stop_id_val = platform_lookup.get(key1) or platform_lookup.get(key2)
            if not stop_id_val:
                plist = station_to_platforms.get(station_code) or []
                if plist:
                    first_platform = plist[0]
                    stop_id_val = platform_amenity_to_stop_id.get(first_platform['amenity_id'])
                else:
                    # no platform known for this station; skip this stop_time row
                    continue

            mtr_stoptimes_rows.append({
                'trip_id': trip_id,
                'arrival_time': t_str,
                'departure_time': t_str,
                'stop_id': stop_id_val,
                'stop_sequence': idx
            })
            prev_code = station_code
    mtr_stoptimes_df = pd.DataFrame(mtr_stoptimes_rows)

    # -- Light Rail --
    if not silent:
        print("Processing Light Rail routes, trips, stop_times, and frequencies...")
    lr_route_cols = ['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']
    lr_trip_cols = ['route_id', 'agency_id', 'route_short_name', 'service_id', 'trip_id', 'direction_id', 'bound', 'trip_headsign']
    lr_stoptime_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
    lr_freq_cols = ['trip_id', 'start_time', 'end_time', 'headway_secs', 'exact_times']
    lr_routes_df = pd.DataFrame(columns=lr_route_cols)
    lr_trips_df = pd.DataFrame(columns=lr_trip_cols)
    lr_stoptimes_df = pd.DataFrame(columns=lr_stoptime_cols)
    lr_frequencies_df = pd.DataFrame(columns=lr_freq_cols)
    lr_schedule_path = Path(__file__).resolve().parents[2] / 'lightrail_legacy' / 'lightrailschedule.json'
    try:
        lr_data = build_light_rail_gtfs_data(str(lr_schedule_path), silent=silent)
        if lr_data.routes is not None and not lr_data.routes.empty:
            lr_routes_df = lr_data.routes.copy()
        if lr_data.trips is not None and not lr_data.trips.empty:
            lr_trips_df = lr_data.trips.copy()
        if lr_data.stop_times is not None and not lr_data.stop_times.empty:
            lr_stoptimes_df = lr_data.stop_times.copy()
        if lr_data.frequencies is not None and not lr_data.frequencies.empty:
            lr_frequencies_df = lr_data.frequencies.copy()
    except Exception as e:
        if not silent:
            print(f"Warning: Light Rail GTFS generation failed: {e}")

    # -- Frequency Processing --
    # Standardize merge keys to string to prevent type errors
    kmb_trips_df['original_service_id'] = kmb_trips_df['original_service_id'].astype(str)
    kmb_trips_df['route_short_name'] = kmb_trips_df['route_short_name'].astype(str)
    ctb_trips_df['original_service_id'] = ctb_trips_df['original_service_id'].astype(str)
    ctb_trips_df['route_short_name'] = ctb_trips_df['route_short_name'].astype(str)
    gmb_trips_df['original_service_id'] = gmb_trips_df['original_service_id'].astype(str)
    gmb_trips_df['route_short_name'] = gmb_trips_df['route_short_name'].astype(str)
    mtrbus_trips_df['original_service_id'] = mtrbus_trips_df['original_service_id'].astype(str)
    mtrbus_trips_df['route_short_name'] = mtrbus_trips_df['route_short_name'].astype(str)
    nlb_trips_df['original_service_id'] = nlb_trips_df['original_service_id'].astype(str)
    nlb_trips_df['route_short_name'] = nlb_trips_df['route_short_name'].astype(str)

    master_duplicates_map = {
        **kmb_duplicates_map,
        **ctb_duplicates_map,
        **gmb_duplicates_map,
        **mtrbus_duplicates_map,
        **nlb_duplicates_map
    }

    if not silent:
        print("Creating a reverse map for unified stops to original stops...")
    unified_to_original_map = {}
    for original, unified in master_duplicates_map.items():
        if unified not in unified_to_original_map:
            unified_to_original_map[unified] = []
        unified_to_original_map[unified].append(original)


    with PhaseTimer('KMB stop_times generation', phase_timings, silent):
        kmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('KMB', na=False)]
        kmb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(kmb_gov_routes_df['route_id'])]
        kmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(kmb_gov_trips_df['trip_id'])]
        kmb_stoptimes_df = generate_stop_times_for_agency(
            'KMB', kmb_trips_df, kmb_stoptimes_df, kmb_gov_routes_df, kmb_gov_trips_df, kmb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
        )

    with PhaseTimer('CTB stop_times generation', phase_timings, silent):
        ctb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('CTB', na=False)]
        ctb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(ctb_gov_routes_df['route_id'])]
        ctb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(ctb_gov_trips_df['trip_id'])]
        ctb_stoptimes_df = generate_stop_times_for_agency(
            'CTB', ctb_trips_df, ctb_stoptimes_df, ctb_gov_routes_df, ctb_gov_trips_df, ctb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
        )

    with PhaseTimer('GMB stop_times generation', phase_timings, silent):
        gmb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('GMB', na=False)]
        gmb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(gmb_gov_routes_df['route_id'])]
        gmb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(gmb_gov_trips_df['trip_id'])]
        gmb_stoptimes_df = generate_stop_times_for_agency(
            'GMB', gmb_trips_df, gmb_stoptimes_df, gmb_gov_routes_df, gmb_gov_trips_df, gmb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
        )
    print(gmb_stoptimes_df.head())

    with PhaseTimer('MTRB stop_times generation', phase_timings, silent):
        mtrbus_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('LRTFeeder', na=False)]
        mtrbus_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(mtrbus_gov_routes_df['route_id'])]
        mtrbus_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(mtrbus_gov_trips_df['trip_id'])]
        mtrbus_stoptimes_df = generate_stop_times_for_agency(
            'MTRB', mtrbus_trips_df, mtrbus_stoptimes_df, mtrbus_gov_routes_df, mtrbus_gov_trips_df, mtrbus_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
        )

    with PhaseTimer('NLB stop_times generation', phase_timings, silent):
        nlb_gov_routes_df = gov_routes_df[gov_routes_df['agency_id'].str.contains('NLB', na=False)]
        nlb_gov_trips_df = gov_trips_with_route_info[gov_trips_with_route_info['route_id'].isin(nlb_gov_routes_df['route_id'])]
        nlb_gov_frequencies_df = gov_frequencies_df[gov_frequencies_df['trip_id'].isin(nlb_gov_trips_df['trip_id'])]
        nlb_stoptimes_df = generate_stop_times_for_agency(
            'NLB', nlb_trips_df, nlb_stoptimes_df, nlb_gov_routes_df, nlb_gov_trips_df, nlb_gov_frequencies_df, journey_time_data, unified_to_original_map, silent=silent
        )

    # --- Combine & Standardize--
    if not silent:
        print("Combining and standardizing data for final GTFS files...")
    final_routes_df = pd.concat([final_kmb_routes, final_ctb_routes, final_gmb_routes, final_mtrbus_routes, final_nlb_routes, final_ferry_routes, mtr_routes_df, lr_routes_df], ignore_index=True)
    color_map = {'KMB': 'EE171F', 'CTB': '0053B9', 'NLB': '8AB666', 'MTRB': 'AE2A42', 'GMB': '34C759', 'MTRR': '003DA5', 'LR': 'FF8800', 'FERRY': '0099CC'}
    
    def get_route_color(agency_id):
        if not isinstance(agency_id, str):
            return None
        # For co-op routes, use the color of the first agency listed.
        first_agency = agency_id.split('+')[0]
        return color_map.get(first_agency)

    final_routes_df['route_color'] = final_routes_df['agency_id'].apply(get_route_color)
    final_routes_df['route_text_color'] = 'FFFFFF'
    if 'lr_routes_df' in locals() and not lr_routes_df.empty:
        if 'route_color' in lr_routes_df.columns:
            lr_color_lookup = (
                lr_routes_df[['route_id', 'route_color']]
                .dropna(subset=['route_id', 'route_color'])
                .set_index('route_id')['route_color']
            )
            if not lr_color_lookup.empty:
                final_routes_df.loc[
                    final_routes_df['route_id'].isin(lr_color_lookup.index),
                    'route_color'
                ] = final_routes_df['route_id'].map(lr_color_lookup)
        if 'route_text_color' in lr_routes_df.columns:
            lr_text_lookup = (
                lr_routes_df[['route_id', 'route_text_color']]
                .dropna(subset=['route_id', 'route_text_color'])
                .set_index('route_id')['route_text_color']
            )
            if not lr_text_lookup.empty:
                final_routes_df.loc[
                    final_routes_df['route_id'].isin(lr_text_lookup.index),
                    'route_text_color'
                ] = final_routes_df['route_id'].map(lr_text_lookup)
    final_routes_df['network_id'] = final_routes_df['route_id'].astype(str)

    routes_output_cols = [
        'route_id',
        'agency_id',
        'route_short_name',
        'route_long_name',
        'route_type',
        'network_id',
        'route_color',
        'route_text_color'
    ]
    for col in routes_output_cols:
        if col not in final_routes_df.columns:
            final_routes_df[col] = None
    final_routes_output_df = final_routes_df[routes_output_cols]

    all_trips_df = pd.concat([kmb_trips_df, ctb_trips_df, gmb_trips_df, mtrbus_trips_df, nlb_trips_df, ferry_trips_df, mtr_trips_df, lr_trips_df], ignore_index=True)

    # --- Check for flipped co-op routes ---
    if not silent:
        print("Checking for flipped directions in co-op routes...")
    
    try:
        # Get co-op routes from government GTFS
        coop_routes_df = pd.read_sql(
            "SELECT DISTINCT route_short_name FROM gov_gtfs_routes WHERE agency_id = 'KMB+CTB'",
            engine
        )
        coop_route_numbers = coop_routes_df['route_short_name'].tolist()
        
        if coop_route_numbers:
            flipped_routes = []
            
            # Get all stop_times DataFrames
            all_stoptimes_dfs = {
                'KMB': kmb_stoptimes_df,
                'CTB': ctb_stoptimes_df
            }
            
            for route_num in coop_route_numbers:
                # Get KMB and CTB trips for this route
                kmb_route_trips = kmb_trips_df[kmb_trips_df['route_short_name'] == route_num]
                ctb_route_trips = ctb_trips_df[ctb_trips_df['route_short_name'] == route_num]
                
                if kmb_route_trips.empty or ctb_route_trips.empty:
                    continue
                
                # For each direction, check if coordinates are flipped
                for direction in [0, 1]:
                    kmb_dir_trips = kmb_route_trips[kmb_route_trips['direction_id'] == direction]
                    ctb_dir_trips = ctb_route_trips[ctb_route_trips['direction_id'] == direction]
                    
                    if kmb_dir_trips.empty or ctb_dir_trips.empty:
                        continue
                    
                    # Get the most frequent trip (by service_id frequency)
                    kmb_trip_id = kmb_dir_trips.iloc[0]['trip_id']
                    ctb_trip_id = ctb_dir_trips.iloc[0]['trip_id']
                    
                    # Get stop sequences for these trips
                    kmb_stops = all_stop_times_df[all_stop_times_df['trip_id'] == kmb_trip_id].sort_values('stop_sequence')
                    ctb_stops = all_stop_times_df[all_stop_times_df['trip_id'] == ctb_trip_id].sort_values('stop_sequence')
                    
                    if len(kmb_stops) < 2 or len(ctb_stops) < 2:
                        continue
                    
                    # Get first and last stop_ids
                    kmb_first_stop = kmb_stops.iloc[0]['stop_id']
                    kmb_last_stop = kmb_stops.iloc[-1]['stop_id']
                    ctb_first_stop = ctb_stops.iloc[0]['stop_id']
                    ctb_last_stop = ctb_stops.iloc[-1]['stop_id']
                    
                    # Get coordinates from unified stops
                    stops_coords_query = """
                    SELECT stop_id, stop_lat, stop_lon 
                    FROM (
                        SELECT 'KMB-' || stop_id as stop_id, stop_lat, stop_lon FROM kmb_stops
                        UNION ALL
                        SELECT 'CTB-' || stop_id as stop_id, stop_lat, stop_lon FROM citybus_stops
                    ) unified_stops
                    WHERE stop_id IN (%s, %s, %s, %s)
                    """
                    
                    coords_df = pd.read_sql_query(
                        stops_coords_query,
                        engine,
                        params=(kmb_first_stop, kmb_last_stop, ctb_first_stop, ctb_last_stop)
                    )
                    
                    if len(coords_df) < 4:
                        continue
                    
                    # Get coordinates
                    kmb_first = coords_df[coords_df['stop_id'] == kmb_first_stop].iloc[0] if not coords_df[coords_df['stop_id'] == kmb_first_stop].empty else None
                    kmb_last = coords_df[coords_df['stop_id'] == kmb_last_stop].iloc[0] if not coords_df[coords_df['stop_id'] == kmb_last_stop].empty else None
                    ctb_first = coords_df[coords_df['stop_id'] == ctb_first_stop].iloc[0] if not coords_df[coords_df['stop_id'] == ctb_first_stop].empty else None
                    ctb_last = coords_df[coords_df['stop_id'] == ctb_last_stop].iloc[0] if not coords_df[coords_df['stop_id'] == ctb_last_stop].empty else None
                    
                    if None in (kmb_first, kmb_last, ctb_first, ctb_last):
                        continue
                    
                    # Calculate distances
                    # Distance from KMB first to CTB first
                    dist_first_to_first = ((kmb_first['stop_lat'] - ctb_first['stop_lat'])**2 + 
                                          (kmb_first['stop_lon'] - ctb_first['stop_lon'])**2)**0.5
                    
                    # Distance from KMB first to CTB last (this indicates a flip)
                    dist_first_to_last = ((kmb_first['stop_lat'] - ctb_last['stop_lat'])**2 + 
                                         (kmb_first['stop_lon'] - ctb_last['stop_lon'])**2)**0.5
                    
                    # Distance from KMB last to CTB last
                    dist_last_to_last = ((kmb_last['stop_lat'] - ctb_last['stop_lat'])**2 + 
                                        (kmb_last['stop_lon'] - ctb_last['stop_lon'])**2)**0.5
                    
                    # Distance from KMB last to CTB first (this indicates a flip)
                    dist_last_to_first = ((kmb_last['stop_lat'] - ctb_first['stop_lat'])**2 + 
                                         (kmb_last['stop_lon'] - ctb_first['stop_lon'])**2)**0.5
                    
                    # If flipped: KMB first is closer to CTB last, and KMB last is closer to CTB first
                    normal_match = dist_first_to_first + dist_last_to_last
                    flipped_match = dist_first_to_last + dist_last_to_first
                    
                    # Threshold: if flipped match is significantly better (20% difference)
                    if flipped_match < normal_match * 0.8:
                        flipped_routes.append({
                            'route': route_num,
                            'direction_id': direction,
                            'kmb_first_stop': kmb_first_stop,
                            'kmb_last_stop': kmb_last_stop,
                            'ctb_first_stop': ctb_first_stop,
                            'ctb_last_stop': ctb_last_stop,
                            'kmb_first_lat': kmb_first['stop_lat'],
                            'kmb_first_lon': kmb_first['stop_lon'],
                            'kmb_last_lat': kmb_last['stop_lat'],
                            'kmb_last_lon': kmb_last['stop_lon'],
                            'ctb_first_lat': ctb_first['stop_lat'],
                            'ctb_first_lon': ctb_first['stop_lon'],
                            'ctb_last_lat': ctb_last['stop_lat'],
                            'ctb_last_lon': ctb_last['stop_lon'],
                            'normal_distance': normal_match,
                            'flipped_distance': flipped_match,
                            'confidence': (normal_match - flipped_match) / normal_match
                        })
            
            # Write flipped routes to CSV
            if flipped_routes:
                flipped_df = pd.DataFrame(flipped_routes)
                output_path = output_dir / 'flipped_coop_routes.csv'
                flipped_df.to_csv(output_path, index=False)
                if not silent:
                    print(f"⚠️  Found {len(flipped_routes)} potentially flipped direction(s) in co-op routes")
                    print(f"    Details saved to: {output_path}")
            else:
                if not silent:
                    print("✓ No flipped directions detected in co-op routes")
    
    except Exception as e:
        if not silent:
            print(f"Warning: Co-op route flip detection failed: {e}")

    # Attach agency to government trips for scoping
    gov_trips_with_route_info['route_id'] = gov_trips_with_route_info['route_id'].astype(str)
    gov_trips_with_route_info['service_id'] = gov_trips_with_route_info['service_id'].astype(str)
    gov_trips_with_route_info['route_short_name'] = gov_trips_with_route_info['route_short_name'].astype(str)
    gov_trips_with_route_info.rename(columns={'agency_id': 'gov_agency_id'}, inplace=True)

    # Normalize our trips fields for safe joins
    for col in ['original_service_id', 'route_short_name']:
        if col in all_trips_df.columns:
            all_trips_df[col] = all_trips_df[col].astype(str)
    if 'gov_route_id' in all_trips_df.columns:
        all_trips_df['gov_route_id'] = all_trips_df['gov_route_id'].astype(str)

    # Map of allowed government agencies per unified operator
    allowed_gov_by_unified = {
        'KMB': {'KMB', 'LWB', 'KMB+CTB', 'KWB'},
        'CTB': {'CTB', 'NWFB'},
        'GMB': {'GMB'},
        'MTRB': {'LRTFeeder'},
        'NLB': {'NLB'},
        'FERRY': {'FERRY'}
    }

    def build_trip_id_mapping_for(unified_prefix: str) -> pd.DataFrame:
        ours = all_trips_df[all_trips_df['trip_id'].str.startswith(f'{unified_prefix}-')].copy()
        if ours.empty:
            return pd.DataFrame(columns=['original_trip_id', 'new_trip_id'])

        # Prefer exact mapping by gov_route_id when available
        if 'gov_route_id' in ours.columns and ours['gov_route_id'].notna().any():
            left = ours[['trip_id', 'original_service_id', 'direction_id', 'gov_route_id']].dropna()
            left.rename(columns={'trip_id': 'new_trip_id', 'gov_route_id': 'route_id'}, inplace=True)
            right = gov_trips_with_route_info[['trip_id', 'service_id', 'direction_id', 'route_id', 'gov_agency_id']]
            m = left.merge(
                right,
                left_on=['original_service_id', 'direction_id', 'route_id'],
                right_on=['service_id', 'direction_id', 'route_id'],
                how='inner'
            )
        else:
            # Fallback: route_short_name scoped by allowed agencies
            allowed = allowed_gov_by_unified.get(unified_prefix, {unified_prefix})
            left = ours[['trip_id', 'original_service_id', 'direction_id', 'route_short_name']].dropna()
            left.rename(columns={'trip_id': 'new_trip_id'}, inplace=True)
            right = gov_trips_with_route_info[
                gov_trips_with_route_info['gov_agency_id'].isin(allowed)
            ][['trip_id', 'service_id', 'direction_id', 'route_short_name', 'gov_agency_id']]
            m = left.merge(
                right,
                left_on=['original_service_id', 'direction_id', 'route_short_name'],
                right_on=['service_id', 'direction_id', 'route_short_name'],
                how='inner'
            )

        if m.empty:
            return pd.DataFrame(columns=['original_trip_id', 'new_trip_id'])

        return m[['trip_id', 'new_trip_id']].rename(columns={'trip_id': 'original_trip_id'}).drop_duplicates()

    # Build and combine mappings for relevant agencies
    mappings = [
        build_trip_id_mapping_for('KMB'),
        build_trip_id_mapping_for('CTB'),
        build_trip_id_mapping_for('GMB'),
        build_trip_id_mapping_for('MTRB'),
        build_trip_id_mapping_for('NLB'),
        build_trip_id_mapping_for('FERRY')
    ]
    trip_id_mapping = pd.concat([m for m in mappings if m is not None and not m.empty], ignore_index=True).drop_duplicates()

    # Update the trip_id in the frequencies dataframe using scoped mapping
    final_frequencies_df = gov_frequencies_df.merge(
        trip_id_mapping,
        left_on='trip_id',
        right_on='original_trip_id',
        how='inner'
    )
    final_frequencies_df['trip_id'] = final_frequencies_df['new_trip_id']
    final_frequencies_df = final_frequencies_df.drop(columns=['new_trip_id', 'original_trip_id'])

    fallback_frequencies_df = generate_frequencies_from_schedule(
        trip_id_mapping,
        final_frequencies_df,
        gov_stop_times_df,
        silent=silent
    )
    if not fallback_frequencies_df.empty:
        final_frequencies_df = pd.concat([final_frequencies_df, fallback_frequencies_df], ignore_index=True)
        if not silent:
            missing_trip_count = fallback_frequencies_df['trip_id'].nunique()
            print(f"Generated fallback frequencies for {missing_trip_count} trip(s) using schedule-derived headways.")

    # --- Ferry Frequencies ---
    if not ferry_frequencies_df.empty:
        final_frequencies_df = pd.concat([final_frequencies_df, ferry_frequencies_df], ignore_index=True)
        if not silent:
            print(f"Added {len(ferry_frequencies_df)} ferry frequency patterns")

    # --- MTR/LR Frequencies ---
    try:
        # Remove any pre-existing MTR/LR frequencies from government feed (they don't apply to our synthetic rail trips)
        if not final_frequencies_df.empty:
            final_frequencies_df = final_frequencies_df[~final_frequencies_df['trip_id'].str.startswith('MTR-')]
            final_frequencies_df = final_frequencies_df[~final_frequencies_df['trip_id'].str.startswith('LR-')]

        if not lr_frequencies_df.empty:
            lr_freq_export = lr_frequencies_df[['trip_id', 'start_time', 'end_time', 'headway_secs']].copy()
            final_frequencies_df = pd.concat([final_frequencies_df, lr_freq_export], ignore_index=True)

        # Collect trips per line
        mtr_line_trips = {}
       
        if not mtr_trips_df.empty:
            tdf = mtr_trips_df.copy()
            tdf['line_code'] = tdf['route_id'].str.replace('MTR-','', regex=False)
            for lc, grp in tdf.groupby('line_code'):
                mtr_line_trips[lc] = grp['trip_id'].unique().tolist()
        lr_line_trips = {}
        if not lr_trips_df.empty:
            ldf = lr_trips_df.copy()
            ldf['line_code'] = ldf['route_id'].str.replace('LR-','', regex=False)
            for lc, grp in ldf.groupby('line_code'):
                lr_line_trips[lc] = grp['trip_id'].unique().tolist()
        lr_trips_with_freq = set()
        if not lr_frequencies_df.empty and 'trip_id' in lr_frequencies_df.columns:
            lr_trips_with_freq = set(lr_frequencies_df['trip_id'].dropna().astype(str))

        abbr_map = {
            'EAL': 'East Rail', 'TML': 'Tuen Ma', 'TWL': 'Tsuen Wan', 'KTL': 'Kwun Tong',
            'ISL': 'Island', 'TKL': 'Tseung Kwan O', 'SIL': 'South Island', 'AEL': 'Airport Express',
            'TCL': 'Tung Chung', 'DRL': 'Disneyland Resort'
        }

        # Normalize scraped headway data
        scraped_info = {}
        if mtr_headway_data:
            for raw_key, data in mtr_headway_data.items():
                if not isinstance(data, dict):
                    continue
                key_norm = raw_key.lower().replace(' line','').strip()
                target = None
                for abbr, base in abbr_map.items():
                    bl = base.lower()
                    if key_norm == abbr.lower() or key_norm == bl or bl in key_norm or abbr.lower() in key_norm:
                        target = abbr
                        break
                if not target:
                    continue
                scraped_info[target] = {
                    'morning_peak': parse_headway_to_avg_secs(data.get('weekdays', {}).get('morning_peak')),
                    'non_peak': parse_headway_to_avg_secs(data.get('weekdays', {}).get('non_peak')),
                    'saturdays': parse_headway_to_avg_secs(data.get('saturdays'))
                }

        # Default daily slices (extended 25:00 for after midnight)
        default_slices = [
            ('05:30:00','07:00:00',360),
            ('07:00:00','10:00:00',180),
            ('10:00:00','17:00:00',300),
            ('17:00:00','20:00:00',180),
            ('20:00:00','23:00:00',360),
            ('23:00:00','25:00:00',600)
        ]

        def build_slices(line_code):
            info = scraped_info.get(line_code, {})
            slices = []
            for start, end, hw in default_slices:
                adj = hw
                # Override peaks
                if start in ('07:00:00','17:00:00') and info.get('morning_peak'):
                    adj = info['morning_peak']
                # Override mid / evening non-peak
                if start in ('10:00:00','20:00:00') and info.get('non_peak'):
                    adj = info['non_peak']
                # Optionally shorten late night if scraped has Saturday spec (ignore for now)
                slices.append((start, end, int(adj)))
            return slices

        new_rows = []
        # Heavy rail
        for lc, trips in mtr_line_trips.items():
            slices = build_slices(lc)
            for trip in trips:
                for start, end, headway in slices:
                    new_rows.append({'trip_id': trip, 'start_time': start, 'end_time': end, 'headway_secs': headway})
        # Light Rail (use defaults only for trips without explicit frequencies)
        for lc, trips in lr_line_trips.items():
            missing_trips = [trip for trip in trips if trip not in lr_trips_with_freq]
            if not missing_trips:
                continue
            slices = build_slices(lc)  # no scraped overrides expected
            for trip in missing_trips:
                for start, end, headway in slices:
                    new_rows.append({'trip_id': trip, 'start_time': start, 'end_time': end, 'headway_secs': headway})

        if new_rows:
            mtr_lr_freq_df = pd.DataFrame(new_rows)
            final_frequencies_df = pd.concat([final_frequencies_df, mtr_lr_freq_df], ignore_index=True)
            if not silent:
                missing_defaults = [lc for lc in mtr_line_trips.keys() if lc not in scraped_info]
                print(f"Applied frequencies for MTR lines {sorted(mtr_line_trips.keys())}; defaults used for {missing_defaults}")
    except Exception as e:
        if not silent:
            print(f"Warning: MTR/LR frequency generation failed: {e}")
    # --- END MTR/LR Frequencies ---

    # Remove true duplicates only - preserve different services for the same route
    final_trips_df = all_trips_df.drop_duplicates(subset=['trip_id'], keep='first')

    stop_times_cols = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
    final_kmb_stoptimes = kmb_stoptimes_df.rename(columns={'seq': 'stop_sequence'})
    final_ctb_stoptimes = ctb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_gmb_stoptimes = gmb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtrbus_stoptimes = mtrbus_stoptimes_df.rename(columns={'station_seqno': 'stop_sequence'})
    final_nlb_stoptimes = nlb_stoptimes_df.rename(columns={'sequence': 'stop_sequence'})
    final_mtr_stoptimes = mtr_stoptimes_df
    final_lr_stoptimes = lr_stoptimes_df
    final_stop_times_df = pd.concat([final_kmb_stoptimes, final_ctb_stoptimes, final_gmb_stoptimes, final_mtrbus_stoptimes, final_nlb_stoptimes, ferry_stoptimes_df, final_mtr_stoptimes, final_lr_stoptimes], ignore_index=True)
    final_stop_times_df['stop_id'] = final_stop_times_df['stop_id'].replace(master_duplicates_map)

    final_stop_times_output_df = final_stop_times_df[stop_times_cols].copy()

    # Enforce unique stop_sequence for each trip
    final_stop_times_output_df.sort_values(['trip_id', 'stop_sequence'], inplace=True)
    final_stop_times_output_df['stop_sequence'] = final_stop_times_output_df.groupby('trip_id').cumcount() + 1
    final_stop_times_output_df.drop_duplicates(subset=['trip_id', 'stop_sequence'], keep='first', inplace=True)

    # Filter stop_times to ensure foreign key constraint with stops.txt
    valid_stop_ids = set(all_stops_output_df['stop_id'])
    original_stop_times_count = len(final_stop_times_output_df)
    final_stop_times_output_df = final_stop_times_output_df[final_stop_times_output_df['stop_id'].isin(valid_stop_ids)]
    if not silent:
        filtered_count = original_stop_times_count - len(final_stop_times_output_df)
        if filtered_count > 0:
            print(f"Warning: Removed {filtered_count} stop_times records that referenced non-existent stops.")

    # --- Government direction inversion warnings ---
    try:
        if not silent:
            print("Checking for government direction inversions (start/end mismatches)...")
            print("Building operator trip endpoints...")

        # Build operator first/last stop coordinates per unified trip
        opst = final_stop_times_output_df.copy()
        opst = opst.sort_values(['trip_id', 'stop_sequence'])
        first_stops = opst.groupby('trip_id').first().reset_index()[['trip_id', 'stop_id']].rename(columns={'stop_id': 'first_stop_id'})
        last_stops = opst.groupby('trip_id').last().reset_index()[['trip_id', 'stop_id']].rename(columns={'stop_id': 'last_stop_id'})
        op_endpoints = first_stops.merge(last_stops, on='trip_id', how='inner')

        if not silent:
            print(f"Found {len(op_endpoints)} operator trips, merging stop coordinates...")

        stops_coords = all_stops_output_df[['stop_id', 'stop_lat', 'stop_lon']].copy()
        op_endpoints = (
            op_endpoints
            .merge(stops_coords.add_prefix('first_'), left_on='first_stop_id', right_on='first_stop_id', how='left')
            .merge(stops_coords.add_prefix('last_'), left_on='last_stop_id', right_on='last_stop_id', how='left')
        )

        # Build government first/last stop coordinates per original gov trip
        if 'original_trip_id' in locals() or 'trip_id_mapping' in locals():
            timap = trip_id_mapping.copy()
            timap.columns = [c.lower() for c in timap.columns]
            if {'new_trip_id', 'original_trip_id'}.issubset(timap.columns):
                gov_needed = timap['original_trip_id'].dropna().unique().tolist()
                if gov_needed:
                    if not silent:
                        print(f"Loading government data for {len(gov_needed)} matched trips...")
                    
                    # Read only required gov stop_times rows
                    qids = "','".join(gov_needed)
                    gov_st = pd.read_sql(f"SELECT trip_id, stop_id, stop_sequence FROM gov_gtfs_stop_times WHERE trip_id IN ('{qids}')", engine)
                    
                    if not silent:
                        print(f"Processing government endpoints...")
                    
                    gov_st = gov_st.sort_values(['trip_id', 'stop_sequence'])
                    gov_first = gov_st.groupby('trip_id').first().reset_index().rename(columns={'stop_id': 'gov_first_stop_id'})
                    gov_last = gov_st.groupby('trip_id').last().reset_index().rename(columns={'stop_id': 'gov_last_stop_id'})
                    gov_endpoints = gov_first.merge(gov_last, on='trip_id', how='inner')
                    gov_stops = pd.read_sql("SELECT stop_id, stop_lat, stop_lon FROM gov_gtfs_stops", engine)
                    gov_endpoints = (
                        gov_endpoints
                        .merge(gov_stops.add_prefix('gov_first_'), left_on='gov_first_stop_id', right_on='gov_first_stop_id', how='left')
                        .merge(gov_stops.add_prefix('gov_last_'), left_on='gov_last_stop_id', right_on='gov_last_stop_id', how='left')
                    )
                    
                    if not silent:
                        print(f"Merging operator and government trip data...")
                    
                    # Combine mapping with endpoints
                    combo = (
                        timap.merge(op_endpoints, left_on='new_trip_id', right_on='trip_id', how='inner')
                             .merge(gov_endpoints, left_on='original_trip_id', right_on='trip_id', how='inner', suffixes=('', '_gov'))
                    )

                    import numpy as np
                    from tqdm import tqdm

                    if not combo.empty:
                        if not silent:
                            print(f"Extracting coordinate arrays for {len(combo)} trip pairs...")
                        
                        # Extract coordinate arrays
                        f_lat = combo['first_stop_lat'].to_numpy()
                        f_lon = combo['first_stop_lon'].to_numpy()
                        l_lat = combo['last_stop_lat'].to_numpy()
                        l_lon = combo['last_stop_lon'].to_numpy()
                        gf_lat = combo['gov_first_stop_lat'].to_numpy()
                        gf_lon = combo['gov_first_stop_lon'].to_numpy()
                        gl_lat = combo['gov_last_stop_lat'].to_numpy()
                        gl_lon = combo['gov_last_stop_lon'].to_numpy()

                        # Split data into chunks for parallel processing
                        num_cores = min(16, os.cpu_count() or 1)
                        chunk_size = max(1000, len(combo) // num_cores)
                        n_chunks = (len(combo) + chunk_size - 1) // chunk_size
                        
                        if not silent:
                            print(f"Processing {len(combo)} trips using {num_cores} cores in {n_chunks} chunks...")
                        
                        chunks = []
                        for i in range(n_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, len(combo))
                            chunks.append((
                                f_lat[start_idx:end_idx],
                                f_lon[start_idx:end_idx],
                                l_lat[start_idx:end_idx],
                                l_lon[start_idx:end_idx],
                                gf_lat[start_idx:end_idx],
                                gf_lon[start_idx:end_idx],
                                gl_lat[start_idx:end_idx],
                                gl_lon[start_idx:end_idx]
                            ))
                        
                        # Parallel processing with progress bar
                        aligned_results = []
                        reversed_results = []
                        
                        if not silent:
                            print("Starting parallel distance computation...")
                        
                        with ProcessPoolExecutor(max_workers=num_cores) as executor:
                            futures = [executor.submit(_compute_direction_distances_chunk, chunk) for chunk in chunks]
                            
                            if not silent:
                                # Use tqdm to show progress as futures complete
                                from concurrent.futures import as_completed
                                for future in tqdm(as_completed(futures), total=len(futures), desc="Computing distances", unit="chunk"):
                                    aligned_chunk, reversed_chunk = future.result()
                                    aligned_results.append(aligned_chunk)
                                    reversed_results.append(reversed_chunk)
                            else:
                                for future in futures:
                                    aligned_chunk, reversed_chunk = future.result()
                                    aligned_results.append(aligned_chunk)
                                    reversed_results.append(reversed_chunk)
                        
                        if not silent:
                            print("Combining results and detecting inversions...")
                        
                        # Combine results
                        aligned = np.concatenate(aligned_results)
                        reversed_sum = np.concatenate(reversed_results)

                        candidates_mask = reversed_sum + 200 < aligned
                        if candidates_mask.any():
                            wdf = pd.DataFrame({
                                'unified_trip_id': combo.loc[candidates_mask, 'new_trip_id'].values,
                                'gov_trip_id': combo.loc[candidates_mask, 'original_trip_id'].values,
                                'aligned_total_m': aligned[candidates_mask].astype(int),
                                'reversed_total_m': reversed_sum[candidates_mask].astype(int)
                            })
                        else:
                            wdf = pd.DataFrame()
                    else:
                        wdf = pd.DataFrame()

                    if not wdf.empty:
                        # Auto-invert if reversed_total_m < 500 (heuristic threshold)
                        auto_invert_ids = wdf[wdf['reversed_total_m'] >= 0][wdf['reversed_total_m'] < 500]['unified_trip_id'].tolist()
                        if auto_invert_ids:
                            if not silent:
                                print(f"Auto-inverting direction for {len(auto_invert_ids)} trip(s) (reversed_total_m < 500).")
                            def swap_bound(val: str) -> str:
                                if not isinstance(val, str):
                                    return val
                                low = val.lower()
                                if low == 'o' or low == 'outbound':
                                    return 'I' if val.isupper() else ('inbound' if low == 'outbound' else 'I')
                                if low == 'i' or low == 'inbound':
                                    return 'O' if val.isupper() else ('outbound' if low == 'inbound' else 'O')
                                return val
                            # Update final_trips_df direction_id & bound (if present) IN-PLACE
                            mask = final_trips_df['trip_id'].isin(auto_invert_ids)
                            if 'direction_id' in final_trips_df.columns:
                                final_trips_df.loc[mask, 'direction_id'] = final_trips_df.loc[mask, 'direction_id'].apply(lambda d: 1 - int(d) if pd.notna(d) and str(d).isdigit() else d)
                            for bcol in ['bound', 'dir']:
                                if bcol in final_trips_df.columns:
                                    final_trips_df.loc[mask, bcol] = final_trips_df.loc[mask, bcol].apply(swap_bound)
                            # Mark actions
                            wdf['action'] = wdf['unified_trip_id'].apply(lambda x: 'AUTO_INVERTED' if x in auto_invert_ids else 'WARN_ONLY')
                        else:
                            wdf['action'] = 'WARN_ONLY'
                        warn_path = os.path.join(final_output_dir, 'warnings_direction_inversions.txt')
                        wdf.to_csv(warn_path, index=False)
                        if not silent:
                            print(f"Direction inversion warnings written: {len(wdf)} (auto-inverted {wdf[wdf['action']=='AUTO_INVERTED'].shape[0]}) -> {warn_path}")
                    else:
                        if not silent:
                            print("No direction inversion cases detected.")
    except Exception as e:
        if not silent:
            print(f"Warning: direction inversion check failed: {e}")
        import traceback
        if not silent:
            traceback.print_exc()

    # --- Shapes --- 
    if not silent:
        print("Generating shapes from CSDI data...")
    if no_regenerate_shapes and os.path.exists(os.path.join(final_output_dir, 'shapes.txt')):
        if not silent:
            print("Skipping shape generation as --no-regenerate-shapes is set and shapes.txt already exists.")
        success = True
        with open(os.path.join(final_output_dir, 'shapes.txt'), 'r') as f:
            # this is a hack to get the shape_info
            shape_info = []
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                shape_id = parts[0]
                if shape_id not in [s['shape_id'] for s in shape_info]:
                    gov_route_id, bound = shape_id.split('-')[1:]
                    shape_info.append({'shape_id': shape_id, 'gov_route_id': gov_route_id, 'bound': bound})
    else:
        success, shape_info = generate_shapes_from_csdi_files(os.path.join(final_output_dir, 'shapes.txt'), engine, silent=silent)
    if success:
        final_trips_df = match_trips_to_csdi_shapes(final_trips_df, shape_info, engine, silent=silent)

    # Standardize trips.txt output & add headsigns
    if not silent:
        print("Generating trip headsigns (EN)...")
    with PhaseTimer('Headsigns', phase_timings, silent):
        final_trips_df = generate_trip_headsigns(engine, final_trips_df, silent=silent)
    final_trips_df['trip_headsign_tc'] = generate_trip_headsigns_tc(
        engine,
        final_trips_df,
        english_headsigns=final_trips_df.get('trip_headsign'),
        silent=silent
    )
    gtfs_trips_cols = ['route_id', 'service_id', 'trip_id', 'trip_headsign', 'direction_id', 'shape_id']
    # Ensure all columns exist, fill with None if they don't
    for col in gtfs_trips_cols:
        if col not in final_trips_df.columns:
            final_trips_df[col] = None
    final_trips_output_df = final_trips_df[gtfs_trips_cols]


    with PhaseTimer('Write core tables', phase_timings, silent):
        final_routes_output_df.to_csv(os.path.join(final_output_dir, 'routes.txt'), index=False)
        final_trips_output_df.to_csv(os.path.join(final_output_dir, 'trips.txt'), index=False)
        final_stop_times_output_df.to_csv(os.path.join(final_output_dir, 'stop_times.txt'), index=False)

    # Resolve overlapping frequencies before saving
    final_frequencies_df = resolve_overlapping_frequencies(final_frequencies_df)
    frequencies_cols = ['trip_id', 'start_time', 'end_time', 'headway_secs']
    final_frequencies_output_df = final_frequencies_df[frequencies_cols]
    with PhaseTimer('Write frequencies', phase_timings, silent):
        final_frequencies_output_df.to_csv(os.path.join(final_output_dir, 'frequencies.txt'), index=False)

    # --- Pathways (platform-to-exit, platform-to-platform) ---
    try:
        pathways_rows = []

        # 1) Load platform->exit journeys
        def _norm_text(val):
            if val is None:
                return None
            try:
                return str(val).strip().upper()
            except Exception:
                return None

        def _parse_exit_code(exrec: dict) -> str:
            if not isinstance(exrec, dict):
                return None
            ref = exrec.get('ref')
            if ref is not None and str(ref).strip() != '':
                return str(ref).strip()
            name_en = exrec.get('name_en') or ''
            name_zh = exrec.get('name_zh') or ''
            import re
            m = re.search(r"Exit\s+([A-Z][0-9]*)", str(name_en), flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m2 = re.match(r"([A-Z][0-9]*)", str(name_zh), flags=re.IGNORECASE)
            if m2:
                return m2.group(1).upper()
            return None
        try:
            j_url = "https://github.com/wheelstransit/mtr-platform-exits-crawler/raw/refs/heads/main/data/output/journeys.json"
            journeys = requests.get(j_url, timeout=100).json()
        except Exception:
            journeys = []

        # Map (station_name_en, exit_code) -> stop_id using DB entrances
        db_entrance_map = {}
        if 'stop_id' in mtr_entrances_df.columns and not mtr_entrances_df.empty:
            for rec in mtr_entrances_df.to_dict('records'):
                try:
                    tail = rec.get('stop_id', '')
                    if 'MTR-ENTRANCE-' in tail:
                        # Parse either new format (MTR-ENTRANCE-{station_code}-{exit}) or old format (MTR-ENTRANCE-{station_name}-{exit})
                        station_identifier, exit_code = tail.split('MTR-ENTRANCE-')[1].rsplit('-', 1)
                        key = (_norm_text(station_identifier), _norm_text(exit_code))
                        db_entrance_map[key] = rec.get('stop_id')
                except Exception:
                    continue

        scode_to_name = dict(mtr_stations_df[['station_code','stop_name']].values)
        external_exits = (mtr_meta.get('exits') or {}) if isinstance(mtr_meta, dict) else {}

        for rec in journeys:
            p_aid = rec.get('platform_amenity_id')
            e_aid = rec.get('exit_amenity_id')
            if not p_aid or not e_aid:
                continue
            from_id = platform_amenity_to_stop_id.get(p_aid)

            to_id = None
            scode = rec.get('station_code')
            station_name_en = scode_to_name.get(scode)
            if scode:
                # find exit ref by amenity id
                try:
                    ex_key = next((k for k, v in external_exits.items() if v.get('amenity_id') == e_aid), None)
                    if ex_key:
                        exrec = external_exits.get(ex_key, {})
                        exit_code = _parse_exit_code(exrec)
                        if exit_code:
                            # Try station_code first (new format), fall back to station_name_en (old format)
                            to_id = db_entrance_map.get((_norm_text(scode), _norm_text(exit_code)))
                            if not to_id and station_name_en:
                                to_id = db_entrance_map.get((_norm_text(station_name_en), _norm_text(exit_code)))
                except Exception:
                    to_id = None

            if not from_id or not to_id:
                continue

            length_m = rec.get('walk_distance_metres')
            time_s = None
            try:
                time_s = float(rec.get('walk_time_minutes', 0)) * 60.0
            except Exception:
                time_s = None

            pathways_rows.append({
                'pathway_id': f"MTR-PATH-PEX-{p_aid}-{e_aid}",
                'from_stop_id': from_id,
                'to_stop_id': to_id,
                'pathway_mode': 1,
                'is_bidirectional': 1,
                'length': length_m,
                'traversal_time': time_s,
                'signposted_as': None,
                'reversed_signposted_as': None
            })

        # 2) Load platform->platform journeys
        try:
            pp_url = "https://github.com/wheelstransit/mtr-platform-exits-crawler/raw/refs/heads/main/data/output/platform_journeys.json"
            pjourneys = requests.get(pp_url, timeout=20).json()
        except Exception:
            pjourneys = []

        for rec in pjourneys:
            s_aid = rec.get('start_platform_amenity_id')
            t_aid = rec.get('end_platform_amenity_id')
            if not s_aid or not t_aid:
                continue
            from_id = platform_amenity_to_stop_id.get(s_aid)
            to_id = platform_amenity_to_stop_id.get(t_aid)
            if not from_id or not to_id:
                continue
            length_m = rec.get('walk_distance_metres')
            try:
                time_s = float(rec.get('walk_time_minutes', 0)) * 60.0
            except Exception:
                time_s = None
            pathways_rows.append({
                'pathway_id': f"MTR-PATH-PP-{s_aid}-{t_aid}",
                'from_stop_id': from_id,
                'to_stop_id': to_id,
                'pathway_mode': 1,
                'is_bidirectional': 1,
                'length': length_m,
                'traversal_time': time_s,
                'signposted_as': None,
                'reversed_signposted_as': None
            })

        # 3) Fill gaps with dummies (60s) between all platforms and entrances inside a station, and between platforms
        existing_pairs = {tuple(sorted((r.get('from_stop_id'), r.get('to_stop_id')))) for r in pathways_rows}
        station_platform_ids = {scode: [platform_amenity_to_stop_id.get(p['amenity_id']) for p in plist if platform_amenity_to_stop_id.get(p['amenity_id'])] for scode, plist in station_to_platforms.items()}
        station_entrance_ids = {}
        if 'parent_station' in mtr_entrances_df.columns and not mtr_entrances_df.empty:
            for scode, group in mtr_entrances_df.groupby(mtr_entrances_df['parent_station'].str.replace('MTR-','')):
                station_entrance_ids[scode] = group['stop_id'].tolist()

        for scode, pids in station_platform_ids.items():
            for pid in pids:
                for eid in station_entrance_ids.get(scode, []):
                    key = tuple(sorted((pid, eid)))
                    if key not in existing_pairs:
                        pathways_rows.append({'pathway_id': f"DUMMY-PEX-{pid}-{eid}", 'from_stop_id': pid, 'to_stop_id': eid, 'pathway_mode': 1, 'is_bidirectional': 1, 'traversal_time': 60})

        from itertools import combinations
        for scode, pids in station_platform_ids.items():
            for p1, p2 in combinations(pids, 2):
                key = tuple(sorted((p1, p2)))
                if key not in existing_pairs:
                    pathways_rows.append({'pathway_id': f"DUMMY-PP-{p1}-{p2}", 'from_stop_id': p1, 'to_stop_id': p2, 'pathway_mode': 1, 'is_bidirectional': 1, 'traversal_time': 60})

        if pathways_rows:
            pathways_df = pd.DataFrame(pathways_rows)
            for col in ['length', 'traversal_time']:
                pathways_df[col] = pd.to_numeric(pathways_df[col], errors='coerce').round().astype('Int64')
            # Persist only pathways.txt (transfers.txt omitted)
            pathways_df[['pathway_id','from_stop_id','to_stop_id','pathway_mode','is_bidirectional','length','traversal_time']].to_csv(os.path.join(final_output_dir,'pathways.txt'), index=False)
            if not silent:
                print(f"Generated pathways.txt with {len(pathways_df)} pathways.")
    except Exception as e:
        if not silent:
            print(f"Warning: failed to build pathways.txt: {e}")

    # --- 4. Handle `calendar.txt` and `calendar_dates.txt` ---
    if not silent:
        print("Processing calendar and calendar_dates...")

    # Get the base calendar and calendar_dates data
    gov_calendar_df = pd.read_sql("SELECT * FROM gov_gtfs_calendar", engine)
    gov_calendar_df['service_id'] = gov_calendar_df['service_id'].astype(str)
    gov_calendar_df = gov_calendar_df.set_index('service_id')

    gov_calendar_dates_df = pd.read_sql("SELECT * FROM gov_gtfs_calendar_dates", engine)
    gov_calendar_dates_df['service_id'] = gov_calendar_dates_df['service_id'].astype(str)
    gov_calendar_dates_df = gov_calendar_dates_df.set_index('service_id')

    # Create a dataframe with the mapping between new and original service IDs
    service_id_mapping_df = final_trips_df[['service_id', 'original_service_id']].drop_duplicates()

    # Merge the mapping with the calendar data
    # Use a left merge to ensure all service_ids from trips.txt are kept.
    final_calendar_df = service_id_mapping_df.merge(
        gov_calendar_df,
        left_on='original_service_id',
        right_index=True,
        how='left'
    )

    # For trips that didn't have a matching calendar entry (i.e., our default services),
    # create a default calendar entry that runs every day.
    default_calendar_values = {
        'monday': 1, 'tuesday': 1, 'wednesday': 1, 'thursday': 1,
        'friday': 1, 'saturday': 1, 'sunday': 1,
        'start_date': '20230101', 'end_date': '20251231'
    }
    # Fill NaN values for the default services
    for col, val in default_calendar_values.items():
        final_calendar_df[col] = final_calendar_df[col].fillna(val)

    # Ensure integer types for date and day columns
    date_cols = ['start_date', 'end_date']
    day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for col in date_cols + day_cols:
        # Convert to numeric first to handle potential floats from NaN fill, then to int
        final_calendar_df[col] = pd.to_numeric(final_calendar_df[col]).astype(int)

    final_calendar_df = final_calendar_df.drop(columns=['original_service_id'])

    # For calendar_dates, we only want entries that actually existed in the gov data.
    # An inner join is correct here, as we don't want to create default exception dates.
    final_calendar_dates_df = service_id_mapping_df.merge(
        gov_calendar_dates_df,
        left_on='original_service_id',
        right_index=True,
        how='inner'
    )
    final_calendar_dates_df = final_calendar_dates_df.drop(columns=['original_service_id'])

    # why are there dupes??
    final_calendar_df = final_calendar_df.drop_duplicates(subset=['service_id'])
    final_calendar_dates_df = final_calendar_dates_df.drop_duplicates(subset=['service_id', 'date'])

    # Add dummy "NEVER" service entries for unmatched routes (for GTFS-RT activation)
    # These services have dates in the past so they never actually run
    dummy_service_ids = final_trips_df[final_trips_df['original_service_id'] == 'NEVER']['service_id'].unique()
    if len(dummy_service_ids) > 0:
        if not silent:
            print(f"Creating {len(dummy_service_ids)} dummy calendar entries for NEVER services...")
        dummy_calendar_entries = []
        for service_id in dummy_service_ids:
            dummy_calendar_entries.append({
                'service_id': service_id,
                'monday': 0,
                'tuesday': 0,
                'wednesday': 0,
                'thursday': 0,
                'friday': 0,
                'saturday': 0,
                'sunday': 0,
                'start_date': 19700101,  # Very old date, service never runs
                'end_date': 19700101
            })
        dummy_calendar_df = pd.DataFrame(dummy_calendar_entries)
        final_calendar_df = pd.concat([final_calendar_df, dummy_calendar_df], ignore_index=True)
        final_calendar_df = final_calendar_df.drop_duplicates(subset=['service_id'])

    with PhaseTimer('Write calendar', phase_timings, silent):
        final_calendar_df.to_csv(os.path.join(final_output_dir, 'calendar.txt'), index=False)
        final_calendar_dates_df.to_csv(os.path.join(final_output_dir, 'calendar_dates.txt'), index=False)

    ## --- Translations (Traditional Chinese) ---
    if not silent:
        print("Building translations.txt (Traditional Chinese)...")
    translations = []
    lang_tc = "zh-Hant"
    
    # Stop name translations
    try:
        if 'kmb_stops_gdf' in locals() and 'name_tc' in kmb_stops_gdf.columns:
            kmb_tc_clean = (
                kmb_stops_gdf['name_tc']
                .str.replace(r'\s*\([A-Za-z0-9]{5}\)', '', regex=True)
                .str.replace(r'\s*-\s*', ' - ', regex=True)
                .str.replace(r'([^\s])(\([A-Za-z0-9]+\))', r'\1 \2', regex=True)
            )
            for stop_id, name_tc in zip(kmb_stops_gdf['stop_id'], kmb_tc_clean):
                if pd.notna(name_tc) and str(name_tc).strip():
                    translations.append({
                        'table_name':'stops.txt',
                        'field_name':'stop_name',
                        'language':lang_tc,
                        'record_id':stop_id,
                        'translation':name_tc
                    })
        if 'ctb_stops_gdf' in locals() and 'name_tc' in ctb_stops_gdf.columns:
            for r in ctb_stops_gdf[['stop_id','name_tc']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.name_tc})
        if 'gmb_stops_gdf' in locals() and 'stop_name_tc' in gmb_stops_gdf.columns:
            for r in gmb_stops_gdf[['stop_id','stop_name_tc']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.stop_name_tc})
        if 'nlb_stops_gdf' in locals() and 'stopName_c' in nlb_stops_gdf.columns:
            for r in nlb_stops_gdf[['stop_id','stopName_c']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.stopName_c})
        if 'mtr_stations_gdf' in locals() and {'Station Code','Chinese Name'}.issubset(mtr_stations_gdf.columns):
            station_names_df = (
                mtr_stations_gdf[['Station Code', 'Chinese Name']]
                .rename(columns={'Station Code': 'station_code', 'Chinese Name': 'name_tc'})
                .dropna(subset=['station_code', 'name_tc'])
                .drop_duplicates(subset=['station_code'])
            )
            for r in station_names_df.itertuples(index=False):
                name_tc = str(r.name_tc).strip()
                if name_tc:
                    translations.append({
                        'table_name': 'stops.txt',
                        'field_name': 'stop_name',
                        'language': lang_tc,
                        'record_id': f"MTR-{r.station_code}",
                        'translation': name_tc
                    })
        if 'mtr_exits_gdf' in locals() and 'station_name_zh' in mtr_exits_gdf.columns:
            for r in mtr_exits_gdf[['stop_id','station_name_zh','exit']].dropna(subset=['station_name_zh']).itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':f"{r.station_name_zh} 出口 {r.exit}"})
        if 'mtrbus_stops_gdf' in locals() and 'name_zh' in mtrbus_stops_gdf.columns:
            for r in mtrbus_stops_gdf[['stop_id','name_zh']].dropna(subset=['name_zh']).itertuples():
                name_tc = str(r.name_zh).strip()
                if name_tc:
                    translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':name_tc})
        if 'lr_stops_gdf' in locals() and 'name_tc' in lr_stops_gdf.columns:
            for r in lr_stops_gdf[['stop_id','name_tc']].dropna().itertuples():
                translations.append({'table_name':'stops.txt','field_name':'stop_name','language':lang_tc,'record_id':r.stop_id,'translation':r.name_tc})
    except Exception as e:
        if not silent:
            print(f"Stop translations warning: {e}")
    
    # Route long name translations
    try:
        if 'kmb_routes_df' in locals() and not kmb_routes_df.empty:
            for route_num, grp in kmb_routes_df.groupby('route'):
                fo = grp[grp['bound']=='O']
                fi = grp[grp['bound']=='I']
                fo_row = fo.iloc[0] if not fo.empty else (grp.iloc[0] if not grp.empty else None)
                fi_row = fi.iloc[0] if not fi.empty else None
                if fo_row is not None and fi_row is not None:
                    cn_long = f"{fo_row.get('orig_tc','')} - {fi_row.get('orig_tc','')}"
                elif fo_row is not None:
                    cn_long = f"{fo_row.get('orig_tc','')} - {fo_row.get('dest_tc','')}"
                elif fi_row is not None:
                    cn_long = f"{fi_row.get('orig_tc','')} - {fi_row.get('dest_tc','')}"
                else:
                    cn_long = ''
                translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':f"KMB-{route_num}",'translation':cn_long})
        if 'ctb_routes_df' in locals() and not ctb_routes_df.empty:
            for route_num, grp in ctb_routes_df.groupby('route'):
                fo = grp[grp['dir']=='outbound']
                fi = grp[grp['dir']=='inbound']
                fo_row = fo.iloc[0] if not fo.empty else grp.iloc[0]
                fi_row = fi.iloc[0] if not fi.empty else fo_row
                cn_long = f"{fo_row.get('orig_tc','')} - {fi_row.get('orig_tc','')}"
                translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':f"CTB-{route_num}",'translation':cn_long})
        if 'nlb_routes_df' in locals() and not nlb_routes_df.empty and 'routeName_c' in nlb_routes_df.columns:
            for r in nlb_routes_df[['routeNo','routeName_c']].dropna().itertuples():
                translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':f"NLB-{r.routeNo}",'translation':r.routeName_c.replace(' > ', ' - ')})
        if 'mtr_routes_df' in locals() and 'mtr_stations_gdf' in locals() and 'Chinese Name' in mtr_stations_gdf.columns:
            try:
                cn_pairs = {}
                for lc, grp in mtr_stations_gdf.groupby('Line Code'):
                    ordered = grp.sort_values('Sequence', key=lambda s: s.astype(int) if s.dtype==object else s)
                    if ordered.empty:
                        continue
                    first_cn = ordered.iloc[0]['Chinese Name']
                    last_cn = ordered.iloc[-1]['Chinese Name'] if len(ordered)>1 else first_cn
                    cn_pairs[lc] = f"{first_cn} - {last_cn}"
                for r in mtr_routes_df.itertuples():
                    translations.append({'table_name':'routes.txt','field_name':'route_long_name','language':lang_tc,'record_id':r.route_id,'translation':cn_pairs.get(r.route_short_name, r.route_long_name)})
            except Exception:
                pass
        if 'lr_routes_df' in locals() and 'route_long_name_tc' in lr_routes_df.columns:
            lr_cn = (
                lr_routes_df[['route_id', 'route_long_name_tc']]
                .dropna(subset=['route_id', 'route_long_name_tc'])
            )
            for r in lr_cn.itertuples(index=False):
                translation = str(r.route_long_name_tc).strip()
                if translation:
                    translations.append({
                        'table_name': 'routes.txt',
                        'field_name': 'route_long_name',
                        'language': lang_tc,
                        'record_id': r.route_id,
                        'translation': translation
                    })
    except Exception as e:
        if not silent:
            print(f"Route translations warning: {e}")

    # Trip headsign translations
    try:
        if 'final_trips_df' in locals() and 'trip_headsign_tc' in final_trips_df.columns:
            trip_tc = final_trips_df[['trip_id', 'trip_headsign_tc']].dropna(subset=['trip_id'])
            trip_tc = trip_tc[trip_tc['trip_headsign_tc'].astype(str).str.strip() != '']
            for r in trip_tc.itertuples(index=False):
                translations.append({
                    'table_name': 'trips.txt',
                    'field_name': 'trip_headsign',
                    'language': lang_tc,
                    'record_id': r.trip_id,
                    'translation': r.trip_headsign_tc
                })
    except Exception as e:
        if not silent:
            print(f"Trip headsign translations warning: {e}")
    
    if translations:
        translations_df = pd.DataFrame(translations).drop_duplicates(subset=['table_name','field_name','language','record_id'])
        with PhaseTimer('Write translations', phase_timings, silent):
            translations_df.to_csv(os.path.join(final_output_dir,'translations.txt'), index=False)
        if not silent:
            print(f"Generated translations.txt with {len(translations_df)} records.")
    else:
        if not silent:
            print("No translation records generated.")
    
    # --- Generate Fare Files (ezfares format) ---
    if not silent:
        print("Generating fare_stages.csv...")
    with PhaseTimer('Generate fare_stages', phase_timings, silent):
        # use the final filtered/normalized stop_times table produced above
        fare_stages_df = generate_fare_stages(engine, final_trips_df, final_stop_times_output_df, silent=silent)
        if not fare_stages_df.empty:
            fare_stages_df.to_csv(os.path.join(final_output_dir, 'fare_stages.csv'), index=False)
            if not silent:
                print(f"Generated fare_stages.csv with {len(fare_stages_df)} records.")
        else:
            if not silent:
                print("No fare stages generated.")

    if not silent:
        print("Generating special_fare_rules.csv...")
    with PhaseTimer('Generate special_fare_rules', phase_timings, silent):
        # Generate government GTFS special fares (trip-level)
        special_fare_rules_df = generate_special_fare_rules(engine, final_trips_df, final_stop_times_output_df, silent=silent)
        
        # Generate MTR heavy rail agency-level fares
        mtr_fare_rules_df = generate_mtr_special_fare_rules(engine, silent=silent)
        
        # Generate Light Rail agency-level fares
        lr_fare_rules_df = generate_light_rail_special_fare_rules(engine, silent=silent)
        
        # Combine all special fare rules
        all_fare_rules = []
        if not special_fare_rules_df.empty:
            all_fare_rules.append(special_fare_rules_df)
        if not mtr_fare_rules_df.empty:
            all_fare_rules.append(mtr_fare_rules_df)
        if not lr_fare_rules_df.empty:
            all_fare_rules.append(lr_fare_rules_df)
        
        if all_fare_rules:
            combined_special_fare_rules_df = pd.concat(all_fare_rules, ignore_index=True)
            combined_special_fare_rules_df.to_csv(os.path.join(final_output_dir, 'special_fare_rules.csv'), index=False)
            if not silent:
                print(f"Generated special_fare_rules.csv with {len(combined_special_fare_rules_df)} total records:")
                if not special_fare_rules_df.empty:
                    print(f"  - {len(special_fare_rules_df)} trip-level rules from government GTFS")
                if not mtr_fare_rules_df.empty:
                    print(f"  - {len(mtr_fare_rules_df)} agency-level rules for MTR heavy rail")
                if not lr_fare_rules_df.empty:
                    print(f"  - {len(lr_fare_rules_df)} agency-level rules for Light Rail")
        else:
            if not silent:
                print("No special fare rules generated (empty file created as placeholder).")
            # Create empty DataFrame with proper schema
            empty_df = pd.DataFrame(columns=['special_fare_id', 'rule_type', 'trip_id', 'onboarding_stop_id', 'offboarding_stop_id', 'price', 'currency'])
            empty_df.to_csv(os.path.join(final_output_dir, 'special_fare_rules.csv'), index=False)
    
    # --- 5. Zip the feed ---
    if not silent:
        print("Zipping the GTFS feed...")
    zip_path = os.path.join(output_dir, 'hk.gtfs.zip')
    with PhaseTimer('Zip feed', phase_timings, silent):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in os.listdir(final_output_dir):
                zf.write(os.path.join(final_output_dir, filename), arcname=filename)

    # i'll implament gtfs dense encoding later

    if not silent:
        print(f"--- Unified GTFS Build Complete. Output at {zip_path} ---")
        timings_path = os.path.join(final_output_dir, 'build_timings.json')
        with open(timings_path, 'w') as tf:
            json.dump(phase_timings, tf, indent=2)
        print(f"Timing details written to {timings_path}")
