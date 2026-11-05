import logging
import math
from typing import Optional, Tuple, Union

import pandas as pd

from ..db import Store
from .text import smart_title_case

log = logging.getLogger(__name__)

def _load_mtr_terminal_maps(db: Store) -> Tuple[dict, dict]:
    """(line_code, direction variant) -> terminal station name, as (English, Chinese) lookups."""
    try:
        query = '''
            SELECT "Line Code" AS line_code,
                   "Direction" AS direction,
                   "English Name" AS station_name_en,
                   "Chinese Name" AS station_name_tc,
                   "Sequence" AS sequence
            FROM mtr_lines_and_stations
        '''
        df = db.read(query)
    except Exception:
        return {}, {}

    if df.empty:
        return {}, {}

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

        if en_value and en_value.lower() != 'nan':
            en_map[(rec.line_code, dir_raw)] = en_value
        if tc_value and tc_value.lower() != 'nan':
            tc_map[(rec.line_code, dir_raw)] = tc_value

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

def generate_trip_headsigns(db: Store, trips_df: pd.DataFrame) -> pd.DataFrame:
    """English trip_headsign per trip: the operator's destination for the trip's direction, else the route number."""
    if trips_df.empty:
        trips_df['trip_headsign'] = ''
        return trips_df

    work_df = trips_df.copy()
    if 'trip_headsign' not in work_df.columns:
        work_df['trip_headsign'] = ''

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

    try:
        kmb_mask = work_df['agency_id'] == 'KMB'
        if kmb_mask.any():
            kmb_subset = work_df.loc[kmb_mask]
            if 'service_type' not in kmb_subset.columns:
                service_type_series = kmb_subset['trip_id'].str.split('-').str[3]
            else:
                service_type_series = kmb_subset['service_type']
            kmb_routes = db.read("SELECT unique_route_id, dest_en FROM kmb_routes")
            parts = kmb_routes['unique_route_id'].str.split('_', expand=True)
            kmb_routes['route'] = parts[0]; kmb_routes['bound'] = parts[1]; kmb_routes['service_type'] = parts[2]
            kmb_routes['dest_en'] = kmb_routes['dest_en'].apply(smart_title_case)
            kmb_lookup = kmb_routes.drop_duplicates(['route','bound','service_type']).set_index(['route','bound','service_type'])['dest_en'].to_dict()
            dests = []
            for idx, row in kmb_subset.iterrows():
                key = (row.get('route_short_name'), row.get('bound'), service_type_series.loc[idx])
                dests.append(kmb_lookup.get(key, ''))
            work_df.loc[kmb_subset.index, 'trip_headsign'] = dests
    except Exception as e:
        log.warning(f"Headsign warning (KMB map): {e}")

    try:
        ctb_mask = work_df['agency_id'] == 'CTB'
        if ctb_mask.any():
            ctb_subset = work_df.loc[ctb_mask]
            city_routes = db.read("SELECT route, direction, dest_en FROM citybus_routes")
            city_lookup = city_routes.drop_duplicates(['route','direction']).set_index(['route','direction'])['dest_en'].to_dict()
            mapped = []
            for idx, row in ctb_subset.iterrows():
                direction_str = {'O':'outbound','I':'inbound'}.get(row.get('bound'),'outbound')
                mapped.append(city_lookup.get((row.get('route_short_name'), direction_str), work_df.at[idx,'trip_headsign']))
            work_df.loc[ctb_subset.index, 'trip_headsign'] = mapped
    except Exception as e:
        log.warning(f"Headsign warning (CTB map): {e}")

    try:
        nlb_mask = work_df['agency_id'] == 'NLB'
        if nlb_mask.any():
            nlb_subset = work_df.loc[nlb_mask]
            nlb_routes = db.read('SELECT "routeNo" as route_no, "routeName_e" as route_name FROM nlb_routes')
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
        log.warning(f"Headsign warning (NLB map): {e}")

    try:
        mtrb_mask = work_df['agency_id'] == 'MTRB'
        if mtrb_mask.any():
            mtr_subset = work_df.loc[mtrb_mask]
            mtr_routes_df = db.read("SELECT route_id, route_name_eng FROM mtrbus_routes")
            mtr_routes_df['route_short_name'] = mtr_routes_df['route_id'].astype(str)
            mtr_routes_df = mtr_routes_df.drop_duplicates('route_short_name', keep='first')
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
        log.warning(f"Headsign warning (MTRB map): {e}")

    try:
        gmb_mask = work_df['agency_id'] == 'GMB'
        if gmb_mask.any():
            gmb_routes = db.read('SELECT region, route_code, orig_en_primary, dest_en_primary, is_circular_any FROM gmb_routes')
            gmb_routes['route_short_name'] = gmb_routes['region'].astype(str)+'-'+gmb_routes['route_code'].astype(str)
            gmb_routes = gmb_routes.drop_duplicates('route_short_name', keep='first')
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
        log.warning(f"Headsign warning (GMB map): {e}")

    try:
        mtrr_mask = work_df['agency_id'] == 'MTRR'
        if mtrr_mask.any():
            terminal_map_en, _ = _load_mtr_terminal_maps(db)
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
        log.warning(f"Headsign warning (MTR Rail map): {e}")

    if 'route_short_name' in work_df.columns:
        empty_mask = (work_df['trip_headsign'].isna()) | (work_df['trip_headsign'].astype(str).str.strip()=='')
        work_df.loc[empty_mask, 'trip_headsign'] = work_df.loc[empty_mask, 'route_short_name'].astype(str)
    else:
        work_df['__tmp_route_short_name'] = work_df['route_id'].astype(str).apply(lambda rid: rid.split('-', 1)[1] if isinstance(rid, str) and '-' in rid else str(rid))
        empty_mask = (work_df['trip_headsign'].isna()) | (work_df['trip_headsign'].astype(str).str.strip()=='')
        work_df.loc[empty_mask, 'trip_headsign'] = work_df.loc[empty_mask, '__tmp_route_short_name'].astype(str)

    try:
        gmb_sec_mask = work_df['agency_id']=='GMB'
        if gmb_sec_mask.any():
            gmb_routes = db.read('SELECT region, route_code, orig_en_primary, dest_en_primary FROM gmb_routes')
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
        log.warning(f"Headsign second-pass warning (GMB): {e}")
    work_df['trip_headsign'] = work_df['trip_headsign'].fillna('')
    work_df['trip_headsign'] = work_df['trip_headsign'].astype(str)
    work_df['trip_headsign'] = work_df['trip_headsign'].replace({'nan': '', 'None': ''})
    return work_df

def generate_trip_headsigns_tc(db: Store, trips_df: pd.DataFrame, english_headsigns: Optional[pd.Series] = None) -> pd.Series:
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

    try:
        kmb_mask = work_df['agency_id'] == 'KMB'
        if kmb_mask.any():
            kmb_subset = work_df.loc[kmb_mask]
            if 'service_type' not in kmb_subset.columns:
                service_type_series = kmb_subset['trip_id'].str.split('-').str[3]
            else:
                service_type_series = kmb_subset['service_type']
            kmb_routes = db.read("SELECT unique_route_id, dest_tc FROM kmb_routes")
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
        log.warning(f"Headsign TC warning (KMB map): {e}")

    try:
        ctb_mask = work_df['agency_id'] == 'CTB'
        if ctb_mask.any():
            ctb_subset = work_df.loc[ctb_mask]
            city_routes = db.read("SELECT route, direction, dest_tc FROM citybus_routes")
            city_lookup = city_routes.drop_duplicates(['route','direction']).set_index(['route','direction'])['dest_tc'].to_dict()
            mapped = []
            for idx, row in ctb_subset.iterrows():
                direction_str = {'O':'outbound','I':'inbound'}.get(row.get('bound'),'outbound')
                mapped.append(city_lookup.get((row.get('route_short_name'), direction_str), headsigns.at[idx]))
            headsigns.loc[ctb_subset.index] = mapped
    except Exception as e:
        log.warning(f"Headsign TC warning (CTB map): {e}")

    try:
        nlb_mask = work_df['agency_id'] == 'NLB'
        if nlb_mask.any():
            nlb_subset = work_df.loc[nlb_mask]
            nlb_routes = db.read('SELECT "routeNo" as route_no, "routeName_c" as route_name FROM nlb_routes')
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
        log.warning(f"Headsign TC warning (NLB map): {e}")

    try:
        mtrb_mask = work_df['agency_id'] == 'MTRB'
        if mtrb_mask.any():
            mtr_subset = work_df.loc[mtrb_mask]
            mtr_routes_df = db.read("SELECT route_id, route_name_chi FROM mtrbus_routes")
            mtr_routes_df['route_short_name'] = mtr_routes_df['route_id'].astype(str)
            mtr_routes_df = mtr_routes_df.drop_duplicates('route_short_name', keep='first')

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
        log.warning(f"Headsign TC warning (MTRB map): {e}")

    try:
        gmb_mask = work_df['agency_id'] == 'GMB'
        if gmb_mask.any():
            gmb_routes = db.read('SELECT region, route_code, orig_tc_primary, dest_tc_primary FROM gmb_routes')
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
        log.warning(f"Headsign TC warning (GMB map): {e}")

    try:
        mtrr_mask = work_df['agency_id'] == 'MTRR'
        if mtrr_mask.any():
            _, terminal_map_tc = _load_mtr_terminal_maps(db)
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
        log.warning(f"Headsign TC warning (MTR Rail map): {e}")

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

    try:
        gmb_sec_mask = work_df['agency_id']=='GMB'
        if gmb_sec_mask.any():
            gmb_routes = db.read('SELECT region, route_code, orig_tc_primary, dest_tc_primary FROM gmb_routes')
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
        log.warning(f"Headsign TC second-pass warning (GMB): {e}")

    return headsigns.fillna('').astype(str)
