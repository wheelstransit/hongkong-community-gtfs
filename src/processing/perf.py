import os, math
import numpy as np

def resolve_parallel_config():
    enabled = os.getenv('HKGTFSPARALLEL', '1') == '1'
    procs_env = os.getenv('HKGTFSPROCS')
    try:
        max_procs = int(procs_env) if procs_env else min(8, (os.cpu_count() or 1))
    except ValueError:
        max_procs = min(8, (os.cpu_count() or 1))
    max_procs = max(1, max_procs)
    return enabled, max_procs

def parallel_route_match_enabled():
    return os.getenv('HKGTFSPARALLEL_ROUTE_MATCH', '0') == '1'

def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine returning meters. Inputs are numpy arrays (broadcastable)."""
    lat1 = np.asarray(lat1, dtype='float64')
    lon1 = np.asarray(lon1, dtype='float64')
    lat2 = np.asarray(lat2, dtype='float64')
    lon2 = np.asarray(lon2, dtype='float64')
    mask = (~np.isnan(lat1)) & (~np.isnan(lon1)) & (~np.isnan(lat2)) & (~np.isnan(lon2))
    out = np.full(np.broadcast(lat1, lat2).shape, np.inf, dtype='float64')
    if mask.any():
        R = 6371000.0
        dlat = np.radians(lat2[mask] - lat1[mask])
        dlon = np.radians(lon2[mask] - lon1[mask])
        a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1[mask])) *
             np.cos(np.radians(lat2[mask])) * np.sin(dlon/2)**2)
        out_valid = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        out[mask] = out_valid
    return out
