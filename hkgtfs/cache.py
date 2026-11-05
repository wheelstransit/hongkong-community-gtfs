import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .sources import SOURCES

log = logging.getLogger(__name__)


def load_sources(cache: Path, refresh: set[str], offline: bool = False) -> dict:
    cache.mkdir(parents=True, exist_ok=True)

    def load(name):
        path = cache / f"{name}.pkl"
        if path.exists() and name not in refresh:
            return pickle.loads(path.read_bytes())
        if offline:
            raise RuntimeError(f"{name} is not cached and --offline was given")
        start = time.monotonic()
        data = SOURCES[name](cache)
        path.write_bytes(pickle.dumps(data))
        log.info("fetched %s in %.0fs", name, time.monotonic() - start)
        return data

    with ThreadPoolExecutor(len(SOURCES)) as pool:
        return dict(zip(SOURCES, pool.map(load, SOURCES)))
