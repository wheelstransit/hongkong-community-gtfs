import argparse
import logging
import time
from pathlib import Path

from .cache import load_sources
from .db import Store
from .feed import build_feed
from .load import load_all
from .sources import SOURCES


def main():
    parser = argparse.ArgumentParser(prog="python -m hkgtfs", description="Build the Hong Kong community GTFS feed.")
    parser.add_argument("--out", type=Path, default=Path("output"), help="output directory (default: output)")
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="raw data cache directory (default: .cache)")
    parser.add_argument("--refresh", default="", metavar="SOURCES",
                        help=f"comma-separated sources to re-fetch, or 'all' ({', '.join(SOURCES)})")
    parser.add_argument("--offline", action="store_true", help="fail instead of fetching anything that is not cached")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    refresh = set(SOURCES) if args.refresh == "all" else {s for s in args.refresh.split(",") if s}
    if unknown := refresh - set(SOURCES):
        parser.error(f"unknown sources: {', '.join(sorted(unknown))}")

    start = time.monotonic()
    raw = load_sources(args.cache, refresh, args.offline)
    db = Store()
    load_all(db, raw)
    build_feed(db, raw, args.out, args.cache / "waypoints", time.strftime("%Y%m%d"))
    logging.info("done in %.0fs", time.monotonic() - start)


main()
