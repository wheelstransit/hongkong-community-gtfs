<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/logomark-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="images/logomark-light.png">
  <img alt="Hong Kong Community GTFS" src="images/logomark-light.png" width="600">
</picture>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Try the feed in action with [Wheels Router Nano](https://router-nano.justusewheels.com/)

An open-source, community-driven project to generate accurate and up-to-date procedural GTFS and GTFS-RT feeds for Hong Kong's public transport network.

## Why?
Public transport data provided by official sources in Hong Kong is often fragmented, inaccurate, or incomplete. This project aims to solve that problem by creating a single, high-quality, and reliable set of GTFS feeds.

## Who?
Currently realtime information as well as *the code of* Hong Kong Community GTFS is maintained by Wheels, an upcoming transit app by riders, for riders. For more information on Wheels, follow [@anscg_](https://threads.com/anscg_) 😉

## How
Instead of manual editing, this project generates the feeds **procedurally**. We combine data from open sources like OpenStreetMap with official data from portals like `data.gov.hk` to algorithmically build the GTFS files. This makes the process transparent, repeatable, and easier to keep up-to-date.

## Hosted GTFS
https://feed.justusewheels.com/

## Development
Needs Python 3.11+ and `mdbtools` (`apt install mdbtools` / `brew install mdbtools`).

```sh
pip install -r requirements.txt
python -m hkgtfs                      # fetch whatever is not cached yet, build output/hk.gtfs.zip
python -m hkgtfs --offline            # rebuild from .cache only (about a minute)
python -m hkgtfs --refresh kmb,mtr    # re-fetch some sources; --refresh all for everything
```

Raw data is cached per source in `.cache/`. Green minibus data is fetched incrementally using the API's last-update timestamps, so only the first fetch is slow.

`scripts/postprocess.sh` turns `output/hk.gtfs.zip` into the published feeds (needs `gtfstidy`, `osmium` and pfaedle, or Docker to run pfaedle).

| Path | What |
| --- | --- |
| `hkgtfs/sources/` | one module per upstream data source |
| `hkgtfs/load.py` | raw data → in-memory DuckDB tables |
| `hkgtfs/feed/` | GTFS tables, one module per operator or GTFS concern |
| `hkgtfs/realtime/` | GTFS-RT service alerts from `data/rt/events` |

## License
This code of the project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

However, the data produced by this project is a derivative of multiple sources and is licensed under the **Open Database License (ODbL)**.

## Data Source Attribution

This project would not be possible without the generous open data contributions from the following sources. In accordance with their terms, we provide the following attribution.

### 1. OpenStreetMap

This project uses data from OpenStreetMap. The data is available under the Open Database License (ODbL).

*   **(C) OpenStreetMap contributors**
*   For more information on OSM's copyright and license, please visit [https://www.openstreetmap.org/copyright](https://www.openstreetmap.org/copyright).

### 2. HKSAR Government's [DATA.GOV.HK](https://data.gov.hk)

This project uses datasets from the Hong Kong Special Administrative Region Government's public sector information portal, `data.gov.hk`.

*   The data is provided by various government departments and public/private organisations.
*   We acknowledge that the intellectual property rights in the data obtained from `data.gov.hk` are owned by the Government of the HKSAR.
*   Use of this data is subject to the "Terms and Conditions of Use" as published on [https://data.gov.hk](https://data.gov.hk).

### 3. [hkbus/hk-bus-crawling](https://github.com/hkbus/hk-bus-crawling)

A quite large portion of the code was taken indirectly from `hkbus/hk-bus-crawling` and the waypoint part of it.

### 4. [HK-Bus-ETA/hk-bus-time-between-stops](https://github.com/HK-Bus-ETA/hk-bus-time-between-stops)

Journey time data (i.e. stop_times.txt) are generated from data generated with this code.
