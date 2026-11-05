from . import citybus, gmb, gov_gtfs, journey_time, kmb, mtr, mtr_bus, nlb, td_routes, waypoints

SOURCES = {
    "kmb": lambda cache: kmb.fetch(),
    "citybus": lambda cache: citybus.fetch(),
    "gmb": lambda cache: gmb.fetch(cache / "gmb.json"),
    "nlb": lambda cache: nlb.fetch(),
    "mtr_bus": lambda cache: mtr_bus.fetch(),
    "mtr": lambda cache: mtr.fetch(),
    "gov_gtfs": lambda cache: gov_gtfs.fetch(),
    "journey_time": lambda cache: journey_time.fetch(),
    "td_routes": lambda cache: td_routes.fetch(),
    "waypoints": lambda cache: waypoints.fetch(cache / "waypoints"),
}
