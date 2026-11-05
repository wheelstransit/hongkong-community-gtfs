from ..http import get_json


def fetch():
    return get_json("https://raw.githubusercontent.com/HK-Bus-ETA/hk-bus-time-between-stops/pages/times/all.json")
