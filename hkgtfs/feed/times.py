import re
from datetime import timedelta
from typing import Optional


def format_timedelta(td) -> str:
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def to_seconds(timestr) -> Optional[int]:
    if timestr is None:
        return None
    try:
        parts = str(timestr).strip().split(":")
        if len(parts) != 3:
            return None
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError):
        return None


def to_hhmmss(total_seconds: int) -> str:
    return format_timedelta(timedelta(seconds=int(total_seconds)))


def parse_headway(headway_str: str) -> Optional[int]:
    """Average seconds from MTR headway strings like '2.1', '3.6-5' or '2.5 / 4'."""
    if not headway_str or headway_str == "-":
        return None
    headway_str = re.sub(r"\s*\(.*\)$", "", headway_str).strip()
    headway_str = re.sub(r"\s*#.*$", "", headway_str).strip()
    headway_str = re.sub(r"\s*~.*$", "", headway_str).strip()
    try:
        if "-" in headway_str:
            low, high = map(float, headway_str.split("-"))
            minutes = (low + high) / 2
        elif "/" in headway_str:
            parts = [float(p.strip()) for p in headway_str.split("/")]
            minutes = sum(parts) / len(parts)
        else:
            minutes = float(headway_str)
        return int(minutes * 60)
    except (ValueError, TypeError):
        return None
