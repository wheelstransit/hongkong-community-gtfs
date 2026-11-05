import csv
import html
import io

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
session.mount("https://", HTTPAdapter(
    pool_maxsize=32,
    max_retries=Retry(total=6, backoff_factor=2, status_forcelist=(403, 429, 500, 502, 503, 504)),
))


def get(url: str, timeout: float = 60, **kwargs) -> requests.Response:
    response = session.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response


def get_json(url: str, **kwargs):
    return get(url, **kwargs).json()


def get_csv(url: str, unescape: bool = False) -> list[dict]:
    text = get(url).content.decode("utf-8-sig")
    if unescape:
        text = html.unescape(text)
    return list(csv.DictReader(io.StringIO(text)))
