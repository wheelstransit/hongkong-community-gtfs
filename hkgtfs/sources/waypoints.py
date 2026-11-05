import io
import shutil
import zipfile
from pathlib import Path

from ..http import get

URL = "https://codeload.github.com/hkbus/route-waypoints/zip/refs/heads/gh-pages"


def fetch(dest: Path):
    archive = zipfile.ZipFile(io.BytesIO(get(URL, timeout=300).content))
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for name in archive.namelist():
        if name.lower().endswith(".json"):
            (dest / Path(name).name).write_bytes(archive.read(name))
    if not any(dest.glob("*.json")):
        raise RuntimeError("waypoints archive contained no JSON files")
    return dest
