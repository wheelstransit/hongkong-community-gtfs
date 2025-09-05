from __future__ import annotations

import json
import os

from .alert_models import Event


def main() -> str:
    schema = Event.model_json_schema()
    out_dir = os.path.join("data", "rt", "events")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "schema.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {out_file}")
    return out_file


if __name__ == "__main__":
    main()
