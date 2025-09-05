# How to add/edit GTFS‑RT Alerts (no live ETA/positions)

This guide explains how to create and maintain GTFS‑RT Alerts in this repo without running any server. The CI builds `alerts.pb` and uploads it to your CDN (S3 or Cloudflare R2).

## Where things live
- Input events (JSON): `data/rt/events/*.json`
- JSON Schema: `data/rt/events/schema.json`
- Builder script: `src/realtime/build_alerts.py`
- Pydantic model: `src/realtime/alert_models.py`
- Output feed: `data/rt/build/alerts.pb`

## Event JSON shape
Each file can contain a single object or an array of objects. Required fields are minimal; English/Chinese text is strongly encouraged.

Required
- `id` (string): stable, unique per alert (e.g., `kmb-12345-20250905` or a UUID)
- `type` (enum): one of `stop_moved`, `stop_closed`, `detour`, `modified_service`, `network_notice`, `accessibility_notice`

Common fields
- `header` (object lang->text) — e.g., `{ "en": "Stop moved", "zh": "巴士站臨時搬遷" }`
- `description` (object lang->text)
- `url` (object lang->url) — link to more details
- `route_ids` (array<string>) — scope to specific routes
- `stop_ids` (array<string>) — scope to specific stops
- `trip_ids` (array<string>) — use sparingly for highly specific alerts
- `start`, `end` (RFC3339 timestamps, with timezone)
- `cause` (GTFS‑RT cause name, e.g., `CONSTRUCTION`, `WEATHER`, `DEMONSTRATION`)
- `effect` (GTFS‑RT effect name; defaults are applied if omitted)
- `replacement_stop_id` (string) — for `stop_moved`/`stop_closed`

Defaults (if `effect` omitted)
- `stop_moved` → `STOP_MOVED`
- `stop_closed` → `NO_SERVICE`
- `detour` → `DETOUR`
- `modified_service` → `MODIFIED_SERVICE`
- `network_notice` → `UNKNOWN_EFFECT`
- `accessibility_notice` → `OTHER_EFFECT`

## Examples
See `data/rt/events/sample.json` for three typical cases.

## Authoring tips

## Writing good headers and descriptions

Keep it rider-first, short, and clear. Use both English and Chinese when possible.

Header (1 line)
- Purpose: quick summary users can scan in lists.
- Keep to ~45–70 characters.
- Structure: [Entity] + [Action] [+ Scope] [+ When]
  - Examples: "Stop 12345 temporarily moved", "Route 101 detour via Queensway",
    "E11 suspended due to weather".
- Avoid: internal codes, long addresses, multiple clauses.

Description (1–3 short sentences)
- Sentence 1: What + Why + Where.
- Sentence 2: How to ride now (detour streets, replacement stop, headway).
- Sentence 3: When (start–end), and expectation (e.g., delays).
- Optional: link to official post or map.

Time and wording
- Use local time (HKT, +08:00). Example: "from 09:00 to 20:00 on 5 Sep".
- If end time unknown, say "until further notice".
- Use present/imperative: "Use temporary stop 12345T." not "Passengers are advised to...".

Common phrases (EN)
- Stop moved: "Stop {stop_name} moved {distance/direction}." If the temporary stop isn’t in static GTFS, do not invent a stop_id—describe the new boarding point (landmark/cross street) and optionally add a map URL.
- Stop closed: "Stop {stop_code} closed due to {reason}."
- Detour: "Detour via {streets}. {1–2 impacts}."
- Additional service: "More frequent service ~{headway} mins during {period}."
- Reduced/No service: "Service {reduced/suspended} due to {reason}."
- Accessibility: "Lift/escalator at {station/exit} out of service."

Mini examples
- Stop moved
  - Header: Route E11 stop Admiralty Centre moved
  - Description: Stop “Admiralty Centre” moved 50m east to outside XYZ Mall due to roadworks. See map: https://example.com/map. 5–12 Sep, 08:00–23:00.
- Detour
  - Header: Route 101 detour via Queensway
  - Description: Temporary detour via Queensway due to construction. Expect 5–10 min delays. 09:00–20:00, 5 Sep.
- Suspension (weather)
  - Header: E11 suspended (extreme weather)
  - Description: Service suspended due to black rain signal. Resumes when conditions improve.

Quality checklist
- Clear type picked (moved/closed/detour/etc.).
- Correct route_id/stop_id scope.
- English and Chinese provided (if possible).
- Time window present or "until further notice".
- Replacement stop or detour details included when relevant.

## Validate locally
1) Generate JSON Schema
- `python -m src.realtime.dump_schema` (writes `data/rt/events/schema.json`)

2) Build the feed
- `python -m src.realtime.build_alerts` (writes `data/rt/build/alerts.pb`)

Optional: run the GTFS‑RT validator in your CI against your static GTFS.

## CI/CD upload
- GitHub Actions workflow: `.github/workflows/build-alerts.yml`.
- Set repo secrets for either S3 or R2:
  - S3: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET`
  - R2: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
- The workflow uploads `alerts.pb` with `Cache-Control: public,max-age=60`.

## FAQ
- Do I need a server? No, CDN hosting of the static `alerts.pb` works fine.
- Can I add cancellations without ETA? Yes, but that uses TripUpdates; this guide covers Alerts only.
- How do I find correct `stop_id`/`route_id`? Look them up in your static GTFS or project CSVs/DB.
