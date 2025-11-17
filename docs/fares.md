## Fare Calculation Logic

To determine the correct fare for a passenger's journey, the system must check for applicable rules in a specific order of priority. This ensures that the most specific rule is always applied.

The calculation hierarchy is as follows:

1.  **Highest Priority: Agency-Level Special Fare**
    *   First, check `special_fare_rules.csv` for a rule with `rule_type=agency` that exactly matches the passenger's boarding and alighting stops accorss consecutive trips. If a match is found, this price is used, and the calculation stops.

2.  **Medium Priority: Trip-Level Special Fare**
    *   If no agency-level rule applies, check `special_fare_rules.csv` for a rule with `rule_type=trip` that matches the passenger's `trip_id` and where their journey falls within the rule's defined stop range. If a match is found, this price is used, and the calculation stops.

3.  **Default: Standard Stage Fare**
    *   If no special rules apply, the system falls back to `fare_stages.csv`. The fare is determined by the price of the last fare stage the passenger's boarding stop is at or has passed for that specific trip.

---

## File Definitions

### `fare_stages.csv`

This file defines the default fare structure for trips. It is designed to be efficient by only requiring a new entry at the beginning of each "fare stage"—that is, whenever the price changes along a trip. The price defined for a stop applies to passengers boarding at that stop or any subsequent stop until the next defined fare stage begins.

| Field Name | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| `trip_id` | Yes | An identifier from `trips.txt` that uniquely identifies a trip. | `trip_A` |
| `from_stop_id` | Yes | The stop from `stops.txt` at which this fare becomes active. | `stop_a` |
| `price` | Yes | The cost to ride from this fare stage onwards. | `8.00` |
| `currency` | Yes | The currency of the fare, using an ISO 4217 code. | `USD` |

#### Example `fare_stages.csv`:

```csv
trip_id,from_stop_id,price,currency
trip_A,stop_a,8.00,USD
trip_A,stop_d,5.00,USD
trip_A,stop_f,4.00,USD
trip_A,stop_g,3.00,USD
```

#### How to Interpret `fare_stages.csv`

The entries for each trip should be interpreted based on the `stop_sequence` from the GTFS `stop_times.txt` file.

Using the example above for `trip_A`:
*   A passenger boarding at `stop_a`, `stop_b`, or `stop_c` will be charged **$8.00**.
*   A passenger boarding at `stop_d` or `stop_e` will be charged **$5.00**.
*   A passenger boarding at `stop_f` will be charged **$4.00**.
*   A passenger boarding at `stop_g` or any subsequent stop on `trip_A` will be charged **$3.00**.

### `special_fare_rules.csv`

This file defines exceptions that override the default stage fares. It can specify high-priority, system-wide fares between two points (e.g., "Airport to Downtown") or trip-specific promotional fares.

| Field Name | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| `special_fare_id` | Yes | A unique identifier for the special fare rule. | `light_rail_1` |
| `rule_type` | Yes | The scope of the rule. Valid values are `agency` or `trip`. | `agency` |
| `trip_id` | If `rule_type=trip` | The `trip_id` this rule applies to. Must be empty if `rule_type=agency`. | `trip_B` |
| `onboarding_stop_id` | Yes | The boarding stop for this rule. | `stop_tuen_mun` |
| `offboarding_stop_id`| Yes | The alighting stop for this rule. | `stop_tin_shui_wai`|
| `price` | Yes | The special fare price. | `15.00` |
| `currency` | Yes | The currency of the fare (ISO 4217 code). | `USD` |

#### Rule Types Explained

*   **`rule_type=agency`**: This rule defines a direct, point-to-point fare that applies to **any trip** in the feed that services the exact `onboarding_stop_id` and `offboarding_stop_id`. This has the highest priority and overrides all other rules.

*   **`rule_type=trip`**: This rule defines a special fare zone for a **single trip**. It applies if a passenger boards at or after `onboarding_stop_id` and alights at or before `offboarding_stop_id` on the specified `trip_id`. This has medium priority.

#### Example `special_fare_rules.csv`:

```csv
special_fare_id,rule_type,trip_id,onboarding_stop_id,offboarding_stop_id,price,currency
light_rail_1,agency,,stop_tuen_mun,stop_tin_shui_wai,15.00,USD
summer_promo,trip,trip_B,stop_x,stop_y,6.50,USD
```

---

## Consumer

so let's assume the example files above are being used

#### Use Case 1: Standard Stage Fare

*   **Journey**: A passenger boards `trip_A` at `stop_c` and gets off at `stop_h`.
*   **Calculation**:
    1.  The system checks `special_fare_rules.csv`. No `agency` rule matches stops `c` to `h`.
    2.  No `trip` rule exists for `trip_A`.
    3.  The system uses `fare_stages.csv`. The passenger boarded at `stop_c`. The last applicable stage before or at this stop is the one starting at `from_stop_id: stop_a`.
*   **Final Fare**: **$8.00**

#### Use Case 2: Trip-Specific Special Fare

*   **Journey**: A passenger boards `trip_B` at `stop_x` and gets off at `stop_y`.
*   **Calculation**:
    1.  The system checks `special_fare_rules.csv` for `agency` rules. None match.
    2.  It then checks for `trip` rules. It finds the `summer_promo` rule for `trip_B` which covers travel between `stop_x` and `stop_y`. This rule matches.
*   **Final Fare**: **$6.50** (The default stage fare for `trip_B` is ignored).

#### Use Case 3: Agency-Wide Special Fare

*   **Journey**: A passenger boards a Light Rail (let's say `trip_C`) at `stop_tuen_mun` and gets off at `stop_tin_shui_wai`.
*   **Calculation**:
    1.  The system checks `special_fare_rules.csv` for `agency` rules. It finds the `light_rail_1` rule for the exact journey from `stop_tuen_mun` to `stop_tin_shui_wai`. This is the highest priority rule.
*   **Final Fare**: **$15.00** (Any `trip`-level rules for `trip_C` or its stage fares are ignored).