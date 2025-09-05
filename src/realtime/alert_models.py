from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    stop_moved = "stop_moved"
    stop_closed = "stop_closed"
    detour = "detour"
    modified_service = "modified_service"
    network_notice = "network_notice"
    accessibility_notice = "accessibility_notice"


class Event(BaseModel):
    """Canonical alert event shape accepted from crowdsourcing.

    - id: deterministic, unique per alert (controls FeedEntity.id)
    - header/description/url: lang->text mappings (e.g., {"en": "..", "zh": ".."})
    - route_ids/stop_ids/trip_ids: scoping of the informed_entity
    - start/end: RFC3339 timestamps (with timezone) for active window
    - cause/effect: optional explicit GTFS-RT names; sensible defaults applied by generator
    - replacement_stop_id: hint for stop moved/closed scenarios
    """

    id: str = Field(..., description="Stable unique ID for this alert")
    type: EventType

    header: Dict[str, str] = Field(default_factory=dict)
    description: Dict[str, str] = Field(default_factory=dict)
    url: Optional[Dict[str, str]] = Field(default=None)

    route_ids: List[str] = Field(default_factory=list)
    stop_ids: List[str] = Field(default_factory=list)
    trip_ids: List[str] = Field(default_factory=list)

    start: Optional[str] = Field(default=None, description="RFC3339 timestamp with timezone")
    end: Optional[str] = Field(default=None, description="RFC3339 timestamp with timezone")

    cause: Optional[str] = Field(default=None)
    effect: Optional[str] = Field(default=None)

    replacement_stop_id: Optional[str] = None

    @field_validator("start", "end")
    @classmethod
    def must_be_rfc3339(cls, v: Optional[str]) -> Optional[str]:
        # Lightweight guard; full parsing occurs in generator.
        if v is None:
            return v
        if "T" not in v:
            raise ValueError("Timestamp must be RFC3339 (include 'T')")
        return v
