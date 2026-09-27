"""Uploaded route geometry and transparent nearest-segment matching (not HMM)."""
import math
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    source_url: str = Field(pattern=r"^https://", max_length=2000)
    coordinates: list[tuple[float, float]] = Field(min_length=2, max_length=10000)
    @model_validator(mode="after")
    def valid(self):
        url=urlsplit(self.source_url)
        if not url.hostname or url.username or url.password:
            raise ValueError('Expected a public HTTPS source URL without credentials')
        if not all(36 <= lon <= 39 and 54.5 <= lat <= 57 for lon, lat in self.coordinates):
            raise ValueError("Expected [longitude, latitude] coordinates in Moscow bounds")
        if len(set(self.coordinates)) < 2:
            raise ValueError("A route needs at least two different points")
        return self


class RouteNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routes: list[Route] = Field(max_length=100)
    @model_validator(mode="after")
    def valid(self):
        if len({r.id for r in self.routes}) != len(self.routes):
            raise ValueError("Duplicate route IDs")
        if sum(len(r.coordinates) for r in self.routes) > 50000:
            raise ValueError("At most 50000 points per network")
        return self


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lat: float = Field(ge=54.5, le=57)
    lon: float = Field(ge=36, le=39)
    max_distance_m: float = Field(default=100, ge=5, le=1000)
    route_id: str | None = None


def match(network: RouteNetwork, point: MatchRequest):
    candidates = []
    # Local equirectangular projection about the observation, metres.
    sx = 6371000 * math.pi / 180 * math.cos(math.radians(point.lat))
    sy = 6371000 * math.pi / 180
    for route in network.routes:
        if point.route_id and route.id != point.route_id:
            continue
        best = None
        for i, (a, b) in enumerate(zip(route.coordinates, route.coordinates[1:])):
            ax, ay = (a[0]-point.lon)*sx, (a[1]-point.lat)*sy
            bx, by = (b[0]-point.lon)*sx, (b[1]-point.lat)*sy
            dx, dy = bx-ax, by-ay
            if dx*dx + dy*dy == 0:
                continue
            u = max(0, min(1, -(ax*dx+ay*dy)/(dx*dx+dy*dy)))
            d = math.hypot(ax+u*dx, ay+u*dy)
            item = {"route_id": route.id, "segment_index": i, "distance_m": d,
                    "coordinate": [a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1])], "source_url": route.source_url}
            if best is None or d < best["distance_m"]:
                best = item
        if best and best["distance_m"] <= point.max_distance_m:
            candidates.append(best)
    candidates.sort(key=lambda c: c["distance_m"])
    ambiguous = len(candidates) > 1 and candidates[1]["distance_m"] - candidates[0]["distance_m"] < 20
    return {"status": "ambiguous" if ambiguous else "matched" if candidates else "unmatched",
            "match": None if ambiguous or not candidates else candidates[0], "candidates": candidates[:5],
            "method": "nearest_segment_v1", "formula": "u=clip(−A·(B−A)/|B−A|²,0,1); d=|A+u(B−A)|",
            "limitation": "Геометрическая близость; не учитывает направление, историю, развязки и принадлежность ТС маршруту"}
