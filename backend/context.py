"""External evidence, kept separate from ML. All timestamps are timezone-aware.

No incident feed is inferred from map tiles. Missing feeds never mean zero events.
"""
import asyncio
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow():
    return datetime.now(timezone.utc)


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=160)
    kind: Literal["accident", "roadworks", "closure"]
    title: str = Field(min_length=1, max_length=500)
    lat: float = Field(ge=54.5, le=57)
    lon: float = Field(ge=36, le=39)
    starts_at: datetime
    ends_at: datetime
    published_at: datetime
    source_url: str = Field(max_length=2000)

    @model_validator(mode="after")
    def valid(self):
        from urllib.parse import urlsplit
        if any(t.tzinfo is None for t in (self.starts_at, self.ends_at, self.published_at)):
            raise ValueError("Event timestamps must contain a timezone")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        u = urlsplit(self.source_url)
        if u.scheme != "https" or not u.hostname or u.username or u.password:
            raise ValueError("source_url must be a public HTTPS citation")
        return self


class EventFeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    source_name: str = Field(min_length=1, max_length=200)
    # An imported file is a user supplied export, never a verified live CODD connection.
    generated_at: datetime
    events: list[Event] = Field(max_length=5000)

    @model_validator(mode="after")
    def valid(self):
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must contain a timezone")
        if len({e.id for e in self.events}) != len(self.events):
            raise ValueError("Duplicate event IDs")
        if any(e.published_at > self.generated_at for e in self.events):
            raise ValueError("An event cannot be published after feed generation")
        return self


def distance_m(lat1, lon1, lat2, lon2):
    a, b = math.radians(lat1), math.radians(lat2)
    h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(lon2-lon1)/2)**2
    return 6371000 * 2 * math.asin(math.sqrt(min(1, max(0, h))))


class ExternalContext:
    def __init__(self, cache_path=None):
        self.lock = RLock()
        self.cache_path = Path(cache_path or os.getenv("CONTEXT_CACHE_PATH", "artifacts/context-cache.json"))
        self.weather = None
        self.feed = None
        self.feed_meta = None
        self.weather_error = None
        self.codd_error = None
        self.weather_enabled = os.getenv("WEATHER_ENABLED", "true").lower() == "true"
        self.codd_url = os.getenv("CODD_FEED_URL", "").strip()
        self.last_attempt = None
        self.load()

    def load(self):
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.weather = data.get("weather")
            if data.get("feed"):
                self.feed = EventFeed.model_validate(data["feed"])
                self.feed_meta = data["feed_meta"]
        except (OSError, ValueError, KeyError, TypeError):
            self.weather, self.feed, self.feed_meta = None, None, None

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.cache_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"weather": self.weather, "feed": self.feed.model_dump(mode="json") if self.feed else None,
                                    "feed_meta": self.feed_meta}, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.cache_path)

    def import_feed(self, feed, transport="manual_import", now=None):
        now = now or utcnow()
        if feed.generated_at > now:
            raise ValueError("Future feed generation time is not allowed")
        digest = hashlib.sha256(feed.model_dump_json().encode()).hexdigest()
        with self.lock:
            self.feed = feed
            self.feed_meta = {"transport": transport, "fetched_at": now.isoformat(), "sha256": digest}
            self.codd_error = None
            self.save()
        return {"accepted": len(feed.events), **self.feed_meta}

    async def refresh_weather(self):
        if not self.weather_enabled:
            return
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                response = await client.get("https://api.open-meteo.com/v1/forecast", params={
                    "latitude": 55.7558, "longitude": 37.6173, "timezone": "UTC", "wind_speed_unit": "ms",
                    "current": "temperature_2m,precipitation,rain,snowfall,weather_code,wind_speed_10m"})
                response.raise_for_status()
                data = response.json()
            c = data["current"]
            for k in ("temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m", "weather_code"):
                if not isinstance(c[k], (float, int)) or not math.isfinite(c[k]):
                    raise ValueError("Invalid weather measurement")
            observed = datetime.fromisoformat(c["time"]).replace(tzinfo=timezone.utc)
            now = utcnow()
            if observed > now:
                raise ValueError("Future weather timestamp")
            with self.lock:
                self.weather = {"values": c, "units": data["current_units"], "observed_at": observed.isoformat(),
                                "fetched_at": now.isoformat(), "source": "Open-Meteo", "source_url": "https://open-meteo.com/en/docs",
                                "latitude": data["latitude"], "longitude": data["longitude"], "kind": "weather_model",
                                "scope": "Одна модельная ячейка около центра Москвы; не измерения для каждой улицы"}
                self.weather_error = None
                self.save()
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError):
            # Do not expose request URLs, headers or credentials in responses.
            with self.lock:
                self.weather_error = "Не удалось обновить погоду; последнее значение сохранено"

    async def refresh_codd(self):
        if not self.codd_url:
            return
        try:
            from urllib.parse import urlsplit
            u = urlsplit(self.codd_url)
            if u.scheme != "https" or not u.hostname or u.username or u.password:
                raise ValueError("HTTPS server configuration required")
            headers = {"Authorization": "Bearer " + os.environ["CODD_API_TOKEN"]} if os.getenv("CODD_API_TOKEN") else {}
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                async with client.stream("GET", self.codd_url, headers=headers) as response:
                    response.raise_for_status()
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > 5_000_000:
                            raise ValueError("Feed exceeds 5 MB")
            self.import_feed(EventFeed.model_validate_json(chunks), "configured_adapter")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError):
            with self.lock:
                self.codd_error = "Источник не ответил или не соответствует контракту EventFeed 1.0"

    async def run(self):
        while True:
            self.last_attempt = utcnow().isoformat()
            await asyncio.gather(self.refresh_weather(), self.refresh_codd())
            await asyncio.sleep(300)

    def snapshot(self, lat=55.7558, lon=37.6173, radius_m=1000, mode="LIVE", now=None):
        now = now or utcnow()
        with self.lock:
            weather = copy.deepcopy(self.weather)
            feed, meta = self.feed, copy.deepcopy(self.feed_meta)
            weather_error, codd_error = self.weather_error, self.codd_error
        wa = (now - datetime.fromisoformat(weather["observed_at"])).total_seconds() if weather else None
        fa = (now - feed.generated_at).total_seconds() if feed else None
        weather_status = "disabled" if not self.weather_enabled else "unavailable" if not weather else "stale" if wa < 0 or wa > 3600 or weather_error else "ok"
        feed_status = "unavailable" if not feed else "stale" if fa < 0 or fa > 900 or codd_error else "ok"
        if not feed and not self.codd_url:
            feed_status = "not_configured"
        events = []
        if feed:
            for event in feed.events:
                if event.published_at <= now and event.starts_at <= now < event.ends_at:
                    distance = distance_m(lat, lon, event.lat, event.lon)
                    if distance <= radius_m:
                        events.append({**event.model_dump(mode="json"), "distance_m": round(distance, 1)})
        eligible = mode == "LIVE"
        return {"as_of": now.isoformat(), "mode": mode, "location": {"lat": lat, "lon": lon, "radius_m": radius_m},
                "weather": {"status": weather_status, "data": weather, "age_seconds": wa, "error": weather_error},
                "road_events": {"status": feed_status, "events": sorted(events, key=lambda e: e["distance_m"]),
                                "source_name": feed.source_name if feed else None, "generated_at": feed.generated_at.isoformat() if feed else None,
                                "metadata": meta, "age_seconds": fa, "error": codd_error,
                                "coverage": "Полнота покрытия неизвестна; пустой список не доказывает отсутствие ДТП",
                                "official_reference": "https://transport.mos.ru/mostrans/closures"},
                "usable_with_telemetry": eligible,
                "ml_features_applied": False,
                "explanation": "Внешний контекст не изменяет ML-прогноз. Близость события не доказывает причинность.",
                "replay_notice": None if eligible else "Текущие погода и события показаны отдельно и исключены из исторического replay",
                "formulas": {"distance": "d = 2·6371000·asin(√(sin²(Δφ/2) + cos φ₁·cos φ₂·sin²(Δλ/2)))",
                             "event_filter": "published_at ≤ now; starts_at ≤ now < ends_at; d ≤ radius_m",
                             "freshness": "age = now − source_time; weather ≤ 3600 s; events ≤ 900 s"}}
