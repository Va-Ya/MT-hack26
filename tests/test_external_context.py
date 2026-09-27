import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from pydantic import ValidationError

from backend.context import EventFeed, ExternalContext, distance_m
from backend.routes import RouteNetwork, MatchRequest, match
from backend.scenarios import Scenario, calculate

NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)


def feed(now=NOW):
    return EventFeed.model_validate({"source_name": "Test fixture, NOT real CODD", "generated_at": now,
        "events": [{"id": "test-only", "kind": "roadworks", "title": "Synthetic test event",
        "lat": 55.7558, "lon": 37.6173, "starts_at": now-timedelta(hours=1),
        "ends_at": now+timedelta(hours=1), "published_at": now-timedelta(hours=2),
        "source_url": "https://example.org/test-only"}]})


def test_missing_feed_is_not_zero_incidents(tmp_path, monkeypatch):
    monkeypatch.setenv('WEATHER_ENABLED', 'true')
    monkeypatch.delenv('CODD_FEED_URL', raising=False)
    ctx = ExternalContext(tmp_path/'cache.json')
    result = ctx.snapshot(now=NOW)
    assert result['road_events']['status'] == 'not_configured'
    assert result['weather']['data'] is None
    assert result['ml_features_applied'] is False


def test_feed_expiry_spatial_filter_and_replay(tmp_path):
    ctx = ExternalContext(tmp_path/'cache.json')
    ctx.import_feed(feed(), now=NOW)
    live = ctx.snapshot(now=NOW)
    assert live['road_events']['status'] == 'ok'
    assert live['road_events']['events'][0]['distance_m'] == 0
    assert ctx.snapshot(lat=56, now=NOW)['road_events']['events'] == []
    replay = ctx.snapshot(mode='REPLAY', now=NOW)
    assert replay['usable_with_telemetry'] is False
    assert replay['replay_notice']
    expired = ctx.snapshot(now=NOW+timedelta(hours=1))
    assert expired['road_events']['status'] == 'stale'
    assert expired['road_events']['events'] == []
    restored = ExternalContext(tmp_path/'cache.json')
    assert restored.snapshot(now=NOW)['road_events']['metadata']['sha256'] == live['road_events']['metadata']['sha256']


def test_future_and_malformed_feed_rejected(tmp_path):
    ctx = ExternalContext(tmp_path/'cache.json')
    with pytest.raises(ValueError):
        ctx.import_feed(feed(NOW+timedelta(minutes=1)), now=NOW)
    raw = feed().model_dump()
    raw['events'][0]['source_url'] = 'javascript:alert(1)'
    with pytest.raises(ValidationError):
        EventFeed.model_validate(raw)
    raw = feed().model_dump()
    raw['generated_at'] = NOW.replace(tzinfo=None)
    with pytest.raises(ValidationError):
        EventFeed.model_validate(raw)


def test_upstream_failure_keeps_cache_and_marks_stale(tmp_path, monkeypatch):
    ctx = ExternalContext(tmp_path/'cache.json')
    ctx.import_feed(feed(), now=NOW)
    ctx.codd_url = 'https://example.org/feed'
    async def failure(*args, **kwargs):
        raise httpx.ConnectError('secret token must not leak')
    monkeypatch.setattr(httpx.AsyncClient, '__aenter__', failure)
    asyncio.run(ctx.refresh_codd())
    result = ctx.snapshot(now=NOW)
    assert result['road_events']['status'] == 'stale'
    assert len(result['road_events']['events']) == 1
    assert 'secret' not in result['road_events']['error']


def test_weather_contract_and_source_time(tmp_path, monkeypatch):
    import backend.context as module
    monkeypatch.setenv('WEATHER_ENABLED', 'true')
    monkeypatch.setattr(module, 'utcnow', lambda: NOW)
    ctx = ExternalContext(tmp_path/'cache.json')
    payload = {'latitude':55.75,'longitude':37.625,'current_units':{'temperature_2m':'°C','precipitation':'mm','wind_speed_10m':'m/s'},
               'current':{'time':'2026-09-27T12:00','temperature_2m':10,'precipitation':1,'rain':1,'snowfall':0,'weather_code':61,'wind_speed_10m':5}}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, params):
            assert params['wind_speed_unit'] == 'ms'
            return httpx.Response(200, json=payload, request=httpx.Request('GET',url))
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    asyncio.run(ctx.refresh_weather())
    result = ctx.snapshot(now=NOW)
    assert result['weather']['status'] == 'ok'
    assert result['weather']['data']['kind'] == 'weather_model'
    assert result['weather']['data']['values']['wind_speed_10m'] == 5
    payload['current']['time'] = '2026-09-28T12:00'
    asyncio.run(ctx.refresh_weather())
    assert ctx.snapshot(now=NOW)['weather']['status'] == 'stale'
    assert ctx.snapshot(now=NOW)['weather']['data']['observed_at'] == NOW.isoformat()


@pytest.mark.parametrize('change', [
    {'lat': float('nan')}, {'lat': 0}, {'ends_at': NOW-timedelta(hours=2)},
    {'published_at': NOW+timedelta(minutes=1)}, {'starts_at': NOW.replace(tzinfo=None)},
])
def test_event_contract_rejects_bad_values(change):
    raw = feed().model_dump()
    raw['events'][0].update(change)
    with pytest.raises(ValidationError): EventFeed.model_validate(raw)


def test_what_if_units_and_reserve_horizon():
    args = dict(cycle_minutes=60, fleet=10, reserve=1, segment_km=2,
                baseline_speed_kmh=30, scenario_speed_kmh=15, added_dwell_seconds=30,
                horizon_minutes=15, dispatch_minutes=5)
    r = calculate(Scenario(**args))
    assert r['segment_before_seconds'] == 240
    assert r['segment_after_seconds'] == 510
    assert r['delta_seconds'] == 270
    assert r['headway_after_minutes'] == pytest.approx(64.5/11)
    assert r['wait_after_minutes'] == pytest.approx(64.5/22)
    args['dispatch_minutes'] = 16
    assert calculate(Scenario(**args))['effective_reserve'] == 0
    args['scenario_speed_kmh'] = 0
    with pytest.raises(ValidationError): Scenario(**args)


def test_nearest_segment_ambiguity_and_distance():
    network = RouteNetwork(routes=[dict(id='r1',name='Test line',source_url='https://example.org',coordinates=[(37.6,55.75),(37.62,55.75)])])
    point = MatchRequest(lat=55.7501, lon=37.61)
    r = match(network, point)
    assert r['status'] == 'matched'
    assert r['match']['coordinate'] == pytest.approx([37.61,55.75])
    assert r['match']['distance_m'] == pytest.approx(11.1195, abs=.01)
    assert match(network, MatchRequest(lat=56,lon=38))['status'] == 'unmatched'
    network.routes.append(network.routes[0].model_copy(update={'id':'r2'}))
    ambiguous = match(network, point)
    assert ambiguous['status'] == 'ambiguous' and ambiguous['match'] is None
    assert distance_m(55.75,37.6,55.75,37.6) == 0


def test_external_api_without_dataset(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import app as module
    monkeypatch.setattr(module, 'DATA', tmp_path/'missing')
    monkeypatch.setenv('PREDICTION_LOG_PATH', str(tmp_path/'log.parquet'))
    monkeypatch.setenv('CONTEXT_CACHE_PATH', str(tmp_path/'context.json'))
    monkeypatch.setenv('WEATHER_ENABLED', 'false')
    monkeypatch.setenv('INGEST_TOKEN', 'test-only')
    with TestClient(module.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/external/status').json()['schedule_available'] is False
        assert client.get('/external/context').status_code == 200
        assert client.get('/external/context?lat=nan').status_code == 422
        assert client.get('/docs').status_code == 200
        assert client.post('/external/events/import', json=feed().model_dump(mode='json')).status_code == 401
        assert client.post('/replay/control', headers={'X-Ingest-Token':'test-only'},json={'action':'play'}).status_code == 409
