from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend import app as backend
from backend.web import app

def test_same_origin_api_and_protected_writes(monkeypatch):
    monkeypatch.setenv('DATA_DIR','demo/data')
    monkeypatch.setenv('DEMO_ENABLED','1')
    monkeypatch.setenv('INGEST_TOKEN','test-private-token')
    monkeypatch.setenv('WEATHER_ENABLED','false')
    monkeypatch.setattr(backend,'DATA',Path('demo/data'))
    backend._demo_commands.clear()
    with TestClient(app) as client:
        health=client.get('/api/health')
        assert health.status_code==200
        assert health.json()['model_version']=='e44988f343f4'
        assert health.json()['demo_enabled']
        assert client.get('/api/docs').status_code==200
        assert client.get('/api/openapi.json').status_code==200
        assert client.get('/api/openapi.json').json()['servers'][0]['url']=='/api'
        assert client.get('/api/does-not-exist').status_code==404
        assert client.post('/api/replay/control',json={'action':'live'}).status_code==401
        assert client.post('/api/demo/control',json={'action':'live'}).status_code==403
        assert client.post('/api/demo/control',json={'action':'seek','timestamp':'2026-01-07 12:00:00'}).status_code==422
        assert client.post('/api/demo/control',json={'action':'pause'}).status_code==200
        backend._demo_commands.extend([__import__('time').monotonic()]*60)
        assert client.post('/api/demo/control',json={'action':'pause'}).status_code==429
    backend._demo_commands.clear()

def test_public_demo_requires_private_operator_token(monkeypatch):
    monkeypatch.setenv('DEMO_ENABLED','1')
    monkeypatch.delenv('INGEST_TOKEN',raising=False)
    with pytest.raises(RuntimeError,match='INGEST_TOKEN'):
        with TestClient(app):pass
