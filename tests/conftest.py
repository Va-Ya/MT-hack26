"""Unit/API tests must not call live providers or depend on an optional private dataset."""
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_external_providers(monkeypatch, tmp_path):
    monkeypatch.setenv('NDTP_ENABLED', 'false')
    monkeypatch.setenv('WEATHER_ENABLED', 'false')
    monkeypatch.delenv('CODD_FEED_URL', raising=False)
    monkeypatch.setenv('CONTEXT_CACHE_PATH', str(tmp_path/'context.json'))


def pytest_collection_modifyitems(items):
    if not Path('data/raw/test/traffic.csv').exists():
        for item in items:
            if item.name == 'test_online_offline_parity':
                item.add_marker(pytest.mark.skip(reason='Original hackathon CSV dataset is not present; run again after DATA_DIR setup'))
