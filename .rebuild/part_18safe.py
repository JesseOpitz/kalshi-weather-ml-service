from pathlib import Path

def write(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')

write('tests/conftest.py', '''from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DB_PATH = Path('/tmp/kalshi_weather_ml_pytest.sqlite3')
DB_PATH.unlink(missing_ok=True)
os.environ.update({
    'ML_SERVICE_API_KEYS': secrets.token_urlsafe(32),
    'ADMIN_API_KEY': secrets.token_urlsafe(32),
    'DATABASE_URL': f'sqlite:///{DB_PATH}',
    'MODEL_STORE_URI': '/tmp/kalshi_weather_ml_pytest_artifacts',
    'TASK_ALWAYS_EAGER': 'true',
    'AUTO_CREATE_TABLES': 'true',
})

from app.main import app  # noqa: E402

@pytest.fixture(scope='session')
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope='session')
def api_headers() -> dict[str, str]:
    return {'X-API-Key': os.environ['ML_SERVICE_API_KEYS']}
''')
write('tests/test_api.py', '''from __future__ import annotations

def test_health_is_public(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'online'

def test_private_endpoint_rejects_missing_key(client):
    assert client.get('/v1/model-health').status_code == 401

def test_model_health_is_authenticated(client, api_headers):
    response = client.get('/v1/model-health', headers=api_headers)
    assert response.status_code == 200
    assert 'model_status' in response.json()
''')
write('tests/test_ml.py', '''from __future__ import annotations

from app.ml.distributions import bracket_probability

def test_bracket_probability_is_coherent():
    below = bracket_probability(70, 2, None, 68)
    middle = bracket_probability(70, 2, 68, 72)
    above = bracket_probability(70, 2, 72, None)
    assert abs((below + middle + above) - 1.0) < 1e-4
    assert middle > below
    assert middle > above
''')
