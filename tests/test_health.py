import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["service"] == "jd-role-classifier"


@pytest.mark.asyncio
async def test_health_db(client: AsyncClient):
    resp = await client.get("/health/db")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
