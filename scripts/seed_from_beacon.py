"""
Seed jd-role-classifier from the beacon job-search-pipeline database.

Pulls raw job description text from the beacon PostgreSQL DB and submits
each JD to the classifier API for NER extraction + SOC code prediction.

Usage:
    python scripts/seed_from_beacon.py \
        --beacon-url postgresql://user:pass@localhost:5432/beacon \
        --api-url http://localhost:8002 \
        --limit 100
"""
import asyncio
import argparse
import httpx
import sys

try:
    import asyncpg
except ImportError:
    print("asyncpg not installed. Run: pip install asyncpg", file=sys.stderr)
    sys.exit(1)


BEACON_QUERY = """
    SELECT
        id::text AS source_id,
        title,
        company,
        description AS raw_text
    FROM job_postings
    WHERE description IS NOT NULL
      AND length(description) > 100
    ORDER BY created_at DESC
    LIMIT $1
"""


async def fetch_beacon_jds(beacon_url: str, limit: int) -> list[dict]:
    conn = await asyncpg.connect(beacon_url)
    rows = await conn.fetch(BEACON_QUERY, limit)
    await conn.close()
    return [dict(row) for row in rows]


async def submit_jd(client: httpx.AsyncClient, api_url: str, jd: dict) -> bool:
    try:
        resp = await client.post(
            f"{api_url}/api/v1/jd",
            json={
                "text": jd["raw_text"],
                "source_id": jd["source_id"],
                "title": jd.get("title"),
                "top_k": 5,
            },
            timeout=30.0,
        )
        if resp.status_code == 201:
            body = resp.json()
            top_pred = body["predictions"][0] if body["predictions"] else None
            print(
                f"[{jd.get('title', 'N/A')[:50]}] "
                f"status={body['status']} "
                f"top_soc={top_pred['soc_code'] if top_pred else 'none'} "
                f"conf={top_pred['confidence'] if top_pred else 0:.3f}"
            )
            return True
        else:
            print(f"ERROR {resp.status_code}: {resp.text[:100]}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"EXCEPTION: {e}", file=sys.stderr)
        return False


async def main(beacon_url: str, api_url: str, limit: int) -> None:
    print(f"Fetching up to {limit} JDs from beacon DB...")
    jds = await fetch_beacon_jds(beacon_url, limit)
    print(f"Fetched {len(jds)} job descriptions.")

    async with httpx.AsyncClient() as client:
        # Health check
        health = await client.get(f"{api_url}/health", timeout=5.0)
        if health.status_code != 200:
            print(f"API not healthy at {api_url}", file=sys.stderr)
            sys.exit(1)

        success = 0
        for jd in jds:
            if await submit_jd(client, api_url, jd):
                success += 1
            await asyncio.sleep(0.05)

    print(f"\nSeeded {success}/{len(jds)} job descriptions successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed jd-role-classifier from beacon DB")
    parser.add_argument("--beacon-url", required=True, help="Beacon PostgreSQL connection URL")
    parser.add_argument("--api-url", default="http://localhost:8002", help="jd-role-classifier API base URL")
    parser.add_argument("--limit", type=int, default=100, help="Max JDs to seed")
    args = parser.parse_args()
    asyncio.run(main(args.beacon_url, args.api_url, args.limit))
