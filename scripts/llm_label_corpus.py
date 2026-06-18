"""Bootstrap a labeled JD corpus by calling Claude Sonnet 4.6 over the beacon DB.

Produces ``data/labeled_jds.jsonl`` in the exact shape expected by
``scripts/fine_tune.py``:

    {"text": "<JD text>", "soc_code": "15-2053.00", "source_id": "...",
     "title": "...", "confidence": 0.87, "rationale": "..."}

Design choices
--------------
* **Model**: Claude Sonnet 4.6. For structured-output classification
  picking from a 30-code enum, the schema constraint does most of the
  heavy lifting — Sonnet closes most of the quality gap with Opus on this
  shape of task at ~40% of the cost. Haiku 4.5 is cheaper still but
  struggles on the lookalike classes (Data Scientist vs MLE vs Data
  Engineer) where the labels matter most.
* **Structured outputs** (``output_config.format``) with ``soc_code`` as an
  enum of the 30 O*NET codes in ``app/data/onet_soc_codes.json``. The model
  cannot emit a code outside the taxonomy.
* **Prompt caching** on the system prompt (the 30-code taxonomy plus
  rubric, ~3-5K tokens). One ``max_tokens=0`` pre-warm call writes the
  cache; every real call afterward reads at ~0.1x base price.
* **Resume**: existing ``source_id`` values in the output JSONL are
  skipped, so a Ctrl-C and re-run picks up where it stopped.
* **Confidence threshold** filters low-quality labels into a separate
  ``_uncertain.jsonl`` for human review (don't poison training with them).

Usage
-----
::

    python scripts/llm_label_corpus.py \
        --beacon-url "postgresql://jsp_user:changeme@127.0.0.1:15436/job_search" \
        --output data/labeled_jds.jsonl \
        --limit 2700 \
        --min-confidence 0.6

Reads ``ANTHROPIC_API_KEY`` (and optionally ``BEACON_DATABASE_URL``) from
``.env`` in the repo root. Falls back to the live process environment if
the file is missing — the SDK also accepts an ``ant auth login`` profile.
Beacon DB access goes through the SSH tunnel — see
``portfolio-infra/scripts/beacon-tunnel.ps1``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Run: pip install -e .", file=sys.stderr)
    sys.exit(1)

# Load .env from the repo root (parent of this file's directory) before any
# code reads os.environ — anthropic.Anthropic() picks up ANTHROPIC_API_KEY at
# construction time, so this must run first.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

try:
    import anthropic
except ImportError:
    print("anthropic SDK not installed. Run: pip install -e .[labeling]", file=sys.stderr)
    sys.exit(1)

try:
    import asyncpg
except ImportError:
    print("asyncpg not installed. Run: pip install asyncpg", file=sys.stderr)
    sys.exit(1)


# Sonnet 4.6 pricing — $3/MTok input, $15/MTok output, cache writes 1.25x, reads 0.1x.
# Used for the end-of-run cost summary so the operator knows what was spent.
INPUT_USD_PER_MTOK = 3.00
OUTPUT_USD_PER_MTOK = 15.00
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def load_soc_codes() -> list[dict]:
    """Read the 30-code taxonomy that lives alongside the running classifier."""
    here = Path(__file__).resolve().parent
    path = here.parent / "app" / "data" / "onet_soc_codes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_system_prompt(soc_codes: list[dict]) -> str:
    """Render the SOC taxonomy + labeling rubric. Stable across all calls so it
    caches; do NOT interpolate per-request data (timestamps, JD text) here."""
    lines = [
        "You are labeling job descriptions with O*NET Standard Occupational",
        "Classification (SOC) codes. You will be given a job title and JD",
        "text. Pick the SINGLE code that best describes the role.",
        "",
        "Rules:",
        "- Choose based on the JD's day-to-day responsibilities and tech stack,",
        "  not on flashy buzzwords or company self-description.",
        "- If the role spans multiple codes (e.g. 'ML Engineer doing data",
        "  engineering 60% of the time'), pick the code that matches the",
        "  majority of the work.",
        "- Confidence reflects how clearly the JD points at one code vs.",
        "  others. Above 0.8 means unambiguous; 0.5-0.7 means there's a",
        "  reasonable second choice; below 0.5 means the JD is too generic",
        "  or contradictory to label reliably.",
        "- Rationale is 1-2 sentences citing the specific JD content that",
        "  drove the choice (e.g. 'JD lists PyTorch, LoRA, model training',",
        "  not 'role sounds like MLE').",
        "",
        "O*NET SOC codes you may select from:",
        "",
    ]
    for c in soc_codes:
        lines.append(f"- {c['soc_code']} — {c['title']}")
    return "\n".join(lines)


def build_output_schema(soc_codes: list[dict]) -> dict:
    """JSON schema constraining the response. soc_code is an enum so the model
    physically cannot emit a code outside the taxonomy."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "soc_code": {
                "type": "string",
                "enum": [c["soc_code"] for c in soc_codes],
            },
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["soc_code", "confidence", "rationale"],
    }


def soc_title_map(soc_codes: list[dict]) -> dict[str, str]:
    return {c["soc_code"]: c["title"] for c in soc_codes}


BEACON_QUERY = """
    SELECT
        id::text AS source_id,
        title,
        company,
        COALESCE(description_clean, description_raw) AS raw_text
    FROM job_listings
    WHERE COALESCE(description_clean, description_raw) IS NOT NULL
      AND length(COALESCE(description_clean, description_raw)) > 200
    ORDER BY created_at DESC
    LIMIT $1
"""


async def fetch_beacon_jds(beacon_url: str, limit: int) -> list[dict]:
    conn = await asyncpg.connect(beacon_url)
    try:
        rows = await conn.fetch(BEACON_QUERY, limit)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def load_existing_source_ids(output_path: Path, uncertain_path: Path) -> set[str]:
    """Resume support — collect source_ids from prior runs so we skip them."""
    seen: set[str] = set()
    for p in (output_path, uncertain_path):
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("source_id")
                if sid:
                    seen.add(sid)
    return seen


def prewarm_cache(
    client: anthropic.Anthropic, system_prompt: str, output_schema: dict
) -> None:
    """Write the cached system prompt with max_tokens=0 so the first real
    request reads (~0.1x) instead of writing (~1.25x). See the prompt-caching
    pre-warm pattern.

    Note: max_tokens=0 is incompatible with output_config.format, so the
    pre-warm request omits the schema. The next request that DOES include
    the schema writes a fresh cache entry (schema is rendered after system,
    so it's still on the cacheable prefix), then everything after that reads.
    """
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=0,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": "warmup"}],
    )


def classify_one(
    client: anthropic.Anthropic,
    system_prompt: str,
    output_schema: dict,
    title: str | None,
    jd_text: str,
) -> tuple[dict, Any]:
    """Single JD → labeled dict + usage object. Trusts the SDK's automatic
    retries on 429/5xx (default max_retries=2)."""
    user_content = f"Job title: {title or '(none provided)'}\n\nJob description:\n{jd_text}"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": output_schema}},
        messages=[{"role": "user", "content": user_content}],
    )
    # output_config.format guarantees a single text block with valid JSON.
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    return parsed, response.usage


def estimate_cost(
    input_tokens: int, output_tokens: int, cache_write: int, cache_read: int
) -> float:
    cost = (
        (input_tokens / 1_000_000) * INPUT_USD_PER_MTOK
        + (output_tokens / 1_000_000) * OUTPUT_USD_PER_MTOK
        + (cache_write / 1_000_000) * INPUT_USD_PER_MTOK * CACHE_WRITE_MULT
        + (cache_read / 1_000_000) * INPUT_USD_PER_MTOK * CACHE_READ_MULT
    )
    return cost


async def main(
    beacon_url: str,
    output: str,
    uncertain_output: str,
    limit: int,
    min_confidence: float,
    dry_run: bool,
) -> None:
    output_path = Path(output)
    uncertain_path = Path(uncertain_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uncertain_path.parent.mkdir(parents=True, exist_ok=True)

    soc_codes = load_soc_codes()
    system_prompt = build_system_prompt(soc_codes)
    output_schema = build_output_schema(soc_codes)
    titles = soc_title_map(soc_codes)

    seen = load_existing_source_ids(output_path, uncertain_path)
    if seen:
        print(f"Resume: skipping {len(seen)} already-labeled JDs.")

    print(f"Fetching up to {limit} JDs from beacon...")
    jds = await fetch_beacon_jds(beacon_url, limit)
    jds = [j for j in jds if j["source_id"] not in seen]
    print(f"{len(jds)} JDs to label (after dedupe).")

    if dry_run:
        print("--dry-run: skipping API calls. Sample of first 3 JDs:")
        for jd in jds[:3]:
            print(f"  {jd['source_id']}  {jd.get('title', '?')[:60]}")
        return

    if not jds:
        print("Nothing to do.")
        return

    client = anthropic.Anthropic()

    print("Pre-warming prompt cache...")
    prewarm_cache(client, system_prompt, output_schema)

    totals = {
        "input": 0,
        "output": 0,
        "cache_write": 0,
        "cache_read": 0,
        "errors": 0,
        "uncertain": 0,
        "labeled": 0,
    }
    started = time.monotonic()

    with output_path.open("a", encoding="utf-8") as out_f, uncertain_path.open(
        "a", encoding="utf-8"
    ) as uncertain_f:
        for i, jd in enumerate(jds, start=1):
            jd_text = jd["raw_text"]
            try:
                parsed, usage = classify_one(
                    client, system_prompt, output_schema, jd.get("title"), jd_text
                )
            except anthropic.APIError as e:
                # Retries are exhausted at this point — log and move on rather
                # than crashing the whole batch.
                totals["errors"] += 1
                print(f"  [{i}/{len(jds)}] FAIL {jd['source_id']}: {e}", file=sys.stderr)
                continue
            except (KeyError, json.JSONDecodeError) as e:
                totals["errors"] += 1
                print(
                    f"  [{i}/{len(jds)}] PARSE-FAIL {jd['source_id']}: {e}",
                    file=sys.stderr,
                )
                continue

            totals["input"] += usage.input_tokens
            totals["output"] += usage.output_tokens
            totals["cache_write"] += usage.cache_creation_input_tokens or 0
            totals["cache_read"] += usage.cache_read_input_tokens or 0

            record = {
                "text": jd_text,
                "soc_code": parsed["soc_code"],
                "soc_title": titles.get(parsed["soc_code"], "unknown"),
                "source_id": jd["source_id"],
                "title": jd.get("title"),
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
            }
            target = out_f if parsed["confidence"] >= min_confidence else uncertain_f
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()  # crash-safe — never lose a labeled record

            if parsed["confidence"] >= min_confidence:
                totals["labeled"] += 1
            else:
                totals["uncertain"] += 1

            if i % 25 == 0:
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed else 0
                cost = estimate_cost(
                    totals["input"],
                    totals["output"],
                    totals["cache_write"],
                    totals["cache_read"],
                )
                print(
                    f"  [{i}/{len(jds)}] {rate:.2f} jd/s  "
                    f"labeled={totals['labeled']}  uncertain={totals['uncertain']}  "
                    f"errors={totals['errors']}  spent=${cost:.2f}"
                )

    elapsed = time.monotonic() - started
    cost = estimate_cost(
        totals["input"], totals["output"], totals["cache_write"], totals["cache_read"]
    )
    print("\n--- Summary ---")
    print(f"  Processed:      {len(jds)} JDs in {elapsed:.0f}s")
    print(f"  Labeled:        {totals['labeled']}  -> {output_path}")
    print(f"  Uncertain:      {totals['uncertain']}  -> {uncertain_path}")
    print(f"  Errors:         {totals['errors']}")
    print(
        f"  Tokens:         input={totals['input']:,}  output={totals['output']:,}  "
        f"cache_write={totals['cache_write']:,}  cache_read={totals['cache_read']:,}"
    )
    print(f"  Estimated cost: ${cost:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--beacon-url",
        default=os.environ.get(
            "BEACON_DATABASE_URL",
            "postgresql://jsp_user:changeme@127.0.0.1:15436/job_search",
        ),
        help=(
            "Postgres URL for beacon DB. Defaults to BEACON_DATABASE_URL "
            "from .env if set, else the local SSH tunnel default."
        ),
    )
    ap.add_argument(
        "--output",
        default="data/labeled_jds.jsonl",
        help="JSONL output (passed to scripts/fine_tune.py).",
    )
    ap.add_argument(
        "--uncertain-output",
        default="data/labeled_jds_uncertain.jsonl",
        help="Low-confidence labels routed here for human review.",
    )
    ap.add_argument("--limit", type=int, default=5000, help="Max JDs to fetch.")
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Labels below this confidence go to --uncertain-output instead.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch JDs and exit without calling the API.",
    )
    args = ap.parse_args()

    asyncio.run(
        main(
            beacon_url=args.beacon_url,
            output=args.output,
            uncertain_output=args.uncertain_output,
            limit=args.limit,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
        )
    )
