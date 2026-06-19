"""Re-label JDs that the initial run parked under 13-1111.00 Management Analysts.

The initial labeling run (``llm_label_corpus.py``) dumped 786 of 987
uncertain records under ``13-1111.00 Management Analysts``. That code's
rubric is **TPM / Program Manager / Delivery Manager only**, but the
model used it as a fallback for technical IC roles it couldn't slot
confidently elsewhere. Training BERT+LoRA on that mapping would teach
"ambiguous JD -> Management Analysts", which is the opposite of useful.

This script:
1. Reads ``data/labeled_jds_uncertain.jsonl``,
2. Filters to records currently labeled ``13-1111.00``,
3. Re-runs the classifier with a sharper system prompt that explicitly
   bans ``13-1111.00`` as a fallback for technical IC work,
4. Writes the revised labels to ``data/relabeled_ma.jsonl`` for review
   and merging with the main corpus.

Reads the original JD text from the uncertain JSONL -- no DB tunnel
required. Pure file-in / file-out. Resumes by ``source_id``.

Cost on ~786 records with caching: ~$2.

Usage::

    python scripts/relabel_uncertain_ma.py
    # then review data/relabeled_ma.jsonl before merging
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Run: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

try:
    import anthropic
except ImportError:
    print("anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)


INPUT_USD_PER_MTOK = 3.00
OUTPUT_USD_PER_MTOK = 15.00
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

TARGET_SOC_CODE = "13-1111.00"


def load_soc_codes() -> list[dict]:
    path = _REPO_ROOT / "app" / "data" / "onet_soc_codes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_system_prompt(soc_codes: list[dict]) -> str:
    lines = [
        "You are labeling job descriptions with O*NET Standard Occupational",
        "Classification (SOC) codes. You will be given a job title and JD",
        "text. Pick the SINGLE code that best describes the role.",
        "",
        "CRITICAL -- DO NOT use 13-1111.00 Management Analysts as a fallback.",
        "That code is ONLY for true Technical Program Manager / Program",
        "Manager / Delivery Manager roles -- people whose day-to-day is",
        "coordinating people, timelines, and dependencies across teams,",
        "NOT building software, models, or infrastructure themselves.",
        "If the JD describes any technical IC work (writing code, training",
        "models, building pipelines, designing systems, security analysis,",
        "QA), DO NOT label it Management Analysts even if the role is",
        "ambiguous. Instead, pick the closest technical IC code from the",
        "list below and use a lower confidence (0.4-0.6) if you're not",
        "sure. Confidence below 0.5 is acceptable -- a noisy IC label is",
        "more useful than a confidently-wrong management label.",
        "",
        "Rules for all roles:",
        "- Choose based on the JD's day-to-day responsibilities and tech",
        "  stack, not on flashy buzzwords or company self-description.",
        "- If the role spans multiple codes, pick the one that matches the",
        "  majority of the work.",
        "- Confidence reflects how clearly the JD points at one code vs.",
        "  others. Above 0.8 means unambiguous; 0.5-0.7 means there's a",
        "  reasonable second choice; below 0.5 means the JD is too generic",
        "  or contradictory to label cleanly.",
        "- Rationale is 1-2 sentences citing the specific JD content that",
        "  drove the choice (e.g. 'JD lists PyTorch, LoRA, model training',",
        "  not 'role sounds like MLE').",
        "",
        "O*NET SOC codes you may select from:",
        "",
    ]
    for c in soc_codes:
        lines.append(f"- {c['soc_code']} -- {c['title']}")
    return "\n".join(lines)


def build_output_schema(soc_codes: list[dict]) -> dict:
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


def load_existing_relabeled(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen: set[str] = set()
    with out_path.open(encoding="utf-8") as f:
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


def load_ma_records(uncertain_path: Path, exclude_ids: set[str]) -> list[dict]:
    out: list[dict] = []
    with uncertain_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("soc_code") != TARGET_SOC_CODE:
                continue
            if r.get("source_id") in exclude_ids:
                continue
            out.append(r)
    return out


def classify_one(client, system_prompt, output_schema, title, jd_text):
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
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def prewarm_cache(client, system_prompt):
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


def estimate_cost(input_t, output_t, c_write, c_read):
    return (
        (input_t / 1_000_000) * INPUT_USD_PER_MTOK
        + (output_t / 1_000_000) * OUTPUT_USD_PER_MTOK
        + (c_write / 1_000_000) * INPUT_USD_PER_MTOK * CACHE_WRITE_MULT
        + (c_read / 1_000_000) * INPUT_USD_PER_MTOK * CACHE_READ_MULT
    )


def main(uncertain_path: str, output_path: str, dry_run: bool) -> None:
    uncertain = Path(uncertain_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    soc_codes = load_soc_codes()
    system_prompt = build_system_prompt(soc_codes)
    output_schema = build_output_schema(soc_codes)
    titles = {c["soc_code"]: c["title"] for c in soc_codes}

    seen = load_existing_relabeled(out)
    if seen:
        print(f"Resume: skipping {len(seen)} already-relabeled records.")

    records = load_ma_records(uncertain, seen)
    print(f"Loaded {len(records)} '{TARGET_SOC_CODE}' records to re-label.")

    if dry_run:
        print("--dry-run: skipping API calls. First 3 titles:")
        for r in records[:3]:
            print(f"  {r.get('source_id')}  {(r.get('title') or '?')[:70]}")
        return
    if not records:
        print("Nothing to do.")
        return

    client = anthropic.Anthropic()
    print("Pre-warming prompt cache...")
    prewarm_cache(client, system_prompt)

    totals = {
        "input": 0,
        "output": 0,
        "cache_write": 0,
        "cache_read": 0,
        "errors": 0,
        "changed": 0,
        "unchanged": 0,
    }
    started = time.monotonic()

    with out.open("a", encoding="utf-8") as out_f:
        for i, r in enumerate(records, start=1):
            try:
                parsed, usage = classify_one(
                    client, system_prompt, output_schema, r.get("title"), r["text"]
                )
            except anthropic.APIError as e:
                totals["errors"] += 1
                print(
                    f"  [{i}/{len(records)}] FAIL {r.get('source_id')}: {e}",
                    file=sys.stderr,
                )
                continue
            except (KeyError, json.JSONDecodeError) as e:
                totals["errors"] += 1
                print(
                    f"  [{i}/{len(records)}] PARSE-FAIL {r.get('source_id')}: {e}",
                    file=sys.stderr,
                )
                continue

            totals["input"] += usage.input_tokens
            totals["output"] += usage.output_tokens
            totals["cache_write"] += usage.cache_creation_input_tokens or 0
            totals["cache_read"] += usage.cache_read_input_tokens or 0

            new_code = parsed["soc_code"]
            if new_code != r.get("soc_code"):
                totals["changed"] += 1
            else:
                totals["unchanged"] += 1

            record = {
                "text": r["text"],
                "soc_code": new_code,
                "soc_title": titles.get(new_code, "unknown"),
                "source_id": r.get("source_id"),
                "title": r.get("title"),
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
                "previous_soc_code": r.get("soc_code"),
                "previous_confidence": r.get("confidence"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

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
                    f"  [{i}/{len(records)}] {rate:.2f} jd/s  "
                    f"changed={totals['changed']}  unchanged={totals['unchanged']}  "
                    f"errors={totals['errors']}  spent=${cost:.2f}"
                )

    elapsed = time.monotonic() - started
    cost = estimate_cost(
        totals["input"], totals["output"], totals["cache_write"], totals["cache_read"]
    )
    print("\n--- Summary ---")
    print(f"  Processed:      {len(records)} records in {elapsed:.0f}s")
    print(f"  Changed label:  {totals['changed']}  -> different SOC code than original MA")
    print(f"  Kept MA:        {totals['unchanged']}  -> still 13-1111.00 (likely genuine TPM)")
    print(f"  Errors:         {totals['errors']}")
    print(
        f"  Tokens:         input={totals['input']:,}  output={totals['output']:,}  "
        f"cache_write={totals['cache_write']:,}  cache_read={totals['cache_read']:,}"
    )
    print(f"  Estimated cost: ${cost:.2f}")
    print(f"\nReview {output_path} before merging into the main corpus.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--uncertain",
        default="data/labeled_jds_uncertain.jsonl",
        help="Input -- the uncertain JSONL from the initial labeling run.",
    )
    ap.add_argument(
        "--output",
        default="data/relabeled_ma.jsonl",
        help="Where to write revised labels for review.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and report counts without calling the API.",
    )
    args = ap.parse_args()
    main(args.uncertain, args.output, args.dry_run)
