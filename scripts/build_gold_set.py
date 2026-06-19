"""Interactive CLI for hand-labeling a gold test set.

The labeling pipeline produced ~1675 LLM-labeled training records, but
fine-tune evaluation needs human-labeled ground truth that the model
never saw. This walker:

1. Stratified-samples ~200 JDs across SOC codes (so the test set has
   representation, not just the SWE majority class).
2. Shows you one JD at a time with the LLM's label as a suggestion.
3. Accepts Enter to agree, a code to override, or a few quick commands
   (m = show full text, ? = list codes, s = skip, q = save and quit).
4. Writes accepted labels to data/gold_test_set.jsonl.

Resume by source_id -- a Ctrl-C and re-run picks up where you stopped.

IMPORTANT: source_ids in the gold set must be EXCLUDED from training
in scripts/fine_tune.py to avoid evaluation leakage. The walker writes
gold only; the exclusion is enforced when fine-tuning starts.

Usage::

    python scripts/build_gold_set.py
    # default: stratified 15-per-class sample, target ~200 total
    # writes data/gold_test_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_soc_codes() -> list[dict]:
    return json.loads(
        (_REPO_ROOT / "app" / "data" / "onet_soc_codes.json").read_text(encoding="utf-8")
    )


def stratified_sample(
    records: list[dict],
    per_class: int,
    min_class_size: int,
    seed: int,
) -> list[dict]:
    """Pick up to per_class records from each class. Skip classes with
    fewer than min_class_size members (can't evaluate per-class accuracy
    with N=1 or N=2 anyway)."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["soc_code"]].append(r)

    picked: list[dict] = []
    for code, items in sorted(by_class.items()):
        if len(items) < min_class_size:
            continue
        n = min(per_class, len(items))
        picked.extend(rng.sample(items, n))

    rng.shuffle(picked)
    return picked


def term_width() -> int:
    try:
        return max(60, shutil.get_terminal_size().columns)
    except Exception:
        return 80


def render_jd(jd_text: str, truncate: bool) -> str:
    """Wrap JD text to terminal width, optionally truncate to fit a screen."""
    width = term_width()
    wrapped = textwrap.fill(jd_text, width=width)
    if not truncate:
        return wrapped
    lines = wrapped.splitlines()
    cap = 20
    if len(lines) <= cap:
        return wrapped
    return "\n".join(lines[:cap]) + f"\n... [{len(lines) - cap} more lines, press 'm']"


def print_jd_screen(
    idx: int,
    total: int,
    record: dict,
    show_full: bool,
) -> None:
    width = term_width()
    print("\n" + "=" * width)
    title = record.get("title") or "(no title)"
    company = record.get("company") or ""
    header = f"[{idx}/{total}] {title}"
    if company:
        header += f"  @  {company}"
    print(header)
    print("=" * width)
    print(render_jd(record["text"], truncate=not show_full))
    print("-" * width)
    print(
        f"LLM suggestion: {record['soc_code']} ({record.get('soc_title') or '?'})  "
        f"conf {record.get('confidence', 0):.2f}"
    )
    if record.get("rationale"):
        print(textwrap.fill(f"  > {record['rationale']}", width=width))
    print()


def print_help(soc_codes: list[dict]) -> None:
    print("\nO*NET SOC codes:")
    for c in soc_codes:
        print(f"  {c['soc_code']}  {c['title']}")
    print()


def prompt(record: dict, soc_codes: list[dict]) -> tuple[str, dict | None]:
    """Returns (action, extra). action in {'label', 'skip', 'quit', 'more'}.

    For 'label', extra is the gold record dict to write.
    """
    valid_codes = {c["soc_code"]: c["title"] for c in soc_codes}
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return ("quit", None)
        if raw == "":
            chosen = record["soc_code"]
            agreed = True
        elif raw.lower() in {"q", "quit"}:
            return ("quit", None)
        elif raw.lower() in {"s", "skip"}:
            return ("skip", None)
        elif raw.lower() in {"m", "more"}:
            return ("more", None)
        elif raw == "?":
            print_help(soc_codes)
            continue
        elif raw in valid_codes:
            chosen = raw
            agreed = chosen == record["soc_code"]
        else:
            print(
                f"  unknown input '{raw}'. Press Enter to accept, type a code, "
                f"or use ? / m / s / q."
            )
            continue

        gold = {
            "text": record["text"],
            "source_id": record.get("source_id"),
            "title": record.get("title"),
            "true_soc_code": chosen,
            "true_soc_title": valid_codes[chosen],
            "llm_soc_code": record["soc_code"],
            "llm_confidence": record.get("confidence"),
            "agreed_with_llm": agreed,
            "labeled_at": datetime.now(UTC).isoformat(),
        }
        return ("label", gold)


def main(
    source_path: str,
    output_path: str,
    per_class: int,
    min_class_size: int,
    target: int,
    seed: int,
) -> None:
    source = _REPO_ROOT / source_path
    out = _REPO_ROOT / output_path

    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(source)
    soc_codes = load_soc_codes()

    existing = load_jsonl(out)
    seen_ids = {r.get("source_id") for r in existing if r.get("source_id")}

    pool = stratified_sample(records, per_class, min_class_size, seed)
    pool = [r for r in pool if r.get("source_id") not in seen_ids]
    if target and len(pool) > target - len(existing):
        # honor the overall target across resumed sessions
        remaining = max(0, target - len(existing))
        pool = pool[:remaining]

    print(f"Source corpus:        {len(records)} records")
    print(f"Already labeled gold: {len(existing)}")
    print(f"This session's pool:  {len(pool)} records to walk")
    print(f"Output:               {out}")
    print()
    print("Controls: <Enter>=agree  <SOC code>=override  ?=list codes")
    print("          m=show full JD  s=skip  q=save and quit")
    print()

    if not pool:
        print("Nothing left to label. Done!")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    written_this_session = 0
    show_full = False

    with out.open("a", encoding="utf-8") as f:
        for i, record in enumerate(pool, start=1):
            show_full = False
            while True:
                print_jd_screen(
                    len(existing) + written_this_session + 1,
                    len(existing) + len(pool),
                    record,
                    show_full,
                )
                action, gold = prompt(record, soc_codes)
                if action == "more":
                    show_full = True
                    continue
                break

            if action == "quit":
                break
            if action == "skip":
                continue
            assert gold is not None
            f.write(json.dumps(gold, ensure_ascii=False) + "\n")
            f.flush()
            written_this_session += 1

    final = load_jsonl(out)
    print("\n--- Summary ---")
    print(f"  This session:        +{written_this_session} labels")
    print(f"  Total gold labels:    {len(final)}")
    if final:
        agreed = sum(1 for r in final if r.get("agreed_with_llm"))
        print(f"  Agreement with LLM:   {agreed}/{len(final)} ({100*agreed/len(final):.0f}%)")
        dist = Counter(r["true_soc_code"] for r in final)
        print(f"  Class coverage:       {len(dist)} of 30 SOC codes")
    print(f"\nFile: {out}")
    print("Reminder: exclude these source_ids from training when fine-tuning.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        default="data/labeled_jds_final.jsonl",
        help="Source corpus (LLM-labeled records to draw a stratified sample from).",
    )
    ap.add_argument(
        "--output",
        default="data/gold_test_set.jsonl",
        help="Where to write hand-labeled gold records.",
    )
    ap.add_argument(
        "--per-class",
        type=int,
        default=15,
        help="Max records to sample from each SOC class.",
    )
    ap.add_argument(
        "--min-class-size",
        type=int,
        default=3,
        help="Skip classes with fewer than this many records in the source corpus.",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=200,
        help="Aim for this many gold labels total across sessions (0 = no cap).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the stratified sample (stable across sessions).",
    )
    args = ap.parse_args()
    main(
        args.source,
        args.output,
        args.per_class,
        args.min_class_size,
        args.target,
        args.seed,
    )
