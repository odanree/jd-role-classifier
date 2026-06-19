"""Merge labeling runs into a clean training corpus for scripts/fine_tune.py.

Reads:
  data/labeled_jds.jsonl              -- high-confidence labels from llm_label_corpus.py
  data/relabeled_ma.jsonl             -- re-labels from relabel_uncertain_ma.py

Writes:
  data/labeled_jds_final.jsonl        -- the merged training corpus

Cleaning rules (applied in order):

  1. DROP every record whose soc_code is in --drop-codes. Default drops
     13-1111.00 (Management Analysts). The analysis after the initial
     run showed the original MA bucket was ~99% non-technical jobs
     (PMs, marketing managers, media buyers) that don't belong in a
     software-role classifier's training data. The handful of genuine
     TPMs are not worth the noise of the rest.

  2. For each record in relabeled_ma.jsonl: only accept the re-label
     if confidence >= --min-confidence AND new soc_code is not in
     --drop-codes. The relabel run found 6 of 228 reassignments were
     confident -- the rest were the model picking least-bad fallbacks
     when the sharper rubric forbade MA. Confidence is the clean signal
     for separating real reassignments from dumping-ground reshuffling.

  3. Records from labeled_jds.jsonl whose source_id appears in a
     confident relabel: pick the relabel (it had more rubric context).

  4. Records from relabeled_ma.jsonl whose source_id is NOT in
     labeled_jds.jsonl: include if it cleared the bar in rule 2
     (these came from the uncertain bucket, so any high-confidence
     relabel is a net gain).

Output schema matches what scripts/fine_tune.py expects:
  {"text": "...", "soc_code": "..."}
plus the labeling provenance fields kept for traceability:
  {"source_id", "title", "confidence", "rationale", "soc_title"}

Usage::

    python scripts/merge_labels.py
    # inspect: head data/labeled_jds_final.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default drop list. 13-1111.00 was the dumping-ground code in the initial
# run; analysis confirmed >99% of records under it were non-technical jobs
# that don't belong in a software-classifier corpus. Override with
# --drop-codes if reanalysis later changes this judgment.
DEFAULT_DROP_CODES = {"13-1111.00"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a torn final line from a crashed run.
                continue
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_record(r: dict) -> dict:
    """Strip non-training fields not expected by fine_tune.py, but keep
    provenance fields for spot-checking later."""
    return {
        "text": r["text"],
        "soc_code": r["soc_code"],
        "soc_title": r.get("soc_title"),
        "source_id": r.get("source_id"),
        "title": r.get("title"),
        "confidence": r.get("confidence"),
        "rationale": r.get("rationale"),
    }


def merge(
    labeled: list[dict],
    relabeled: list[dict],
    drop_codes: set[str],
    min_confidence: float,
) -> tuple[list[dict], dict]:
    """Return (merged_records, stats)."""
    # Index confident relabels by source_id. A confident relabel is one
    # that cleared the threshold AND landed on a non-dropped code.
    confident_relabels: dict[str, dict] = {}
    rejected_relabels = 0
    for r in relabeled:
        sid = r.get("source_id")
        if sid is None:
            continue
        if r.get("confidence", 0) < min_confidence:
            rejected_relabels += 1
            continue
        if r.get("soc_code") in drop_codes:
            rejected_relabels += 1
            continue
        confident_relabels[sid] = r

    merged: list[dict] = []
    stats = {
        "from_labeled_kept": 0,
        "from_labeled_dropped_code": 0,
        "from_labeled_replaced_by_relabel": 0,
        "from_relabel_added_new": 0,
        "relabel_records_rejected": rejected_relabels,
    }

    seen_source_ids: set[str] = set()

    # Pass 1: walk the main labeled file; drop, replace, or keep.
    for r in labeled:
        sid = r.get("source_id")
        if r.get("soc_code") in drop_codes:
            stats["from_labeled_dropped_code"] += 1
            continue
        if sid and sid in confident_relabels:
            # Prefer the relabel - it had more rubric context.
            merged.append(normalize_record(confident_relabels[sid]))
            stats["from_labeled_replaced_by_relabel"] += 1
            if sid:
                seen_source_ids.add(sid)
            continue
        merged.append(normalize_record(r))
        stats["from_labeled_kept"] += 1
        if sid:
            seen_source_ids.add(sid)

    # Pass 2: add confident relabels whose source_id wasn't in the main
    # labeled file (these came from the uncertain bucket originally).
    for sid, r in confident_relabels.items():
        if sid in seen_source_ids:
            continue
        merged.append(normalize_record(r))
        stats["from_relabel_added_new"] += 1

    return merged, stats


def main(
    labeled_path: str,
    relabeled_path: str,
    output_path: str,
    drop_codes: set[str],
    min_confidence: float,
    dry_run: bool,
) -> None:
    labeled = load_jsonl(_REPO_ROOT / labeled_path)
    relabeled = load_jsonl(_REPO_ROOT / relabeled_path)

    print(f"Loaded labeled:   {len(labeled):>5}  from {labeled_path}")
    print(f"Loaded relabeled: {len(relabeled):>5}  from {relabeled_path}")
    print(f"Drop codes:       {sorted(drop_codes)}")
    print(f"Min confidence:   {min_confidence}")
    print()

    merged, stats = merge(labeled, relabeled, drop_codes, min_confidence)

    print("=== Merge stats ===")
    print(f"  Kept from labeled (unchanged):         {stats['from_labeled_kept']}")
    print(f"  Replaced from labeled (had relabel):   {stats['from_labeled_replaced_by_relabel']}")
    print(f"  Dropped from labeled (dropped code):   {stats['from_labeled_dropped_code']}")
    print(f"  Added new from relabel:                {stats['from_relabel_added_new']}")
    print(f"  Rejected relabels (low conf / drop):   {stats['relabel_records_rejected']}")
    print(f"  -------")
    print(f"  Final corpus:                          {len(merged)}")
    print()

    print("=== Final class distribution ===")
    c = Counter(r["soc_code"] for r in merged)
    titles = {r["soc_code"]: r["soc_title"] for r in merged}
    for code, n in c.most_common():
        print(f"  {n:>4}  {code}  {titles.get(code)}")

    if dry_run:
        print()
        print("--dry-run: skipping write.")
        return

    out = _REPO_ROOT / output_path
    write_jsonl(out, merged)
    print()
    print(f"Wrote {len(merged)} records to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labeled",
        default="data/labeled_jds.jsonl",
        help="Primary labeled JSONL from llm_label_corpus.py",
    )
    ap.add_argument(
        "--relabeled",
        default="data/relabeled_ma.jsonl",
        help="Relabel JSONL from relabel_uncertain_ma.py",
    )
    ap.add_argument(
        "--output",
        default="data/labeled_jds_final.jsonl",
        help="Where to write the cleaned merged corpus",
    )
    ap.add_argument(
        "--drop-codes",
        nargs="*",
        default=sorted(DEFAULT_DROP_CODES),
        help="SOC codes to exclude from the final corpus",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for relabel records to override or augment the main corpus",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats without writing the output file",
    )
    args = ap.parse_args()
    main(
        args.labeled,
        args.relabeled,
        args.output,
        set(args.drop_codes),
        args.min_confidence,
        args.dry_run,
    )
