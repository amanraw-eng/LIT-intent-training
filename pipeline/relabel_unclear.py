"""Re-label transcripts stuck in UNCLEAR_INPUT (or another target intent).

Many UNCLEAR_INPUT rows are actually short, generic utterances (greetings,
backchannel acknowledgements like "haan"/"hoon", bare "no") that either fit an
existing intent on closer look, or recur often enough to deserve a brand-new
general intent. This script asks the LLM to re-examine each UNIQUE transcript
currently in the target intent(s), in batches, and for each one either:
  1. reassign it to an existing intent, or
  2. propose a new general intent (only for patterns that recur a lot), or
  3. leave it as-is (genuinely unclear/noise).
New-intent proposals from different batches are then consolidated into one
clean set (deduping synonyms) before anything is written.

Run with the venv's python as a module, from a dry run first:

    .venv/bin/python -m pipeline.relabel_unclear                     # dry run, Gemini, default output dir
    .venv/bin/python -m pipeline.relabel_unclear --use-openai         # dry run with OpenAI instead
    .venv/bin/python -m pipeline.relabel_unclear --apply              # actually commit the changes

--apply does four things:
  - rewrites data.jsonl: only rows whose CURRENT intent is one of --target-intents
    and whose transcript got reassigned have their `intent` field updated.
  - appends any new consolidated intents to intents.md as a new table section,
    with real example transcripts.
  - deletes intents.json so intents.py re-parses intents.md next import -
    every other script (build_dataset.py, app.py) then automatically picks up
    the updated taxonomy the next time it runs, since they all read from there.
  - recomputes intent counts post-relabel and drops any that are STILL at zero
    from intents.md (opt out with --keep-zero-count), then refreshes
    metadata.json's intents list.
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

from . import build_dataset as bd
from . import config, intents
from .intents import INTENT_NAMES, build_classifier


class RelabelItem(BaseModel):
    index: int
    # Exact existing intent name, OR a brand-new UPPER_SNAKE_CASE name, OR
    # UNCLEAR_INPUT/SILENCE_NOISE to leave as-is. Plain str (not the closed
    # Intent enum) since a new name by definition isn't in it yet.
    final_intent: str
    # Required (non-empty) only when final_intent is a brand-new name.
    new_intent_definition: str = ""


class RelabelBatchResult(BaseModel):
    items: list[RelabelItem]


class CanonicalIntent(BaseModel):
    canonical_name: str
    definition: str
    absorbs: list[str]  # raw proposed names this canonical intent covers (incl. itself)


class ConsolidationResult(BaseModel):
    canonical_intents: list[CanonicalIntent]
    discarded_names: list[str]  # raw proposed names not worth keeping -> fold back to UNCLEAR_INPUT


def _normalize_intent_name(raw):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", raw.strip().upper())
    return re.sub(r"_+", "_", name).strip("_")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _with_retries(fn, label):
    last_error = None
    total_attempts = 1 + config.CLASSIFY_MAX_RETRIES
    for attempt in range(1, total_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < total_attempts:
                print(
                    f"[relabel] {label} attempt {attempt}/{total_attempts} failed "
                    f"({type(e).__name__}: {e}), retrying in {config.CLASSIFY_RETRY_DELAY_S}s..."
                )
                time.sleep(config.CLASSIFY_RETRY_DELAY_S)
    raise last_error


def extract_target_rows(data_path, target_intents):
    """Returns {transcript_text: [chunk_path, ...]} for every row currently
    labeled with one of target_intents (skips empty transcripts - those are
    legitimately SILENCE_NOISE territory, not worth re-examining)."""
    target_intents = set(target_intents)
    groups = defaultdict(list)
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("intent") in target_intents:
                text = (rec.get("transcript") or "").strip()
                if text:
                    groups[text].append(rec["chunk_path"])
    return groups


def _relabel_prompt(transcripts):
    numbered = "\n".join(f'{i}: """{t}"""' for i, t in enumerate(transcripts))
    return (
        "These transcripts are currently labeled UNCLEAR_INPUT, meaning none of "
        "the intents listed above were judged to fit. Re-examine each one "
        "independently (they are unrelated turns from different calls):\n"
        "1. If it now clearly fits one of the EXISTING intents listed above, set "
        "final_intent to that EXACT existing name.\n"
        "2. Otherwise, if it doesn't fit any existing intent but represents a "
        "common, general communicative function likely shared by MANY other "
        "transcripts (e.g. a bare greeting, a filler acknowledgement/backchannel, "
        "a generic bare negative response with no other content) that deserves "
        "its own label, propose ONE new intent: set final_intent to a brand-new "
        "UPPER_SNAKE_CASE name that does not match any existing intent, and fill "
        "new_intent_definition with a short 'Fires when ...' style definition. "
        "Only do this for genuinely general, recurring patterns - never invent a "
        "new intent for a single one-off phrase.\n"
        "3. Otherwise - genuinely unintelligible, off-topic, crosstalk, or too "
        "ambiguous without more call context - set final_intent to UNCLEAR_INPUT "
        "(or SILENCE_NOISE if it is essentially empty/noise).\n\n"
        f"Return exactly one item per index, {len(transcripts)} items total.\n\n"
        f"{numbered}"
    )


def propose_relabels(classifier, unique_transcripts, batch_size):
    """Returns list of (transcript_text, RelabelItem), one per unique transcript."""
    system_prompt = intents.build_system_prompt()
    results = []
    n_batches = (len(unique_transcripts) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = unique_transcripts[b * batch_size : (b + 1) * batch_size]
        parsed = _with_retries(
            lambda: classifier.generate_structured(
                system_prompt, _relabel_prompt(batch), RelabelBatchResult
            ),
            f"relabel batch {b + 1}/{n_batches}",
        )
        by_index = {item.index: item for item in parsed.items}
        missing = [i for i in range(len(batch)) if i not in by_index]
        if missing:
            raise intents.ClassificationError(
                f"relabel batch {b + 1} response missing indices: {missing}"
            )
        for i, text in enumerate(batch):
            results.append((text, by_index[i]))
        print(f"[relabel] proposed batch {b + 1}/{n_batches} ({len(batch)} transcripts)")
    return results


def _consolidation_prompt(raw_proposals):
    lines = [
        "Below are new-intent proposals gathered from independent batches while "
        "re-labeling transcripts that didn't fit any existing intent in a loan "
        "collections call dataset. Many are near-duplicates or synonyms of each "
        "other (e.g. GREETING vs HELLO_GREETING vs GREETING_OPENING).",
        "",
        "Merge them into a small, clean, final set of GENERAL intents - each one "
        "clearly distinct from the others and from the existing taxonomy, each "
        "with a short 'Fires when ...' definition. If a proposal is too narrow, "
        "redundant with another, or on reflection not general enough to deserve "
        "its own intent, put its raw name in discarded_names instead (those "
        "transcripts will stay UNCLEAR_INPUT). Every raw name below must appear "
        "in exactly one canonical intent's `absorbs` list, or in discarded_names.",
        "",
        "Raw proposals (raw_name: definition [n=how many transcripts proposed it] example transcripts):",
    ]
    for p in raw_proposals:
        examples = "; ".join(p["examples"][:3])
        lines.append(f"- {p['raw_name']}: {p['definition']} [n={p['count']}] e.g. {examples}")
    return "\n".join(lines)


def consolidate_new_intents(classifier, raw_proposals):
    if not raw_proposals:
        return ConsolidationResult(canonical_intents=[], discarded_names=[])
    parsed = _with_retries(
        lambda: classifier.generate_structured(
            "You are cleaning up a proposed extension to a closed intent taxonomy.",
            _consolidation_prompt(raw_proposals),
            ConsolidationResult,
        ),
        "consolidation",
    )
    return parsed


def _apply_relabels(data_path, final_map, target_intents):
    """Rewrites data.jsonl. Only rows whose CURRENT intent is in target_intents
    are eligible - so a transcript text that also appears under an unrelated,
    already-correct intent elsewhere is left untouched."""
    target_intents = set(target_intents)
    tmp_path = data_path + ".tmp"
    changed = 0
    with open(data_path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("intent") in target_intents:
                new_intent = final_map.get((rec.get("transcript") or "").strip())
                if new_intent and new_intent != rec["intent"]:
                    rec["intent"] = new_intent
                    changed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, data_path)
    return changed


def _add_intents_to_taxonomy(md_path, canonical_defs, examples_by_name):
    lines = [
        "",
        "### General group (auto-added by relabel_unclear.py)",
        "",
        "| Intent | Fires when | Example utterances |",
        "|---|---|---|",
    ]
    for name, defn in canonical_defs.items():
        examples = examples_by_name.get(name, [])[:3]
        examples_str = ", ".join(f'"{e}"' for e in examples) if examples else "(see dataset)"
        lines.append(f"| `{name}` | {defn} | {examples_str} |")

    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content.rstrip("\n") + "\n" + "\n".join(lines) + "\n")


def _remove_intents_from_taxonomy(md_path, names_to_remove):
    names_to_remove = set(names_to_remove)
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            m = re.match(r"\|\s*`([A-Z0-9_]+)`", stripped)
            if m and m.group(1) in names_to_remove:
                continue
        kept.append(line)
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(kept)


def _count_intents(data_path, intent_names):
    counts = Counter({name: 0 for name in intent_names})
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts[rec.get("intent", "NO_INTENT_FOUND")] += 1
    return counts


def _write_intent_analytics(counts, path):
    """Same format as pipeline/intent_analytics.py's own output, kept in sync
    here too so it doesn't go stale after a relabel until someone re-runs it by hand."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("{")
        for i, (intent, count) in enumerate(counts.most_common()):
            f.write(("\n" if i == 0 else ",\n") + f'"{intent}": {count}')
        f.write("\n}")


def run_relabel(
    output_dir=config.OUTPUT_DIR,
    target_intents=("UNCLEAR_INPUT",),
    batch_size=50,
    use_openai=False,
    apply=False,
    drop_zero_count=True,
):
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    groups = extract_target_rows(data_path, target_intents)
    unique_transcripts = sorted(groups, key=lambda t: -len(groups[t]))  # most-repeated first
    total_rows = sum(len(v) for v in groups.values())
    print(
        f"[relabel] {total_rows} rows across {len(unique_transcripts)} unique transcripts "
        f"currently in {list(target_intents)}"
    )
    if not unique_transcripts:
        print("[relabel] nothing to do")
        return {"applied": False, "rows_changed": 0}

    classifier = build_classifier(use_openai=use_openai)
    print(f"[relabel] using {classifier.name} ({classifier.model})")

    proposals = propose_relabels(classifier, unique_transcripts, batch_size)

    existing_names = set(INTENT_NAMES)
    target_set = set(target_intents)
    text_to_existing = {}
    text_to_new_raw = {}
    raw_new_defs = {}
    raw_new_examples = defaultdict(list)
    kept_unclear = 0

    for text, item in proposals:
        name = _normalize_intent_name(item.final_intent)
        if name in target_set:
            # Genuinely still belongs to whichever target intent it came from -
            # not a reassignment, checked first so this never gets counted as one.
            kept_unclear += 1
        elif name in existing_names:
            text_to_existing[text] = name
        else:
            text_to_new_raw[text] = name
            if name not in raw_new_defs:
                raw_new_defs[name] = item.new_intent_definition.strip() or "(no definition given)"
            raw_new_examples[name].append(text)

    raw_proposals = [
        {
            "raw_name": name,
            "definition": raw_new_defs[name],
            "count": len(raw_new_examples[name]),
            "examples": raw_new_examples[name],
        }
        for name in raw_new_defs
    ]
    print(
        f"[relabel] {len(text_to_existing)} -> existing intents, "
        f"{len(text_to_new_raw)} unique transcripts -> {len(raw_proposals)} raw new-intent "
        f"proposals, {kept_unclear} left unclear"
    )

    consolidation = consolidate_new_intents(classifier, raw_proposals)

    canonical_map = {}
    canonical_defs = {}
    examples_by_canonical = defaultdict(list)
    for ci in consolidation.canonical_intents:
        canon_name = _normalize_intent_name(ci.canonical_name)
        canonical_defs[canon_name] = ci.definition.strip()
        for raw in ci.absorbs:
            canonical_map[raw] = canon_name
            examples_by_canonical[canon_name].extend(raw_new_examples.get(raw, []))
    for raw in consolidation.discarded_names:
        canonical_map.setdefault(raw, None)

    final_map = dict(text_to_existing)
    for text, raw_name in text_to_new_raw.items():
        canon = canonical_map.get(raw_name)
        final_map[text] = canon if canon else "UNCLEAR_INPUT"

    # A discarded new-intent proposal falls back to "UNCLEAR_INPUT" above, which
    # is a no-op for any text whose current intent already is that - fold those
    # into "unchanged" too rather than reporting them as a change.
    actual_changes = {t: v for t, v in final_map.items() if v not in target_set}
    no_op_count = len(final_map) - len(actual_changes)

    change_counts = Counter(actual_changes.values())
    print("\n[relabel] === PROPOSED CHANGES (unique transcripts / rows) ===")
    for name, count in change_counts.most_common():
        rows_affected = sum(len(groups[t]) for t, n in actual_changes.items() if n == name)
        tag = " (NEW)" if name in canonical_defs else ""
        print(f"  -> {name}{tag}: {count} / {rows_affected} rows")
    unchanged = kept_unclear + no_op_count
    print(f"  -> unchanged (stay in {list(target_intents)}): {unchanged} unique transcripts")

    if canonical_defs:
        print("\n[relabel] === NEW INTENTS ===")
        for name, defn in canonical_defs.items():
            examples = examples_by_canonical[name][:4]
            print(f"  {name}: {defn}")
            print(f"    e.g. {examples}")

    if consolidation.discarded_names:
        print(f"\n[relabel] discarded (folded back to UNCLEAR_INPUT): {consolidation.discarded_names}")

    if not apply:
        print("\n[relabel] DRY RUN - no files were changed. Re-run with --apply to commit.")
        return {
            "applied": False,
            "unique_transcripts": len(unique_transcripts),
            "reassigned_to_existing": len(text_to_existing),
            "new_intents_proposed": len(canonical_defs),
            "change_counts": dict(change_counts),
        }

    rows_changed = _apply_relabels(data_path, final_map, target_intents)
    print(f"\n[relabel] rewrote {rows_changed} rows in {data_path}")

    if canonical_defs:
        _add_intents_to_taxonomy(config.INTENTS_MD_PATH, canonical_defs, examples_by_canonical)
        print(f"[relabel] added {len(canonical_defs)} new intents to {config.INTENTS_MD_PATH}")

    if os.path.exists(config.INTENTS_JSON_PATH):
        os.remove(config.INTENTS_JSON_PATH)

    fresh_rows = intents.parse_intents(config.INTENTS_MD_PATH)
    fresh_names = [r["name"] for r in fresh_rows]
    counts_after = _count_intents(data_path, fresh_names)

    zero_count = [name for name in fresh_names if counts_after[name] == 0]
    if drop_zero_count and zero_count:
        _remove_intents_from_taxonomy(config.INTENTS_MD_PATH, zero_count)
        if os.path.exists(config.INTENTS_JSON_PATH):
            os.remove(config.INTENTS_JSON_PATH)
        fresh_rows = intents.parse_intents(config.INTENTS_MD_PATH)
        fresh_names = [r["name"] for r in fresh_rows]
        counts_after = _count_intents(data_path, fresh_names)
        print(f"[relabel] removed {len(zero_count)} zero-count intents from taxonomy: {zero_count}")
    elif zero_count:
        print(f"[relabel] {len(zero_count)} zero-count intents left as-is (--keep-zero-count): {zero_count}")

    # parse_intents() (unlike _load_intent_rows()) doesn't write the cache -
    # write it explicitly so the next import elsewhere doesn't re-parse the md.
    with open(config.INTENTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(fresh_rows, f, ensure_ascii=False, indent=2)

    bd.write_metadata(
        output_dir,
        intents=fresh_names,
        num_chunks_processed=bd._count_lines(data_path),
        status="relabeled",
        last_relabel_at=_now(),
    )

    _write_intent_analytics(counts_after, config.INTENT_ANALYTICS_PATH)
    print(f"[relabel] refreshed {config.INTENT_ANALYTICS_PATH}")

    print("\n[relabel] === FINAL INTENT COUNTS ===")
    for name, count in counts_after.most_common():
        print(f"  {name}: {count}")
    print(
        "\n[relabel] done. The classification prompt (build_dataset.py, "
        "retry-skipped) and app.py's filter list pick up the updated taxonomy "
        "automatically next time they run, since they all read intents.json "
        "fresh on import."
    )

    return {
        "applied": True,
        "rows_changed": rows_changed,
        "new_intents_added": list(canonical_defs.keys()),
        "zero_count_removed": zero_count if drop_zero_count else [],
        "final_counts": dict(counts_after),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR)
    parser.add_argument("--target-intents", nargs="+", default=["UNCLEAR_INPUT"])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument(
        "--apply", action="store_true", help="Actually commit changes (default is dry-run only)"
    )
    parser.add_argument(
        "--keep-zero-count",
        action="store_true",
        help="Don't remove taxonomy intents that have zero rows after relabeling",
    )
    args = parser.parse_args()

    result = run_relabel(
        output_dir=args.output_dir,
        target_intents=tuple(args.target_intents),
        batch_size=args.batch_size,
        use_openai=args.use_openai,
        apply=args.apply,
        drop_zero_count=not args.keep_zero_count,
    )
    print(f"\n[relabel] result -> {result}")


if __name__ == "__main__":
    main()
