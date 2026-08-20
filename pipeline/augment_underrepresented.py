"""Augment EVERY intent with new example sentences via LLM (spread across 10
Hindi regional tones/registers, code-mixed with common English words, varied
length/completeness - real speech, not literary Hindi), synthesize them with
TTS, and distort the audio with real noise extracted from the existing call
corpus, so it matches the domain of the real data.

For each intent: new_needed = max(--min-new-per-intent, max(--target-total -
current_total, 0)), where current_total = real rows in --source-dir +
already-synthesized rows in --output-dir for that intent. So every intent
gets at least --min-new-per-intent new rows, and intents below --target-total
get topped up to it.

    .venv/bin/python -m pipeline.augment_underrepresented run
    .venv/bin/python -m pipeline.augment_underrepresented run --intents DO_NOT_CALL --target-total 3000

Requires TTS_WS_ENDPOINT (and TTS_VOICE_ID) in .env.

Output: call_trascript_intent_data_augmented/data.jsonl + audio/ subfolder,
same row schema as v1/v2/v3 (chunk_path/oid/conversation_id/etc, with
synthetic placeholder ids) so pipeline.push_to_hub works unchanged against it.

Resumable: re-running skips sentences already synthesized for a given intent
(tracked by exact transcript text already present in data.jsonl for that intent).
"""

import argparse
import asyncio
import json
import os
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from pydantic import BaseModel

from . import config
from .intents import INTENT_ROWS, IntentClassifier, OpenAIIntentClassifier, build_system_prompt
from .noise_bank import augment as noise_augment
from .noise_bank import extract_noise_donors
from .tts_client import tts_with_retry


class GeneratedSentences(BaseModel):
    sentences: list[str]


def _now():
    return datetime.now(timezone.utc).isoformat()


def count_intents(data_path):
    counts = Counter()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                counts[json.loads(line).get("intent", "NO_INTENT_FOUND")] += 1
    return counts


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
                    f"[augment] {label} attempt {attempt}/{total_attempts} failed "
                    f"({type(e).__name__}: {e}), retrying in {config.CLASSIFY_RETRY_DELAY_S}s..."
                )
                time.sleep(config.CLASSIFY_RETRY_DELAY_S)
    raise last_error


# Regional/register variety - deliberately spread generation across these so
# the corpus doesn't skew toward literary Standard Hindi, which real borrowers
# on a phone call essentially never speak.
TONES = [
    "Standard Hindi",
    "Brij",
    "Haryanvi",
    "Bhojpuri",
    "Awadhi",
    "Rajasthani",
    "Punjabi-influenced Hindi",
    "Mumbai Hindi",
    "Delhi colloquial",
    "UP colloquial",
]


def _tone_generation_prompt(intent_row, tone, n, avoid_examples):
    avoid_block = ""
    if avoid_examples:
        avoid_block = (
            "\n\nAvoid near-duplicates of sentences already generated for this intent:\n"
            + "\n".join(f'- "{s}"' for s in avoid_examples[-40:])
        )
    return (
        f"Generate {n} REAL, natural-sounding things a BORROWER might actually say "
        f"during an EMI/loan-collections phone call in India, that clearly express "
        f"the `{intent_row['name']}` intent - ALL in the **{tone}** tone/dialect.\n\n"
        f"Intent definition: {intent_row['condition']}\n"
        f"Reference examples (for MEANING only - these are too formal, do not copy "
        f"their register): {intent_row['examples']}\n\n"
        "These must sound like an actual rushed/informal phone call, NOT written or "
        "literary Hindi. Specifically:\n"
        "- Code-mix in common English words the way real Hindi speakers do. For "
        "MOST of these, keep the English word in ACTUAL LATIN SCRIPT embedded "
        "directly inside the Devanagari sentence - e.g. 'हाँ, EMI तो pay कर दी "
        "थी', 'मुझे थोड़ा time चाहिए', 'OK, बता दीजिए' - do NOT transliterate it "
        "into Devanagari (avoid writing it as 'फोन', 'कॉल', 'टाइम' etc. - write "
        "'phone', 'call', 'time' in English letters instead). A good chunk of "
        "sentences should have at least one such English word, but not every "
        "single one, and don't overuse any single word. Borrowers rarely address "
        "the caller as 'सर'/'मैडम' - use those VERY sparingly (a small minority "
        "of sentences at most), most should have no address term at all.\n"
        f"- Actually write in {tone}'s real vocabulary/verb endings/particles where "
        f"it differs from Standard Hindi (not Standard Hindi text merely labeled "
        f"as {tone}).\n"
        "- Vary length A LOT: roughly a third should be very short (1-4 words, a "
        "clipped reaction like a real interrupted caller), a third medium, a third "
        "a longer natural sentence.\n"
        "- Some should be grammatically incomplete or trail off mid-thought, like "
        "real speech - not every sentence needs to be a complete, well-formed "
        "sentence.\n"
        f"{avoid_block}\n\n"
        f"Return exactly {n} sentences."
    )


def generate_sentences_for_intent(classifier, intent_row, total_n, batch_size=None, tones=None):
    batch_size = batch_size or config.AUGMENT_SENTENCE_BATCH_SIZE
    tones = tones or TONES
    system_prompt = build_system_prompt()
    all_sentences = []
    seen = set()

    per_tone = total_n // len(tones)
    remainder = total_n - per_tone * len(tones)

    for i, tone in enumerate(tones):
        tone_target = per_tone + (1 if i < remainder else 0)
        if tone_target <= 0:
            continue
        tone_collected = 0
        while tone_collected < tone_target:
            this_n = min(batch_size, tone_target - tone_collected)
            prompt = _tone_generation_prompt(intent_row, tone, this_n, all_sentences)
            parsed = _with_retries(
                lambda: classifier.generate_structured(system_prompt, prompt, GeneratedSentences),
                f"generate [{tone}] batch for {intent_row['name']}",
            )
            for s in parsed.sentences:
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    all_sentences.append(s)
            tone_collected += this_n  # advance on attempted count, not unique-yield, to bound the loop
        print(f"[augment] {intent_row['name']} [{tone}]: {len(all_sentences)}/{total_n} unique sentences so far")

    return all_sentences


def _read_done_texts(data_path, intent_name):
    done = set()
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("intent") == intent_name:
                    done.add(rec.get("transcript"))
    return done


def load_voice_clone_paths(users_json_path=None):
    """voice_clone_users.json is a flat list of reference-audio paths (see
    pipeline/voice_clone_bulk.py). Missing file/empty list -> no cloning,
    falls back to config.TTS_VOICE_ID."""
    users_json_path = users_json_path or config.VOICE_CLONE_USERS_JSON
    if not os.path.exists(users_json_path):
        return []
    with open(users_json_path, encoding="utf-8") as f:
        paths = json.load(f)
    return [p for p in paths if os.path.exists(p)]


async def _synthesize_and_augment_one(
    text, out_wav_path: Path, sem, donors, rng, voice_paths=None, speed_range=None
):
    voice = rng.choice(voice_paths) if voice_paths else None
    speed = rng.uniform(*speed_range) if speed_range else None
    tmp_tts_path = out_wav_path.with_suffix(".tts.wav")
    _, ok, err = await tts_with_retry(str(out_wav_path), text, tmp_tts_path, sem, voice=voice, speed=speed)
    if not ok:
        return False, err
    try:
        clean, sr = sf.read(str(tmp_tts_path), dtype="float32", always_2d=False)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        mixed, sr_out = noise_augment(clean, sr, donors, rng=rng)
        sf.write(str(out_wav_path), mixed, sr_out)
        return True, len(mixed) / sr_out
    except Exception as e:
        return False, str(e)
    finally:
        tmp_tts_path.unlink(missing_ok=True)


async def run(
    output_dir=config.OUTPUT_DIR_AUGMENTED,
    source_dir=config.OUTPUT_DIR_V3,
    min_new_per_intent=config.AUGMENT_MIN_NEW_PER_INTENT,
    target_total=config.AUGMENT_TARGET_TOTAL,
    intents_filter=None,
    use_openai=True,
    model=config.AUGMENT_MODEL,
    tts_concurrency=config.TTS_CONCURRENCY,
    noise_donor_pool_size=300,
    extra_for_done=0,
    speed_range=(config.AUGMENT_SPEED_MIN, config.AUGMENT_SPEED_MAX),
):
    """For every intent (or just intents_filter if given): new_needed =
    max(min_new_per_intent, max(target_total - current_total, 0)), where
    current_total = real rows in source_dir + already-synthesized rows for
    that intent in output_dir. So every intent gets AT LEAST min_new_per_intent
    new rows, and intents below target_total get topped up to it.

    extra_for_done: one-off override (NOT part of the resumable formula above,
    so it won't re-fire every run) - for intents that already hit their target
    (needed == 0), still generate this many more anyway. Use this to inject a
    deliberate one-time top-up (e.g. to backfill speed variety into intents
    synthesized before speed randomization existed) without permanently
    reintroducing the "always add 500" bug this formula was fixed to avoid.
    """
    if not config.TTS_WS_ENDPOINT:
        raise RuntimeError("TTS_WS_ENDPOINT not set - add it to .env")

    os.makedirs(output_dir, exist_ok=True)
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(output_dir, config.DATA_FILENAME)
    errors_path = os.path.join(output_dir, config.ERRORS_LOG_FILENAME)

    source_data_path = os.path.join(source_dir, config.DATA_FILENAME)
    real_counts = count_intents(source_data_path)

    intent_rows = INTENT_ROWS
    if intents_filter:
        intent_rows = [r for r in INTENT_ROWS if r["name"] in intents_filter]

    plan = []
    print(f"[augment] plan (min_new_per_intent={min_new_per_intent}, target_total={target_total}):")
    for row in intent_rows:
        intent_name = row["name"]
        real_count = real_counts.get(intent_name, 0)
        done_texts = _read_done_texts(data_path, intent_name)
        current_total = real_count + len(done_texts)
        # min_new_per_intent is a floor on TOTAL synth ever generated for this
        # intent, not a flat add-on-every-run - net it against synth already
        # written in prior runs, else every resume re-adds another 500+ on top
        # of already-satisfied intents (e.g. CALL_OPENING_OR_PROMPT already at
        # target kept regenerating another ~500 on every subsequent run).
        needed_for_min = max(min_new_per_intent - len(done_texts), 0)
        needed_for_target = max(target_total - current_total, 0)
        needed = max(needed_for_min, needed_for_target)
        if needed <= 0 and extra_for_done > 0:
            needed = extra_for_done
        plan.append((row, done_texts, needed))
        print(f"  {intent_name}: current={current_total} (real={real_count}, synth={len(done_texts)}) -> +{needed} new")

    print("[augment] building noise donor pool from real call audio...")
    donor_paths = []
    with open(source_data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= noise_donor_pool_size * 5:  # oversample candidates; extraction filters most out
                break
            donor_paths.append(json.loads(line)["chunk_path"])
    donors = extract_noise_donors(donor_paths, max_donors=noise_donor_pool_size)
    print(f"[augment] {len(donors)} noise donors extracted")

    voice_paths = load_voice_clone_paths()
    if voice_paths:
        print(f"[augment] {len(voice_paths)} cloned voices available - picking one at random per sentence")
    else:
        print(f"[augment] no voice_clone_users.json entries found - using default TTS_VOICE_ID")

    classifier = OpenAIIntentClassifier(model=model) if use_openai else IntentClassifier()
    print(f"[augment] sentence-generation backend: {classifier.name} ({classifier.model})")

    sem = asyncio.Semaphore(tts_concurrency)
    rng = random.Random(1234)
    results = {}

    with open(data_path, "a", encoding="utf-8") as data_f, open(errors_path, "a", encoding="utf-8") as err_f:
        for row, done_texts, needed in plan:
            intent_name = row["name"]
            if needed <= 0:
                results[intent_name] = {"generated": 0, "synthesized": 0, "already_done": len(done_texts)}
                continue

            sentences = generate_sentences_for_intent(classifier, row, needed + 20)
            sentences = [s for s in sentences if s not in done_texts][:needed]

            async def _do_one(text):
                rec_uuid = uuid.uuid4().hex[:12]
                out_path = audio_dir / f"{intent_name}_{rec_uuid}.wav"
                ok, info = await _synthesize_and_augment_one(
                    text, out_path, sem, donors, rng, voice_paths=voice_paths, speed_range=speed_range
                )
                return text, out_path, ok, info

            synthesized_this_intent = 0
            for fut in asyncio.as_completed([_do_one(text) for text in sentences]):
                text, out_path, ok, info = await fut
                if not ok:
                    err_f.write(
                        json.dumps({"intent": intent_name, "text": text, "error": info, "time": _now()}) + "\n"
                    )
                    err_f.flush()
                    print(f"[augment] FAILED tts/augment for {intent_name}: {info}")
                    continue
                duration_s = info
                record = {
                    "oid": f"synthetic_{uuid.uuid4().hex}",
                    "conversation_id": f"synthetic_{out_path.stem}",
                    "recording_url": "",
                    "chunk_path": str(out_path),
                    "chunk_index": 0,
                    "start_ms": 0.0,
                    "end_ms": round(duration_s * 1000, 1),
                    "duration_s": duration_s,
                    "transcript": text,
                    "intent": intent_name,
                }
                data_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                data_f.flush()
                synthesized_this_intent += 1

            results[intent_name] = {
                "generated": len(sentences),
                "synthesized": synthesized_this_intent,
                "already_done": len(done_texts),
            }
            print(f"[augment] {intent_name}: synthesized {synthesized_this_intent} new rows")

    print(f"[augment] done -> {results}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Generate + synthesize + augment rows for underrepresented intents")
    run_p.add_argument("--output-dir", default=config.OUTPUT_DIR_AUGMENTED)
    run_p.add_argument("--source-dir", default=config.OUTPUT_DIR_V3, help="dataset to check counts/donors against")
    run_p.add_argument("--min-new-per-intent", type=int, default=config.AUGMENT_MIN_NEW_PER_INTENT)
    run_p.add_argument("--target-total", type=int, default=config.AUGMENT_TARGET_TOTAL)
    run_p.add_argument(
        "--intents", nargs="+", default=None, help="only these intent names, instead of all 20"
    )
    run_p.add_argument("--use-gemini", action="store_true", help="use Gemini instead of the OpenAI default")
    run_p.add_argument("--model", default=config.AUGMENT_MODEL)
    run_p.add_argument("--tts-concurrency", type=int, default=config.TTS_CONCURRENCY)
    run_p.add_argument(
        "--extra-for-done", type=int, default=0,
        help="one-off: generate this many more even for intents already at target (e.g. to backfill speed variety)",
    )
    run_p.add_argument("--speed-min", type=float, default=config.AUGMENT_SPEED_MIN)
    run_p.add_argument("--speed-max", type=float, default=config.AUGMENT_SPEED_MAX)
    args = parser.parse_args()

    asyncio.run(
        run(
            output_dir=args.output_dir,
            source_dir=args.source_dir,
            min_new_per_intent=args.min_new_per_intent,
            target_total=args.target_total,
            intents_filter=args.intents,
            use_openai=not args.use_gemini,
            model=args.model,
            tts_concurrency=args.tts_concurrency,
            extra_for_done=args.extra_for_done,
            speed_range=(args.speed_min, args.speed_max),
        )
    )


if __name__ == "__main__":
    main()
