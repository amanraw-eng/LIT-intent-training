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
from .intents import IntentClassifier, OpenAIIntentClassifier
from .noise_bank import augment as noise_augment
from .noise_bank import extract_noise_donors
from .tts_client import tts_with_retry

# Explicit paths requested
DEFAULT_OUTPUT_AUDIO_DIR = "/home/jovyan/aman_ws/stt/LIT-intent-training/data/augmented_data/audio"
DEFAULT_JSONL_PATH = "/home/jovyan/aman_ws/stt/LIT-intent-training/data/augmented_data/updated_augmented_data17.jsonl"
DEFAULT_VC_SAMPLES_DIR = "/home/jovyan/aman_ws/stt/LIT-intent-training/data/VC_samples"


class GeneratedSentences(BaseModel):
    sentences: list[str]


def _now():
    return datetime.now(timezone.utc).isoformat()


INTENT_ROWS = [
    {
        "name": "AFFIRMATIVE_ACKNOWLEDGEMENT",
        "condition": "Generic positive acknowledgement or agreement. A clearly affirmative 'हम्म' counts. Do NOT infer context.",
        "examples": "हाँ, जी, जी हाँ, हाँ जी, हाँ हाँ, जी जी, ठीक है, बिल्कुल, हम्म",
    },
    {
        "name": "NEGATIVE_ACKNOWLEDGEMENT",
        "condition": "Generic negative acknowledgement or denial. Specific meaning overrides this class.",
        "examples": "नहीं, ना, जी नहीं, नहीं जी, बिल्कुल नहीं",
    },
    {
        "name": "IDENTITY_CONFIRMED",
        "condition": "Customer explicitly identifies themselves as the requested person.",
        "examples": "हाँ मैं ही हूँ, जी हाँ मैं ही हूँ, मैं ही बोल रहा हूँ, हाँ यही हूँ, जी मैं ही हूँ, Yes this is me, Yes speaking",
    },
    {
        "name": "THIRD_PARTY_AVAILABLE",
        "condition": "Customer explicitly indicates that requested person is available, coming to phone, or handed phone.",
        "examples": "फोन दे रहा हूँ, फोन दे रही हूँ, मैं उनको बुलाता हूँ, एक मिनट उनको बुलाता हूँ, वो आ गए हैं, वो आ गए बात कर लीजिए, वो आ रहे हैं, उनसे बात कर सकते हैं",
    },
    {
        "name": "THIRD_PARTY_UNAVAILABLE",
        "condition": "Customer explicitly says requested person is unavailable, absent, or cannot currently talk without requesting a explicit callback.",
        "examples": "वो अभी नहीं हैं, वो बाहर हैं, वो घर पर नहीं हैं, वो available नहीं हैं, वो busy हैं, अभी उनसे बात नहीं हो सकती, वो फोन पर नहीं आ सकते",
    },
    {
        "name": "PAY_NOW_AGREE",
        "condition": "Explicit commitment to make payment immediately/currently. Must contain payment action + immediate timing.",
        "examples": "अभी payment कर देता हूँ, अभी कर दूंगा payment, अभी पैसे जमा कर देता हूँ, अभी pay कर देता हूँ, तुरंत payment कर देता हूँ",
    },
    {
        "name": "PAY_LATER_AGREE",
        "condition": "Explicit commitment to make payment later. Must contain payment action + future timing.",
        "examples": "बाद में payment कर दूंगा, कल payment कर दूंगा, शाम तक कर दूंगा, थोड़ी देर में कर दूंगा, अगले हफ्ते payment कर दूंगा, सोमवार को कर दूंगा",
    },
    {
        "name": "PAID_ALREADY",
        "condition": "Explicitly says payment has already been completed.",
        "examples": "मैंने payment कर दी है, payment हो गई है, मैं pay कर चुका हूँ, पैसे जमा कर दिए हैं, पहले ही payment कर दी, payment successful हो गई, transaction complete हो गया, पैसे भेज दिए हैं",
    },
    {
        "name": "REFUSE_TO_PAY",
        "condition": "Explicit refusal/unwillingness to pay without presenting specific non-payment reason.",
        "examples": "नहीं दूंगा, नहीं दूंगा पैसे, मैं payment नहीं करूंगा, मैं पैसे नहीं दूंगा, payment नहीं करूंगा, मैं भुगतान करने से मना कर रहा हूँ",
    },
    {
        "name": "NON_PAYMENT_REASON",
        "condition": "Reason provided for payment delay/inability or inability to pay currently.",
        "examples": "पैसे नहीं हैं, salary नहीं आई, अभी payment नहीं कर सकता, funds नहीं हैं to dekhuga",
    },
    {
        "name": "END_CALL",
        "condition": "Explicitly ending the current call.",
        "examples": "बाय, ठीक है बाय, ओके बाय, अच्छा बाय, बाय बाय, नमस्ते, अलविदा, चलता हूँ, फिर बात करेंगे",
    },
    {
        "name": "DO_NOT_CALL",
        "condition": "Explicit request not to call or contact again.",
        "examples": "अब कॉल मत करना, दोबारा फोन मत करना, आगे से कॉल मत करना, फिर कभी कॉल मत करना, मुझे फिर कॉल मत करना, इस नंबर पर फोन मत करना",
    },
    {
        "name": "CALL_DEFER",
        "condition": "Explicit request for a later callback or deferred conversation.",
        "examples": "बाद में कॉल करना, बाद में फोन करना, थोड़ी देर बाद कॉल करना, कल कॉल करना, बाद में बात करना, अभी busy हूँ बाद में कॉल करना, अभी समय नहीं है बाद में कॉल करिए, शाम को कॉल करना",
    },
    {
        "name": "BACKCHANNEL_OR_NOISE",
        "condition": "Clearly non-semantic background audio, non-linguistic sound, coughing, breathing, noise.",
        "examples": "[throat-clearing], [cough], [background noise], [breathing], [mic rustle]",
    },
    {
        "name": "UNCLEAR_INPUT",
        "condition": "Ambiguous, dependent on missing context, incomplete, complex, or doubtful inputs.",
        "examples": "हाँ बोलिए, जी बताइए, बोलिए, मैं busy हूँ, मेरा loan नहीं है, ये amount गलत है, penalty नहीं दूंगा, मुझे help चाहिए, steps बताइए, गलत नंबर है, मैं वो नहीं हूँ, शायद लिया था, देखता हूँ, कोशिश करूंगा, कैसे करूँ, क्या करूँ",
    },
]


def count_intents(data_path):
    counts = Counter()
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[json.loads(line).get("intent", "UNCLEAR_INPUT")] += 1
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
        f"Reference examples (for MEANING only - do not copy register): {intent_row['examples']}\n\n"
        "These must sound like an actual rushed/informal phone call, NOT written or literary Hindi:\n"
        "- Code-mix in common English words in LATIN SCRIPT embedded directly inside Devanagari "
        "(e.g., 'हाँ, EMI तो pay कर दी थी', 'मुझे थोड़ा time चाहिए'). Avoid over-transliterating into Devanagari script.\n"
        "- Do NOT overuse address terms like 'सर'/'मैडम'.\n"
        f"- Use genuine {tone} vocabulary, particles, and verb endings where applicable.\n"
        "- Vary sentence lengths significantly (1-4 words short clipped reactions, medium, long natural sentences).\n"
        "- Include grammatically incomplete or mid-thought utterances typical of real human speech.\n"
        f"{avoid_block}\n\n"
        f"Return exactly {n} sentences."
    )


def generate_sentences_for_intent(classifier, intent_row, total_n, batch_size=None, tones=None):
    batch_size = batch_size or config.AUGMENT_SENTENCE_BATCH_SIZE
    tones = tones or TONES
    system_prompt = (
        "You generate realistic, highly specific borrower voice lines for high-precision intent classification datasets."
    )
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
            tone_collected += this_n
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


def load_voice_clone_paths(vc_dir=DEFAULT_VC_SAMPLES_DIR):
    """Scans the designated directory for .wav or .mp3 files to use for voice cloning."""
    if not os.path.exists(vc_dir):
        return []
    valid_extensions = {".wav", ".mp3", ".flac"}
    voice_paths = [
        str(p) for p in Path(vc_dir).rglob("*") if p.suffix.lower() in valid_extensions and p.is_file()
    ]
    return voice_paths


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
    data_path=DEFAULT_JSONL_PATH,
    audio_dir=DEFAULT_OUTPUT_AUDIO_DIR,
    vc_dir=DEFAULT_VC_SAMPLES_DIR,
    source_dir=config.OUTPUT_DIR_V3,
    min_new_per_intent=1000,
    target_total=config.AUGMENT_TARGET_TOTAL,
    intents_filter=None,
    use_openai=True,
    model=config.AUGMENT_MODEL,
    tts_concurrency=config.TTS_CONCURRENCY,
    noise_donor_pool_size=300,
    extra_for_done=0,
    speed_range=(config.AUGMENT_SPEED_MIN, config.AUGMENT_SPEED_MAX),
):
    if not config.TTS_WS_ENDPOINT:
        raise RuntimeError("TTS_WS_ENDPOINT not set - add it to .env")

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    errors_path = os.path.join(os.path.dirname(data_path), config.ERRORS_LOG_FILENAME)

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

        needed_for_min = max(min_new_per_intent - len(done_texts), 0)
        needed_for_target = max(target_total - current_total, 0)
        needed = max(needed_for_min, needed_for_target)
        if needed <= 0 and extra_for_done > 0:
            needed = extra_for_done
        plan.append((row, done_texts, needed))
        print(f"  {intent_name}: current={current_total} (real={real_count}, synth={len(done_texts)}) -> +{needed} new")

    print("[augment] building noise donor pool from real call audio...")
    donor_paths = []
    if os.path.exists(source_data_path):
        with open(source_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= noise_donor_pool_size * 5:
                    break
                donor_paths.append(json.loads(line)["chunk_path"])
    donors = extract_noise_donors(donor_paths, max_donors=noise_donor_pool_size) if donor_paths else []
    print(f"[augment] {len(donors)} noise donors extracted")

    voice_paths = load_voice_clone_paths(vc_dir)
    if voice_paths:
        print(f"[augment] {len(voice_paths)} voice clone samples found in {vc_dir} - picking one at random per sentence")
    else:
        print(f"[augment] No audio samples found in {vc_dir} - fallback to default TTS_VOICE_ID")

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
            print(f"[augment] {intent_name}: synthesized {synthesized_this_intent} new rows appended to {data_path}")

    print(f"[augment] done -> {results}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Generate + synthesize + augment rows for underrepresented intents")
    run_p.add_argument("--data-path", default=DEFAULT_JSONL_PATH, help="Target jsonl path to append data")
    run_p.add_argument("--audio-dir", default=DEFAULT_OUTPUT_AUDIO_DIR, help="Target audio directory")
    run_p.add_argument("--vc-dir", default=DEFAULT_VC_SAMPLES_DIR, help="Voice clone directory containing sample audios")
    run_p.add_argument("--source-dir", default=config.OUTPUT_DIR_V3, help="dataset to check counts/donors against")
    run_p.add_argument("--min-new-per-intent", type=int, default=1000)
    run_p.add_argument("--target-total", type=int, default=config.AUGMENT_TARGET_TOTAL)
    run_p.add_argument("--intents", nargs="+", default=None, help="only these intent names")
    run_p.add_argument("--use-gemini", action="store_true", help="use Gemini instead of the OpenAI default")
    run_p.add_argument("--model", default=config.AUGMENT_MODEL)
    run_p.add_argument("--tts-concurrency", type=int, default=config.TTS_CONCURRENCY)
    run_p.add_argument("--extra-for-done", type=int, default=0)
    run_p.add_argument("--speed-min", type=float, default=config.AUGMENT_SPEED_MIN)
    run_p.add_argument("--speed-max", type=float, default=config.AUGMENT_SPEED_MAX)
    args = parser.parse_args()

    asyncio.run(
        run(
            data_path=args.data_path,
            audio_dir=args.audio_dir,
            vc_dir=args.vc_dir,
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