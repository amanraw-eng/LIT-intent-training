import json
import os
import re
from enum import Enum

from google import genai
from google.genai import types
from google.oauth2 import service_account
from openai import OpenAI
from pydantic import BaseModel

from . import config


def parse_intents(md_path):
    """Parse the closed-set intent taxonomy table out of intents.md.

    Returns a list of {"name", "condition", "examples"} dicts, in document order.
    """
    rows = []
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 3:
                continue
            m = re.match(r"`([A-Z_]+)`", cells[0])
            if not m:
                continue
            rows.append({"name": m.group(1), "condition": cells[1], "examples": cells[2]})
    if not rows:
        raise RuntimeError(f"no intents parsed from {md_path}")
    return rows


def _load_intent_rows(md_path, cache_path):
    """intents.md is only ever parsed once - after that this reads the cached
    JSON. Delete cache_path to force a re-parse (e.g. after editing intents.md)."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    rows = parse_intents(md_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


INTENT_ROWS = _load_intent_rows(config.INTENTS_MD_PATH, config.INTENTS_JSON_PATH)
INTENT_NAMES = [r["name"] for r in INTENT_ROWS]
Intent = Enum("Intent", {name: name for name in INTENT_NAMES})


class IntentResult(BaseModel):
    intent: Intent


class BatchIntentItem(BaseModel):
    index: int
    intent: Intent


class BatchIntentResult(BaseModel):
    items: list[BatchIntentItem]


class ClassificationError(Exception):
    pass


def build_system_prompt():
    lines = [
        "Call context: this is a recorded outbound phone call from a bank/NBFC's "
        "loan collections customer care team to one of their borrowers in India. "
        "The agent is calling to inform the customer about a problem with their "
        "loan - most commonly that their EMI (loan installment) bounced or failed "
        "to auto-debit, but sometimes other dues, late/bounce charges, or overdue "
        "amounts - and is trying to get the customer to acknowledge it and pay, "
        "understand their reason if they can't/won't pay, or handle a related "
        "request (identity check, callback, escalation, etc.).",
        "Each transcript you are given is ONE TURN of speech from that call - "
        "almost always the customer/borrower's side, occasionally a third party "
        "who picked up their phone - in Hindi/Hinglish, possibly with ASR errors "
        "or transliteration noise. Interpret short or ambiguous replies (e.g. "
        '"haan", "theek hai", "nahi abhi nahi", "kar dunga") as a response about '
        "this EMI-bounce / loan-payment call, not as generic small talk.",
        "Classify each transcript into EXACTLY ONE of the following intents. "
        "Judge by customer meaning in this call context, not surface keywords. "
        "The taxonomy is closed: never invent a new intent. If nothing else fits, "
        "use UNCLEAR_INPUT; if the transcript is empty or pure noise, use "
        "SILENCE_NOISE.",
        "",
        "| Intent | Fires when | Example utterances |",
        "|---|---|---|",
    ]
    for row in INTENT_ROWS:
        lines.append(f"| {row['name']} | {row['condition']} | {row['examples']} |")
    return "\n".join(lines)


def _single_prompt(transcript):
    return f'Transcript: """{transcript}"""\n\nReturn the single best-matching intent.'


def _batch_prompt(transcripts):
    numbered = "\n".join(f'{i}: """{t}"""' for i, t in enumerate(transcripts))
    return (
        f"Classify EACH of the following {len(transcripts)} transcripts "
        "independently - they are unrelated turns from different calls. Return "
        "exactly one entry per transcript in `items`, each tagged with the same "
        "`index` shown below.\n\n"
        f"{numbered}"
    )


def _reconcile_batch(parsed, n, backend):
    if parsed is None:
        raise ClassificationError(f"{backend} batch response could not be parsed")
    by_index = {item.index: item.intent.value for item in parsed.items}
    missing = [i for i in range(n) if i not in by_index]
    if missing:
        raise ClassificationError(f"{backend} batch response missing indices: {missing}")
    return [by_index[i] for i in range(n)]


class _BaseClassifier:
    """Shared logic for both backends - only generate_structured() differs."""

    def classify(self, transcript):
        text = (transcript or "").strip()
        if not text:
            return "SILENCE_NOISE"
        parsed = self.generate_structured(self.system_prompt, _single_prompt(text), IntentResult)
        return parsed.intent.value

    def classify_batch(self, transcripts):
        if not transcripts:
            return []
        parsed = self.generate_structured(self.system_prompt, _batch_prompt(transcripts), BatchIntentResult)
        return _reconcile_batch(parsed, len(transcripts), self.name)


class IntentClassifier(_BaseClassifier):
    """Gemini on Vertex AI. Default backend."""

    name = "gemini"

    def __init__(
        self,
        key_path=config.GEMINI_KEY_PATH,
        location=config.GEMINI_LOCATION,
        model=config.GEMINI_MODEL,
    ):
        with open(key_path) as f:
            info = json.load(f)
        creds = service_account.Credentials.from_service_account_file(
            key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.client = genai.Client(
            vertexai=True,
            project=info["project_id"],
            location=location,
            credentials=creds,
        )
        self.model = model
        self.system_prompt = build_system_prompt()

    def generate_structured(self, system_prompt, user_prompt, response_schema, temperature=0.0):
        """Run one structured-output call. Returns the parsed pydantic object."""
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=temperature,
                ),
            )
        except Exception as e:
            raise ClassificationError(f"gemini call failed: {e}") from e
        if resp.parsed is None:
            raise ClassificationError(f"gemini returned unparseable response: {resp.text!r}")
        return resp.parsed


class OpenAIIntentClassifier(_BaseClassifier):
    """Alternative backend, selected with --use-openai. Reads OPENAI_API_KEY
    from the environment (populated from .env by config.py's load_dotenv)."""

    name = "openai"

    # Reasoning-family models only support the default temperature (1) and
    # reject an explicit value (incl. 0) with a 400 error - omit it for these.
    _NO_CUSTOM_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def __init__(self, model=None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ClassificationError(
                "OPENAI_API_KEY not set - add it to .env in the project root "
                f"({config.BASE_DIR}/.env)"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model or config.OPENAI_MODEL
        self.system_prompt = build_system_prompt()

    def generate_structured(self, system_prompt, user_prompt, response_schema, temperature=0.0):
        """Run one structured-output call. Returns the parsed pydantic object."""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_schema,
        )
        if not self.model.startswith(self._NO_CUSTOM_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = temperature
        try:
            resp = self.client.chat.completions.parse(**kwargs)
        except Exception as e:
            raise ClassificationError(f"openai call failed: {e}") from e
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise ClassificationError(
                f"openai returned unparseable response: {resp.choices[0].message.content!r}"
            )
        return parsed


def build_classifier(use_openai=False):
    return OpenAIIntentClassifier() if use_openai else IntentClassifier()
