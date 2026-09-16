"""Gemini classifier for the taxonomy at config.INTENTS10_TAXONOMY_PATH
(currently intents15.json - evolved from the original 12-intent
intents10.json, see that config var's comment for history).

Self-contained (mirrors pipeline/intents.py's structure) rather than sharing
code with the 17-intent classifier there, since this is a different,
general-purpose conversational taxonomy (not specific to loan-collection
calls) used for a separate dataset (see pipeline/build_intents10.py).
"""

import json
import unicodedata
from enum import Enum

from google import genai
from google.genai import types
from google.oauth2 import service_account
from pydantic import BaseModel

from . import config

with open(config.INTENTS10_TAXONOMY_PATH, encoding="utf-8") as _f:
    TAXONOMY = json.load(_f)["intents"]

INTENT_NAMES = [item["name"] for item in TAXONOMY]
Intent = Enum("Intent10", {name: name for name in INTENT_NAMES})
FALLBACK_INTENT = "UNCLEAR_INPUT"


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
        "Each transcript you are given is ONE UTTERANCE of speech from one turn "
        "of a recorded phone conversation between a human caller and an "
        "automated voice bot/agent in India. Calls span many business domains "
        "(collections, support, sales, surveys, etc.) in Hindi/Hinglish/English, "
        "transcribed by an ASR system that may contain errors or transliteration "
        "noise.",
        "Classify the transcript into EXACTLY ONE of the following intents, "
        "judging only by the transcript's own wording - do not assume "
        f"conversational context you were not given. The taxonomy is closed: "
        f"never invent a new intent. If nothing else clearly applies, or the "
        f"transcript is ambiguous/incomplete, use {FALLBACK_INTENT}.",
        "",
    ]
    for item in TAXONOMY:
        lines.append(f"### {item['name']}")
        lines.append(item["description"])
        for cond in item.get("conditions", []):
            lines.append(f"- {cond}")
        if item.get("examples"):
            lines.append("Examples: " + "; ".join(item["examples"]))
        lines.append("")
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


def _reconcile_batch(parsed, n):
    if parsed is None:
        raise ClassificationError("gemini batch response could not be parsed")
    by_index = {item.index: item.intent.value for item in parsed.items}
    missing = [i for i in range(n) if i not in by_index]
    if missing:
        raise ClassificationError(f"gemini batch response missing indices: {missing}")
    return [by_index[i] for i in range(n)]


class GeminiIntentClassifier10:
    """Gemini on Vertex AI - the only backend for this taxonomy."""

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
            # Without this, a stalled connection hangs the call indefinitely -
            # seen in production: a relabel run sat with zero progress for
            # over an hour on one stuck request. A bounded timeout turns that
            # into a normal failure the existing retry loop already handles.
            http_options=types.HttpOptions(timeout=90_000),
        )
        self.model = model
        self.system_prompt = build_system_prompt()

    def _generate_structured(self, user_prompt, response_schema, temperature=0.7):
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=temperature,
                    # 8192 was tight enough for a 50-item batch to sometimes
                    # get cut off mid-JSON, producing unparseable/incomplete
                    # output (surfaced as "missing indices" errors) rather
                    # than a clean failure. More headroom, not a bigger batch.
                    max_output_tokens=32_768,
                ),
            )
        except Exception as e:
            raise ClassificationError(f"gemini call failed: {e}") from e

        if resp.parsed is not None:
            return resp.parsed

        if resp.text:
            try:
                cleaned_text = resp.text.replace(" ", " ").strip()
                cleaned_text = unicodedata.normalize("NFC", cleaned_text)
                return response_schema.model_validate_json(cleaned_text)
            except Exception as e:
                raise ClassificationError(
                    f"gemini returned unparseable response: {resp.text!r}"
                ) from e

        raise ClassificationError("gemini returned empty response")

    def classify(self, transcript):
        text = (transcript or "").strip()
        if not text:
            return FALLBACK_INTENT
        parsed = self._generate_structured(_single_prompt(text), IntentResult)
        return parsed.intent.value

    def classify_batch(self, transcripts):
        if not transcripts:
            return []
        parsed = self._generate_structured(_batch_prompt(transcripts), BatchIntentResult)
        return _reconcile_batch(parsed, len(transcripts))


def build_classifier():
    return GeminiIntentClassifier10()
