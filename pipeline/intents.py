import json
import os
from enum import Enum

from google import genai
from google.genai import types
from google.oauth2 import service_account
from openai import OpenAI
from pydantic import BaseModel

from . import config

# Clean Python-native intent definitions directly matching your custom prompt
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
        "examples": "बाय, ठीक है बाय, ओके बाय, अच्छा बाय, बाय बाय, नमस्ते, अलविदा, चलता हूँ, फिर बात करेंगे, रखो फोन, कॉल काट दो, अब फोन रख दो।",
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
        "use UNCLEAR_INPUT; if the transcript is empty or pure noise, use  very sure noise means needs very high precision for other classes too except UNCLEAR_INPUT if any ambiguity make UNCLEAR_INPUT so give just noise for " 
        "BACKCHANNEL_OR_NOISE.",
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
            return "BACKCHANNEL_OR_NOISE"
        parsed = self.generate_structured(self.system_prompt, _single_prompt(text), IntentResult)
        return parsed.intent.value

    def classify_batch(self, transcripts):
        if not transcripts:
            return []
        parsed = self.generate_structured(self.system_prompt, _batch_prompt(transcripts), BatchIntentResult)
        return _reconcile_batch(parsed, len(transcripts), self.name)


class GeminiIntentClassifier(_BaseClassifier):
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
                    max_output_tokens=8192,
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
    return OpenAIIntentClassifier() if use_openai else GeminiIntentClassifier()