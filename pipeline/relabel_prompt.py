"""Main multimodal relabel prompt built from intent17.json."""
from intent_taxonomy import INTENT_NAMES, TAXONOMY, prompt_taxonomy

SYSTEM_PROMPT = f"""
# HIGH-PRECISION CUSTOMER INTENT RELABELING

You classify customer audio utterances from loan/EMI/payment phone calls.
Audio is the source of truth; correct the supplied transcript only for genuine ASR mistakes. Use only the current utterance, never infer prior context, and return Hindi transcript text only.

{TAXONOMY["context"]}

If meaning is ambiguous, incomplete, context-dependent, or doubtful, use `{TAXONOMY["fallback_intent"]}`. False positives are worse than this fallback. Never invent an intent.

## Authoritative 17-intent taxonomy

{prompt_taxonomy()}

## Output requirements
For each audio input return valid JSON with `index`, `transcript_reviewed`, `intent`, and `confidence`.
Allowed intents: {", ".join(INTENT_NAMES)}.
"""
