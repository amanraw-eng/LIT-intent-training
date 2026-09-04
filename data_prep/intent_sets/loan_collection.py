"""Loan-collection call intent taxonomy.

Thin adapter over the repo's already-canonical taxonomy and prompts rather
than a second copy of them:
  - intent17.json / intent_taxonomy.py - the single authoritative intent map.
  - pipeline/relabel_prompt.py - the multimodal (audio + transcript) review
    prompt, built from that taxonomy.
  - pipeline/relabel_unclear_prompt.py - the UNCLEAR_INPUT second-pass
    promotion prompt, also built from that taxonomy.

This module only adds a data-generation prompt (classifying a fresh,
unlabeled transcript - not something pipeline/*.py has today) and wires
everything into one IntentSetConfig.
"""

from __future__ import annotations

from intent_taxonomy import TAXONOMY, prompt_taxonomy
from pipeline.relabel_prompt import SYSTEM_PROMPT as RELABELING_PROMPT
from pipeline.relabel_unclear_prompt import SYSTEM_PROMPT as UNCLEAR_INPUT_RELABEL_PROMPT

from .base import IntentDefinition, IntentSetConfig
from .. import config

INTENTS: list[IntentDefinition] = [
    IntentDefinition(
        name=item["name"],
        condition=item["definition"] + (f" {item['notes']}" if item.get("notes") else ""),
        examples="; ".join(item.get("examples", [])),
    )
    for item in TAXONOMY["intents"]
]


def _build_generation_prompt(_intents: list[IntentDefinition]) -> str:
    return "\n\n".join(
        [
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
            "who picked up their phone. " + TAXONOMY["context"],
            "Classify each transcript into EXACTLY ONE of the following intents. "
            "The taxonomy is closed: never invent a new intent. If nothing else "
            f"fits, use {TAXONOMY['fallback_intent']}.",
            "## Authoritative taxonomy",
            prompt_taxonomy(),
        ]
    )


LOAN_COLLECTION_INTENT_SET = IntentSetConfig(
    name="loan_collection_v1",
    intents=INTENTS,
    generation_prompt=_build_generation_prompt,
    relabeling_prompt=RELABELING_PROMPT,
    relabel_prompts={"unclear_input": UNCLEAR_INPUT_RELABEL_PROMPT},
    generation_batch_size=config.GENERATION_BATCH_SIZE,
    relabel_batch_size=config.RELABEL_MULTIMODAL_BATCH_SIZE,
    relabel_max_concurrency=config.RELABEL_MULTIMODAL_MAX_CONCURRENCY,
)
