"""Base config for one intent taxonomy + every LLM prompt that operates on it.

Each concrete intent set (see loan_collection.py) is a single file holding:
  - the INTENTS map (name, condition, examples) used for generation and for
    building the closed-taxonomy enum,
  - the data-generation prompt (classifying a fresh, unlabeled transcript),
  - the (multimodal) relabeling prompt (re-reviewing audio + transcript for
    an existing row),
  - any number of named, targeted relabel prompts for re-examining a
    particular already-assigned class (e.g. promoting rows stuck at
    UNCLEAR_INPUT to a more specific intent).

This is deliberately one object per taxonomy version rather than one shared
global taxonomy: different prompts in the same project can legitimately
consider different candidate intent sets (e.g. a second-pass relabel prompt
that also considers a couple of intents the primary generation prompt
doesn't), and forcing them into one map would silently change behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class IntentDefinition:
    name: str
    condition: str
    examples: str = ""


@dataclass
class IntentSetConfig:
    name: str
    intents: list[IntentDefinition]

    # Either a fixed prompt string, or a function that builds one from
    # `intents` (so the table stays in sync with the map above).
    generation_prompt: str | Callable[[list[IntentDefinition]], str]

    # Full system prompt for multimodal (audio + transcript) review/relabel.
    relabeling_prompt: str

    # Named second-pass relabel prompts, e.g. {"unclear_input": "..."} for
    # re-examining rows currently labeled UNCLEAR_INPUT.
    relabel_prompts: dict[str, str] = field(default_factory=dict)

    # Per-taxonomy overrides of the task defaults in data_prep/config.py.
    # None means "use the config.py default".
    generation_batch_size: int | None = None
    relabel_batch_size: int | None = None
    relabel_max_concurrency: int | None = None

    @property
    def intent_names(self) -> list[str]:
        return [i.name for i in self.intents]

    def build_generation_prompt(self) -> str:
        if callable(self.generation_prompt):
            return self.generation_prompt(self.intents)
        return self.generation_prompt

    def relabel_prompt(self, key: str) -> str:
        try:
            return self.relabel_prompts[key]
        except KeyError:
            raise KeyError(
                f"No relabel prompt registered under {key!r} for intent set "
                f"{self.name!r}. Available: {list(self.relabel_prompts)}"
            ) from None
