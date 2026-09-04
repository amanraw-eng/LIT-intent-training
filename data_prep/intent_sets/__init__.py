from .base import IntentDefinition, IntentSetConfig
from .loan_collection import LOAN_COLLECTION_INTENT_SET

INTENT_SETS: dict[str, IntentSetConfig] = {
    LOAN_COLLECTION_INTENT_SET.name: LOAN_COLLECTION_INTENT_SET,
}


def get_intent_set(name: str) -> IntentSetConfig:
    try:
        return INTENT_SETS[name]
    except KeyError:
        raise KeyError(f"Unknown intent set {name!r}. Available: {list(INTENT_SETS)}") from None


__all__ = ["IntentDefinition", "IntentSetConfig", "INTENT_SETS", "get_intent_set", "LOAN_COLLECTION_INTENT_SET"]
