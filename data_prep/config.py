"""Configuration for the data_prep package.

Secrets and non-secret settings both come from the repo-wide shared config
(shared_config.py / shared_settings.py) rather than a separate data_prep
.env or settings file - one place to look, consistent with every other
script in the repo (see pipeline/config.py). Only the env-var *names* and
task-specific batch/concurrency defaults live here, read from
shared_settings.py's "data_prep" section.
"""

from __future__ import annotations

import os

from shared_config import load_environment, project_path, section

load_environment()

_settings = section("data_prep")
_pipeline_settings = section("pipeline")  # reuse the existing Gemini/OpenAI model defaults


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


# ============================================================
# Secrets / per-machine paths - env only, never hardcode these.
# ============================================================

HF_TOKEN = _env("HF_TOKEN")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
GEMINI_KEY_PATH = project_path(_env("GEMINI_KEY_PATH", ".secrets/kapture-gemini-9d92c74fb501.json"))

# ============================================================
# LLM provider selection
# ============================================================

# "gemini" or "openai" - which build_llm_client() returns when no provider
# is passed explicitly.
DEFAULT_LLM_PROVIDER = _env("DATA_PREP_LLM_PROVIDER", _settings["llm_provider"])

GEMINI_PROJECT = _pipeline_settings["gemini_project"]
GEMINI_LOCATION = _pipeline_settings["gemini_location"]
GEMINI_MODEL = _pipeline_settings["gemini_model"]

OPENAI_MODEL = _pipeline_settings["openai_model"]

# ============================================================
# Hugging Face
# ============================================================

HF_DEFAULT_PRIVATE = False

# ============================================================
# Generation task - classifying fresh transcripts (text-only).
# ============================================================

_gen = _settings["generation"]
GENERATION_BATCH_SIZE = int(_env("GENERATION_BATCH_SIZE", _gen["batch_size"]))
GENERATION_MAX_CONCURRENCY = int(_env("GENERATION_MAX_CONCURRENCY", _gen["max_concurrency"]))
GENERATION_MAX_RETRIES = int(_env("GENERATION_MAX_RETRIES", _gen["max_retries"]))
GENERATION_RETRY_DELAY_S = float(_env("GENERATION_RETRY_DELAY_S", _gen["retry_delay_s"]))

# ============================================================
# Relabeling task - text-only re-examination of existing rows
# (e.g. re-checking every unique transcript currently UNCLEAR_INPUT).
# ============================================================

_rt = _settings["relabel_text"]
RELABEL_TEXT_BATCH_SIZE = int(_env("RELABEL_TEXT_BATCH_SIZE", _rt["batch_size"]))
RELABEL_TEXT_MAX_CONCURRENCY = int(_env("RELABEL_TEXT_MAX_CONCURRENCY", _rt["max_concurrency"]))
RELABEL_TEXT_MAX_RETRIES = int(_env("RELABEL_TEXT_MAX_RETRIES", _rt["max_retries"]))
RELABEL_TEXT_RETRY_DELAY_S = float(_env("RELABEL_TEXT_RETRY_DELAY_S", _rt["retry_delay_s"]))

# ============================================================
# Relabeling task - multimodal (audio + transcript) review.
# ============================================================

_rm = _settings["relabel_multimodal"]
RELABEL_MULTIMODAL_BATCH_SIZE = int(_env("RELABEL_MULTIMODAL_BATCH_SIZE", _rm["batch_size"]))
RELABEL_MULTIMODAL_MAX_CONCURRENCY = int(_env("RELABEL_MULTIMODAL_MAX_CONCURRENCY", _rm["max_concurrency"]))
RELABEL_MULTIMODAL_MAX_RETRIES = int(_env("RELABEL_MULTIMODAL_MAX_RETRIES", _rm["max_retries"]))
RELABEL_MULTIMODAL_RETRY_DELAY_S = float(_env("RELABEL_MULTIMODAL_RETRY_DELAY_S", _rm["retry_delay_s"]))
