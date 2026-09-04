"""Version-controlled, non-secret settings for every project script.

Edit values in this file before a run. Credentials remain in the root .env.
"""

SETTINGS = {
    "project": {
        "data_dir": "data",
    },
    "pipeline": {
        "source_dataset_dir": "",
        "lit_ws_base": "ws://localhost:4000/lit/ws",
        "language": "hi",
        "hf_dataset_repo": "kapturecx/call-transcript-intent-data",
        "gemini_project": "kapture-gemini",
        "gemini_location": "us-central1",
        "gemini_model": "gemini-2.5-flash",
        "openai_model": "gpt-4o-mini",
        "vllm_base_url": "http://localhost:5500",
        "vllm_model": "kapturecx/qwen-asr-hindi-3006-ft",
        "relabel": {
            "batch_size": 20,
            "max_concurrency": 10,
            "max_retries": 4,
            "retry_delay_s": 2.0,
        },
        "augmented17": {
            "data_dir": "data/augmented_data",
            "input_jsonl": "data.jsonl",
            "path_updated_jsonl": "path_updated_augmented_data17.jsonl",
            "final_jsonl": "updated_augmented_data17.jsonl",
            "checkpoint_jsonl": "updated_augmented_data17_checkpoint.jsonl",
            "error_jsonl": "updated_augmented_data17_errors.jsonl",
            "old_audio_prefix": "",
            "new_audio_prefix": "",
        },
        "intents10": {
            "source_repo": "kapturecx/bolAIndia",
            "source_config": "combined",
            "source_split": "train",
            # amn-raw (personal, free plan) hit its private-storage cap at
            # ~58k rows / ~5GB - kapturecx is a paid Team-plan org with much
            # more private storage room. amn-raw/S2I-10-v1 is left as-is
            # (untouched, not deleted) with its rows up to that point.
            "hf_dataset_repo": "kapturecx/S2I-10-v1",
            # Rows are streamed and processed in fetch-batches of this size
            # (save audio + queue transcript) before checkpointing stream
            # position; classification itself still happens in the smaller
            # INTENT_BATCH_SIZE sub-batches Gemini handles reliably.
            "fetch_batch_size": 200,
            # Push to the Hub after this many NEW finalized rows have piled up
            # since the last push. append_incremental() (pipeline/push_intents10.py)
            # uploads only the delta as a new shard - cost scales with this
            # number, not total dataset size, so it's safe to keep this fairly
            # low even once the dataset is large.
            "push_every_rows": 2000,
            # Number of Gemini classify_batch calls (each INTENT_BATCH_SIZE
            # transcripts) fired concurrently via a thread pool. 1 = sequential.
            "classify_max_concurrency": 15,
        },
    },
    "data_prep": {
        "llm_provider": "gemini",
        # Transcripts are classified in batches (one LLM call per batch).
        # Gemini's structured-output parsing gets unreliable above ~50
        # items per call for text-only prompts.
        "generation": {
            "batch_size": 50,
            "max_concurrency": 1,
            "max_retries": 2,
            "retry_delay_s": 3.0,
        },
        "relabel_text": {
            "batch_size": 50,
            "max_concurrency": 1,
            "max_retries": 4,
            "retry_delay_s": 2.0,
        },
        # Each item here sends its full audio clip to the model, so batches
        # must be much smaller than the text-only tasks above or structured
        # output truncates and fails to parse.
        "relabel_multimodal": {
            "batch_size": 10,
            "max_concurrency": 10,
            "max_retries": 4,
            "retry_delay_s": 2.0,
        },
    },
    "training": {
        "dataset_repo": "kapturecx/call-transcript-intent-data-v2",
        "experiment_version": "v6-eval2",
        "seed": 100,
        "gpu_device": "0",
        "whisper_size": "small",
        "learning_rate": 1e-5,
        "weight_decay": 1e-2,
        "augmented17": {
            "jsonl_data_path": "data/augmented_data/updated_augmented_data17.jsonl",
        },
    },
}
