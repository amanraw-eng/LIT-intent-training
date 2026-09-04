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
