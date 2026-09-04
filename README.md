# Call Transcript Intent Dataset

Multimodal Hindi/Hinglish customer utterance dataset for loan/EMI/payment call intent classification.

## Configuration

- Edit [`shared_settings.py`](shared_settings.py) for non-secret, versioned settings such as paths, models, GPU choice, and training parameters.
- Copy [`.env.example`](.env.example) to a repository-root `.env` for credentials and machine-specific values. `.env` is ignored by Git.
- Store the Gemini Vertex service-account JSON at `.secrets/gemini-service-account.json`, then keep `GEMINI_KEY_PATH=.secrets/gemini-service-account.json` in `.env`.
- Existing `pipeline/.env` and `intent-training/.env` files remain supported as a migration fallback; the root `.env` takes priority.
- The main pipeline, 17-intent migration/relabel/augmentation scripts, and training entry points read common settings from these shared files. Uncommon knobs are grouped in a `SCRIPT-SPECIFIC RUN OVERRIDES` block at the top of each script.
## Portable JSONL audio paths

Store `chunk_path` relative to the audio folder, for example `PAY_NOW_AGREE_001.wav`, never as an absolute machine path. Convert an existing manifest without overwriting it:

```bash
python -m pipeline.jsonl_tools make-portable --input data.jsonl --output data.portable.jsonl --audio-dir /path/to/audio
```

Use the manifest elsewhere by providing the local audio folder: `python -m pipeline.push_to_hub --output-dir /path/to/dataset --audio-dir /new/path/audio` or `python intent-training/train_augmented.py --audio-dir /new/path/audio`.

Analyze rows, intent counts, total hours, and a `split` field when present:

```bash
python -m pipeline.jsonl_tools analyze --input data.portable.jsonl --audio-dir /new/path/audio
```


## Dataset Summary

| Metric | Value |
|---|---:|
| Total examples | 139,348 |
| Total audio duration | 51.04 h |
| Number of intents | 17 |

## Split Statistics

| Split | Examples | Duration | Hours |
|---|---:|---:|---:|
| train | 126,848 | 2755.14 min | 45.92 h |
| validation | 10,000 | 219.03 min | 3.65 h |
| eval | 2,500 | 88.37 min | 1.47 h |

## Class Distribution

| Intent | Total | Train | Validation | Eval |
|---|---:|---:|---:|---:|
| `UNCLEAR_INPUT` | 75,051 | 68,669 | 5,524 | 858 |
| `AFFIRMATIVE_ACKNOWLEDGEMENT` | 37,719 | 34,652 | 2,710 | 357 |
| `CONTINUE_CONVERSATION` | 6,366 | 5,768 | 477 | 121 |
| `PAY_LATER_AGREE` | 3,971 | 3,621 | 220 | 130 |
| `NEGATIVE_ACKNOWLEDGEMENT` | 3,641 | 3,260 | 254 | 127 |
| `NO_PAYMENT_REASON` | 2,963 | 2,680 | 210 | 73 |
| `BACKCHANNEL_OR_NOISE` | 2,096 | 1,907 | 167 | 22 |
| `PAID_ALREADY` | 1,836 | 1,605 | 114 | 117 |
| `THIRD_PARTY_UNAVAILABLE` | 1,035 | 878 | 63 | 94 |
| `IDENTITY_CONFIRMED` | 1,027 | 846 | 75 | 106 |
| `PAY_NOW_AGREE` | 819 | 663 | 48 | 108 |
| `END_CALL` | 744 | 606 | 33 | 105 |
| `CALL_DEFER` | 622 | 579 | 31 | 12 |
| `REFUSE_TO_PAY` | 462 | 324 | 29 | 109 |
| `THIRD_PARTY_AVAILABLE` | 367 | 322 | 29 | 16 |
| `WRONG_NUMBER` | 343 | 286 | 15 | 42 |
| `DO_NOT_CALL` | 286 | 182 | 1 | 103 |

## Columns

- `audio`
- `id`
- `oid`
- `conversation_id`
- `recording_url`
- `chunk_index`
- `start_ms`
- `end_ms`
- `duration_s`
- `transcript`
- `transcript_reviewed`
- `intent`

