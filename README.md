# Call Transcript Intent Dataset

Multimodal Hindi/Hinglish customer utterance dataset for loan/EMI/payment call intent classification.

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

