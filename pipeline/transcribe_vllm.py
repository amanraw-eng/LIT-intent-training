import os
import time
from collections import deque

import requests
import soundfile as sf

from . import config
from .transcribe import TranscriptionError, TranscriptionTimeout


class _RateLimiter:
    """Sliding-window limiter: blocks as needed to keep calls to at most
    max_calls within any rolling period_s window."""

    def __init__(self, max_calls, period_s):
        self.max_calls = max_calls
        self.period_s = period_s
        self._calls = deque()

    def wait(self):
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self.period_s:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            sleep_s = self.period_s - (now - self._calls[0])
            if sleep_s > 0:
                print(
                    f"[transcribe_vllm] rate limit ({self.max_calls}/{self.period_s:.0f}s) "
                    f"reached, waiting {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.period_s:
                self._calls.popleft()
        self._calls.append(now)


class VLLMTranscriber:
    """Self-hosted transcription via a vLLM server exposing the standard
    OpenAI-compatible /v1/audio/transcriptions endpoint. Selected with --use-vllm.

    Reached one of two ways - whichever is configured wins, ngrok first:
      - via an ngrok tunnel, at NGROK_ENDPOINT + QWEN_ASR_PATH (set NGROK_ENDPOINT
        in .env). This is what --use-vllm uses by default once that's set.
      - locally, at VLLM_BASE_URL + VLLM_LOCAL_PATH, e.g. started with:
        vllm serve kapturecx/qwen-asr-hindi-3006-ft --max-model-len 768 \\
            --max-num-seqs 16 --port 5500 --gpu-memory-utilization 0.6
    """

    name = "vllm"

    def __init__(self, url=None, model=None, timeout_s=None):
        if url:
            self.url = url
        elif config.NGROK_ENDPOINT:
            self.url = config.NGROK_ENDPOINT.rstrip("/") + config.QWEN_ASR_PATH
        else:
            self.url = config.VLLM_BASE_URL.rstrip("/") + config.VLLM_LOCAL_PATH
        self.model = model or config.VLLM_MODEL
        self.default_timeout_s = timeout_s or config.VLLM_TIMEOUT_S
        # A persistent Session with keep-alive means most requests reuse the
        # same underlying TCP connection instead of opening a new one each
        # time. That matters because ngrok's free tier caps *new TCP
        # connections* at 100/min separately from HTTP requests at 4000/min -
        # requests.post() with no Session opens a fresh connection every call,
        # which was hitting the 100/min connection cap (SSLEOFError resets)
        # while nowhere near the request cap. We're strictly sequential (one
        # request in flight at a time) so a small pool is plenty.
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=4)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._rate_limiter = _RateLimiter(config.VLLM_MAX_REQUESTS_PER_MINUTE, 60.0)

    async def prepare(self):
        pass  # plain REST endpoint - nothing to probe/handshake up front

    async def transcribe_chunk(self, path, call_id, flush_timeout_s=None):
        """Returns (transcript_text, duration_s).

        Uses the blocking `requests` library directly - fine here since chunks
        are processed strictly one at a time with nothing else to keep the
        event loop busy concurrently.
        """
        timeout_s = flush_timeout_s or self.default_timeout_s

        try:
            duration_s = sf.info(path).duration
        except Exception as e:
            raise TranscriptionError(f"audio load failed for {path}: {e}") from e

        self._rate_limiter.wait()
        try:
            with open(path, "rb") as f:
                resp = self.session.post(
                    self.url,
                    files={"file": (os.path.basename(path), f, "audio/wav")},
                    data={"model": self.model},
                    timeout=timeout_s,
                )
        except requests.exceptions.Timeout as e:
            raise TranscriptionTimeout(
                f"vllm request timed out for {path} (waited {timeout_s}s)"
            ) from e
        except requests.exceptions.RequestException as e:
            raise TranscriptionError(f"vllm request failed for {path}: {e}") from e

        if resp.status_code != 200:
            raise TranscriptionError(
                f"vllm returned HTTP {resp.status_code} for {path}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise TranscriptionError(
                f"vllm returned non-JSON response for {path}: {resp.text[:300]}"
            ) from e

        text = (data.get("text") or "").strip()
        return text, duration_s
