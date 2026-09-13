"""MiniMax speech-to-text (``asr-1.0``) as a plugin-provided ``stt.provider`` backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.minimax.io"
_DEFAULT_MODEL = "asr-1.0"
_TIMEOUT_SECONDS = 180
_KEY_ENV_VARS = ("MINIMAX_API_KEY", "MINIMAX_STT_API_KEY")


def _config_section() -> Dict[str, Any]:
    """``stt.minimax`` from config.yaml; empty when config cannot be read."""
    try:
        from hermes_cli.config import load_config

        stt = (load_config() or {}).get("stt") or {}
        section = stt.get("minimax") if isinstance(stt, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _api_key() -> str:
    """Configured key, then the env vars. ``key_env`` names a variable, ``api_key`` is literal."""
    section = _config_section()
    literal = str(section.get("api_key") or "").strip()
    if literal:
        return literal
    named = str(section.get("key_env") or section.get("api_key_env") or "").strip()
    candidates = (named, *_KEY_ENV_VARS) if named else _KEY_ENV_VARS
    for var in candidates:
        try:
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = str(get_env_value_prefer_dotenv(var) or "").strip()
        except Exception:
            value = str(os.environ.get(var) or "").strip()
        if value:
            return value
    return ""


def _base_url() -> str:
    section = _config_section()
    configured = str(section.get("base_url") or "").strip()
    return (configured or os.environ.get("MINIMAX_STT_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


class MiniMaxTranscriptionProvider(TranscriptionProvider):
    """Speech-to-text over MiniMax ``/v1/speech_to_text``.

    The endpoint takes multipart form data — ``model`` plus the audio file — and answers
    ``{"text": ...}``. Language is auto-detected server-side; a configured hint is forwarded
    but never synthesised, because pinning a language is what corrupts the other one for a
    bilingual speaker.
    """

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax ASR"

    def is_available(self) -> bool:
        return bool(_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": _DEFAULT_MODEL, "display": "MiniMax asr-1.0"}]

    def transcribe(
        self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None, **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe ``file_path``; never raises, failures come back as the error envelope."""
        key = _api_key()
        if not key:
            return self._error("MINIMAX_API_KEY is not set — add it to .env or stt.minimax.api_key")
        chosen = str(model or "").strip() or _DEFAULT_MODEL
        data: Dict[str, str] = {"model": chosen}
        hint = str(language or "").strip()
        if hint:
            data["language"] = hint
        try:
            import requests

            path = Path(file_path)
            with path.open("rb") as handle:
                response = requests.post(
                    f"{_base_url()}/v1/speech_to_text",
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files={"file": (path.name, handle)},
                    timeout=_TIMEOUT_SECONDS,
                )
            if response.status_code != 200:
                return self._error(f"HTTP {response.status_code}: {response.text[:300]}")
            payload = response.json()
        except Exception as exc:
            return self._error(f"{type(exc).__name__}: {exc}")
        transcript = str(payload.get("text") or "").strip()
        if not transcript:
            failure = payload.get("base_resp") or payload.get("error") or payload
            return self._error(f"empty transcript: {str(failure)[:300]}")
        return {"success": True, "transcript": transcript, "provider": self.name}

    def _error(self, message: str) -> Dict[str, Any]:
        logger.warning("MiniMax STT failed: %s", message)
        return {"success": False, "transcript": "", "provider": self.name, "error": message}


def register(ctx) -> None:
    """Entry point called by the plugin manager."""
    ctx.register_transcription_provider(MiniMaxTranscriptionProvider())
