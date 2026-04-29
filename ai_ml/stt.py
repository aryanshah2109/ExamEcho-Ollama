"""
Speech-to-Text module for ExamEcho.

Supports two backends:
  * ``whisper``  – OpenAI Whisper running locally (default, preloaded at startup)
  * ``hf``       – HuggingFace Whisper pipeline (optional)

Typical usage with a preloaded model (fastest path)::

    from app.core.state import app_state
    from ai_ml.stt import STT

    text = STT.transcribe_with_model(
        model=app_state.whisper_model,
        audio_path="/tmp/answer.wav",
        lang="en",
    )
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore")

from ai_ml.audio_preprocessor import AudioPreprocessor
from ai_ml.exceptions import AudioProcessingError, IllegalModelSelectionError
from ai_ml.model_creator import WhisperModelLoader

logger = logging.getLogger(__name__)


class STT:
    """
    Speech-to-Text helper.

    All methods are **class methods** or **static methods** — there is no
    per-request state to store, so instantiation is optional.

    Args:
        lang:             BCP-47 language code (e.g. ``"en"``, ``"hi"``).
        model:            Backend to use: ``"whisper"`` or ``"hf"``.
        audio_file_path:  Path to the input audio file.
    """

    SUPPORTED_MODELS = ("whisper", "hf")

    def __init__(self, lang: str = "en", model: str = "whisper", audio_file_path: str = "") -> None:
        self.lang = lang.lower()
        self.model = model.lower()
        self.audio_file_path = audio_file_path

    # ── High-level convenience ────────────────────────────────────────────────

    @staticmethod
    def transcribe_with_model(model, audio_path: str, lang: str = "en") -> str:
        """
        Transcribe audio using a **preloaded** Whisper model instance.

        This is the recommended path in production — the model is loaded once
        at startup and reused without re-loading overhead.

        Args:
            model:       A loaded ``whisper.Whisper`` model instance.
            audio_path:  Path to the (preprocessed) audio file.
            lang:        BCP-47 language code.

        Returns:
            Transcribed text string, or empty string on failure.

        Raises:
            AudioProcessingError: If preprocessing fails.
        """
        preprocessor = AudioPreprocessor()
        result = preprocessor.preprocess_file(audio_path)
        processed_path = result.metadata.processed_path

        try:
            output = model.transcribe(processed_path, language=lang)
        except Exception as exc:
            logger.error("Whisper transcription error: %s", exc)
            raise AudioProcessingError(f"Whisper transcription failed: {exc}") from exc

        if isinstance(output, dict) and "text" in output:
            text = output["text"].strip()
            logger.debug("Transcription result (%d chars)", len(text))
            return text

        logger.warning("Unexpected Whisper output format: %s", type(output))
        return ""

    def transcribe(self) -> str:
        """
        Transcribe audio using the backend specified in ``self.model``.

        Returns:
            Transcribed text string, or empty string on failure.

        Raises:
            IllegalModelSelectionError: If the requested model is unsupported.
            AudioProcessingError:       If preprocessing or transcription fails.
        """
        if self.model not in self.SUPPORTED_MODELS:
            raise IllegalModelSelectionError(
                f"Unsupported STT model '{self.model}'. "
                f"Choose one of: {self.SUPPORTED_MODELS}."
            )

        if self.model == "whisper":
            return self._whisper_transcribe()
        return self._hf_transcribe()

    # ── Private backends ──────────────────────────────────────────────────────

    def _whisper_transcribe(self) -> str:
        whisper_model = WhisperModelLoader.get_model()
        return STT.transcribe_with_model(whisper_model, self.audio_file_path, self.lang)

    def _hf_transcribe(self) -> str:
        """HuggingFace Whisper pipeline transcription."""
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            preprocessor = AudioPreprocessor()
            result = preprocessor.preprocess_file(self.audio_file_path)
            processed_path = result.metadata.processed_path

            pipe = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base")
            output = pipe(processed_path)

            if isinstance(output, list) and output and isinstance(output[0], dict):
                return output[0].get("text", "").strip()
            if isinstance(output, dict):
                return output.get("text", "").strip()

            logger.warning("Unexpected HF pipeline output: %s", type(output))
            return ""

        except Exception as exc:
            logger.error("HF Whisper transcription error: %s", exc)
            raise AudioProcessingError(f"HF Whisper transcription failed: {exc}") from exc
