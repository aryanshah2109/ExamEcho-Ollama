"""
Rubric generation engine.

Generates marking criteria with OpenAI structured outputs and caches the
validated result.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field, field_validator

from ai_ml.exceptions import LLMServiceError, RubricsGenerationError
from ai_ml.openai_llm import OpenAIStructuredCaller
from app.config import settings

logger = logging.getLogger(__name__)


class RubricsResult(BaseModel):
    """Validated rubric list for a single question."""

    question_text: str = Field(min_length=1)
    rubrics: List[str] = Field(min_length=1)

    @field_validator("rubrics", mode="before")
    @classmethod
    def _ensure_list(cls, value) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("rubrics must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("Rubrics must contain at least one non-empty item.")
        return cleaned


_SYSTEM_PROMPT = (
    "You are an academic exam evaluator. "
    "Generate concise marking criteria and return only structured data."
)

_PROMPT_VERSION = "openai-rubrics-v1"


class RubricsEngine:
    """Generates marking rubrics for a given question using OpenAI."""

    def __init__(self, client=None) -> None:
        self._caller = OpenAIStructuredCaller(client=client)

    def generate(self, *, question_text: str, max_marks: int) -> RubricsResult:
        user_prompt = (
            f"Question:\n{question_text}\n\n"
            f"Total marks: {max_marks}\n\n"
            "Task: Generate one specific evaluative criterion for roughly every 2 to 3 marks.\n"
            "Constraints:\n"
            "- Each criterion must be a single sentence.\n"
            "- Do not include generic feedback like 'good effort' or 'clear writing'.\n"
            "- Keep the rubric criteria specific and markable.\n"
            "- Echo the question text exactly."
        )

        try:
            result = self._caller.parse(
                cache_namespace="rubrics",
                model=settings.OPENAI_MODEL_RUBRIC,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=RubricsResult,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_RUBRIC,
                cache_inputs={
                    "question_text": question_text,
                    "max_marks": max_marks,
                    "prompt_version": _PROMPT_VERSION,
                },
            )
            if not result.question_text.strip():
                result = result.model_copy(update={"question_text": question_text})
            return result
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.error("OpenAI call failed during rubric generation: %s", exc)
            raise RubricsGenerationError(f"LLM call failed: {exc}") from exc
