"""
Viva / long-answer evaluation engine.

Uses OpenAI structured outputs to score a student answer against a rubric
and returns a validated, deterministic payload.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, field_validator

from ai_ml.exceptions import EvaluationError, LLMServiceError
from ai_ml.openai_llm import OpenAIStructuredCaller
from app.config import settings

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    """Validated structure for a single evaluation response."""

    score: int
    strengths: List[str]
    weakness: List[str]
    justification: str
    suggested_improvement: str

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score must be numeric, got {value!r}") from exc

    @field_validator("strengths", "weakness", mode="before")
    @classmethod
    def _ensure_list(cls, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]


_SYSTEM_PROMPT = (
    "You are a strict academic exam evaluator. "
    "Score answers fairly and return only structured data."
)

_PROMPT_VERSION = "openai-evaluation-v1"


class EvaluationEngine:
    """
    Evaluates a student's answer using OpenAI structured outputs.
    """

    def __init__(self, client=None) -> None:
        self._caller = OpenAIStructuredCaller(client=client)

    def evaluate(
        self,
        *,
        question_text: str,
        student_answer: str,
        rubric: List[str],
        max_marks: float,
    ) -> EvalResult:
        rubric_text = "\n".join(f"- {item}" for item in rubric)
        user_prompt = (
            f"Rubric:\n{rubric_text}\n\n"
            f"Question:\n{question_text}\n\n"
            f"Student answer:\n{student_answer}\n\n"
            f"Maximum marks: {int(max_marks)}\n\n"
            "Rules:\n"
            f"- Score must be an integer between 0 and {int(max_marks)}.\n"
            '- If the answer is "I don\'t know", blank, or off-topic, the score must be 0.\n'
            "- strengths: array of short strings.\n"
            "- weakness: array of short strings.\n"
            "- justification: short explanation of the score.\n"
            "- suggested_improvement: one actionable suggestion."
        )

        try:
            result = self._caller.parse(
                cache_namespace="evaluation",
                model=settings.OPENAI_MODEL_EVAL,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=EvalResult,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_EVAL,
                cache_inputs={
                    "question_text": question_text,
                    "student_answer": student_answer,
                    "rubric": rubric,
                    "max_marks": int(max_marks),
                    "prompt_version": _PROMPT_VERSION,
                },
            )
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.error("OpenAI call failed during evaluation: %s", exc)
            raise EvaluationError(f"LLM call failed: {exc}") from exc

        result = result.model_copy(
            update={"score": max(0, min(result.score, int(max_marks)))}
        )
        return result
