"""
MCQ generation engine for ExamEcho.

Uses OpenAI structured outputs to produce multiple-choice questions with
validated answer options and persistent caching.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator

from ai_ml.exceptions import LLMServiceError, QuestionsGenerationError
from ai_ml.openai_llm import OpenAIStructuredCaller
from app.config import settings

logger = logging.getLogger(__name__)


class _MCQItem(BaseModel):
    question: str = Field(min_length=1)
    options: List[str] = Field(min_length=4, max_length=4)
    correct_option: str = Field(min_length=1)

    @field_validator("options", mode="before")
    @classmethod
    def _clean_options(cls, value) -> List[str]:
        if not isinstance(value, list):
            raise ValueError("options must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if len(cleaned) != 4:
            raise ValueError("options must contain exactly four non-empty items")
        return cleaned

    @field_validator("correct_option", mode="after")
    @classmethod
    def _validate_correct_option(cls, value: str) -> str:
        return value.strip()


class _MCQBatch(BaseModel):
    topic: str = Field(min_length=1)
    mcqs: List[_MCQItem] = Field(min_length=1)

    @field_validator("mcqs", mode="before")
    @classmethod
    def _ensure_mcqs(cls, value) -> List[dict]:
        if not isinstance(value, list):
            raise ValueError("mcqs must be a list")
        return value


class _MCQTopicItem(BaseModel):
    topic: str = Field(min_length=1)
    mcqs: List[_MCQItem] = Field(min_length=1)

    @field_validator("mcqs", mode="before")
    @classmethod
    def _ensure_mcqs(cls, value) -> List[dict]:
        if not isinstance(value, list):
            raise ValueError("mcqs must be a list")
        return value


class _MCQTopicBatch(BaseModel):
    topics: List[_MCQTopicItem] = Field(min_length=1)

    @field_validator("topics", mode="before")
    @classmethod
    def _ensure_topics(cls, value) -> List[dict]:
        if not isinstance(value, list):
            raise ValueError("topics must be a list")
        return value


_SYSTEM_PROMPT = (
    "You are an academic exam setter. "
    "Generate polished MCQs and return only structured data."
)

_PROMPT_VERSION = "openai-mcq-v1"


class MCQGenerator:
    """Generates MCQ exam questions using OpenAI."""

    def __init__(self, client=None) -> None:
        self._caller = OpenAIStructuredCaller(client=client)

    def _generate_batch(
        self,
        *,
        topic: str,
        num_questions: int,
        difficulty: str,
    ) -> List[Dict[str, Any]]:
        user_prompt = (
            f"Topic: {topic}\n"
            f"Difficulty: {difficulty}\n"
            f"Task: Generate exactly {num_questions} MCQs.\n"
            "Constraints:\n"
            "- Each MCQ must have exactly four options.\n"
            "- Options must be prefixed with A:, B:, C:, and D:.\n"
            "- Exactly one option must be correct.\n"
            "- correct_option must match the exact string of one of the options.\n"
            "- Avoid code/programs/algorithms unless the topic strictly requires them.\n"
            "- Stay strictly within the topic.\n"
            "- Do not include numbering, markdown, explanations, or extra keys."
        )

        try:
            result = self._caller.parse(
                cache_namespace="mcq_generation",
                model=settings.OPENAI_MODEL_MCQ,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=_MCQBatch,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_MCQ,
                cache_inputs={
                    "topic": topic,
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "prompt_version": _PROMPT_VERSION,
                },
            )
        except LLMServiceError:
            raise
        except Exception as exc:
            raise QuestionsGenerationError(f"OpenAI call failed: {exc}") from exc

        normalized = self._normalize_mcqs(result.mcqs, num_questions)
        if len(normalized) < num_questions:
            raise QuestionsGenerationError(
                f"Expected {num_questions} MCQs but received only {len(normalized)}."
            )
        return normalized

    def _generate_multi_topic_batch(
        self,
        *,
        topics: Sequence[str],
        num_questions: int,
        difficulty: str,
    ) -> dict[str, List[Dict[str, Any]]]:
        topic_list = [str(topic).strip() for topic in topics if str(topic).strip()]
        if not topic_list:
            return {}

        user_prompt = (
            f"Topics: {', '.join(topic_list)}\n"
            f"Difficulty: {difficulty}\n"
            f"Task: For each topic, generate exactly {num_questions} MCQs.\n"
            "Constraints:\n"
            "- Each MCQ must have exactly four options.\n"
            "- Options must be prefixed with A:, B:, C:, and D:.\n"
            "- Exactly one option must be correct.\n"
            "- correct_option must match the exact string of one of the options.\n"
            "- Avoid code/programs/algorithms unless the topic strictly requires them.\n"
            "- Stay strictly within each topic.\n"
            "- Do not include numbering, markdown, explanations, or extra keys.\n"
            "- Return one entry per topic."
        )

        try:
            result = self._caller.parse(
                cache_namespace="mcq_generation_multi",
                model=settings.OPENAI_MODEL_MCQ,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=_MCQTopicBatch,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_MCQ,
                cache_inputs={
                    "topics": topic_list,
                    "num_questions": num_questions,
                    "difficulty": difficulty,
                    "prompt_version": _PROMPT_VERSION,
                },
            )
        except LLMServiceError:
            raise
        except Exception as exc:
            raise QuestionsGenerationError(f"OpenAI batch call failed: {exc}") from exc

        mapped: dict[str, List[Dict[str, Any]]] = {}
        for item in result.topics:
            mcqs = self._normalize_mcqs(item.mcqs, num_questions)
            if mcqs:
                mapped[item.topic] = mcqs
        return mapped

    @staticmethod
    def _normalize_mcqs(raw_mcqs: list, num_questions: int) -> List[Dict[str, Any]]:
        normalized = []
        for mcq in raw_mcqs:
            if hasattr(mcq, "model_dump"):
                mcq = mcq.model_dump()
            if not isinstance(mcq, dict):
                continue

            question = str(mcq.get("question", "")).strip()
            options = mcq.get("options", [])
            correct_option = str(mcq.get("correct_option", "")).strip()

            if not question or not isinstance(options, list) or len(options) != 4 or not correct_option:
                continue

            options = [str(opt).strip() for opt in options if str(opt).strip()]
            if len(options) != 4:
                continue

            if correct_option not in options:
                found = False
                for opt in options:
                    if correct_option in opt or opt in correct_option:
                        correct_option = opt
                        found = True
                        break
                if not found:
                    continue

            normalized.append(
                {
                    "question": question,
                    "options": options,
                    "correct_option": correct_option,
                }
            )

        return normalized[:num_questions]

    def generate(self, *, topic: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        logger.debug(
            "Generating %d '%s' MCQs for topic: %s",
            num_questions,
            difficulty,
            topic,
        )

        chunk_size = max(1, min(settings.OPENAI_GENERATION_CHUNK_SIZE, num_questions))
        results: List[Dict[str, Any]] = []
        remaining = num_questions

        while remaining > 0:
            batch_size = min(chunk_size, remaining)
            results.extend(
                self._generate_batch(
                    topic=topic,
                    num_questions=batch_size,
                    difficulty=difficulty,
                )
            )
            remaining -= batch_size

        return results[:num_questions]

    def generate_many(
        self,
        *,
        topics: Sequence[str],
        num_questions: int,
        difficulty: str,
    ) -> dict[str, List[Dict[str, Any]]]:
        """
        Generate MCQs for multiple topics using batched requests.
        """
        results: dict[str, List[Dict[str, Any]]] = {}
        batch_size = max(1, settings.OPENAI_TOPIC_BATCH_SIZE)
        topic_list = [str(topic).strip() for topic in topics if str(topic).strip()]

        for start in range(0, len(topic_list), batch_size):
            batch = topic_list[start:start + batch_size]
            batch_result = self._generate_multi_topic_batch(
                topics=batch,
                num_questions=num_questions,
                difficulty=difficulty,
            )
            results.update(batch_result)

        return results
