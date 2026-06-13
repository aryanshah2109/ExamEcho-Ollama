"""
Question generation engine for ExamEcho.

Uses OpenAI structured outputs to generate theory-based exam questions
with deterministic formatting and persistent caching.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import List

from pydantic import BaseModel, Field, field_validator

from ai_ml.exceptions import LLMServiceError, QuestionsGenerationError
from ai_ml.openai_llm import OpenAIStructuredCaller
from app.config import settings

logger = logging.getLogger(__name__)


class _QuestionBatch(BaseModel):
    topic: str = Field(min_length=1)
    questions: List[str] = Field(min_length=1)

    @field_validator("questions", mode="before")
    @classmethod
    def _clean_questions(cls, value) -> List[str]:
        if not isinstance(value, list):
            raise ValueError("questions must be a list")
        cleaned = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("questions must contain at least one non-empty item")
        return cleaned


class _QuestionTopicItem(BaseModel):
    topic: str = Field(min_length=1)
    questions: List[str] = Field(min_length=1)

    @field_validator("questions", mode="before")
    @classmethod
    def _clean_questions(cls, value) -> List[str]:
        if not isinstance(value, list):
            raise ValueError("questions must be a list")
        cleaned = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("questions must contain at least one non-empty item")
        return cleaned


class _QuestionTopicBatch(BaseModel):
    topics: List[_QuestionTopicItem] = Field(min_length=1)

    @field_validator("topics", mode="before")
    @classmethod
    def _clean_topics(cls, value) -> List[dict]:
        if not isinstance(value, list):
            raise ValueError("topics must be a list")
        return value


_SYSTEM_PROMPT = (
    "You are an academic exam setter. "
    "Generate concise, high-quality theory questions and return only structured data."
)

_PROMPT_VERSION = "openai-question-v1"


class QuestionGenerator:
    """
    Generates theory exam questions using OpenAI structured outputs.

    Args:
        client: Preloaded OpenAI client instance (optional).
    """

    def __init__(self, client=None) -> None:
        self._client = client
        self._caller = OpenAIStructuredCaller(client=client)

    def _generate_batch(self, *, topic: str, num_questions: int, difficulty: str) -> List[str]:
        user_prompt = (
            f"Topic: {topic}\n"
            f"Difficulty: {difficulty}\n"
            f"Task: Generate exactly {num_questions} theory-based exam questions.\n"
            "Constraints:\n"
            "- Questions must be verbally answerable.\n"
            "- Do not produce code, programs, or algorithms unless the topic strictly requires them.\n"
            "- Each question must be a complete sentence ending with a question mark.\n"
            "- Stay strictly within the topic.\n"
            "- Do not include numbering, markdown, explanations, or extra keys."
        )

        try:
            result = self._caller.parse(
                cache_namespace="question_generation",
                model=settings.OPENAI_MODEL_QUESTION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=_QuestionBatch,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_QUESTION,
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

        questions = self._normalize_questions(result.questions, num_questions)
        if len(questions) < num_questions:
            raise QuestionsGenerationError(
                f"Expected {num_questions} questions but received only {len(questions)}."
            )
        return questions

    def _generate_multi_topic_batch(
        self,
        *,
        topics: Sequence[str],
        num_questions: int,
        difficulty: str,
    ) -> dict[str, List[str]]:
        topic_list = [str(topic).strip() for topic in topics if str(topic).strip()]
        if not topic_list:
            return {}

        user_prompt = (
            f"Topics: {', '.join(topic_list)}\n"
            f"Difficulty: {difficulty}\n"
            f"Task: For each topic, generate exactly {num_questions} theory-based exam questions.\n"
            "Constraints:\n"
            "- Questions must be verbally answerable.\n"
            "- Do not produce code, programs, or algorithms unless the topic strictly requires them.\n"
            "- Each question must be a complete sentence ending with a question mark.\n"
            "- Stay strictly within each topic.\n"
            "- Do not include numbering, markdown, explanations, or extra keys.\n"
            "- Return one entry per topic."
        )

        try:
            result = self._caller.parse(
                cache_namespace="question_generation_multi",
                model=settings.OPENAI_MODEL_QUESTION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=_QuestionTopicBatch,
                max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_QUESTION,
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

        mapped: dict[str, List[str]] = {}
        for item in result.topics:
            questions = self._normalize_questions(item.questions, num_questions)
            if questions:
                mapped[item.topic] = questions
        return mapped

    def generate_many(
        self,
        *,
        topics: Sequence[str],
        num_questions: int,
        difficulty: str,
    ) -> dict[str, List[str]]:
        """
        Generate questions for multiple topics.

        Topics are processed in batches to reduce request count and cost.
        If a batch fails validation, callers may fall back to single-topic
        generation for just that batch.
        """
        results: dict[str, List[str]] = {}
        batch_size = max(1, settings.OPENAI_TOPIC_BATCH_SIZE)
        topic_list = [str(topic).strip() for topic in topics if str(topic).strip()]

        for start in range(0, len(topic_list), batch_size):
            batch = topic_list[start:start + batch_size]
            try:
                batch_result = self._generate_multi_topic_batch(
                    topics=batch,
                    num_questions=num_questions,
                    difficulty=difficulty,
                )
                results.update(batch_result)
            except LLMServiceError:
                # Let the service layer decide whether to fall back to single-topic retries.
                raise

        return results

    @staticmethod
    def _normalize_questions(raw_questions: list, num_questions: int) -> List[str]:
        normalized = []
        for q in raw_questions:
            if isinstance(q, str):
                q = q.strip()
                if q:
                    normalized.append(q)
        return normalized[:num_questions]

    def generate(self, *, topic: str, num_questions: int, difficulty: str) -> List[str]:
        """
        Generate exam questions for a topic.

        Large requests are chunked to reduce output-token pressure and to
        improve cache reuse across repeated calls.
        """
        logger.debug(
            "Generating %d '%s' questions for topic: %s",
            num_questions,
            difficulty,
            topic,
        )

        chunk_size = max(1, min(settings.OPENAI_GENERATION_CHUNK_SIZE, num_questions))
        results: List[str] = []
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
