"""
MCQ generation service: bridges FastAPI route -> MCQGenerator.

Multi-topic requests are batched to reduce paid calls. If a batch fails,
the service falls back to per-topic generation for just that batch.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from ai_ml.exceptions import LLMServiceError, QuestionsGenerationError
from ai_ml.mcq_generator import MCQGenerator
from app.config import settings
from app.core.state import app_state
from app.schemas.mcq_generation import MCQGenerationRequest, MCQGenerationResponse, MCQItem

logger = logging.getLogger(__name__)

_generator: MCQGenerator | None = None


def _get_generator() -> MCQGenerator:
    global _generator
    if _generator is None:
        _generator = MCQGenerator(client=app_state.openai_client)
    return _generator


def _chunk_topics(topics: Sequence[str]) -> list[list[str]]:
    batch_size = max(1, settings.OPENAI_TOPIC_BATCH_SIZE)
    topic_list = [str(topic).strip() for topic in topics if str(topic).strip()]
    return [topic_list[i : i + batch_size] for i in range(0, len(topic_list), batch_size)]


async def generate_mcqs(payload: MCQGenerationRequest) -> MCQGenerationResponse:
    """
    Generate MCQs for one or more topics.
    """
    generator = _get_generator()

    def _generate_topic(topic: str):
        raw_mcqs = generator.generate(
            topic=topic,
            num_questions=payload.num_questions,
            difficulty=payload.difficulty,
        )
        return topic, [MCQItem(**m) for m in raw_mcqs]

    def _generate_batch(topics: Sequence[str]):
        batch_results = generator.generate_many(
            topics=topics,
            num_questions=payload.num_questions,
            difficulty=payload.difficulty,
        )
        payload_map: dict[str, list[MCQItem]] = {}
        for topic in topics:
            mcqs = batch_results.get(topic)
            if mcqs is not None:
                payload_map[topic] = [MCQItem(**m) for m in mcqs]
        return payload_map

    results: dict[str, list[MCQItem] | dict[str, str]] = {}
    batch_groups = _chunk_topics(payload.topics)

    batch_tasks = [asyncio.to_thread(_generate_batch, batch) for batch in batch_groups]
    batch_outputs = await asyncio.gather(*batch_tasks, return_exceptions=True)
    for batch, output in zip(batch_groups, batch_outputs):
        if isinstance(output, Exception):
            logger.warning("MCQ batch failed for topics %s: %s", batch, output)
            fallback_tasks = [asyncio.to_thread(_generate_topic, topic) for topic in batch]
            fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
            for topic, result in zip(batch, fallback_results):
                if isinstance(result, Exception):
                    if isinstance(result, (QuestionsGenerationError, LLMServiceError)):
                        logger.error("MCQ generation failed for topic '%s': %s", topic, result)
                        results[topic] = {"error": str(result)}
                    else:
                        logger.exception("Unexpected error generating MCQs for topic '%s'", topic)
                        results[topic] = {"error": "Unexpected internal error."}
                else:
                    topic_name, topic_result = result
                    results[topic_name] = topic_result
            continue
        results.update(output)

    missing_topics = [topic for topic in payload.topics if topic not in results]
    if missing_topics:
        logger.warning("Falling back to single-topic MCQ generation for omitted topics: %s", missing_topics)
        tasks = [asyncio.to_thread(_generate_topic, topic) for topic in missing_topics]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        for topic, result in zip(missing_topics, results_list):
            if isinstance(result, Exception):
                if isinstance(result, (QuestionsGenerationError, LLMServiceError)):
                    results[topic] = {"error": str(result)}
                else:
                    results[topic] = {"error": "Unexpected internal error."}
            else:
                topic_name, topic_result = result
                results[topic_name] = topic_result

    return MCQGenerationResponse(topics=results)
