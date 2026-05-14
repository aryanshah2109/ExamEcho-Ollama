"""
Question generation service: bridges FastAPI route → QuestionGenerator.

Supports multi-topic generation — each topic in the request is processed
independently and results are returned as a unified dict keyed by topic.
"""

from __future__ import annotations

import logging
from typing import Dict
import asyncio
from functools import partial

from ai_ml.question_generator import QuestionGenerator
from ai_ml.exceptions import QuestionsGenerationError
from app.core.state import app_state
from app.schemas.question_generation import QuestionGenerationRequest, QuestionGenerationResponse

logger = logging.getLogger(__name__)

_generator: QuestionGenerator | None = None


def _get_generator() -> QuestionGenerator:
    global _generator
    if _generator is None:
        _generator = QuestionGenerator(model=app_state.ollama_model)
    return _generator


async def generate_questions(payload: QuestionGenerationRequest) -> QuestionGenerationResponse:
    """
    Generate questions for one or more topics.

    Each topic is processed independently.  If a single topic fails, the
    error is logged and that topic's entry will contain an ``error`` key
    rather than ``questions`` — so other topics still succeed.

    Args:
        payload: Request containing topic list, question count, and difficulty.

    Returns:
        :class:`QuestionGenerationResponse` mapping topic → question dict.
    """
    
    generator = _get_generator()
    loop = asyncio.get_running_loop()

    def _generate_one(topic):
        try:
            questions = generator.generate(
                topic=topic,
                num_questions=payload.num_questions,
                difficulty=payload.difficulty,
            )
            return topic, {str(idx + 1): q for idx, q in enumerate(questions)}
        except QuestionsGenerationError as exc:
            logger.error("Question generation failed for topic '%s': %s", topic, exc)
            return topic, {"error": str(exc)}

    tasks = [loop.run_in_executor(None, _generate_one, topic) for topic in payload.topics]
    results_list = await asyncio.gather(*tasks)
    results = dict(results_list)
    return QuestionGenerationResponse(topics=results)