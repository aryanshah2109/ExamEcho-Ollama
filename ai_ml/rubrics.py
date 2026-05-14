"""
Rubric generation engine.

Given a question and its maximum marks, uses the local Ollama model
(mistral:7b) to produce a list of marking criteria (rubrics) that an
evaluator should check when scoring a student's answer.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from ai_ml.exceptions import RubricsGenerationError
from ai_ml.model_creator import OllamaModelLoader
from app.utils.json_utils import extract_json
from app.config import settings

logger = logging.getLogger(__name__)


# Output schema

class RubricsResult(BaseModel):
    """Validated rubric list for a single question."""

    question_text: str = Field(min_length=1)
    rubrics: List[str] = Field(min_length=1)

    @field_validator("rubrics", mode="before")
    @classmethod
    def _ensure_list(cls, v) -> List[str]:
        if isinstance(v, str):
            return [v]
        cleaned = [str(item).strip() for item in v if str(item).strip()]
        if not cleaned:
            raise ValueError("Rubrics must contain at least one non-empty item.")
        return cleaned


# Prompt

_RUBRICS_TEMPLATE = """\
[INST]
You are an academic exam evaluator. Respond with ONLY a valid JSON object. No markdown, no explanation, no text before or after.

Rules:
- Generate one rubric criterion per 2-3 marks (e.g. 10 marks = 4-5 rubrics).
- Each criterion must be a single, specific, evaluative sentence.
- Do NOT include generic rubrics like "Good effort" or "Clear writing".

Question:
{question_text}

Total Marks: {max_marks}

Respond with ONLY this JSON:
{{"question_text":"{question_text}","rubrics":["criterion 1","criterion 2","criterion 3"]}}
[/INST]"""



class RubricsEngine:
    """
    Generates marking rubrics for a given question using a local Ollama model.

    Args:
        model: Pre-loaded ChatOllama instance (optional; lazy-loaded if omitted).
    """

    def __init__(self, model=None) -> None:
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = OllamaModelLoader.get_model()
        return self._model

    def _build_chain(self):
        prompt = PromptTemplate(
            template=_RUBRICS_TEMPLATE,
            input_variables=["question_text", "max_marks"],
        )
        model = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.OLLAMA_TEMPERATURE,
            num_ctx=settings.OLLAMA_NUM_CTX,
            options={
                "num_predict": settings.OLLAMA_MAX_TOKENS_MCQ
            }
        )
        return prompt | model

    def generate(self, *, question_text: str, max_marks: int) -> RubricsResult:
        """
        Generate rubrics for a question.

        Args:
            question_text: The exam question.
            max_marks:     The maximum marks allocated to this question.

        Returns:
            :class:`RubricsResult` with validated rubric list.

        Raises:
            RubricsGenerationError: If the Ollama call or response parsing fails.
        """
        try:
            chain = self._build_chain()
            raw = chain.invoke({"question_text": question_text, "max_marks": max_marks})
        except Exception as exc:
            logger.error("Ollama call failed during rubric generation: %s", exc)
            raise RubricsGenerationError(f"LLM call failed: {exc}") from exc

        content = raw.content if hasattr(raw, "content") else str(raw)
        logger.debug("Rubrics raw output (%d chars)", len(content))

        try:
            data = extract_json(content)
            # Ensure question_text is echoed correctly (model may paraphrase it)
            if "question_text" not in data or not data["question_text"]:
                data["question_text"] = question_text
            return RubricsResult(**data)
        except Exception as exc:
            logger.error(
                "Failed to parse rubrics response. Raw output (first 500 chars): %s",
                content[:500],
            )
            raise RubricsGenerationError(
                f"Could not parse rubrics response: {exc}"
            ) from exc
