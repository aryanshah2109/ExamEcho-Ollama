"""
MCQ generation engine for ExamEcho.

Generates Multiple Choice Questions for a given topic and difficulty
using a local Ollama model (mistral:7b).
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from ai_ml.exceptions import ChainCreationError, QuestionsGenerationError
from ai_ml.model_creator import OllamaModelLoader
from app.config import settings

logger = logging.getLogger(__name__)


_MCQ_TEMPLATE = """\
[INST]
You are an academic exam question setter. Respond with ONLY a valid JSON object. No markdown, no explanation, no text before or after.

Rules:
- Generate EXACTLY {num_questions} MCQs for the TOPIC below.
- Each question must have exactly 4 options prefixed with A:, B:, C:, D:.
- Exactly one option must be correct.
- correct_option must be the exact full string of the correct option (e.g. "A: some text").
- NO code, NO programs, NO algorithms unless the topic demands it.
- Stay strictly within the TOPIC.
- Difficulty guidelines:
  EASY   : definitions, meanings, purposes
  MEDIUM : explanations, reasoning, simple examples
  HARD   : critical thinking, limitations, trade-offs, applications

TOPIC: {topic}
DIFFICULTY: {difficulty}

Respond with ONLY this JSON:
{{"topic":"{topic}","mcqs":[{{"question":"<question text>","options":["A: <option>","B: <option>","C: <option>","D: <option>"],"correct_option":"A: <option>"}}]}}
[/INST]"""

class MCQGenerator:
    """
    Generates MCQ exam questions using a local Ollama model.

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
        try:
            prompt = PromptTemplate(
                template=_MCQ_TEMPLATE,
                input_variables=["num_questions", "topic", "difficulty"],
            )
            model = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL_NAME,
                temperature=settings.OLLAMA_TEMPERATURE,
                num_ctx=settings.OLLAMA_NUM_CTX,
                options={
                    "num_predict": settings.OLLAMA_MAX_TOKENS_RUBRICS
                }
            )
            return prompt | model
        except Exception as exc:
            raise ChainCreationError(
                f"Could not build MCQ generation chain: {exc}"
            ) from exc

    @staticmethod
    def _sanitize_json(text: str) -> str:
        """
        Strip markdown fences, [INST]/[/INST] tags, and extract the first
        valid JSON object from raw model output.
        """
        text = re.sub(r"\[/?INST\]", "", text)
        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "").strip()

        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    return json.dumps(obj)
                except json.JSONDecodeError:
                    continue
        raise ValueError("No valid JSON object found in model output.")

    @staticmethod
    def _normalize_mcqs(raw_mcqs: list, num_questions: int) -> List[Dict[str, Any]]:
        """Validate and normalize the generated MCQs."""
        normalized = []
        for mcq in raw_mcqs:
            if not isinstance(mcq, dict):
                continue
            
            question = mcq.get("question", "").strip()
            options = mcq.get("options", [])
            correct_option = mcq.get("correct_option", "").strip()

            if not question or not isinstance(options, list) or len(options) != 4 or not correct_option:
                continue
                
            # Basic validation to ensure correct option is in the options list
            if correct_option not in options:
                # Attempt to find it if it was stripped
                found = False
                for opt in options:
                    if correct_option in opt or opt in correct_option:
                        correct_option = opt
                        found = True
                        break
                if not found:
                    continue

            normalized.append({
                "question": question,
                "options": [str(opt).strip() for opt in options],
                "correct_option": correct_option
            })

        return normalized[:num_questions]

    def generate(self, *, topic: str, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """
        Generate MCQ questions for a topic.

        Args:
            topic:         Subject topic.
            num_questions: Number of questions to generate.
            difficulty:    One of "easy", "medium", "hard".

        Returns:
            List of dictionaries representing the MCQs.

        Raises:
            ChainCreationError:      If the LangChain chain cannot be built.
            QuestionsGenerationError: If generation fails or returns fewer
                                      questions than requested.
        """
        logger.debug(
            "Generating %d '%s' MCQs for topic: %s",
            num_questions,
            difficulty,
            topic,
        )

        try:
            chain = self._build_chain()
            raw = chain.invoke({
                "topic": topic,
                "num_questions": num_questions,
                "difficulty": difficulty,
            })
        except ChainCreationError:
            raise
        except Exception as exc:
            raise QuestionsGenerationError(f"Ollama call failed: {exc}") from exc

        content = raw.content if hasattr(raw, "content") else str(raw)

        try:
            cleaned = self._sanitize_json(content)
            data = json.loads(cleaned)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse MCQs JSON. Raw output (first 500 chars): %s", content[:500])
            raise QuestionsGenerationError(
                f"Invalid JSON from model. Original error: {exc}"
            ) from exc

        raw_list = data.get("mcqs", [])
        if not isinstance(raw_list, list):
            raise QuestionsGenerationError("'mcqs' field is not a list in model response.")

        mcqs = self._normalize_mcqs(raw_list, num_questions)

        if not mcqs:
            raise QuestionsGenerationError("Model returned an empty or invalid MCQs list.")

        if len(mcqs) < num_questions:
            logger.warning(
                "Expected %d MCQs but received %d for topic '%s'.",
                num_questions,
                len(mcqs),
                topic,
            )
            raise QuestionsGenerationError(
                f"Expected {num_questions} MCQs but received only {len(mcqs)}. "
                "Try reducing num_questions or simplifying the topic."
            )

        logger.debug("Generated %d MCQs for '%s'.", len(mcqs), topic)
        return mcqs
