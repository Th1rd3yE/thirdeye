from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from groq import Groq

from agent.tools import ToolRegistry

logger = logging.getLogger("thirdeye.agent")

SYSTEM_PROMPT = """You are ThirdEye, a multilingual news and information verification AI.

Your job is to determine whether a news claim is credible by calling the right tools
and then responding to the user in their native language.

━━━ STRICT VERIFICATION WORKFLOW ━━━

You will receive a pre-analyzed query payload (JSON) containing the user's question,
detected keywords, language, country, and date.

STEP 1 — Call `get_from_data_sources` with the full JSON payload string.
  • Parse the response and check the `classification` field.
  • If classification = true  → Proceed to STEP 3.
  • If classification = false → Proceed to STEP 2.

STEP 2 — Call `get_from_vertex_search` with the SAME JSON payload string.
  • Parse the response and check the `classification` field.
  • Proceed to STEP 3 regardless of the result.

STEP 3 — Call `reanalyse` with:
  • `question`       — the original user question
  • `classification` — the classification obtained from STEP 1 or STEP 2
  • `explanation`    — the explanation text from whichever source classified it
  This tool cross-checks whether the classification is logically consistent with
  the explanation and the question. Use its returned `classification` as the
  FINAL classification going forward.

STEP 4 — Call `get_recommended_next_actions` with:
  • `question`       — the original user question
  • `classification` — the FINAL classification from STEP 3 ("TRUE", "FALSE", or "UNCERTAIN")
  • `explanation`    — the explanation text
  • `native_language`— from the payload

STEP 5 — Respond to the user in their native_language:
  • If verified (TRUE): present the findings clearly.
  • If unverified (FALSE / UNCERTAIN): inform the user the claim could not be verified
    and they should treat it as unverified until proven otherwise.
  • Always include the recommended next actions in your response.
  • Your final answer MUST be written in plain, human-readable prose only.
    It MUST NOT contain any JSON, raw data payloads, file paths, tool outputs,
    internal field names, or any machine-readable content whatsoever.

━━━ RULES ━━━
• Always respond in the `native_language` field from the payload.
• Pass the full JSON payload string unchanged to each tool.
• Never skip STEP 1. Never call vertex search before data sources.
• Always call `reanalyse` before `get_recommended_next_actions`.
• Always call `get_recommended_next_actions` before giving your final answer.
• Be concise and factual in your final answer.
• FINAL ANSWER FORMAT: Plain human-readable text only. Never include JSON,
  raw data, file contents, internal keys, or any structured data in your
  final answer. Summarise all findings in natural language sentences."""


class StepType(str, Enum):
    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"
    ANSWER = "Answer"


@dataclass
class AgentStep:
    step_type: StepType
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None

    def __str__(self) -> str:
        header = f"[{self.step_type.value}]"
        if self.tool_name:
            header += f" {self.tool_name}"
            if self.tool_args:
                args_str = ", ".join(f"{k}={v!r}" for k, v in self.tool_args.items())
                header += f"({args_str})"
        return f"{header}\n{self.content}"


_VALID_CLASSIFICATIONS = {"TRUE", "FALSE", "UNCERTAIN", "UNVERIFIED"}


def _extract_structured_result(
    observation: str,
) -> tuple[str | None, str, list[str], list[str]]:
    """Parse a tool observation JSON and extract classification, explanation, sources, recommended_next_actions."""
    try:
        data = json.loads(observation)
    except (json.JSONDecodeError, ValueError):
        return None, "", [], []

    # Handle recommended_next_actions-only payloads (from RecommendedNextActionTool)
    if "recommended_next_actions" in data and "classification" not in data:
        actions = data.get("recommended_next_actions") or []
        return None, "", [], [str(a) for a in actions]

    raw = data.get("classification")
    if isinstance(raw, bool):
        classification: str | None = "TRUE" if raw else "FALSE"
    elif isinstance(raw, str) and raw.upper() in _VALID_CLASSIFICATIONS:
        classification = raw.upper()
    else:
        return None, "", [], []

    explanation_native = data.get("explanation_native", "") or ""
    explanation_en = (
        data.get("explanation", "") or data.get("explanation_en", "") or ""
    )
    explanation = explanation_native if explanation_native.strip() else explanation_en
    if not explanation.strip():
        explanation = data.get("reason", "")

    sources: list[str] = []
    if "results" in data:
        for r in data.get("results") or []:
            url = r.get("url", "")
            if url:
                sources.append(url)
    elif data.get("data"):
        url = data["data"].get("source_url", "")
        if url:
            sources.append(url)
    # Handle flat "sources" list returned by DS tool
    if not sources:
        for s in data.get("sources") or []:
            if isinstance(s, str) and s:
                sources.append(s)
            elif isinstance(s, dict):
                url = s.get("url", "")
                if url:
                    sources.append(url)

    return classification, explanation, sources, []


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    classification: str | None = None
    explanation: str = ""
    sources: list[str] = field(default_factory=list)
    recommended_next_actions: list[str] = field(default_factory=list)

    def pretty_print(self) -> None:
        print("\n" + "=" * 60)
        print("REACT AGENT TRACE")
        print("=" * 60)
        for i, step in enumerate(self.steps, 1):
            print(f"\nStep {i}: {step}")
            print("-" * 40)
        print(f"\nFINAL ANSWER:\n{self.answer}")
        print("=" * 60)


def _is_raw_data(text: str) -> bool:
    """Return True if `text` looks like a raw JSON payload or data dump rather than prose."""
    stripped = text.strip()
    # Starts with a JSON object or array
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    # Contains a large inline JSON blob (heuristic: a key-value pair pattern)
    if stripped.count('":') >= 3 or stripped.count("': ") >= 3:
        return True
    return False


class ReactAgent:
    """
    ReAct loop backed by the Groq chat-completions API.

    The loop runs up to `max_iterations` rounds of:
      Thought -> Action (tool call) -> Observation
    and terminates when the model produces a plain-text answer (no tool call).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        model_name: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.0,
    ) -> None:
        self.registry = registry
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.model_name = model_name or os.environ.get(
            "GROQ_MODEL", "llama-3.3-70b-versatile"
        )
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def run(self, query: str, verbose: bool = True, req_id: str = "-") -> AgentResult:
        """Execute the ReAct loop for the given user query."""
        steps: list[AgentStep] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        tools = self.registry.to_groq_tools()

        result_classification: str | None = None
        result_explanation: str = ""
        result_sources: list[str] = []
        result_recommended_actions: list[str] = []

        if verbose:
            print(f"\nQuery: {query}\n{chr(9472) * 60}")

        logger.info("[%s] ReAct loop started | model=%s | max_iterations=%d",
                    req_id, self.model_name, self.max_iterations)

        for iteration in range(1, self.max_iterations + 1):
            logger.info("[%s] ── Iteration %d/%d: calling LLM …",
                        req_id, iteration, self.max_iterations)
            llm_start = time.perf_counter()

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
            )

            llm_elapsed = time.perf_counter() - llm_start
            message = response.choices[0].message
            messages.append(message.to_dict())

            tool_calls = message.tool_calls or []
            content = message.content or ""

            logger.info("[%s]   LLM response in %.2fs | tool_calls=%d | has_content=%s",
                        req_id, llm_elapsed, len(tool_calls), bool(content))

            if content:
                thought_step = AgentStep(StepType.THOUGHT, content)
                steps.append(thought_step)
                logger.info("[%s]   [THOUGHT] %s", req_id, content[:200])
                if verbose:
                    print(thought_step)

            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args: dict[str, Any] = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    _TOOL_LABELS = {
                        "get_from_data_sources": "DS (Internal Data Sources)",
                        "get_from_vertex_search": "Vertex AI Search",
                    }
                    tool_label = _TOOL_LABELS.get(tool_name, tool_name)

                    logger.info(
                        "[%s] ┌─────────────────────────────────────────────────",
                        req_id,
                    )
                    logger.info("[%s] │  TOOL CALL ▶ %s", req_id, tool_label)
                    logger.info(
                        "[%s] │  INPUT: %s",
                        req_id,
                        json.dumps(tool_args, ensure_ascii=False),
                    )
                    logger.info(
                        "[%s] └─────────────────────────────────────────────────",
                        req_id,
                    )

                    action_step = AgentStep(
                        StepType.ACTION,
                        f"Calling tool with args: {tool_args}",
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                    steps.append(action_step)
                    if verbose:
                        print(action_step)

                    tool_start = time.perf_counter()
                    try:
                        tool = self.registry.get(tool_name)
                        observation = tool.run(**tool_args)
                        tool_elapsed = time.perf_counter() - tool_start
                        logger.info(
                            "[%s] ┌─────────────────────────────────────────────────",
                            req_id,
                        )
                        logger.info(
                            "[%s] │  TOOL RESULT ◀ %s  (%.2fs)",
                            req_id, tool_label, tool_elapsed,
                        )
                        logger.info("[%s] │  OUTPUT: %s", req_id, observation[:500])
                        logger.info(
                            "[%s] └─────────────────────────────────────────────────",
                            req_id,
                        )
                    except Exception as exc:
                        tool_elapsed = time.perf_counter() - tool_start
                        observation = f"Tool error: {exc}"
                        logger.error(
                            "[%s] ┌─────────────────────────────────────────────────",
                            req_id,
                        )
                        logger.error(
                            "[%s] │  TOOL ERROR ✗ %s  (%.2fs): %s",
                            req_id, tool_label, tool_elapsed, exc,
                        )
                        logger.error(
                            "[%s] └─────────────────────────────────────────────────",
                            req_id,
                        )

                    obs_step = AgentStep(StepType.OBSERVATION, observation)
                    steps.append(obs_step)
                    if verbose:
                        print(obs_step)

                    cls, expl, srcs, actions = _extract_structured_result(observation)
                    if actions:
                        result_recommended_actions = actions
                    if cls is not None:
                        result_classification = cls
                        # Reanalyser only corrects the classification; preserve the
                        # original explanation from the data/search source so that
                        # internal reanalyser reasoning never surfaces to the user.
                        if tool_name != "reanalyse":
                            result_explanation = expl
                            result_sources = srcs
                        if cls in ("TRUE", "UNCERTAIN"):
                            logger.info(
                                "[%s] ★ DECISIVE ANSWER from %s — classification=%s",
                                req_id, tool_label, cls,
                            )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": observation,
                        }
                    )

                continue

            logger.info("[%s]   [ANSWER] Reached final answer on iteration %d | preview=%r",
                        req_id, iteration, content[:200])

            # Guard: if the model returned raw JSON or data instead of prose,
            # reject it and continue the loop so the model can self-correct.
            if _is_raw_data(content):
                logger.warning(
                    "[%s]   Final answer appears to contain raw data — injecting correction prompt",
                    req_id,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response contained raw data or JSON. "
                            "Please rewrite your answer as plain, human-readable prose "
                            "in the user's native language. Do not include any JSON, "
                            "data fields, file contents, or structured data."
                        ),
                    }
                )
                continue

            answer_step = AgentStep(StepType.ANSWER, content)
            steps.append(answer_step)
            if verbose:
                print(answer_step)

            return AgentResult(
                answer=content,
                steps=steps,
                iterations=iteration,
                classification=result_classification,
                explanation=result_explanation,
                sources=result_sources,
                recommended_next_actions=result_recommended_actions,
            )

        logger.warning("[%s] Max iterations (%d) reached without a final answer",
                       req_id, self.max_iterations)
        last_answer = next(
            (s.content for s in reversed(steps) if s.step_type == StepType.THOUGHT),
            "Reached maximum iterations without a final answer.",
        )
        return AgentResult(
            answer=last_answer,
            steps=steps,
            iterations=self.max_iterations,
            classification=result_classification,
            explanation=result_explanation,
            sources=result_sources,
            recommended_next_actions=result_recommended_actions,
        )
