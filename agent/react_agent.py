from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from agent.tools import ToolRegistry

# System prompt that instructs Gemini to follow the ReAct pattern.
# When function calling is enabled Gemini's "Thought" is implicit in the
# function it chooses to call; we surface it explicitly in the step log.
SYSTEM_PROMPT = """You are a helpful AI assistant that solves tasks step by step.
You have access to a set of tools. Use them whenever they help you give a better,
more accurate answer.

Follow this reasoning loop:
1. Think about what you know and what you still need to find out.
2. If you need more information, call the appropriate tool.
3. After receiving the tool result, incorporate it into your reasoning.
4. Repeat until you have enough information to answer confidently.
5. Provide a clear, concise final answer.

Always be transparent about your reasoning process."""


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


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0

    def pretty_print(self) -> None:
        print("\n" + "=" * 60)
        print("REACT AGENT TRACE")
        print("=" * 60)
        for i, step in enumerate(self.steps, 1):
            print(f"\nStep {i}: {step}")
            print("-" * 40)
        print(f"\nFINAL ANSWER:\n{self.answer}")
        print("=" * 60)


class ReactAgent:
    """
    ReAct loop backed by a Vertex AI Gemini model.

    The loop runs up to `max_iterations` rounds of:
      Thought → Action (tool call) → Observation
    and terminates when the model produces a plain-text answer (no tool call).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        project: str | None = None,
        location: str | None = None,
        model_name: str | None = None,
        max_iterations: int = 10,
        temperature: float = 0.0,
    ) -> None:
        self.registry = registry
        self.max_iterations = max_iterations

        project = project or os.environ["GOOGLE_CLOUD_PROJECT"]
        location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        model_name = model_name or os.environ.get(
            "VERTEX_AI_MODEL", "gemini-2.0-flash-001"
        )

        vertexai.init(project=project, location=location)

        self._model = GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            tools=[registry.to_vertex_tool()],
            generation_config=GenerationConfig(temperature=temperature),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str, verbose: bool = True) -> AgentResult:
        """Execute the ReAct loop for the given user query."""
        steps: list[AgentStep] = []
        history: list[Content] = []

        if verbose:
            print(f"\nQuery: {query}\n{'─' * 60}")

        history.append(Content(role="user", parts=[Part.from_text(query)]))

        for iteration in range(1, self.max_iterations + 1):
            response = self._model.generate_content(history)
            candidate = response.candidates[0]
            model_content = candidate.content

            # Accumulate the model turn into history
            history.append(model_content)

            function_calls = [
                part for part in model_content.parts
                if part.function_call and part.function_call.name
            ]
            text_parts = [
                part.text
                for part in model_content.parts
                if hasattr(part, "text") and part.text
            ]

            # ── Thought ───────────────────────────────────────────────
            if text_parts:
                thought_text = "\n".join(text_parts)
                thought_step = AgentStep(StepType.THOUGHT, thought_text)
                steps.append(thought_step)
                if verbose:
                    print(thought_step)

            # ── Action + Observation ──────────────────────────────────
            if function_calls:
                tool_responses: list[Part] = []

                for fc in function_calls:
                    tool_name = fc.function_call.name
                    tool_args = dict(fc.function_call.args)

                    action_step = AgentStep(
                        StepType.ACTION,
                        f"Calling tool with args: {tool_args}",
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                    steps.append(action_step)
                    if verbose:
                        print(action_step)

                    # Execute the tool
                    try:
                        tool = self.registry.get(tool_name)
                        observation = tool.run(**tool_args)
                    except Exception as exc:  # pylint: disable=broad-except
                        observation = f"Tool error: {exc}"

                    obs_step = AgentStep(StepType.OBSERVATION, observation)
                    steps.append(obs_step)
                    if verbose:
                        print(obs_step)

                    tool_responses.append(
                        Part.from_function_response(
                            name=tool_name,
                            response={"result": observation},
                        )
                    )

                # Feed all tool results back to the model in one turn
                history.append(Content(role="user", parts=tool_responses))
                continue  # next iteration

            # ── Final Answer ──────────────────────────────────────────
            # No function calls → the model is done
            answer = "\n".join(
                part.text
                for part in model_content.parts
                if hasattr(part, "text") and part.text
            )
            answer_step = AgentStep(StepType.ANSWER, answer)
            steps.append(answer_step)
            if verbose:
                print(answer_step)

            return AgentResult(answer=answer, steps=steps, iterations=iteration)

        # Exceeded max_iterations – return whatever the last model text was
        last_answer = next(
            (s.content for s in reversed(steps) if s.step_type == StepType.THOUGHT),
            "Reached maximum iterations without a final answer.",
        )
        return AgentResult(answer=last_answer, steps=steps, iterations=self.max_iterations)
