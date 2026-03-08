"""
ThirdEye — Federated Agentic News Verification System
Run:  python main.py

Pipeline per query
──────────────────
1. Local pre-processing  →  language detection + keyword extraction
2. Build standardized JSON payload (no LLM needed for this step)
3. Agent calls get_from_data_sources  →  check classification
4. If classification=false, agent calls get_from_vertex_search  →  check classification
5. If both fail  →  report unverifiable / false

Demo scenarios
──────────────
Scenario A: Chinese query  →  Tool 1 (data sources) classifies TRUE  →  answer in Chinese
Scenario B: English query  →  Tool 1 FALSE, Tool 2 (Vertex) classifies TRUE  →  answer in English
Scenario C: Vague query    →  both tools FALSE  →  unverifiable response
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

from agent import (
    ReactAgent,
    ToolRegistry,
    build_query_payload,
    GetFromDataSourcesTool,
    GetFromVertexSearchTool,
    ReanalyserTool,
    RecommendedNextActionTool,
)

load_dotenv()


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GetFromDataSourcesTool())
    registry.register(GetFromVertexSearchTool())
    registry.register(ReanalyserTool())
    registry.register(RecommendedNextActionTool())
    return registry


def verify(user_input: str, agent: ReactAgent, verbose: bool = True) -> None:
    """Pre-process user input locally, then run the verification agent."""

    # ── Step 1 & 2: Local processing ─────────────────────────────────────
    payload = build_query_payload(user_input)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    print("\n" + "━" * 60)
    print("USER INPUT:")
    print(f"  {user_input}")
    print("\nLOCAL PRE-PROCESSING OUTPUT:")
    print(payload_json)
    print("━" * 60)

    # ── Step 3+: Agent takes over ─────────────────────────────────────────
    agent_message = (
        f"Please verify the following news/information claim.\n\n"
        f"Use this pre-analyzed payload when calling the tools "
        f"(pass each field as a separate argument):\n"
        f"{payload_json}\n\n"
        f"Original user question: {user_input}\n\n"
        f"Follow the verification workflow and respond in: {payload['native_language']}"
    )

    result = agent.run(agent_message, verbose=verbose)

    print(f"\n✔ Verification complete in {result.iterations} iteration(s).")
    print("━" * 60 + "\n")


def main() -> None:
    registry = build_registry()
    agent = ReactAgent(registry=registry)

    scenarios = [
        # Scenario A: Chinese → country=China → Tool 1 classifies TRUE
        "这条关于中国2026年GDP增长5.2%的新闻是否属实？",

        # Scenario B: English, question > 30 chars → Tool 1 FALSE, Tool 2 classifies TRUE
        "Is it true that the G20 summit held in Tokyo in February 2026 reached a landmark climate agreement?",

        # Scenario C: Vague English, ≤ 30 chars → both tools FALSE
        "Is bigfoot real?",
    ]

    for query in scenarios:
        verify(query, agent, verbose=True)


if __name__ == "__main__":
    main()
