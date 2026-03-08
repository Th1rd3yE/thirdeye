"""
Tool implementations for ThirdEye's two verification sources.

get_from_data_sources        — real HTTP call to the DS endpoint (generate-and-fetch)
get_from_vertex_search       — real HTTP call to the Vertex AI Search endpoint
get_recommended_next_actions — LLM-generated recommended actions based on classification
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from groq import Groq

from agent.tools import ParameterProperty, Tool, ToolSchema

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_DS_PAYLOAD_SCHEMA = ToolSchema(
    properties={
        "country": ParameterProperty(
            type="string",
            description="Country context of the query, e.g. 'China' or 'Global'.",
        ),
        "date": ParameterProperty(
            type="string",
            description="Query date, e.g. 'March 2026'.",
        ),
        "questions": ParameterProperty(
            type="string",
            description="The original user question to verify.",
        ),
        "key_words": ParameterProperty(
            type="array",
            items={"type": "string"},
            description="Keywords extracted from the question.",
        ),
        "native_language": ParameterProperty(
            type="string",
            description="User's native language, e.g. 'Chinese (Simplified)' or 'English'.",
        ),
        "english_language": ParameterProperty(
            type="string",
            description="Language used for source analysis, typically 'English'.",
        ),
    },
    required=["country", "questions", "native_language"],
)

_VERTEX_PAYLOAD_SCHEMA = ToolSchema(
    properties={
        "country": ParameterProperty(
            type="string",
            description="Country context of the query, e.g. 'China' or 'Global'.",
        ),
        "date": ParameterProperty(
            type="string",
            description="Query date, e.g. 'March 2026'.",
        ),
        "questions": ParameterProperty(
            type="string",
            description="The original user question to verify.",
        ),
        "key_words": ParameterProperty(
            type="array",
            items={"type": "string"},
            description="Keywords extracted from the question.",
        ),
        "native_language": ParameterProperty(
            type="string",
            description="User's native language, e.g. 'Chinese (Simplified)' or 'English'.",
        ),
        "english_language": ParameterProperty(
            type="string",
            description="Language used for source analysis, typically 'English'.",
        ),
    },
    required=["country", "questions", "native_language"],
)


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

class GetFromDataSourcesTool(Tool):
    """Queries internal verified data sources for news/information credibility."""

    name = "get_from_data_sources"
    description = (
        "Queries ThirdEye's internal verified data sources (institutional databases "
        "and curated records) using the provided query fields. "
        "Returns structured data with a 'classification' key: "
        "true = information is credible and verified; false = no match found. "
        "ALWAYS call this tool FIRST before get_from_vertex_search."
    )
    schema = _DS_PAYLOAD_SCHEMA

    _ENDPOINT = os.environ.get(
        "DS_ENDPOINT", "https://hands-38uo.onrender.com/generate-and-fetch"
    )

    def run(
        self,
        *args: object,
        country: str = "Global",
        date: str = "",
        questions: str = "",
        key_words: list[str] | None = None,
        native_language: str = "English",
        english_language: str = "English",
        **kwargs: object,
    ) -> str:
        payload = {
            "country": country,
            "date": date,
            "context": questions,
            "key_words": key_words or [],
            "native_language": native_language,
            "english_language": english_language,
        }
        try:
            resp = requests.post(self._ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            print(data)
        except requests.RequestException as exc:
            return json.dumps(
                {
                    "claim": questions,
                    "classification": "FALSE",
                    "confidence": 0.0,
                    "explanation_en": f"Data sources endpoint error: {exc}",
                    "explanation_native": f"Data sources endpoint error: {exc}",
                    "sources": [],
                },
                ensure_ascii=False,
                indent=2,
            )

        raw_classification = str(data.get("classification", "FALSE")).upper()
        if raw_classification == "TRUE":
            classification = "TRUE"
        elif raw_classification == "UNCERTAIN":
            classification = "UNCERTAIN"
        else:
            classification = "FALSE"

        return json.dumps(
            {
                "claim": questions,
                "classification": classification,
                "confidence": data.get("confidence", 0.9 if classification == "TRUE" else 0.1),
                "explanation_en": data.get("explanation_en", ""),
                "explanation_native": data.get("explanation_native", ""),
                "sources": data.get("sources", []),
            },
            ensure_ascii=False,
            indent=2,
        )


class GetFromVertexSearchTool(Tool):
    """Queries Google Vertex AI Search (managed web index) for credibility."""

    name = "get_from_vertex_search"
    description = (
        "Queries Google Vertex AI Search (a managed living web index) using the provided "
        "query fields. Returns top-K relevant snippets with a 'classification' key: "
        "true = credible web sources found; false = no credible sources exist. "
        "Call this ONLY when get_from_data_sources returns classification=false."
    )
    schema = _VERTEX_PAYLOAD_SCHEMA

    _ENDPOINT = os.environ.get(
        "VERTEX_SEARCH_ENDPOINT", "https://hands-b1s0.onrender.com/verify"
    )

    def run(
        self,
        *args: object,
        country: str = "Global",
        date: str = "",
        questions: str = "",
        key_words: list[str] | None = None,
        native_language: str = "English",
        english_language: str = "English",
        **kwargs: object,
    ) -> str:
        context = " ".join(key_words) if key_words else questions
        payload = {
            "country": country,
            "date": date,
            "questions": questions,
            "key_words": key_words or [],
            "native_language": native_language,
            "english_language": english_language,
            "context": context,
        }
        max_retries = 4
        backoff = 5.0  # seconds; doubles each retry
        last_exc: requests.RequestException | None = None
        data: dict[str, Any] | None = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(self._ENDPOINT, json=payload, timeout=60)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    time.sleep(retry_after)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2

        if data is None:
            return json.dumps(
                {
                    "source": "Google Vertex AI Search",
                    "classification": False,
                    "confidence": 0.0,
                    "results": [],
                    "reason": f"Vertex AI Search endpoint error: {last_exc}",
                },
                ensure_ascii=False,
                indent=2,
            )

        raw_classification = str(data.get("classification", "FALSE")).upper()
        # Normalise to one of the three valid values
        if raw_classification == "TRUE":
            classification = "TRUE"
        elif raw_classification == "UNCERTAIN":
            classification = "UNCERTAIN"
        else:
            classification = "FALSE"

        sources = data.get("sources", [])
        results = (
            [
                {"url": url, "relevance_score": 1.0}
                for url in sources
                if "vertexaisearch" in url or "google.com" not in url
            ]
            if classification == "TRUE"
            else []
        )
        if classification == "TRUE" and not results:
            results = [{"url": url} for url in sources[:5]]

        explanation_native = data.get("explanation_native", "") if payload.get("native_language") != "English" else data.get("explanation_en", "")
        return json.dumps(
            {
                "source": "Google Vertex AI Search",
                "classification": classification,
                "confidence": data.get("truth_score"),
                "results": results,
                "explanation": data.get("explanation_en", ""),
                "explanation_native": explanation_native,
                "reason": (
                    "Vertex AI Search found credible sources confirming the claim."
                    if classification == "TRUE"
                    else data.get("explanation_en")
                    or "Vertex AI Search returned no credible matching sources."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# RecommendedNextActionTool
# ---------------------------------------------------------------------------

_RECOMMENDED_ACTION_SCHEMA = ToolSchema(
    properties={
        "question": ParameterProperty(
            type="string",
            description="The original user question that was verified.",
        ),
        "classification": ParameterProperty(
            type="string",
            description="Verification result: 'TRUE', 'FALSE', or 'UNCERTAIN'.",
            enum=["TRUE", "FALSE", "UNCERTAIN"],
        ),
        "explanation": ParameterProperty(
            type="string",
            description="Explanation from the verification step.",
        ),
        "native_language": ParameterProperty(
            type="string",
            description="User's native language for the response, e.g. 'English' or 'Chinese (Simplified)'.",
        ),
    },
    required=["question", "classification"],
)

_ACTIONS_PROMPT = """\
You are an advisor helping users respond to news or information they just had verified.

Given:
- Question: {question}
- Verification result: {classification}
- Explanation: {explanation}

Produce exactly 3 brief, practical recommended actions the user should take.
Each action must be a single sentence (≤ 20 words).
Respond ONLY with a valid JSON array of 3 strings, nothing else.
Example: ["Action one.", "Action two.", "Action three."]
"""


class RecommendedNextActionTool(Tool):
    """Generates 3 recommended next actions based on the verification result."""

    name = "get_recommended_next_actions"
    description = (
        "Generates 3 short, practical recommended actions for the user based on the "
        "classification result (TRUE / FALSE / UNCERTAIN) and the verification explanation. "
        "ALWAYS call this tool as the LAST step after the classification is determined."
    )
    schema = _RECOMMENDED_ACTION_SCHEMA

    _MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def run(
        self,
        *args: object,
        question: str = "",
        classification: str = "FALSE",
        explanation: str = "",
        native_language: str = "English",
        **kwargs: object,
    ) -> str:
        prompt = _ACTIONS_PROMPT.format(
            question=question,
            classification=classification,
            explanation=explanation or "No additional explanation provided.",
        )
        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model=self._MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            raw = (response.choices[0].message.content or "").strip()
            actions: list[str] = json.loads(raw)
            if not isinstance(actions, list):
                raise ValueError("Expected a JSON array")
            actions = [str(a) for a in actions[:3]]
        except Exception as exc:
            actions = _fallback_actions(classification)

        return json.dumps(
            {"recommended_next_actions": actions},
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# ReanalyserTool
# ---------------------------------------------------------------------------

_REANALYSER_SCHEMA = ToolSchema(
    properties={
        "question": ParameterProperty(
            type="string",
            description="The original user question that was verified.",
        ),
        "classification": ParameterProperty(
            type="string",
            description="The classification from the verification tool: 'TRUE', 'FALSE', or 'UNCERTAIN'.",
            enum=["TRUE", "FALSE", "UNCERTAIN"],
        ),
        "explanation": ParameterProperty(
            type="string",
            description="The explanation text returned by the verification tool.",
        ),
    },
    required=["question", "classification", "explanation"],
)

_REANALYSER_PROMPT = """\
You are a fact-checking validation assistant. Your job is to determine whether a \
given classification (TRUE/FALSE/UNCERTAIN) is logically consistent with the \
explanation and the original question.

Question: {question}
Classification: {classification}
Explanation: {explanation}

Analysis rules:
1. Read the question carefully — is it asking whether something POSITIVE or NEGATIVE is true?
2. Read the explanation — does it support or contradict the claim in the question?
3. A classification of TRUE means "yes, the claim in the question is correct".
   A classification of FALSE means "no, the claim in the question is incorrect".

Common mistake to catch:
  - Question: "Is it true that [place/thing] is BAD?"
    Explanation: "[place/thing] is actually diverse/good/well-regarded."
    → The explanation CONTRADICTS the claim, so the answer should be FALSE, not TRUE.

Respond ONLY with a valid JSON object with exactly these fields:
{{
  "classification": "TRUE" | "FALSE" | "UNCERTAIN",
  "changed": true | false,
  "reason": "One or two sentences describing what the explanation actually says about the topic — factual content only. Do NOT mention the classification values, do NOT say things like 'the classification should be X rather than Y', and do NOT describe your reasoning process."
}}
"""


class ReanalyserTool(Tool):
    """Cross-checks a classification against the explanation and original question."""

    name = "reanalyse"
    description = (
        "Validates whether the classification (TRUE/FALSE/UNCERTAIN) is logically "
        "consistent with the explanation and the original question. "
        "If the explanation contradicts the classification, this tool corrects it. "
        "ALWAYS call this tool AFTER obtaining a classification but BEFORE calling "
        "get_recommended_next_actions."
    )
    schema = _REANALYSER_SCHEMA

    _MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def run(
        self,
        *args: object,
        question: str = "",
        classification: str = "FALSE",
        explanation: str = "",
        **kwargs: object,
    ) -> str:
        prompt = _REANALYSER_PROMPT.format(
            question=question,
            classification=classification,
            explanation=explanation or "No explanation provided.",
        )
        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model=self._MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = (response.choices[0].message.content or "").strip()
            result: dict[str, Any] = json.loads(raw)
            final_cls = str(result.get("classification", classification)).upper()
            if final_cls not in {"TRUE", "FALSE", "UNCERTAIN"}:
                final_cls = classification
            changed = bool(result.get("changed", False))
            reason = str(result.get("reason", ""))
        except Exception as exc:
            return json.dumps(
                {
                    "classification": classification,
                    "changed": False,
                    "reason": f"Reanalysis failed, keeping original classification: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            {
                "classification": final_cls,
                "changed": changed,
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        )


def _fallback_actions(classification: str) -> list[str]:
    """Rule-based fallback when LLM call fails."""
    if classification == "TRUE":
        return [
            "You can share this information with confidence as it has been verified.",
            "Refer to the cited sources for more details.",
            "Stay updated by following reputable news outlets on this topic.",
        ]
    if classification == "UNCERTAIN":
        return [
            "Treat this information with caution until further verified.",
            "Cross-check with multiple reputable sources before sharing.",
            "Avoid making decisions based solely on this unconfirmed information.",
        ]
    return [
        "Do not share this information as it could not be verified.",
        "Seek information from official or trusted government/news sources.",
        "Report the claim to a fact-checking organisation if you encounter it again.",
    ]
