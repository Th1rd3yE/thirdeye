"""
Tool implementations for ThirdEye's two verification sources.

get_from_data_sources  — real HTTP call to the DS endpoint (generate-and-fetch)
get_from_vertex_search — real HTTP call to the Vertex AI Search endpoint
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from agent.tools import ParameterProperty, Tool, ToolSchema

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_DS_PAYLOAD_SCHEMA = ToolSchema(
    properties={
        "question": ParameterProperty(
            type="string",
            description="The original user question to verify.",
        ),
        "keywords": ParameterProperty(
            type="array",
            items={"type": "string"},
            description="Keywords extracted from the question.",
        ),
        "country": ParameterProperty(
            type="string",
            description="Country context of the query, e.g. 'Singapore' or 'Global'.",
        ),
        "languages": ParameterProperty(
            type="array",
            items={"type": "string"},
            description="Languages to search in, e.g. ['English', 'Chinese'].",
        ),
        "date": ParameterProperty(
            type="string",
            description="Query date in YYYY-MM-DD format, e.g. '2026-03-10'.",
        ),
    },
    required=["question", "country", "languages"],
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
        question: str = "",
        keywords: list[str] | None = None,
        country: str = "Global",
        languages: list[str] | None = None,
        date: str = "",
        **kwargs: object,
    ) -> str:
        payload = {
            "question": question,
            "keywords": keywords or [],
            "country": country,
            "languages": languages or ["English"],
            "date": date,
        }
        try:
            resp = requests.post(self._ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except requests.RequestException as exc:
            return json.dumps(
                {
                    "source": "ThirdEye Internal Data Sources",
                    "classification": "FALSE",
                    "confidence": 0.0,
                    "data": None,
                    "reason": f"Data sources endpoint error: {exc}",
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
                "source": "ThirdEye Internal Data Sources",
                "classification": classification,
                "confidence": data.get("confidence", 0.9 if classification == "TRUE" else 0.1),
                "data": data.get("data"),
                "explanation": data.get("explanation_en", ""),
                "explanation_native": data.get("explanation_native", ""),
                "reason": (
                    "Internal data sources confirmed the claim."
                    if classification == "TRUE"
                    else data.get("explanation_en")
                    or "No matching verified records found in internal database."
                ),
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
        try:
            resp = requests.post(self._ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except requests.RequestException as exc:
            return json.dumps(
                {
                    "source": "Google Vertex AI Search",
                    "classification": False,
                    "confidence": 0.0,
                    "results": [],
                    "reason": f"Vertex AI Search endpoint error: {exc}",
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
                "confidence": data.get("confidence", 0.8 if classification == "TRUE" else 0.1),
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
