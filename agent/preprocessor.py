"""Local pre-processing: language detection and keyword extraction via LLM."""

from __future__ import annotations

import datetime
import json
import os

from groq import Groq

_ANALYSIS_PROMPT = """\
Analyse the text below and respond with a single JSON object — no markdown, no extra text.

Fields to return:
- "native_language": full language name (e.g. "Burmese", "Chinese (Simplified)", "English")
- "country": most likely country or region the text relates to (e.g. "Myanmar", "China", "Global")
- "keywords": array of up to 6 concise, meaningful keywords extracted from the text

Text: {text}"""


def _call_llm(prompt: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def build_query_payload(user_input: str) -> dict:
    """Build the standardised JSON payload from raw user input using a single LLM call."""
    raw = _call_llm(_ANALYSIS_PROMPT.format(text=user_input))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    native_language = parsed.get("native_language") or "English"
    country = parsed.get("country") or "Global"
    keywords = parsed.get("keywords")
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords[:6]]

    date_str = datetime.datetime.now().strftime("%B %Y")

    return {
        "country": country,
        "date": date_str,
        "questions": user_input,
        "key_words": keywords,
        "native_language": native_language,
        "english_language": "English",
    }
