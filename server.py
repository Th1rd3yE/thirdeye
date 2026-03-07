"""
ThirdEye — FastAPI server
Run:  uvicorn server:app --reload

Endpoints
─────────
POST /verify   { "query": "<claim to check>" }
GET  /health   liveness probe
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import (
    ReactAgent,
    ToolRegistry,
    build_query_payload,
    GetFromDataSourcesTool,
    GetFromVertexSearchTool,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("thirdeye")

# ── Shared agent (initialised once at startup) ────────────────────────────────

_registry: ToolRegistry | None = None
_agent: ReactAgent | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _registry, _agent
    _registry = ToolRegistry()
    _registry.register(GetFromDataSourcesTool())
    _registry.register(GetFromVertexSearchTool())
    _agent = ReactAgent(registry=_registry)
    logger.info("ThirdEye agent initialised.")
    yield
    _registry = None
    _agent = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ThirdEye",
    description="Federated Agentic News Verification API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="News claim or question to verify")


class AgentStepOut(BaseModel):
    step_type: str
    tool_name: str | None
    content: str


class VerifyResult(BaseModel):
    classification: str | None  # "TRUE" | "FALSE" | "UNCERTAIN"
    explanation: str
    sources: list[str]


class VerifyResponse(BaseModel):
    result: VerifyResult
    answer: str
    iterations: int
    payload: dict[str, Any]
    steps: list[AgentStepOut]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    logger.debug("Health check requested")
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse, tags=["verification"])
async def verify(body: VerifyRequest, request: Request) -> VerifyResponse:
    """
    Verify a news claim through the ReAct agent pipeline.

    1. Pre-processes the query locally (language detection + keyword extraction).
    2. Runs the ReAct agent which calls internal data sources and/or Vertex Search.
    3. Returns the final answer, intermediate steps, and the pre-processed payload.
    """
    req_id = uuid.uuid4().hex[:8]
    client_ip = request.client.host if request.client else "unknown"
    start_time = time.perf_counter()

    logger.info("[%s] POST /verify from %s | query=%r", req_id, client_ip, body.query)

    if _agent is None:
        logger.error("[%s] Agent not ready — returning 503", req_id)
        raise HTTPException(status_code=503, detail="Agent not ready.")

    # ── Local pre-processing ──────────────────────────────────────────────────
    payload = build_query_payload(body.query)
    payload_json = json.dumps(payload, ensure_ascii=False)

    logger.info(
        "[%s] Payload built | language=%s | country=%s | keywords=%s",
        req_id,
        payload.get("native_language"),
        payload.get("country"),
        payload.get("keywords"),
    )

    agent_message = (
        f"Please verify the following news/information claim.\n\n"
        f"Use this pre-analyzed payload when calling the tools "
        f"(pass each field as a separate argument):\n"
        f"{payload_json}\n\n"
        f"Original user question: {body.query}\n\n"
        f"Follow the verification workflow and respond in: {payload['native_language']}"
    )

    # ── Agent run (blocking — wrap in thread pool for production) ─────────────
    logger.info("[%s] Handing off to ReAct agent", req_id)
    try:
        result = _agent.run(agent_message, verbose=False, req_id=req_id)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("[%s] Agent error after %.2fs", req_id, elapsed)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - start_time
    logger.info(
        "[%s] Request complete | iterations=%d | total_elapsed=%.2fs",
        req_id,
        result.iterations,
        elapsed,
    )

    steps_out = [
        AgentStepOut(
            step_type=step.step_type.value,
            tool_name=step.tool_name,
            content=step.content,
        )
        for step in result.steps
    ]

    return VerifyResponse(
        result=VerifyResult(
            classification=result.classification,
            explanation=result.explanation,
            sources=result.sources,
        ),
        answer=result.answer,
        iterations=result.iterations,
        payload=payload,
        steps=steps_out,
    )
