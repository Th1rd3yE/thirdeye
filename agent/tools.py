from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParameterProperty:
    type: str
    description: str
    enum: list[str] | None = None
    items: dict[str, Any] | None = None  # used when type == "array"


@dataclass
class ToolSchema:
    """JSON-Schema-style description of a tool's input parameters."""

    properties: dict[str, ParameterProperty]
    required: list[str] = field(default_factory=list)


class Tool(ABC):
    """Base class for all ReAct tools."""

    name: str
    description: str
    schema: ToolSchema

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text observation."""

    def to_groq_schema(self) -> dict[str, Any]:
        """Return the tool definition dict expected by the Groq chat API."""
        properties: dict[str, Any] = {}
        for prop_name, prop in self.schema.properties.items():
            spec: dict[str, Any] = {
                "type": prop.type,
                "description": prop.description,
            }
            if prop.enum:
                spec["enum"] = prop.enum
            if prop.type == "array" and prop.items:
                spec["items"] = prop.items
            properties[prop_name] = spec

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.schema.required,
                },
            },
        }


class ToolRegistry:
    """Holds all registered tools and converts them for the Groq API."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"No tool named '{name}' is registered.")
        return self._tools[name]

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_groq_tools(self) -> list[dict[str, Any]]:
        return [t.to_groq_schema() for t in self._tools.values()]
