"""
ThirdEye – ReAct loop demo
Run:  python main.py
"""

from __future__ import annotations

import math
import datetime

from dotenv import load_dotenv

from agent import ReactAgent, Tool, ToolRegistry
from agent.tools import ParameterProperty, ToolSchema

load_dotenv()


# ---------------------------------------------------------------------------
# Built-in demo tools
# ---------------------------------------------------------------------------

class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluates a mathematical expression and returns the numeric result. "
        "Supports standard Python math operators and the math module. "
        "Use this for any arithmetic, algebra, or numeric computation."
    )
    schema = ToolSchema(
        properties={
            "expression": ParameterProperty(
                type="string",
                description="A valid Python math expression, e.g. '2 ** 10' or 'math.sqrt(144)'.",
            )
        },
        required=["expression"],
    )

    def run(self, *args: object, expression: str = "", **kwargs: object) -> str:  # pylint: disable=arguments-differ
        try:
            # eval is intentional here; __builtins__ is stripped for safety
            result = eval(expression, {"__builtins__": {}}, {"math": math})  # pylint: disable=eval-used
            return str(result)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Error evaluating expression: {exc}"


class CurrentTimeTool(Tool):
    name = "get_current_time"
    description = (
        "Returns the current date and time in UTC, and optionally in a requested timezone."
    )
    schema = ToolSchema(
        properties={
            "timezone": ParameterProperty(
                type="string",
                description="IANA timezone name, e.g. 'America/New_York'. Defaults to 'UTC'.",
            )
        },
        required=[],
    )

    def run(self, *args: object, timezone: str = "UTC", **kwargs: object) -> str:  # pylint: disable=arguments-differ
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if timezone.upper() == "UTC":
            return now_utc.strftime("UTC: %Y-%m-%d %H:%M:%S %Z")
        # Basic named offset support without third-party deps
        offsets: dict[str, int] = {
            "America/New_York": -5,
            "America/Los_Angeles": -8,
            "Europe/London": 0,
            "Europe/Paris": 1,
            "Asia/Tokyo": 9,
            "Asia/Singapore": 8,
            "Australia/Sydney": 11,
        }
        offset_h = offsets.get(timezone)
        if offset_h is None:
            return (
                f"UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(timezone '{timezone}' not recognised; showing UTC)"
            )
        local = now_utc + datetime.timedelta(hours=offset_h)
        return f"{timezone}: {local.strftime('%Y-%m-%d %H:%M:%S')} (UTC{offset_h:+d})"


class UnitConverterTool(Tool):
    name = "unit_converter"
    description = "Converts a numeric value between common units of measurement."
    schema = ToolSchema(
        properties={
            "value": ParameterProperty(
                type="number", description="The numeric value to convert."
            ),
            "from_unit": ParameterProperty(
                type="string",
                description="Source unit (e.g. 'km', 'miles', 'kg', 'lbs', 'celsius', 'fahrenheit').",
            ),
            "to_unit": ParameterProperty(
                type="string",
                description="Target unit (e.g. 'km', 'miles', 'kg', 'lbs', 'celsius', 'fahrenheit').",
            ),
        },
        required=["value", "from_unit", "to_unit"],
    )

    # Conversion factors relative to a SI base unit
    _FACTORS: dict[str, tuple[str, float]] = {
        # length (base: metre)
        "m": ("length", 1.0),
        "km": ("length", 1000.0),
        "miles": ("length", 1609.344),
        "mile": ("length", 1609.344),
        "ft": ("length", 0.3048),
        "feet": ("length", 0.3048),
        "inch": ("length", 0.0254),
        "inches": ("length", 0.0254),
        # mass (base: gram)
        "g": ("mass", 1.0),
        "kg": ("mass", 1000.0),
        "lbs": ("mass", 453.592),
        "lb": ("mass", 453.592),
        "oz": ("mass", 28.3495),
    }

    def run(self, *args: object, value: float = 0, from_unit: str = "", to_unit: str = "", **kwargs: object) -> str:  # pylint: disable=arguments-differ
        fu = from_unit.lower()
        tu = to_unit.lower()

        # Temperature handled separately (non-linear)
        temp_result = self._convert_temperature(value, fu, tu)
        if temp_result is not None:
            return temp_result

        if fu not in self._FACTORS or tu not in self._FACTORS:
            return f"Unknown units: '{from_unit}' or '{to_unit}'."
        cat_from, factor_from = self._FACTORS[fu]
        cat_to, factor_to = self._FACTORS[tu]
        if cat_from != cat_to:
            return f"Cannot convert between '{from_unit}' ({cat_from}) and '{to_unit}' ({cat_to})."
        result = value * factor_from / factor_to
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    @staticmethod
    def _convert_temperature(value: float, fu: str, tu: str) -> str | None:
        if fu == "celsius" and tu == "fahrenheit":
            return f"{value}°C = {value * 9/5 + 32:.4g}°F"
        if fu == "fahrenheit" and tu == "celsius":
            return f"{value}°F = {(value - 32) * 5/9:.4g}°C"
        if fu == "celsius" and tu == "kelvin":
            return f"{value}°C = {value + 273.15:.4g} K"
        if fu == "kelvin" and tu == "celsius":
            return f"{value} K = {value - 273.15:.4g}°C"
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())
    registry.register(UnitConverterTool())
    return registry


def main() -> None:
    registry = build_registry()
    agent = ReactAgent(registry=registry)

    queries = [
        "What is 2 to the power of 32, and what is the square root of that result?",
        "Convert 100 miles to kilometres, then convert that distance from km to metres.",
        "What time is it right now in Tokyo and in New York?",
    ]

    for query in queries:
        result = agent.run(query, verbose=True)
        print(f"\nCompleted in {result.iterations} iteration(s).\n")


if __name__ == "__main__":
    main()
