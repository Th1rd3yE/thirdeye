# thirdeye

Backend of ThirdEye — a ReAct (Reasoning + Acting) agent powered by Google Vertex AI.

## Architecture

```
thirdeye/
├── agent/
│   ├── __init__.py
│   ├── tools.py          # Tool base class, ToolSchema, ToolRegistry
│   └── react_agent.py    # Core ReAct loop (Vertex AI Gemini)
├── main.py               # Demo with built-in tools
├── requirements.txt
└── .env.example
```

## How it works

The agent follows the **ReAct** loop (Yao et al., 2022):

```
User Query
    │
    ▼
 [Thought]  ← Model reasons about what it knows / needs
    │
    ▼
 [Action]   ← Model calls a tool (Vertex AI function calling)
    │
    ▼
[Observation] ← Tool result is returned to the model
    │
    └──► repeat until no tool call → [Answer]
```

Vertex AI's native **function calling** maps naturally to the Action step.  
The model's reasoning text between calls is surfaced as explicit Thought steps.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your GCP project ID and desired region
```

### 3. Authenticate

Either set `GOOGLE_APPLICATION_CREDENTIALS` to a service account key, or use:

```bash
gcloud auth application-default login
```

### 4. Run the demo

```bash
python main.py
```

## Adding your own tools

Subclass `Tool`, declare `name`, `description`, `schema`, and implement `run`:

```python
from agent.tools import Tool, ToolSchema, ParameterProperty

class WeatherTool(Tool):
    name = "get_weather"
    description = "Returns the current weather for a given city."
    schema = ToolSchema(
        properties={
            "city": ParameterProperty(type="string", description="City name"),
        },
        required=["city"],
    )

    def run(self, *args, city: str = "", **kwargs) -> str:
        # call your weather API here
        return f"Sunny, 24°C in {city}"
```

Then register it:

```python
registry = ToolRegistry()
registry.register(WeatherTool())
agent = ReactAgent(registry=registry)
result = agent.run("What's the weather in Tokyo?")
```
