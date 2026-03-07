# ThirdEye — Federated Agentic News Verification System

ThirdEye is a multilingual, agentic news and information verification system. It uses a **ReAct (Reasoning + Acting)** loop powered by a large language model to systematically determine whether a news claim is credible — responding to the user in their native language.

---

## How It Works

ThirdEye follows a strict, deterministic verification pipeline for every query:

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  1. Local Pre-processing    │  Language detection + keyword extraction (no LLM needed)
│     → Build JSON Payload    │  Detect language, infer country, extract keywords, stamp date
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. get_from_data_sources   │  Query ThirdEye's internal verified institutional databases
│     classification=true?   ├─── YES ──► Return verified result to user (in native language)
└────────────┬────────────────┘
             │ NO
             ▼
┌─────────────────────────────┐
│  3. get_from_vertex_search  │  Query Google Vertex AI Search (live web index)
│     classification=true?   ├─── YES ──► Return verified result to user (in native language)
└────────────┬────────────────┘
             │ NO
             ▼
     Both sources failed
     → Report unverifiable / false (in native language)
```

The agent never skips steps and never calls Vertex Search before the internal data sources.

---

## Architecture

```
thirdeye/
├── main.py                  # Entry point — wires everything together and runs demo scenarios
└── agent/
    ├── __init__.py          # Public exports for the agent package
    ├── react_agent.py       # ReAct loop (Thought → Action → Observation → Answer)
    ├── tools.py             # Base Tool / ToolRegistry abstractions
    ├── data_tools.py        # GetFromDataSourcesTool + GetFromVertexSearchTool
    └── preprocessor.py      # Language detection + keyword extraction (local, pre-LLM)
```

### Key Components

| Component | Role |
|---|---|
| `preprocessor.py` | Runs locally before any agent call. Detects the user's language and country, extracts up to 6 keywords via Groq, and stamps the current date into a standardised JSON payload. |
| `ReactAgent` | Implements the ReAct loop using the Groq chat-completions API. Iterates Thought → Action (tool call) → Observation until the model produces a final plain-text answer. |
| `ToolRegistry` | Holds all registered tools and serialises them to the Groq function-calling schema. |
| `GetFromDataSourcesTool` | First-pass verification against ThirdEye's internal institutional databases. |
| `GetFromVertexSearchTool` | Second-pass verification using Google Vertex AI Search (managed live web index). |

---

## Demo Scenarios

The `main.py` ships with three illustrative scenarios:

| Scenario | Input | Outcome |
|---|---|---|
| **A** | Chinese-language query about China's GDP | Tool 1 (Data Sources) classifies `true` → answer returned in Chinese |
| **B** | Detailed English query about a G20 summit | Tool 1 `false`, Tool 2 (Vertex) classifies `true` → answer returned in English |
| **C** | Vague query ("Is bigfoot real?") | Both tools `false` → reported as unverifiable |

---

## Supported Languages

ThirdEye's preprocessor auto-detects and maps the following languages to their country context:

English, Chinese (Simplified/Traditional), Japanese, Korean, French, German, Spanish, Arabic, Russian, Portuguese, Italian, Thai, Vietnamese, Malay, Indonesian, Burmese — with graceful fallback to English/Global for unrecognised languages.

---

## Setup

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key

### Installation

```bash
git clone https://github.com/your-org/thirdeye.git
cd thirdeye
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default
```

### Run

```bash
python main.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | LLM inference and function-calling via the Groq API |
| `langdetect` | Lightweight, offline language detection |
| `python-dotenv` | `.env` file loading |

---

## Extending ThirdEye

### Adding a New Verification Tool

1. Create a class that inherits from `Tool` in `agent/tools.py`.
2. Define `name`, `description`, and `schema` (a `ToolSchema` instance).
3. Implement the `run()` method — return a plain-text or JSON string observation.
4. Register it in `main.py`:

```python
registry.register(YourNewTool())
```

The agent will automatically discover and use it via the Groq function-calling interface.

### Replacing Mock Tools with Real APIs

`GetFromDataSourcesTool.run()` and `GetFromVertexSearchTool.run()` in `agent/data_tools.py` contain placeholder mock logic. Replace their bodies with real HTTP calls to your internal database and Google Vertex AI Search respectively — the rest of the pipeline requires no changes.

---

## License

See [LICENSE](LICENSE) for details.
