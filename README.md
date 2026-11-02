# PATHFINDER // AI Technical Curriculum & Roadmap Architect

An interactive, DAG-based learning roadmap generator built with **Python**, **Streamlit**, and **Google Gemini 3.8 Flash** (`gemini-3.8-flash` via the official `google-genai` SDK).

Unlike generic AI wrappers that dump unformatted walls of bullet points, **Pathfinder** models any discipline as a **Directed Acyclic Graph (DAG)** of engineering milestones, complete with prerequisite gating, time-budget telemetry, mental models, anti-pattern warnings, proof-of-work capstone rubrics, and one-click exports.

## Key Architectural Features

- **Gemini 3.8 Flash Structured Outputs (`google-genai`)**: Enforces a strict Pydantic schema (`RoadmapResponseSchema`) with `response_mime_type="application/json"` so every generated curriculum is deterministically parsed and validated.
- **Bespoke "Obsidian & Amber" Blueprint UI**: Custom dark theme engineered with `Space Grotesk`, `JetBrains Mono`, and `Inter` typography, radial blueprint grid surfaces, and high-contrast status badges.
- **Interactive SVG Dependency Graph (DAG)**: Automatically computes topological layers and renders an inline vector graph illustrating **Mastered (Emerald)**, **Unlocked/Ready (Amber)**, and **Prerequisite-Locked (Slate)** phases.
- **Proof-of-Work Capstones & Mental Models**: Every module includes concrete acceptance criteria, a verification benchmark test, core mental models, and common beginner pitfalls.
- **Zero-Key Instant Blueprint Inspection**: Ships with a built-in *Distributed Systems & Consensus Engineering* blueprint so you can explore the DAG, progress tracker, and exporters immediately before entering an API key.
- **Multi-Format Exporters**: Export any roadmap to **Markdown (`.md`)**, **Graphviz (`.dot`)**, or **JSON Schema (`.json`)**.

## Project Structure

| File | Responsibility |
| :--- | :--- |
| `app.py` | Complete application containing Pydantic domain schemas, `GeminiRoadmapService`, DAG topological layout engine, SVG renderer, exporters, and Streamlit UI |
| `requirements.txt` | Minimal, production-pinned dependencies (`streamlit`, `google-genai`, `pydantic`) |
| `readme.md` | Setup, configuration, architecture, and verification documentation |

## Quickstart

### 1. Create a Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Your Gemini API Key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/) and either export it in your shell or paste it directly into the Pathfinder sidebar:

```bash
export GEMINI_API_KEY="your-google-ai-studio-api-key"
```

### 3. Launch the Application

```bash
streamlit run app.py
```

## Code Quality & Architecture

`app.py` is structured into clean, decoupled layers to ensure testability and maintainability:

1. **Domain & Validation Layer (`RoadmapResponseSchema`, `GenerationRequest`)**: Enforces input bounds and output schema invariants (compatible with both Pydantic v1 and v2).
2. **DAG Normalization Layer (`normalize_roadmap_dag`, `compute_progress_metrics`)**: Guarantees valid prerequisite references (stripping cycles or forward references) and computes real-time mastery velocity and unlocked modules.
3. **AI Service Layer (`GeminiRoadmapService`)**: Wraps `genai.Client` with dependency injection (`client_factory`) for deterministic unit testing and automatic model fallback (`gemini-3.8-flash` -> `gemini-3.5-flash-lite` -> `gemini-2.5-flash`).
4. **Presentation & Export Layer (`render_svg_topology_graph`, `export_roadmap_to_markdown`, `export_roadmap_to_dot`)**: Pure functions that transform validated roadmap dictionaries into SVG diagrams, Markdown study packs, and Graphviz specifications.
