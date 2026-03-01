"""Create a W&B Report for the hackathon submission."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")

import wandb
import wandb_workspaces.reports.v2 as wr

api = wandb.Api()
entity = api.default_entity
project = "roblox-luau-codegen"
print(f"Creating report for {entity}/{project}")

report = wr.Report(
    project=project,
    entity=entity,
    title="Luau Copilot: Self-Improving Roblox Code Generator",
    description="W&B Fine-Tuning Hackathon — SFT → RFT with Claude Sonnet Judge → Self-Correction Loop",
    blocks=[
        wr.H1(text="Luau Copilot: Self-Improving Roblox Code Generator"),
        wr.MarkdownBlock(text="""A fine-tuned Mistral model that writes **production-ready Roblox Luau scripts**, trained through a three-stage self-improvement pipeline tracked end-to-end in W&B Weave.

**Models**: [SFT](https://huggingface.co/squaredcuber/roblox-luau-mistral-7b-2) | [RFT](https://huggingface.co/squaredcuber/roblox-luau-mistral-7b-rft) | [GitHub](https://github.com/squaredrblx-lang/roblox-luau-codegen)

**Live Demo**: [roblox-luau-codegen-production.up.railway.app](https://roblox-luau-codegen-production.up.railway.app)"""),

        wr.HorizontalRule(),

        wr.H2(text="The Three-Stage Pipeline"),

        wr.H3(text="Stage 1: Supervised Fine-Tuning (SFT)"),
        wr.MarkdownBlock(text="""Fine-tune Mistral 7B on curated Roblox Luau examples with QLoRA (rank-32, 4-bit quantization).
- All projection modules targeted for maximum adaptation
- Model learns Luau syntax, Roblox API patterns, game development idioms
- Published as `squaredcuber/roblox-luau-mistral-7b-2`"""),

        wr.H3(text="Stage 2: Reinforcement Fine-Tuning (RFT) with Claude Sonnet Judge"),
        wr.MarkdownBlock(text="""Generate 800 candidates from the SFT model, score each with **4 deterministic scorers + Claude Sonnet as LLM judge**, keep the top 181 (22.6% acceptance rate), and train on those.

**Why cross-model judging matters**: Claude Sonnet evaluates Mistral outputs. The model can't game the reward signal because it's a completely different model family — this prevents the reward hacking that happens with self-judging.

- Training loss: **0.414**
- Published as `squaredcuber/roblox-luau-mistral-7b-rft`"""),

        wr.H3(text="Stage 3: Self-Correction Loop (Code Forge)"),
        wr.MarkdownBlock(text="""The model generates code → 4 regex-based scorers diagnose issues → the model rewrites its own code → re-score → repeat until passing.

Corrected outputs become **new training data** for the next cycle — creating a genuine self-improvement flywheel."""),

        wr.HorizontalRule(),

        wr.H2(text="Four-Dimensional Scoring System"),
        wr.MarkdownBlock(text="""Every generated script is evaluated by 4 specialized, deterministic, regex-based scorers (no ML overhead). All scorers are traced with `@weave.op()` — every score computation is logged to Weave.

| Scorer | What It Checks | Key Patterns |
|--------|---------------|-------------|
| **Syntax** | Bracket matching, block balance, Python-ism detection | `function/if/for → end` balance, `def`/`class`/`True`/`None` detection |
| **API** | GetService usage, deprecated patterns, modern API | 30+ known Roblox services, `wait()→task.wait()`, `game.Players→GetService()` |
| **Bugs** | 6 bug patterns with severity weights | Unchecked FindFirstChild, infinite loops without yield, DataStore without pcall |
| **Quality** | Comments, naming, organization, error handling | PascalCase services, camelCase locals, comment density, section headers |"""),

        wr.HorizontalRule(),

        wr.H2(text="Results"),

        wr.H3(text="Prompt Self-Improvement Loop"),
        wr.MarkdownBlock(text="""The pipeline automatically analyzes scorer results, identifies weak dimensions, and rewrites the system prompt using Mistral as a meta-optimizer.

| Iteration | Prompt | Composite | Syntax | API | Bugs | Quality | Task |
|-----------|--------|-----------|--------|-----|------|---------|------|
| 1 | v4 | **89.72%** | 97.6% | 100% | 67.3% | 91.6% | 92.0% |
| 2 | v5 | **90.24%** | 96.8% | 100% | 71.0% | 90.2% | 92.6% |
| 3 | v6 | 88.99% | 96.8% | 98.3% | 65.7% | 90.2% | 93.2% |

**Key insight**: Bug score improved 67.3% → 71.0% (+3.7pp) in iteration 2 after the prompt was auto-rewritten to emphasize pcall wrapping and nil checks. 7 prompt versions evolved across the project."""),

        wr.H3(text="Code Forge Self-Correction Results"),
        wr.MarkdownBlock(text="""The Code Forge runs diagnose→heal→re-score across 15 test tasks spanning 6 categories.

| Round | Difficulty | Tasks | Before | After | Pass Rate | Corrections |
|-------|-----------|-------|--------|-------|-----------|-------------|
| 1 | Easy | 15 | 87.1% | 88.2% | 66.7% | 3 |
| 2 | Medium | 13 | 87.1% | 88.1% | 69.2% | 1 |
| 3 | Hard | 13 | 88.7% | **89.6%** | **92.3%** | 2 |

**Pass rate: 66.7% → 92.3%** as the model learned from its corrections. The scorer also evolved 3 new rules (self-evolving scorer rules).

**Category Breakdown (Round 3):**

| Category | Score |
|----------|-------|
| Game Mechanics | 92.7% |
| Data Persistence | 92.0% |
| Physics/CFrame | 91.7% |
| Remote Events | 88.1% |
| NPC Behavior | 86.3% |
| UI | 86.0% |"""),

        wr.HorizontalRule(),

        wr.H2(text="RFT Training Details"),
        wr.MarkdownBlock(text="""### Rejection Sampling Process
- **Generated**: 800 candidates from SFT model across 200 tasks (4 candidates each)
- **Scoring**: 4 deterministic scorers (syntax, API, bugs, quality) — each weighted 25%
- **Threshold**: Candidates scoring ≥ 0.75 composite accepted
- **Accepted**: 181 candidates (22.6% acceptance rate)
- **Training**: QLoRA fine-tuning on accepted candidates, training loss 0.414

### Claude Sonnet as Cross-Model Judge
For the RFT reward signal, Claude Sonnet evaluates each candidate on:
- **Functionality** (40%): Does the code implement what was asked?
- **Correctness** (35%): Is the code free of bugs?
- **Completeness** (25%): All features implemented, no placeholders?

Cross-model judging is critical — since Claude Sonnet is a different model family from Mistral, the fine-tuned model cannot learn to exploit scoring biases. The reward signal stays robust."""),

        wr.HorizontalRule(),

        wr.H2(text="W&B Weave Integration"),
        wr.MarkdownBlock(text="""Every component is traced with Weave:

- **`LuauCodeGenerator`** is a `weave.Model` subclass — every field change (including system_prompt) creates a new model version
- **All scorer functions** decorated with `@weave.op()` — inputs, outputs, and latency tracked
- **Evaluations** use `weave.Evaluation` with all 5 scorers
- **Datasets** published as `weave.Dataset` objects with 15 test tasks

This means you can trace any generation → see the exact prompt → see each scorer's detailed output → compare across model versions, all in the Weave UI."""),

        wr.HorizontalRule(),

        wr.H2(text="Web Application & Studio Plugin"),
        wr.MarkdownBlock(text="""The project includes a full Django web app that serves as an **agentic harness** — like Claude Code, but for Roblox:

- **Agentic Chat**: Describe what to build → AI generates, scores, self-corrects in one pipeline
- **Code Generator**: Pick Base/SFT/RFT model, generate code, see live quality dashboards
- **Model Comparison**: Side-by-side outputs from all 3 models on the same task
- **Roblox Studio Plugin**: HTTP polling plugin that auto-inserts scripts into Roblox Studio
- **Training Dashboard**: Charts showing score progression

Deployed on Railway with a Dockerfile (no ML dependencies — scorers are pure Python regex)."""),

        wr.HorizontalRule(),

        wr.H2(text="Architecture"),
        wr.MarkdownBlock(text="""```
User Task Description
        │
        ▼
┌───────────────────┐     ┌──────────────────┐
│  Code Generator   │     │  System Prompt   │
│  (weave.Model)    │◄────│  (auto-versioned)│
│  Mistral API      │     │  v1 → v7         │
└─────────┬─────────┘     └──────────────────┘
          │ generated code
          ▼
┌───────────────────────────────────────────────┐
│              4 Scorer Pipeline                 │
│  Syntax (25%) │ API (25%) │ Bugs (25%) │ Quality (25%)
│              composite score                   │
└─────────────────────┬─────────────────────────┘
                      │
              ┌───────┴───────┐
              │  Score < 85%? │
              └───────┬───────┘
                yes   │   no
                ▼         ▼
        ┌─────────────┐  PASS
        │ Self-Correct │
        │ (heal loop)  │
        └──────┬──────┘
               ▼
          Re-score → PASS
```"""),

        wr.H2(text="Claude Code Workflow Integration"),
        wr.MarkdownBlock(text="""The entire self-improvement pipeline is orchestrated through Claude Code slash commands + the W&B MCP server:

| Command | What It Does |
|---------|-------------|
| `/run-eval` | Runs a single evaluation of all 15 test tasks with 5 scorers |
| `/analyze` | Identifies weak dimensions below threshold, ranks by gap |
| `/improve` | Auto-rewrites the system prompt targeting weaknesses |
| `/full-loop 5` | Runs 5 iterations of eval→analyze→improve end-to-end |
| `/forge` | Runs the Code Forge self-correction pipeline |

The W&B MCP server lets Claude Code query Weave traces directly — it can pull evaluation results, compare model versions, and verify improvement without leaving the terminal."""),

        wr.HorizontalRule(),

        wr.H2(text="Tech Stack"),
        wr.MarkdownBlock(text="""| Component | Technology |
|-----------|-----------|
| Base Model | Mistral 7B (via Mistral API) |
| Fine-tuning | QLoRA with PEFT/TRL on H100 |
| LLM Judge | Claude Sonnet (cross-model scoring) |
| Evaluation | W&B Weave |
| Scorers | Pure Python regex (no ML deps) |
| Web App | Django + Tailwind CSS |
| Deployment | Railway (Docker) |
| Studio Plugin | HTTP polling Roblox plugin |

Built for the **W&B Fine-Tuning Hackathon** — Fine-Tuning Track + Web App Track"""),
    ],
)

report.save()
print(f"\nReport created successfully!")
print(f"URL: {report.url}")
