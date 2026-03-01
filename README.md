# GamePulse — AI-Powered Roblox Game Analytics & Luau Code Generation

> **Built for the Mistral AI Hackathon** | Powered by `mistral-small-latest` | Traced with W&B Weave

GamePulse is a unified platform that combines **AI-powered game analytics** with an **intelligent Luau code generator** — both powered by Mistral AI with self-improving prompt pipelines and full observability through Weights & Biases Weave.

**Live Demo:** [https://web-production-fe47d.up.railway.app](https://web-production-fe47d.up.railway.app)

---

## What It Does

### 1. GamePulse Dashboard — AI Game Analytics
A consulting-grade analytics dashboard for Roblox developers. Enter any Roblox game and get instant AI-powered analysis across **10 dimensions**:

| Feature | What Mistral Analyzes |
|---|---|
| **Portfolio Overview** | AI-contextualized stats with health score |
| **Daily Briefing** | Personalized morning briefing with quick wins |
| **Deep Dive Insights** | Game health diagnosis, growth actions, competitor notes |
| **Game Loop Analysis** | Core loop rating, retention hooks, bottlenecks, monetization gaps |
| **Competitor Radar** | Competitive positioning, threats, feature gaps, "steal this" recommendations |
| **Trend Analysis** | Platform trends mapped to your specific games |
| **Revenue Estimator** | Monetization analysis with revenue percentile ranking |
| **Social & Marketing** | AI-generated tweets, Discord posts, YouTube descriptions |
| **Update Roadmap** | 4-week prioritized development plan |
| **AI Chat** | Free-form conversation about your games with full context |

Every AI call is traced through **W&B Weave** with `@weave.op()` decorators for full observability.

### 2. Luau Copilot — Self-Improving Code Generator
An AI code generation system for Roblox Luau that **improves itself** through an automated evaluation pipeline:

- **Fine-tuned models**: SFT (Supervised Fine-Tuned) and RFT (Reinforcement Fine-Tuned) LoRA adapters on Devstral
- **5 automated scorers**: Syntax validation, Roblox API correctness, bug detection, code quality, and LLM-as-judge task completion
- **Self-correction loop**: Generate → Score → Fix → Re-score (up to 3 rounds)
- **Agentic pipeline**: Full generate → score → self-correct → deploy-to-Studio flow
- **Roblox Studio plugin**: Live code injection from web UI directly into Studio

---

## The Self-Improvement Pipeline

The core innovation — an automated loop where Mistral improves its own system prompts:

```
FOR each iteration:
  1. GENERATE  → Mistral generates Luau code for 15 test tasks
  2. EVALUATE  → 5 scorers grade each output (traced in Weave)
  3. ANALYZE   → Identify dimensions below threshold
  4. IMPROVE   → Mistral rewrites its own system prompt targeting weaknesses
  5. SAVE      → Version the prompt (v1→v7) + log scores
  6. REPEAT    → Run again with the improved prompt
```

**Results**: 7 prompt versions evolved, with scores tracked in `results/iteration_log.json`. The champion prompt (v7) feeds back into both the code generator and the analytics dashboard.

### Champion Prompt Evolution
The analytics prompts were evolved through a **genetic algorithm**:
- Genome `83e98d52` — **83.9% fitness** (97.9% accuracy, 97.5% completeness, 80% insight quality)
- 6 genes: identity, analysis framework, format rules, few-shot examples, interaction patterns, benchmark context
- Survived 3 generations of crossover + AI-driven mutation

---

## Architecture

```
gamepulse/
├── webapp/                    # Django application
│   ├── core/                  # Game analytics (10 AI features)
│   ├── copilot/               # Luau code generator + Studio integration
│   └── templates/             # Tailwind CSS dark-mode UI
├── scorers/                   # 5 automated code scorers
├── pipeline/                  # Self-improvement loop (eval→analyze→improve)
├── training/                  # SFT + GRPO/RFT training scripts
├── src/                       # Generator core + champion prompt
├── configs/prompt_versions/   # v1.txt → v7.txt (evolved prompts)
├── results/                   # Iteration logs + forge corrections
├── studio_plugin/             # Roblox Studio Lua plugin
└── scripts/                   # Pipeline runners + W&B report generation
```

**Stack**: Django · Mistral AI (`mistral-small-latest`) · W&B Weave · Tailwind CSS · Railway

---

## Mistral AI Usage

Every AI feature runs through Mistral's chat API:
- **`mistral-small-latest`** for all analytics, code generation, self-correction, and prompt evolution
- **Champion prompt system** — evolved prompts with diagnostic frameworks, benchmark context, and few-shot examples
- **LLM-as-judge** — Mistral evaluates its own code output for task completion scoring
- **Meta-prompting** — Mistral rewrites its own system prompts based on evaluation feedback

---

## W&B Weave Observability

All Mistral calls are traced with `@weave.op()`:
- Every API call logged with inputs, outputs, latency, and token usage
- Evaluations run through `weave.Evaluation` with 5 scorer dimensions
- Model versions auto-tracked via `weave.Model` (prompt changes = new version)
- Dashboard: [wandb.ai/carpediemhari-n-a/gamepulse](https://wandb.ai/carpediemhari-n-a/gamepulse)

---

## Quick Start

```bash
git clone https://github.com/Hcoder10/gamepulse.git
cd gamepulse
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your MISTRAL_API_KEY and WANDB_API_KEY

# Run the web app
cd webapp
python manage.py runserver

# Run the self-improvement pipeline (optional)
python scripts/run_loop.py -n 5
```

---

## Team

Built by **Hari** for the Mistral AI Hackathon.

---

*All AI features powered by [Mistral AI](https://mistral.ai) · Observability by [Weights & Biases Weave](https://wandb.ai/site/weave)*
