# Who is the richest club in the championship?
## Detecting and Rewriting Underspecified Questions Improve QA Performance

**Yunchong Huang¹, Gianni Barlacchi²\*, Sandro Pezzelle¹**

¹ ILLC, University of Amsterdam &nbsp;|&nbsp; ² Amazon AGI &nbsp;|&nbsp; \*Work done outside Amazon

*ACL 2026*

---

## Overview

Large language models (LLMs) are routinely blamed for QA failures that are not actually their fault. This paper argues that a key but overlooked culprit is **underspecification (UND)**: questions whose intended meaning *cannot be uniquely determined* without additional context.

> **Example.** "Who is the richest club in the championship?" — Which championship? Richest by what metric? As of when?
>
> GPT-4o (no context) → ~~Leicester City~~ ✗  
> After rewriting to a fully-specified form → **Manchester City** ✓

We show that UND is widespread in standard QA benchmarks, that LLMs consistently underperform on UND questions, and that automatically rewriting these questions yields large, consistent performance gains.

---

## Taxonomy of Underspecification

| # | Type | Example |
|---|------|---------|
| 1 | **Missing components** | "What's the capital?" *(of which country?)* |
| 2 | **Undetermined reference** | "When did the Giants reach the playoffs?" *(football or baseball?)* |
| 3 | **Undetermined granularity** | "When did WWI break out?" *(politically or militarily?)* |
| 4 | **Undetermined standard** | "Best smartwatches in 2023?" *(best by what standard?)* |

---

## Pipeline

```
Step 1 — UND Classifier
  Select best LLM-classifier from 10 candidates (Qwen3-4B wins;
  avg acc 0.74, macro-F1 0.73 on UNDER / UNDER-gold)

Step 2 — Assess QA on FS / UND subsets
  Apply classifier to 3,824 questions across NQ, HotpotQA,
  TriviaQA, FRAMES. Run GPT-4o and Gemini-2.5-Flash — no
  context passages provided.

Step 3 — Rewrite UND questions
  Oracle rewriter (gold answer + classifier reasoning) converts
  UND → FS. 64–86% of rewrites reclassified as fully specified.

Step 4 — Reassess QA
  Cross-assignment setup: GPT-4o rewrites answered by Gemini,
  and vice versa. Compare F1 scores on original vs. rewritten UND.
```

---

## Key Results

### Prevalence of underspecification

| Dataset | % UND | N |
|---------|-------|---|
| TriviaQA | 15.9% | 1,000 |
| NQ | 45.8% | 1,000 |
| HotpotQA | 49.6% | 1,000 |
| FRAMES | **53.4%** | 824 |

LLMs score significantly lower on UND vs. FS subsets across all datasets and models (independent *t*-tests, all *p* < 0.05).

### F1 improvement after rewriting

| Model | NQ | HotpotQA | TriviaQA | FRAMES |
|-------|----|----------|---------|--------|
| GPT-4o (orig.) | 37.0% | 34.6% | 75.8% | 24.4% |
| GPT-4o (rewr.) | **57.3%** (+20.3) | **51.8%** (+17.2) | **83.6%** (+7.8) | **41.6%** (+17.2) |
| Gemini (orig.) | 38.8% | 41.2% | 76.0% | 37.1% |
| Gemini (rewr.) | **50.0%** (+11.2) | **50.6%** (+9.4) | 74.4% (−1.6) | **46.5%** (+9.4) |

*TriviaQA exception is expected — trivia questions are relatively self-contained by design; AA metric still improves for both models.*

---

## Repository Structure

```
.
├── Data/
│   ├── UNDER.csv                          # 855-question annotated UND dataset
│   ├── UNDER_gold.csv                     # 150-question annotated & expert-verified UNDER-gold dataset
│   └── UNDER_gold (tracking annotations).xlsx
│
├── Testing Off-the-Shelf LLMs/            # Step 1: classifier selection
│   ├── helper_functions_testing.py
│   └── [notebooks for DeepSeek / Llama / Qwen3 variants]
│
├── QA_datasets_classified_qa_eval/        # Steps 2: QA evaluation for UND and FS subsets
│   ├── helper_functions_qa.py
│   └── [notebooks per dataset × model]
│
├── Question_Rewriting_new/               # Step 3 & 4: UND → FS rewriting and QA assessment for rewritten UND questions
│   ├── helper_functions_qr.py
│   └── [notebooks per dataset × rewriter model]
│
├── pyproject.toml
└── uv.lock
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python 3.12).

```bash
# Clone the repository
git clone https://github.com/franzyellow/Underspecification-QA-conf-paper.git
cd Underspecification-QA-conf-paper

# Install dependencies
uv sync
```

### API keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

---

## Reproducing the Experiments

All experiments are run from Jupyter notebooks. Launch the environment with:

```bash
uv run jupyter lab
```

| Step | Directory | What to run |
|------|-----------|-------------|
| 1. Classifier selection | `Testing Off-the-Shelf LLMs/` | One notebook per model candidate |
| 2. QA evaluation (orig.) | `QA_datasets_classified_qa_eval/` | `{Dataset}_{Model}_BaseClass.ipynb` |
| 3. Question rewriting | `Question_Rewriting_new/` | `{Dataset}_{RewriterModel}_Rewriting.ipynb` |
| 4. QA evaluation (rewr.) | `Question_Rewriting_new/` | Same notebooks, rewritten-question input |

Visualisations are produced by `visualization.ipynb` inside each directory.

---

## Citation

```bibtex
@inproceedings{huang-etal-2026-underspecification,
  title     = {Who is the richest club in the championship?
               Detecting and Rewriting Underspecified Questions Improve {QA} Performance},
  author    = {Huang, Yunchong and Barlacchi, Gianni and Pezzelle, Sandro},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association
               for Computational Linguistics (ACL 2026)},
  year      = {2026},
}
```

---

## Contact

- **Yunchong Huang** — franzhuang027@gmail.com
- **Gianni Barlacchi** — gbarlac@amazon.com
- **Sandro Pezzelle** — s.pezzelle@uva.nl