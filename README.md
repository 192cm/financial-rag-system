<div align="center">

# 📈 Financial RAG System

**Hybrid RAG Pipeline** — extracts structured financial data from Korean DART filings using dual-path vision and text retrieval with agentic self-correction.

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-Vision_LLM-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ColPali](https://img.shields.io/badge/ColPali-v1.2-412991?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/vidore/colpali-v1.2)
[![BGE-M3](https://img.shields.io/badge/BGE--M3-Reranker-D97706?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/BAAI/bge-reranker-v2-m3)

<br>

[Features](#-features) · [Quick Start](#-quick-start) · [Usage](#-usage) · [Architecture](#-architecture) · [Benchmark](#-benchmark)

</div>

---

## ✨ Features

- **Vision-First Retrieval** — ColPali v1.2 indexes every PDF page as a visual token sequence, recovering tables and charts that OCR-based pipelines misread.
- **Dual-Path Hybrid Search** — a BM25 + semantic text path runs in parallel with the vision path; pages confirmed by both receive a consensus score boost.
- **Neural Reranking** — BAAI/bge-reranker-v2-m3 runs locally on GPU to reorder all merged candidates by semantic relevance to the query.
- **Agentic Self-Correction** — a Gemini vision agent inspects top-ranked page images, detects tables split at page boundaries, and triggers a sliding-window scan to fetch missing rows.
- **Query Routing** — extracts company name, report type, and financial item from the query to scope retrieval to the relevant document subset before any search begins.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/your-repo/Financial_RAG_System.git
cd Financial_RAG_System
pip install -r requirements.txt
```

Requires Python 3.10+ and a GPU with 8 GB+ VRAM (ColPali and the reranker run locally).

### 2. Credentials

Create `.env` in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

A free-tier Gemini API key is available at [Google AI Studio](https://aistudio.google.com/).

### 3. Run

```bash
jupyter notebook notebooks/00_environment_check.ipynb
```

This notebook verifies CUDA availability and VRAM before any model loads.

---

## 📖 Usage

### Notebooks

Run the notebooks in order — each is self-contained and covers one methodology:

| Notebook | Method | Description |
|----------|--------|-------------|
| `00_environment_check.ipynb` | — | Hardware and CUDA verification |
| `01_baseline_rag.ipynb` | 0 | Text hybrid baseline |
| `02_method1_vision_rag.ipynb` | 1 | ColPali-only retrieval |
| `03_method2_dual_path.ipynb` | 2 | Dual-path without reranker |
| `04_method3_sota_rag.ipynb` | 3 | Full pipeline with corrective agent |
| `05_evaluation_results.ipynb` | — | Benchmark analysis |

> Start with `04_method3_sota_rag.ipynb` to run the highest-accuracy configuration directly.

---

## ⚙ Configuration

Everything is controlled through `config/config.yaml` — no code changes needed to swap API credentials.

| Key | Default | Description |
|-----|---------|-------------|
| `api_keys.google_gemini_env_name` | `GOOGLE_API_KEY` | Environment variable name read at runtime for the Gemini key |

---

## 🏗 Architecture

```
Financial_RAG_System/
├── src/
│   ├── retrieval/
│   │   ├── text_engine.py     # BM25 + semantic ensemble
│   │   ├── vision_engine.py   # ColPali v1.2 page indexing and search
│   │   ├── reranker.py        # BGE-M3 neural reranker (local GPU)
│   │   └── router.py          # query entity extraction and document scoping
│   ├── engines/
│   │   ├── rag_engines.py     # dual-path retrieval orchestration
│   │   └── agent.py           # corrective agent with sliding-window logic
│   ├── evaluation/
│   │   ├── judge.py           # LLM-as-judge scoring
│   │   ├── metrics.py         # exact match, ROUGE-L, BLEU
│   │   └── runner.py          # benchmark runner
│   ├── models.py              # shared data models
│   └── utils/
│       ├── vision.py          # PDF-to-image rendering
│       └── common.py          # shared helpers
├── notebooks/                 # per-methodology experiments (methods 0–3)
├── data/                      # source PDFs and vector indices
└── config/config.yaml         # API keys and retrieval parameters
```

```
User Query
   │  entity extraction: company, report type, financial item
   ▼
router.py ──▶ scoped document set
   │
   ├──▶ text_engine.py   (BM25 + semantic ensemble)  ──┐
   │                                                    │  consensus boost
   └──▶ vision_engine.py  (ColPali page vectors)    ──┘
                    │  merged + scored candidates
                    ▼
             reranker.py  (BGE-M3, local GPU)
                    │  top-K pages with scores
                    ▼
             agent.py ──▶ Gemini Vision (page image analysis)
                    │  sliding-window if table boundary detected
                    ▼
              Final Answer
```

> Consensus boost is the core design decision: pages identified by both retrieval paths rank above any single-path result, reducing hallucination from partial evidence.

---

## ⚡ Benchmark

10-question evaluation set covering numerical calculations and table analysis across real DART annual reports.

| Methodology | Exact Match | ROUGE-L | BLEU | Latency (s) |
|:---|:---:|:---:|:---:|:---:|
| Method 0 — Text Hybrid Baseline | 0.6 | 0.380 | 0.022 | 10.04 |
| Method 1 — ColPali Only | 0.0 | 0.249 | 0.006 | 12.49 |
| Method 2 — Dual-Path (no reranker) | 0.3 | 0.289 | 0.026 | 9.97 |
| Method 3 — Full Pipeline (no reranker) | 0.7 | 0.389 | 0.070 | 41.20 |
| **Method 3 — Full Pipeline (SOTA)** | **0.8** | **0.438** | **0.077** | **133.75** |

SOTA also achieves an LLM Judge Score of **4.2 / 5.0**. The higher latency reflects multi-step agent inspection and sliding-window correction passes.
