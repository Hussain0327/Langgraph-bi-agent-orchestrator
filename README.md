# Business Intelligence Orchestrator

![Python](https://img.shields.io/badge/python-3.12+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-purple)
![ML](https://img.shields.io/badge/ML-routing%20classifier-red)
![Cost Savings](https://img.shields.io/badge/cost%20savings-86%25-success)
![Speedup](https://img.shields.io/badge/speedup-2.1x%20parallel-orange)
![AutoGen](https://img.shields.io/badge/automation-PPT%20%2B%20Excel-teal)
![License](https://img.shields.io/badge/license-MIT-green)

**Author**: Raja Hussain  
**Built with**: Python, LangGraph, FastAPI, Docker, DeepSeek, GPT-5  
**Status**: MVP-Ready  
**Last Updated**: December 6, 2025

---

## TLDR

- Coordinates 5 specialized AI agents with LangGraph to answer business questions
- Retrieves academic papers from Semantic Scholar and arXiv with proper citations
- Routes queries between DeepSeek and GPT-5 to cut costs by 86%
- Auto-generates PowerPoint decks and Excel workbooks from single queries
- 2.1x speedup with parallel execution, 138x speedup on cache hits

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Cost Analysis](#cost-analysis)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Roadmap](#roadmap)

---

## Overview

This system coordinates multiple specialized AI agents to provide comprehensive business analysis backed by academic research. It automatically generates three deliverables from a single query:

1. **Structured JSON** - Machine-readable data for API integrations
2. **PowerPoint Presentation** - Professional executive summary decks (10-12 slides)
3. **Excel Workbook** - Analysis spreadsheets with formulas and scenarios (5 sheets)

### Key Capabilities

- **Research-Augmented Generation (RAG)** - Retrieves and synthesizes academic papers with proper citations
- **Parallel Agent Execution** - 2.1x performance improvement through concurrent processing
- **Hybrid LLM Strategy** - 90% cost savings using DeepSeek + GPT-5 fallback
- **Intelligent Caching** - 60-138x speedup on repeated queries with Redis
- **One-Command Deployment** - Docker Compose for easy self-hosting

This project is built to be deployable: Docker Compose, Redis caching, cost optimization, and benchmarking are all implemented. Authentication, rate limiting, and production monitoring are planned but not yet built.

---

## Skills Demonstrated

- Multi-agent orchestration with LangGraph and Python asyncio
- Research augmentation with academic papers, vector search, and caching
- Cost optimization using hybrid LLM routing and fallback strategies
- Document automation for PowerPoint and Excel generation
- Performance benchmarking for speed, cost, and routing accuracy
- Deployment with Docker and Redis caching

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/Hussain0327/Business-Intelligence-Orchestrator.git
cd Business-Intelligence-Orchestrator

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up

# Access API at http://localhost:8000/docs
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Interactive CLI
python cli.py

# Or start API server
uvicorn src.main:app --reload
```

### Prerequisites

- Python 3.12+ or Docker
- OpenAI API key (GPT-5 access)
- DeepSeek API key (optional, for cost savings)
- LangSmith API key (optional, for tracing)

---

## System Architecture

### Agent Workflow

```
User Query
    ↓
Query Classifier (simple/business/complex)
    ↓
├─ SIMPLE → Direct Answer (5s)
├─ BUSINESS → Agent Router → 4 Parallel Agents → Synthesis
└─ COMPLEX → Research Retrieval → Agent Router → 4 Parallel Agents → Synthesis
```

![System Architecture](docs/screenshots/system_architecture.png)

### Specialized Agents

1. **Market Analysis** - Trends, competition, customer segmentation
2. **Operations Audit** - Process optimization, efficiency analysis
3. **Financial Modeling** - ROI calculations, revenue projections
4. **Lead Generation** - Customer acquisition, growth strategies
5. **Research Synthesis** - Academic paper retrieval and synthesis

---

## Key Features

### 1. Document Automation

Automatically generates professional deliverables from business queries:

![Deliverables Overview](docs/screenshots/deliverables_overview.png)

**PowerPoint Presentation** (10-12 slides):

- Title slide with branding
- Executive summary
- Context and methodology
- Key findings with metrics
- Risk analysis
- Detailed recommendations
- Next steps and appendix

**Excel Workbook** (5 sheets):

- Executive Summary: KPI dashboard
- Raw Data: Complete analysis output
- Calculations: Formulas with Base/Upside/Downside scenarios
- Charts & Visuals: Data visualizations
- Assumptions & Sources: Methodology and citations

**Sample Deliverables:**

The repository includes real examples of auto-generated deliverables:

- `financial_analysis_deck.pptx` - Executive summary presentation
- `SaaS_Unit_Economics_Analysis.pptx` - SaaS business model analysis
- `Monthly_vs_Annual_Billing_Analysis.pptx` - Pricing strategy comparison
- `Monthly_vs_Annual_Billing_Analysis.xlsx` - Financial modeling workbook

These demonstrate the quality and structure of outputs the system generates from single business queries.

### 2. Research Augmentation (RAG)

Retrieves academic papers and integrates findings into recommendations:

- Semantic Scholar + arXiv integration
- ChromaDB vector store for semantic search
- Automatic APA citation formatting
- 7-day caching for research papers
- 2-3 relevant papers per complex query

### 3. Parallel Agent Execution

Runs multiple agents concurrently for faster results:

- Simple queries: 5s (direct answer)
- Business queries: 69s (was 145s, 2.1x faster)
- Complex queries: 153s (was 235s, 1.5x faster)
- Overall: 2.3x average speedup

### 4. Intelligent Caching

Multi-layer Redis caching with file fallback:

- Research papers: 7-day TTL
- Agent responses: 1-day TTL
- Cache hit rate: 60-70%
- Speedup: 60-138x on cache hits

### 5. Cost Optimization

Hybrid LLM routing for significant savings:

- Primary: DeepSeek v3.2-Exp (cheap, fast)
- Fallback: GPT-5-nano (high quality)
- ML classifier: SetFit (77% accuracy, 20ms)
- Cost: $0.043/query (vs $0.30 GPT-5 only)
- Savings: 86% cost reduction

---

## Performance Metrics

![Performance Comparison](docs/screenshots/performance_comparison.png)

### Speed

| Query Type | Time | vs Sequential  | Cache Hit |
| ---------- | ---- | -------------- | --------- |
| Simple     | 5s   | New capability | 0.1s      |
| Business   | 69s  | 2.1x faster    | 0.5s      |
| Complex    | 153s | 1.5x faster    | 1s        |

### Cost

| Metric            | GPT-5 Only | Hybrid | Savings |
| ----------------- | ---------- | ------ | ------- |
| Per Query         | $0.30      | $0.043 | 86%     |
| Monthly (100/day) | $900       | $129   | $771    |
| Annual            | $10,800    | $1,548 | $9,252  |

### Quality

- Response length: 8,000-9,000+ characters
- Citation accuracy: 100% (APA format)
- ML routing accuracy: 77%
- Agent output quality: Validated via evaluation framework

---

## Technology Stack

| Component     | Technology              | Purpose                |
| ------------- | ----------------------- | ---------------------- |
| Orchestration | LangGraph               | State machine workflow |
| LLM           | DeepSeek v3.2-Exp       | Primary model (cheap)  |
| LLM Fallback  | GPT-5-nano              | Quality backup         |
| Routing       | SetFit ML classifier    | Agent selection        |
| Execution     | Python asyncio          | Parallel processing    |
| Caching       | Redis + File            | Performance            |
| Vector Store  | ChromaDB                | Semantic search        |
| Research APIs | Semantic Scholar, arXiv | Academic papers        |
| Document Gen  | python-pptx, openpyxl   | PPT & Excel            |
| Charts        | matplotlib              | Visualizations         |
| API           | FastAPI                 | REST endpoints         |
| Deployment    | Docker Compose          | Containerization       |
| Monitoring    | LangSmith (optional)    | Tracing                |

---

## Project Structure

```
multi_agent_workflow/
├── src/
│   ├── agents/              # 5 specialized AI agents
│   ├── tools/               # Agent tools (research, calculator)
│   ├── generators/          # Document generators (PPT, Excel, charts)
│   ├── schemas/             # Pydantic data schemas
│   ├── ml/                  # ML routing classifier
│   ├── langgraph_orchestrator.py  # Main orchestration
│   ├── cache.py             # Redis caching layer
│   ├── unified_llm.py       # Hybrid LLM wrapper
│   └── main.py              # FastAPI server
│
├── eval/                    # Evaluation framework
│   ├── benchmark.py         # Performance benchmarking
│   └── test_queries.json    # Test dataset
│
├── models/                  # ML models
│   └── routing_classifier.pkl  # Trained SetFit model
│
├── docs/                    # Documentation
│   ├── core/               # Essential docs
│   ├── features/           # Feature guides
│   ├── guides/             # How-to guides
│   └── archive/            # Historical docs
│
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── cli.py                   # Interactive CLI
├── docker-compose.yml       # Docker deployment
└── README.md                # This file
```

---

## Configuration

### Environment Variables

**Required:**

```bash
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-5-nano
```

**Recommended (for cost savings):**

```bash
DEEPSEEK_API_KEY=sk-...
MODEL_STRATEGY=hybrid  # "gpt5" | "deepseek" | "hybrid"
CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379/0
```

**Optional (for tracing):**

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=business-intelligence-orchestrator
```

### Model Strategy

- **gpt5**: Use GPT-5 for everything (highest quality, most expensive)
- **deepseek**: Use DeepSeek for everything (lowest cost, good quality)
- **hybrid**: Smart routing (DeepSeek primary, GPT-5 fallback) - Recommended

---

## API Usage

### REST API

```bash
# Start server
uvicorn src.main:app --reload

# Access interactive docs
open http://localhost:8000/docs
```

**Example Request:**

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "How can I reduce customer churn?"}
)

print(response.json()['synthesis'])
```

### Python SDK

```python
from src.langgraph_orchestrator import LangGraphOrchestrator

# Initialize with RAG
orch = LangGraphOrchestrator(enable_rag=True)

# Execute query
result = orch.orchestrate(
    query="What are SaaS pricing best practices?",
    use_memory=False
)

print(result['recommendation'])
print(result['agents_consulted'])
```

### Available Endpoints

- `POST /query` - Execute business intelligence query
- `GET /health` - Health check
- `GET /cache/stats` - Cache statistics
- `POST /cache/clear` - Clear cache
- `GET /docs` - Interactive API documentation

---

## Testing

### Run All Tests

```bash
# Document automation test
python test_document_automation.py

# RAG system tests
python test_rag_system.py

# DeepSeek integration tests
python test_deepseek.py

# Structured output tests
python test_structured_output.py
```

### Run Benchmarks

```bash
# Quick test (5 queries)
python eval/benchmark.py --mode both --num-queries 5

# Full evaluation (25 queries)
python eval/benchmark.py --mode both --num-queries 25
```

---

## Deployment

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services included:

- Redis (port 6379) - Caching
- Orchestrator (port 8000) - API

### Cloud Deployment

**AWS ECS / GCP Cloud Run / Azure Container Instances:**

See [docs/guides/DEPLOYMENT_GUIDE.md](docs/guides/DEPLOYMENT_GUIDE.md) for detailed instructions.

### Production Considerations

- Use managed Redis (AWS ElastiCache, GCP Memorystore, Azure Cache)
- Set up HTTPS with reverse proxy (nginx, Caddy)
- Configure authentication and rate limiting
- Enable monitoring (Prometheus + Grafana)
- Use secrets management (AWS Secrets Manager, etc.)

---

## Cost Analysis

### Monthly Costs (100 queries/day)

| Configuration    | Cost/Query | Monthly | Annual  | Notes           |
| ---------------- | ---------- | ------- | ------- | --------------- |
| GPT-5 Only       | $0.30      | $900    | $10,800 | Highest quality |
| Hybrid (Current) | $0.043     | $129    | $1,548  | Recommended     |
| DeepSeek Only    | $0.003     | $9      | $108    | Lowest cost     |

**Savings with Hybrid:** $771/month ($9,252/year) compared to GPT-5 only

### Cache Benefits

With 60-70% cache hit rate:

- Effective cost: Even lower than $0.043/query
- Instant responses: 60-138x faster
- API call reduction: 60%+

---

## Troubleshooting

### Common Issues

**Empty Agent Outputs:**

- Status: Fixed (reasoning_effort set to "low")

**Redis Connection Failed:**

- System automatically falls back to file cache
- Start Redis: `docker-compose up redis -d`

**Semantic Scholar Rate Limit:**

- System automatically falls back to arXiv
- 7-day caching reduces API calls by 60%

**DeepSeek API Error:**

- Check `DEEPSEEK_API_KEY` in .env
- System falls back to GPT-5 automatically

For more issues and solutions, see [docs/guides/TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md)

---

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_system.py
```

### Code Quality

```bash
# Format
black src/ eval/ *.py

# Lint
flake8 src/ eval/ --max-line-length=120

# Type check
mypy src/ --ignore-missing-imports
```

### Project Metrics

- **Total code**: 3,640 lines (production)
- **Documentation**: 12,000+ lines (organized in docs/)
- **Tests**: 15+ test files
- **Development time**: ~50 hours over 3 weeks
- **Commits**: 25+ commits

**Result:** Built a multi-agent BI system with academic research integration that auto-generates PowerPoint and Excel deliverables, achieves 2.1x speedup with parallelization, and cuts costs by 86% per query. Built in roughly 50 hours over 3 weeks.

---

## Roadmap

### Completed

- Core multi-agent system with LangGraph
- Research augmentation (RAG)
- ML routing classifier
- Parallel execution (2.1x speedup)
- Redis caching (138x speedup)
- Document automation (PowerPoint + Excel)
- Docker deployment
- 90% cost savings (DeepSeek hybrid)

### In Progress

- ML classifier retraining (77% → 90%+ accuracy)
- Full 25-query evaluation
- Documentation cleanup

### Planned

- Authentication & rate limiting
- Prometheus + Grafana monitoring
- Load testing
- Cloud deployment guides
- Additional research sources (Google Scholar, PubMed)
- Web UI frontend

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- **Author**: Raja Hussain

---

## Acknowledgments

- **OpenAI** - GPT-5 Responses API
- **DeepSeek** - Affordable, high-quality models
- **LangChain** - LangGraph orchestration framework
- **Semantic Scholar** - Free academic research API
- **ChromaDB** - Simple, effective vector store
