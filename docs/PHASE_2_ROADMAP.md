# Phase 2 Complete Roadmap

**Project**: Business Intelligence Orchestrator v2
**Phase 2 Goal**: Research-augmented generation + ML optimization + production hardening

---

## Phase 2 Structure

Phase 2 has **3 weeks**, not 3 phases:

```
Phase 2: RAG + ML + Production
├─ Week 1: RAG Integration ✅ COMPLETE
├─ Week 2: ML Routing + Evaluation ✅ COMPLETE
└─ Week 3: Production Optimization ⏳ NOT STARTED
```

**No Phase 3 exists.** Week 3 is the final milestone.

---

## Week 1: RAG Integration ✅ COMPLETE

**Completed**: Nov 5, 2025
**Status**: 100% done

### What We Built

**Vector Store** (`src/vector_store.py` - 242 lines)
- ChromaDB wrapper
- Semantic search with OpenAI embeddings
- Persistent storage
- Collection management

**Research Retrieval** (`src/tools/research_retrieval.py` - 405 lines)
- Semantic Scholar API integration
- arXiv API integration
- 7-day caching
- Citation formatting (APA style)
- Relevance ranking

**Research Synthesis Agent** (`src/agents/research_synthesis.py` - 253 lines)
- Retrieves top-3 papers per query
- GPT-5 synthesis with high reasoning
- Extracts key findings
- Creates agent context

**Agent Updates** (4 files)
- `market_analysis.py`
- `operations_audit.py`
- `financial_modeling.py`
- `lead_generation.py`

All agents updated to accept `research_context` and include citations.

**Orchestrator Integration**
- Added research synthesis node to LangGraph
- RAG toggle (`enable_rag=True/False`)
- Sequential agent execution chain
- Citations propagate through workflow

**Testing**
- `test_rag_system.py` - 5 tests, all passing
- End-to-end RAG validation
- Citation formatting tests

### Deliverables

- 8 files created/modified
- ~1,200 lines of code
- RAG retrieval working
- Citations in agent outputs
- Toggle between RAG on/off

### Critical Bug Fixed

**Problem**: GPT-5 `reasoning_effort="medium/high"` used all tokens for reasoning, leaving zero for output.

**Solution**: Changed all agents to `reasoning_effort="low"`.

**Impact**: System went from 0-char outputs to 9,000+ char responses.

---

## Week 2: ML Routing + Evaluation ✅ COMPLETE

**Completed**: Nov 7, 2025
**Status**: 100% done

### What We Built

**ML Routing Classifier** (`src/ml/routing_classifier.py` - 334 lines)
- SetFit model with sentence-transformers
- 4 binary classifiers (one per agent)
- 77% exact match accuracy
- 20ms inference time
- $0 cost per route
- Model saved: `models/routing_classifier.pkl` (349MB)

**Training Data Pipeline**
- `scripts/export_langsmith_data.py` - LangSmith trace export
- `models/training_data.json` - 125 training examples, 22 validation
- Added 20 boundary examples for weak agents (leadgen, operations)
- Expandable to 200+ examples

**Evaluation Framework** (`eval/benchmark.py` - 583 lines)
- 25-query test suite covering all business categories
- LLM-as-judge quality scoring (factuality, helpfulness, comprehensiveness)
- Latency and cost tracking
- Citation detection
- Routing accuracy measurement
- Baseline vs RAG comparison

**Statistical Analysis** (`eval/analysis.py` - 550 lines)
- T-tests for quality comparison
- Cohen's d effect size calculation
- Cost-benefit analysis
- Citation correlation analysis
- P-value significance testing

**A/B Testing Framework** (`src/ab_testing.py` - 425 lines)
- Deterministic user assignment
- Session-based experiment tracking
- Real-time metrics aggregation
- Statistical significance testing
- Multi-variant support

**Routing Comparison** (`eval/routing_comparison.py` - 423 lines)
- ML vs GPT-5 routing benchmarks
- Accuracy comparison
- Latency comparison
- Cost comparison

**Automation Scripts** (7 scripts)
- `add_training_examples.py` - Add boundary examples
- `quick_retrain.py` - Retrain classifier
- `check_accuracy.py` - Validate accuracy
- `run_analysis.py` - Statistical analysis
- `auto_analyze.sh` - Auto-run analysis after eval
- `export_langsmith_data.py` - Training data export
- `try_load_model.py` - Model inspection

**Citation Formatting Fixed**
- Updated research synthesis citation format
- Updated all 4 agents with explicit citation requirements
- Changed from "IMPORTANT" to "CRITICAL CITATION REQUIREMENTS"
- Added citation examples to prompts

**Tests**
- `tests/test_routing_classifier.py` - 176 lines
- `tests/test_ab_testing.py` - 225 lines

### Deliverables

- 19 files created/modified
- 5,478 lines added (latest commit)
- 2,221 lines added today
- ML routing operational (77% accuracy)
- Evaluation framework complete
- Citation formatting fixed

### Performance

**ML Routing**:
- Market: 1.000 F1 (perfect)
- Financial: 0.875 F1 (good)
- Operations: 0.867 F1 (acceptable)
- LeadGen: 0.833 F1 (needs improvement - 71% recall)

**Inference**: 20ms vs 500ms (GPT-5)
**Cost**: $0 vs $0.01 (GPT-5)

### Philosophy

Prototype phase focuses on infrastructure over perfect metrics. 77% accuracy proves concept. Full evaluation deferred until production deployment when metrics matter for business decisions.

---

## Week 3: Production Optimization ⏳ NOT STARTED

**Scheduled**: Nov 20-26, 2025 (or when ready)
**Status**: 0% complete
**Estimated Work**: 40-50 hours over 7 days

### Goals

Transform prototype into production-ready system:
- Reduce latency by 3-5x (34s → 8-15s)
- Add caching layer
- Implement authentication
- Set up monitoring
- Deploy to production

---

### Task 1: Parallel Agent Execution (2 days, HIGH PRIORITY)

**Current Problem**: Agents run sequentially, wasting time
- Market agent: 8s
- Operations agent: 8s
- Financial agent: 8s
- LeadGen agent: 8s
- Total: 32s sequential

**Target**: All agents run simultaneously → ~8-10s total

**Implementation**:

File: `src/langgraph_orchestrator.py`

```python
# Current: Sequential
workflow.add_edge("market_agent", "operations_agent")
workflow.add_edge("operations_agent", "financial_agent")
workflow.add_edge("financial_agent", "leadgen_agent")

# Target: Parallel
from langgraph.graph import parallel

@parallel
def run_agents_parallel(state):
    results = await asyncio.gather(
        self.market_agent.analyze_async(query, research_context),
        self.operations_agent.audit_async(query, research_context),
        self.financial_agent.model_async(query, research_context),
        self.leadgen_agent.generate_async(query, research_context)
    )
    return results
```

**Changes needed**:
- Convert all agent methods to async
- Use `asyncio.gather()` for parallel execution
- Update LangGraph workflow to parallel node
- Update GPT5Wrapper to support async calls
- Test with concurrent API rate limits

**Expected impact**:
- Latency: 34s → 10-12s (3x faster)
- Throughput: 3x more queries per minute
- User experience: Much faster responses

---

### Task 2: Response Caching (1 day)

**Goal**: Cache expensive operations to reduce cost and latency

File: `src/cache.py` (new file)

**Implementation**:

```python
import redis
import hashlib
from typing import Optional

class ResponseCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)

    def get_research(self, query: str) -> Optional[dict]:
        key = f"research:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set_research(self, query: str, papers: dict, ttl: int = 604800):
        # 7 days TTL for research
        key = f"research:{hashlib.sha256(query.encode()).hexdigest()}"
        self.redis.setex(key, ttl, json.dumps(papers))

    def get_agent_response(self, agent: str, query: str) -> Optional[str]:
        key = f"agent:{agent}:{hashlib.sha256(query.encode()).hexdigest()}"
        return self.redis.get(key)

    def set_agent_response(self, agent: str, query: str, response: str, ttl: int = 86400):
        # 1 day TTL for agent responses
        key = f"agent:{agent}:{hashlib.sha256(query.encode()).hexdigest()}"
        self.redis.setex(key, ttl, response)
```

**Cache Strategy**:
- Research queries: 7 days TTL (papers don't change often)
- Agent responses: 1 day TTL (business context changes)
- Routing decisions: No cache (20ms is fast enough)
- Final synthesis: No cache (always unique)

**Dependencies**:
```bash
pip install redis hiredis
```

**Expected impact**:
- Research cache hit rate: 40-60%
- Cost savings: ~30% on repeated topics
- Latency improvement: ~5-10s for cached research

---

### Task 3: Authentication & Rate Limiting (1 day)

**Goal**: Secure API and prevent abuse

File: `src/main.py` (FastAPI updates)

**Implementation**:

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
security = HTTPBearer()

# JWT authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Rate limiting
@app.post("/analyze")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def analyze_query(
    query: str,
    user_id: str = Depends(verify_token)
):
    # Track usage per user
    track_usage(user_id, cost=calculate_cost(query))

    # Run orchestrator
    result = orchestrator.orchestrate(query)
    return result

# API key system
@app.post("/auth/register")
async def register_user(email: str):
    api_key = generate_api_key()
    save_user(email, api_key)
    return {"api_key": api_key}

# Usage tracking
def track_usage(user_id: str, cost: float):
    redis.hincrby(f"usage:{user_id}", "queries", 1)
    redis.hincrbyfloat(f"usage:{user_id}", "cost", cost)
```

**Features**:
- JWT token authentication
- API key system
- Per-user rate limiting (10 requests/min)
- Usage tracking (queries + cost)
- Admin endpoints for management

**Dependencies**:
```bash
pip install pyjwt slowapi
```

**Expected impact**:
- Prevents abuse
- Tracks per-user costs
- Enables tiered pricing
- Production-ready security

---

### Task 4: Monitoring & Observability (1-2 days)

**Goal**: Track system health and performance

File: `src/monitoring.py` (new file)

**Implementation**:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

# Metrics
query_counter = Counter('queries_total', 'Total queries processed', ['agent', 'status'])
latency_histogram = Histogram('query_latency_seconds', 'Query latency', ['agent'])
cost_counter = Counter('query_cost_dollars', 'Query cost in USD', ['agent'])
active_queries = Gauge('active_queries', 'Currently processing queries')
cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])

class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
        start_http_server(9090)  # Prometheus metrics on port 9090

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            active_queries.inc()

            try:
                await self.app(scope, receive, send)
                query_counter.labels(agent="all", status="success").inc()
            except Exception as e:
                query_counter.labels(agent="all", status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                latency_histogram.labels(agent="all").observe(duration)
                active_queries.dec()

# Error tracking
def log_error(error: Exception, context: dict):
    logging.error(f"Error: {str(error)}", extra=context)
    # Send to Sentry/DataDog/etc
    send_to_alerting(error, context)
```

**Dashboards** (Grafana):

1. **Performance Dashboard**
   - P50, P95, P99 latency
   - Queries per minute
   - Error rate
   - Active queries gauge

2. **Cost Dashboard**
   - Cost per query
   - Cost per agent
   - Daily/monthly spend
   - Cost by user

3. **Quality Dashboard**
   - Citation rate
   - Routing accuracy
   - Cache hit rate
   - LLM-as-judge scores

4. **Usage Dashboard**
   - Queries per user
   - Popular agents
   - Peak hours
   - User growth

**Alerting** (PagerDuty/Slack):
- Error rate >5%
- Latency >30s (P95)
- Cost >$100/day
- API rate limit errors

**Dependencies**:
```bash
pip install prometheus-client python-json-logger
```

**Expected impact**:
- Real-time visibility into system health
- Proactive error detection
- Cost tracking and optimization
- SLA monitoring

---

### Task 5: Production Deployment (1-2 days)

**Goal**: Deploy to production environment

**Docker Setup**:

File: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY .env .env

EXPOSE 8000 9090

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    environment:
      - REDIS_HOST=redis
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

  prometheus:
    image: prom/prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: always

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: always

volumes:
  redis_data:
  grafana_data:
```

**Deployment Steps**:

1. **Build and test locally**
   ```bash
   docker-compose build
   docker-compose up -d
   curl http://localhost:8000/health
   ```

2. **Load testing**
   ```bash
   ab -n 100 -c 10 http://localhost:8000/analyze
   ```

3. **Deploy to cloud** (AWS/Azure/GCP)
   - Set up container registry
   - Deploy to ECS/AKS/GKE
   - Configure load balancer
   - Set up auto-scaling

4. **Environment setup**
   ```bash
   # Production .env
   OPENAI_API_KEY=sk-...
   LANGSMITH_API_KEY=ls_...
   REDIS_HOST=prod-redis.internal
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   ```

**Expected impact**:
- Production-ready deployment
- Horizontal scaling capability
- High availability
- Monitoring integrated

---

## Week 3 Deliverables

### Code Files
- `src/cache.py` - Response caching layer
- `src/monitoring.py` - Prometheus metrics
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-service orchestration
- Updated `src/langgraph_orchestrator.py` - Async parallel execution
- Updated `src/main.py` - Auth & rate limiting

### Infrastructure
- Redis cache server
- Prometheus metrics collection
- Grafana dashboards
- Production deployment (Docker)

### Documentation
- `docs/WEEK3_COMPLETE.md` - Week 3 summary
- `docs/DEPLOYMENT.md` - Deployment guide
- `docs/MONITORING.md` - Monitoring setup
- `docs/API.md` - API documentation

### Performance Targets
- Latency: 8-15s (from 34s) ✓ 3-5x improvement
- Cache hit rate: 40-60%
- Uptime: 99.9%
- Error rate: <1%

---

## What's Left Overall

### Phase 2 Remaining Work

**Week 3 tasks** (7-10 days):
1. Parallel agent execution (2 days)
2. Response caching (1 day)
3. Authentication & rate limiting (1 day)
4. Monitoring & observability (1-2 days)
5. Production deployment (1-2 days)
6. Load testing & optimization (1 day)
7. Documentation & handoff (1 day)

**After Week 3 is complete**:
- Phase 2: 100% complete
- Project: 100% complete
- Production-ready system deployed

---

## Beyond Phase 2 (Optional Enhancements)

Not required for completion, but valuable additions:

### ML Routing Improvements
- Collect 200+ training examples per agent
- Retrain with 10 epochs
- Achieve 95%+ accuracy
- Add confidence scoring
- Fallback to GPT-5 for low confidence

### RAG Enhancements
- Add more research sources (Google Scholar, PubMed)
- Implement embedding-based reranking
- Citation validation
- Multi-language support

### Evaluation
- Run full 25-query benchmark
- Prove +18% quality improvement
- Statistical significance testing
- A/B test with real users

### Client Pilot
- Deploy to 2-3 ValtricAI clients
- Collect user feedback
- Measure satisfaction improvement
- Validate premium pricing (+$500-1000/mo)

### NYU Transfer Application
- Publish evaluation results
- Write technical paper
- Document system architecture
- Create portfolio presentation

---

## Timeline Summary

```
Phase 1: Modernization (COMPLETE)
  Nov 4, 2025 | 100% | GPT-5 + LangGraph

Phase 2: RAG + ML + Production
  Week 1 (Nov 5)     | 100% ✅ | RAG Integration
  Week 2 (Nov 6-7)   | 100% ✅ | ML Routing + Evaluation
  Week 3 (Nov 20-26) |   0% ⏳ | Production Optimization

Total: ~20-25 days work
Current: ~15 days complete (60%)
Remaining: ~7-10 days (Week 3)
```

---

## Success Criteria

### Week 3 Complete When:
- [ ] Parallel execution implemented (latency <15s)
- [ ] Caching layer operational (hit rate >40%)
- [ ] Authentication working (JWT + API keys)
- [ ] Monitoring deployed (Prometheus + Grafana)
- [ ] Production deployment successful
- [ ] Load testing passed (100+ concurrent users)
- [ ] Documentation complete

### Project Complete When:
- [x] Phase 1: GPT-5 + LangGraph (DONE)
- [x] Phase 2 Week 1: RAG (DONE)
- [x] Phase 2 Week 2: ML + Eval (DONE)
- [ ] Phase 2 Week 3: Production (PENDING)

---

## Current Status

**Today**: Nov 7, 2025
**Phase**: Phase 2, Week 2
**Status**: 100% COMPLETE ✅

**Latest Commit**: `4da2755` - "Phase 2 Week 2 complete: ML routing + evaluation infrastructure"

**Next Session**: Week 3 - Production optimization

**Time to Completion**: 7-10 days of work

---

**Created**: Nov 7, 2025
**Last Updated**: Nov 7, 2025 22:00
**Status**: Phase 2 Week 2 complete, Week 3 ready to start
