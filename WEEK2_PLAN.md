# Week 2 Implementation Plan: ML Routing + Evaluation

**Start Date**: November 5, 2025
**Duration**: 5-7 days
**Goal**: Replace GPT-5 routing with ML classifier and measure RAG quality improvements

---

## Objectives

1. **Build Evaluation Harness** - Measure quality, cost, latency
2. **Train ML Routing Classifier** - Replace GPT-5 routing (95%+ accuracy target)
3. **A/B Testing Framework** - Compare RAG vs non-RAG quantitatively
4. **Measure Improvements** - Validate +18% quality improvement hypothesis

---

## Task Breakdown

### Phase A: Evaluation Harness (Days 1-2)

#### Task A1: Create Test Query Suite ✅ STARTING
**File**: `eval/test_queries.json`

**Requirements**:
- 20-50 representative business queries
- Cover all agent types (market, operations, financial, leadgen)
- Mix of simple and complex queries
- Include ground truth expected agents

**Format**:
```json
{
  "queries": [
    {
      "id": 1,
      "query": "How can I improve SaaS customer retention?",
      "expected_agents": ["market", "operations", "leadgen"],
      "complexity": "medium",
      "category": "retention"
    }
  ]
}
```

**Time**: 1-2 hours

---

#### Task A2: Build Evaluation Framework
**File**: `eval/benchmark.py`

**Components**:
1. **Query Runner** - Execute queries through orchestrator
2. **Metrics Collector** - Track latency, cost, tokens
3. **LLM-as-Judge** - Quality scoring (GPT-4)
4. **Report Generator** - Markdown/CSV output

**Metrics to Track**:
```python
{
    "query_id": int,
    "mode": "rag" | "no_rag",
    "latency": float,           # seconds
    "cost": float,              # USD
    "tokens_used": int,
    "citation_count": int,
    "has_references": bool,
    "factuality_score": float,  # 0-1 from LLM judge
    "helpfulness_score": float, # 0-1 from LLM judge
    "agents_called": List[str],
    "routing_accuracy": float   # vs expected agents
}
```

**LLM-as-Judge Prompt**:
```
Evaluate this business intelligence recommendation:

Query: {query}
Response: {response}

Rate on:
1. Factuality (0-1): Are claims accurate and well-supported?
2. Helpfulness (0-1): Is the advice actionable and relevant?
3. Comprehensiveness (0-1): Does it address all aspects of the query?

Return JSON: {"factuality": 0.8, "helpfulness": 0.9, "comprehensiveness": 0.85}
```

**Time**: 4-6 hours

---

#### Task A3: Run Baseline Evaluation
**Goal**: Establish Phase 1 baseline metrics

**Process**:
1. Run 20 test queries through **non-RAG mode**
2. Collect all metrics
3. Generate baseline report
4. Save results: `eval/baseline_results.json`

**Expected Baseline** (Phase 1):
- Latency: 10-25s
- Cost: $0.10-0.30
- Factuality: ~0.70 (estimated)
- Helpfulness: ~0.75 (estimated)
- Citations: 0%

**Time**: 1 hour (automated)

---

#### Task A4: Run RAG Evaluation
**Goal**: Measure Phase 2 improvements

**Process**:
1. Run same 20 queries through **RAG mode**
2. Collect all metrics
3. Generate RAG report
4. Save results: `eval/rag_results.json`

**Expected RAG Results**:
- Latency: 30-60s
- Cost: $0.25-0.45
- Factuality: ~0.80-0.85 (+18% target)
- Helpfulness: ~0.85-0.90
- Citations: 60-80%

**Time**: 1 hour (automated)

---

#### Task A5: Statistical Analysis & Report
**File**: `eval/analysis.py`

**Analyses**:
1. **T-test** - Is quality improvement significant? (p < 0.05)
2. **Effect size** - Cohen's d for quality metrics
3. **Cost-benefit** - Quality gain vs cost increase
4. **Citation correlation** - Citations → quality relationship

**Report Output**: `eval/EVALUATION_REPORT.md`

**Time**: 2 hours

---

### Phase B: ML Routing Classifier (Days 3-4)

#### Task B1: Export Training Data from LangSmith
**File**: `scripts/export_langsmith_data.py`

**Requirements**:
- Export 200+ query traces from LangSmith
- Extract: `{query: str, agents_called: List[str]}`
- Clean and validate data
- Split: 70% train / 15% val / 15% test

**Data Format**:
```json
{
  "query": "How can I reduce CAC for B2B SaaS?",
  "agents": ["market", "financial", "leadgen"],
  "timestamp": "2025-11-05T..."
}
```

**Fallback**: If <200 traces, use synthetic data generation

**Time**: 2-3 hours

---

#### Task B2: Train Routing Classifier
**File**: `src/ml/routing_classifier.py`

**Model Choice**: **SetFit** (few-shot learning)
- Why: Works with limited data (50-200 examples)
- Fast inference (~20ms vs 500ms GPT-5)
- 90-95% accuracy achievable
- Small model size (~100MB)

**Alternative**: DistilBERT (if >500 examples available)

**Architecture**:
```python
class RoutingClassifier:
    def __init__(self):
        self.model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def train(self, X_train, y_train):
        # Multi-label classification
        # 4 binary classifiers (one per agent)

    def predict(self, query: str) -> List[str]:
        # Returns: ["market", "financial"]

    def predict_proba(self, query: str) -> Dict[str, float]:
        # Returns: {"market": 0.95, "operations": 0.12, ...}
```

**Training Process**:
1. Prepare data (encode queries)
2. Train 4 binary classifiers (one per agent)
3. Validate on val set
4. Test on test set
5. Save model checkpoint

**Target Metrics**:
- Exact match accuracy: >85%
- Per-agent F1 score: >0.90
- Inference time: <50ms

**Time**: 3-4 hours

---

#### Task B3: Integrate ML Classifier into Orchestrator
**File**: `src/langgraph_orchestrator.py`

**Changes**:
```python
class LangGraphOrchestrator:
    def __init__(self, enable_rag=True, use_ml_routing=False):
        self.use_ml_routing = use_ml_routing
        if use_ml_routing:
            self.ml_router = RoutingClassifier()
            self.ml_router.load("models/routing_classifier.pkl")

    def _router_node(self, state):
        if self.use_ml_routing:
            agents = self.ml_router.predict(state["query"])
        else:
            # Use GPT-5 semantic routing (current)
            agents = self._gpt5_route(state["query"])

        state["agents_to_call"] = agents
        return state
```

**Testing**:
- Compare ML routing vs GPT-5 routing accuracy
- Measure latency improvement
- Verify no quality degradation

**Time**: 2 hours

---

#### Task B4: Benchmark ML vs GPT-5 Routing
**File**: `eval/routing_comparison.py`

**Metrics**:
| Metric | GPT-5 Routing | ML Routing | Improvement |
|--------|--------------|------------|-------------|
| Accuracy | 90% | 95% | +5% |
| Latency | 500ms | 20ms | -96% |
| Cost | $0.01/query | $0.00/query | -100% |

**Time**: 1 hour

---

### Phase C: A/B Testing Framework (Days 5-6)

#### Task C1: Build A/B Test Manager
**File**: `src/ab_testing.py`

**Features**:
1. **Traffic Splitting** - 50/50 random assignment
2. **Session Tracking** - Persistent experiment groups
3. **Metrics Collection** - All queries logged
4. **Statistical Testing** - Auto-calculate significance

**Usage**:
```python
from src.ab_testing import ABTestManager

ab_test = ABTestManager(
    experiment_name="rag_vs_baseline",
    control="no_rag",
    treatment="rag",
    split_ratio=0.5
)

# Assign user to group
group = ab_test.assign_user(user_id="client_123")

# Log result
ab_test.log_result(
    user_id="client_123",
    query="...",
    response="...",
    metrics={...}
)

# Check significance
results = ab_test.analyze()
# Returns: {"control": {...}, "treatment": {...}, "p_value": 0.03, "significant": True}
```

**Time**: 3-4 hours

---

#### Task C2: Run 3-Day A/B Test
**Goal**: Real-world validation with production-like queries

**Setup**:
- Control: Phase 1 (no RAG, GPT-5 routing)
- Treatment 1: Phase 2 RAG (no RAG, GPT-5 routing)
- Treatment 2: Phase 2 Full (RAG + ML routing)

**Sample Size**: 30-50 queries per group (90-150 total)

**Success Criteria**:
- RAG quality improvement: p < 0.05
- User satisfaction: +0.5 points (4.0 → 4.5)
- Cost increase acceptable: <50%

**Time**: 3 days (mostly waiting)

---

### Phase D: Documentation & Reporting (Day 7)

#### Task D1: Create Evaluation Report
**File**: `eval/EVALUATION_REPORT.md`

**Sections**:
1. Executive Summary
2. Methodology
3. Baseline Results (Phase 1)
4. RAG Results (Phase 2)
5. Statistical Analysis
6. ML Routing Performance
7. A/B Test Results
8. Recommendations
9. Next Steps

**Time**: 2 hours

---

#### Task D2: Update Documentation
**Files**:
- `README.md` - Add evaluation results
- `claude.md` - Update with Week 2 findings
- `phase2.md` - Mark Week 2 complete

**Time**: 1 hour

---

## Implementation Schedule

### Day 1 (Today)
- ✅ Create test query suite (1-2 hours)
- ✅ Build evaluation framework (4-6 hours)
- ⏳ Run baseline evaluation (1 hour)

### Day 2
- ⏳ Run RAG evaluation (1 hour)
- ⏳ Statistical analysis (2 hours)
- ⏳ Start LangSmith data export (2 hours)

### Day 3
- ⏳ Train routing classifier (4 hours)
- ⏳ Integrate into orchestrator (2 hours)

### Day 4
- ⏳ Benchmark ML routing (2 hours)
- ⏳ Build A/B testing framework (4 hours)

### Day 5-6
- ⏳ Run A/B test (3 days with breaks)
- ⏳ Monitor and collect data

### Day 7
- ⏳ Analysis and reporting (3 hours)
- ⏳ Documentation updates (2 hours)

**Total Estimated Time**: 30-35 hours over 7 days

---

## Success Metrics

### Must-Have (Critical)
- ✅ Evaluation harness functional
- ✅ Baseline metrics collected
- ✅ RAG metrics collected
- ✅ Statistical significance calculated

### Should-Have (Important)
- ✅ ML classifier trained (>90% accuracy)
- ✅ ML routing integrated
- ✅ A/B test framework functional

### Nice-to-Have (Optional)
- ⏳ A/B test completed (3-day wait)
- ⏳ Publication-ready results
- ⏳ User feedback collected

---

## Risk Mitigation

### Risk 1: Insufficient Training Data (<200 examples)
**Mitigation**:
- Use SetFit (works with 50+ examples)
- Generate synthetic queries
- Start collecting production data now

### Risk 2: Quality Improvement Not Significant
**Mitigation**:
- Test with different query types
- Adjust paper retrieval (more/fewer papers)
- Iterate on synthesis prompts

### Risk 3: ML Routing Lower Accuracy than GPT-5
**Mitigation**:
- Keep GPT-5 routing as fallback
- Use confidence thresholds
- Hybrid approach: ML for high-confidence, GPT-5 for uncertain

---

## Deliverables

### Code Deliverables
1. ✅ `eval/test_queries.json` - Test suite
2. ✅ `eval/benchmark.py` - Evaluation framework
3. ⏳ `scripts/export_langsmith_data.py` - Data export
4. ⏳ `src/ml/routing_classifier.py` - ML model
5. ⏳ `src/ab_testing.py` - A/B testing
6. ⏳ `eval/analysis.py` - Statistical analysis

### Documentation Deliverables
1. ⏳ `eval/EVALUATION_REPORT.md` - Full results
2. ⏳ `eval/baseline_results.json` - Baseline data
3. ⏳ `eval/rag_results.json` - RAG data
4. ⏳ `WEEK2_COMPLETE.md` - Week 2 summary

### Model Deliverables
1. ⏳ `models/routing_classifier.pkl` - Trained model
2. ⏳ `models/training_data.json` - Training dataset
3. ⏳ `models/model_card.md` - Model documentation

---

## Getting Started (Right Now)

### Step 1: Create Directory Structure
```bash
mkdir -p eval models scripts
```

### Step 2: Create Test Query Suite
Start with `eval/test_queries.json`

### Step 3: Build Evaluation Framework
Implement `eval/benchmark.py`

### Step 4: Run First Evaluation
Test on 5 queries to validate framework

---

**Plan Created**: November 5, 2025
**Status**: Ready to start implementation
**First Task**: Create test query suite
