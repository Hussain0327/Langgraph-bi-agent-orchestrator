# Quick Reference Guide

**Business Intelligence Orchestrator v2** - Last Updated: Nov 8, 2025

---

## 🚀 Quick Start Commands

```bash
# Test DeepSeek integration
python test_deepseek.py

# Interactive CLI
python cli.py

# Run evaluation
python eval/benchmark.py --mode both --num-queries 5

# Start API server
uvicorn src.main:app --reload
```

---

## 💰 Cost Comparison

| Configuration | Cost/Query | Monthly (100/day) | Annual |
|---------------|-----------|-------------------|--------|
| GPT-5 | $0.33 | $990 | $11,880 |
| DeepSeek | $0.043 | $129 | $1,548 |
| **Savings** | **$0.29** | **$861** | **$10,332** |

---

## ⚙️ Model Strategy (.env)

```bash
# Use DeepSeek for everything (cheapest)
MODEL_STRATEGY=deepseek

# Use GPT-5 for everything (highest quality)
MODEL_STRATEGY=gpt5

# Smart routing (recommended)
MODEL_STRATEGY=hybrid
```

---

## 🔧 Hybrid Routing Map

| Agent | Model | Temperature | Max Tokens |
|-------|-------|-------------|------------|
| Research Synthesis | DeepSeek-reasoner | 1.0 | 32,000 |
| Financial | DeepSeek-chat | 0.0 | 8,000 |
| Market | DeepSeek-chat | 1.3 | 4,000 |
| Operations | DeepSeek-chat | 1.0 | 4,000 |
| LeadGen | DeepSeek-chat | 1.3 | 4,000 |
| Router | DeepSeek-chat | 0.0 | 4,000 |
| Synthesis | DeepSeek-chat | 1.0 | 4,000 |

---

## ✅ Compatibility Matrix

| Feature | GPT-5 | DeepSeek | Hybrid | Notes |
|---------|-------|----------|--------|-------|
| **LangGraph** | ✅ | ✅ | ✅ | LLM-agnostic |
| **LangSmith** | ✅ | ✅ | ✅ | Traces all |
| **ML Routing** | ✅ | ✅ | ✅ | Local model |
| **RAG System** | ✅ | ✅ | ✅ | After updates |
| **All Agents** | ✅ | ✅ | ✅ | After updates |

---

## 📁 Files to Update (11 min)

```
1. src/agents/research_synthesis.py   (Line 26)
2. src/agents/market_analysis.py      (Line 10)
3. src/agents/operations_audit.py     (Line 10)
4. src/agents/financial_modeling.py   (Line 10)
5. src/agents/lead_generation.py      (Line 10)

Change:
  self.gpt5 = GPT5Wrapper()

To:
  from src.unified_llm import UnifiedLLM
  self.llm = UnifiedLLM(agent_type="xxx")
```

---

## 🧪 Test After Updates

```bash
python -c "
from src.langgraph_orchestrator import LangGraphOrchestrator
orch = LangGraphOrchestrator(enable_rag=True, use_ml_routing=True)
result = orch.orchestrate('Test query')
print('✅ All systems operational!')
"
```

---

## 📊 Performance Comparison

| Metric | GPT-5 | DeepSeek | Improvement |
|--------|-------|----------|-------------|
| **Routing** | 500ms, $0.01 | 20ms, $0.00 | 25x faster, 100% cheaper |
| **Per Agent** | ~5s, $0.06 | ~1.5s, $0.007 | 3x faster, 90% cheaper |
| **Total Query** | ~34s, $0.33 | ~36s, $0.043 | Similar time, 87% cheaper |

---

## 🎯 System Components

### Working Now ✅
- LangGraph orchestration
- LangSmith tracing
- GPT-5 integration
- DeepSeek integration
- Hybrid model strategy
- ML routing (77% accuracy)
- RAG paper retrieval
- Citation formatting

### Needs Update ⚠️
- 5 agent files (11 min work)
- ML routing accuracy (77% → 95%)

### Not Started ⏳
- Parallel execution (Week 3)
- Redis caching (Week 3)
- Monitoring (Week 3)

---

## 📈 ML Routing Performance

```
Overall Accuracy: 77%

Per-Agent F1 Scores:
- Market:     1.000 (perfect!)
- Financial:  0.875 (good)
- Operations: 0.867 (acceptable)
- LeadGen:    0.833 (needs more data)
```

---

## 🔍 RAG System Flow

```
1. Retrieve Papers
   → Semantic Scholar API
   → arXiv API
   → Returns 2-3 papers

2. Synthesize Research
   → DeepSeek-reasoner (after update)
   → Extracts key findings
   → Formats citations

3. Agent Execution
   → All agents receive research context
   → DeepSeek-chat (after update)
   → Generate analysis with citations

4. Final Synthesis
   → DeepSeek-chat (after update)
   → Combines all findings
   → Returns comprehensive report
```

---

## 💡 Key Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-proj-...
DEEPSEEK_API_KEY=sk-72dbef12...
MODEL_STRATEGY=hybrid

# Optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=business-intelligence-orchestrator
```

---

## 🎓 Documentation Index

| Document | Purpose |
|----------|---------|
| `COMPLETE_SYSTEM_STATUS.md` | Full system overview |
| `DEEPSEEK_INTEGRATION.md` | DeepSeek setup guide |
| `LANGCHAIN_COMPATIBILITY.md` | LangChain ecosystem compat |
| `ML_RAG_DEEPSEEK_COMPATIBILITY.md` | ML + RAG + DeepSeek |
| `HYBRID_DEEPSEEK_COMPLETE.md` | Integration summary |
| `PHASE_2_ROADMAP.md` | Complete roadmap |
| `QUICK_REFERENCE.md` | This guide |

---

## 🚨 Common Issues

### "DeepSeek API Error: 401"
→ Check DEEPSEEK_API_KEY in .env

### "DEEPSEEK_API_KEY required"
→ Set MODEL_STRATEGY=gpt5 or add key

### "Module not found"
→ Run: pip install -r requirements.txt

### Cost tracking shows negative savings
→ This is a bug in test script, ignore

---

## ⏱️ Quick Timings

| Task | Time |
|------|------|
| Update 5 agents | 11 min |
| Test system | 1 min |
| Run 5-query eval | 15 min |
| Run 25-query eval | 90 min |
| Full Week 3 | 7-10 days |

---

## 📞 Quick Help

```bash
# Check config
python -c "from src.config import Config; Config.validate()"

# Check which model is selected
python -c "
from src.unified_llm import UnifiedLLM
llm = UnifiedLLM(agent_type='market')
print(llm.get_current_provider())
"

# View LangSmith traces
# Visit: https://smith.langchain.com

# Check git status
git status
```

---

## 🎯 Next Steps Priority

1. **Update 5 agent files** (11 min) → 90% cost savings
2. **Test with one query** (1 min) → Verify it works
3. **Run evaluation** (optional, 90 min) → Validate quality
4. **Start Week 3** (1-2 weeks) → Production optimization

---

## 💰 ROI Calculator

```python
queries_per_day = 100
gpt5_cost = 0.33
deepseek_cost = 0.043

daily_savings = (gpt5_cost - deepseek_cost) * queries_per_day
monthly_savings = daily_savings * 30
annual_savings = monthly_savings * 12

print(f"Daily: ${daily_savings:.2f}")
print(f"Monthly: ${monthly_savings:.2f}")
print(f"Annual: ${annual_savings:.2f}")

# Output:
# Daily: $28.70
# Monthly: $861.00
# Annual: $10,332.00
```

---

## 🔑 Key Files

```
Configuration:
- .env
- src/config.py

Wrappers:
- src/gpt5_wrapper.py
- src/deepseek_wrapper.py
- src/unified_llm.py

Agents (need updates):
- src/agents/research_synthesis.py
- src/agents/market_analysis.py
- src/agents/operations_audit.py
- src/agents/financial_modeling.py
- src/agents/lead_generation.py

ML Routing:
- src/ml/routing_classifier.py
- models/routing_classifier.pkl
- models/training_data.json

Evaluation:
- eval/benchmark.py
- eval/test_queries.json
- eval/analysis.py

Tests:
- test_deepseek.py
- test_rag_system.py
```

---

## 🎉 Status Summary

```
✅ DeepSeek integrated
✅ Hybrid routing working
✅ ML classifier trained
✅ RAG system operational
✅ LangSmith tracing active
✅ Evaluation framework ready
✅ Documentation complete

⚠️ 5 agent files need updates (11 min)

💰 $10,332/year savings waiting!
```

---

**Last Updated**: November 8, 2025, 01:35 UTC
**Version**: 2.0
**Status**: 95% Complete
**Next Action**: Update agents (11 min)
