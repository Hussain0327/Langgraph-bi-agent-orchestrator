# Week 2 Completion Plan - November 7, 2025

**Goal**: Finish Phase 2 Week 2 today
**Current Status**: 80% complete
**Time Available**: ~6-8 hours
**Target**: 100% complete by end of day

---

## Task Breakdown

### Task 1: Verify Citations Work (15 min)
**Status**: Not started
**Priority**: Critical (blocks evaluation)

```bash
# Quick manual test
python cli.py
```

Test query: "What does academic research say about SaaS customer churn reduction strategies?"

Expected output:
- Research retrieval messages
- Citations in format: (Source: Author et al., Year)
- References section at end

If citations missing → debug before proceeding

---

### Task 2: Improve ML Routing (45-60 min)
**Status**: Not started
**Priority**: High

**Current accuracy**: 77.3%
**Target**: 85-90% (good enough for Week 2 completion)

#### Step 2a: Add boundary examples (30 min)

Edit `models/training_data.json` - add 20 examples:
- 5 clear LeadGen examples (boost 71% recall)
- 5 clear Operations examples (reduce false positives)
- 5 boundary clarifications (not operations)
- 5 boundary clarifications (not market)

#### Step 2b: Retrain model (15 min)

```bash
python3 src/ml/routing_classifier.py --retrain
```

#### Step 2c: Test accuracy (15 min)

```bash
python3 eval/routing_comparison.py --num-queries 50
```

Target: 85%+ accuracy

---

### Task 3: Run Full Evaluation (90-120 min)
**Status**: Not started
**Priority**: Critical

```bash
# This will take 60-90 minutes and cost $15-20
python3 eval/benchmark.py --mode both --num-queries 25
```

This generates:
- `eval/results_no_rag_TIMESTAMP.json`
- `eval/results_rag_TIMESTAMP.json`

**While it runs**: Work on Task 4 documentation

---

### Task 4: Statistical Analysis (30 min)
**Status**: Not started
**Priority**: High

After Task 3 completes, run analysis:

```bash
python3 -c "
from eval.analysis import EvaluationAnalyzer
import glob

# Find latest results
no_rag = sorted(glob.glob('eval/results_no_rag_*.json'))[-1]
rag = sorted(glob.glob('eval/results_rag_*.json'))[-1]

analyzer = EvaluationAnalyzer(no_rag, rag)
report = analyzer.generate_full_report()

with open('eval/ANALYSIS_REPORT.md', 'w') as f:
    f.write(report)

print('Analysis complete: eval/ANALYSIS_REPORT.md')
"
```

---

### Task 5: Create Final Report (60 min)
**Status**: Not started
**Priority**: Medium

Create `docs/WEEK2_COMPLETE.md` with:

1. Executive summary
2. What was built
3. Evaluation results
4. Statistical significance
5. ML routing performance
6. Key findings
7. Next steps (Week 3)

Use template from Task 4 analysis output

---

### Task 6: Git Commit (15 min)
**Status**: Not started
**Priority**: Medium

```bash
git status
git add .
git commit -m "Phase 2 Week 2 complete: ML routing + evaluation framework

- Added 20 boundary examples to training data
- Retrained ML classifier (77% → 85%+ accuracy)
- Completed 25-query evaluation (baseline vs RAG)
- Statistical analysis validates quality improvement
- Comprehensive Week 2 completion report

Week 2 deliverables:
- ML routing classifier (improved)
- Evaluation framework (validated)
- Statistical analysis module (tested)
- A/B testing framework (ready)
- Complete evaluation results

Next: Week 3 - Production optimization"

git push origin main
```

---

## Timeline

```
09:00 - 09:15   Task 1: Test citations                    [15 min]
09:15 - 10:15   Task 2: Improve ML routing                [60 min]
10:15 - 12:00   Task 3: Start evaluation (runs async)     [105 min runtime]
10:20 - 11:20   Task 5: Write report (while eval runs)    [60 min]
12:00 - 12:30   Task 4: Statistical analysis              [30 min]
12:30 - 12:45   Task 6: Git commit                        [15 min]
──────────────────────────────────────────────────────────
Total: ~4 hours active work
       ~2 hours waiting for evaluation
```

---

## Code to Write

### 1. Script to add training examples

File: `scripts/add_training_examples.py`

```python
import json

def add_boundary_examples():
    with open('models/training_data.json', 'r') as f:
        data = json.load(f)

    new_examples = [
        # LeadGen - boost recall
        {"query": "Can you follow up with these 5 inbound leads?", "agents": ["leadgen"]},
        {"query": "I need help qualifying these prospects", "agents": ["leadgen"]},
        {"query": "Book sales calls with our top 10 leads", "agents": ["leadgen"]},
        {"query": "Send outreach emails to cold leads", "agents": ["leadgen"]},
        {"query": "Create a lead nurturing sequence", "agents": ["leadgen"]},

        # Operations - reduce false positives
        {"query": "Optimize our manufacturing workflow", "agents": ["operations"]},
        {"query": "Improve team productivity and reduce bottlenecks", "agents": ["operations"]},
        {"query": "How do I automate our onboarding process?", "agents": ["operations"]},
        {"query": "Streamline our supply chain operations", "agents": ["operations"]},
        {"query": "Reduce operational costs in production", "agents": ["operations"]},

        # NOT operations
        {"query": "What's the best pricing model for SaaS?", "agents": ["financial", "market"]},
        {"query": "How can I reduce customer churn?", "agents": ["market"]},
        {"query": "Should I raise prices or increase volume?", "agents": ["financial"]},
        {"query": "What's our target customer profile?", "agents": ["market"]},
        {"query": "How much revenue can we expect next quarter?", "agents": ["financial"]},

        # NOT market
        {"query": "Set up automated email sequences", "agents": ["operations"]},
        {"query": "How much should I budget for paid ads?", "agents": ["financial"]},
        {"query": "Build a sales funnel automation", "agents": ["operations", "leadgen"]},
        {"query": "Calculate our customer lifetime value", "agents": ["financial"]},
        {"query": "Design an onboarding workflow", "agents": ["operations"]},
    ]

    data['train'].extend(new_examples)

    with open('models/training_data.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Added {len(new_examples)} boundary examples")
    print(f"Total training examples: {len(data['train'])}")

if __name__ == '__main__':
    add_boundary_examples()
```

### 2. Quick retrain script

File: `scripts/quick_retrain.py`

```python
import sys
sys.path.insert(0, '/workspaces/multi_agent_workflow')

from src.ml.routing_classifier import RoutingClassifier

classifier = RoutingClassifier()
classifier.train(
    data_path='models/training_data.json',
    num_epochs=5,
    save_path='models/routing_classifier.pkl'
)

print("\nRetraining complete!")
print("New model saved to: models/routing_classifier.pkl")
```

### 3. Quick accuracy check

File: `scripts/check_accuracy.py`

```python
import sys
sys.path.insert(0, '/workspaces/multi_agent_workflow')

from models.inspect_model import route_text
import json

with open('models/training_data.json', 'r') as f:
    data = json.load(f)

val_data = data['val']

correct = 0
total = len(val_data)

for example in val_data:
    query = example['query']
    expected = set(example['agents'])

    result = route_text(query)
    predicted = {result['label']}

    if predicted.intersection(expected):
        correct += 1

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.1%} ({correct}/{total})")
```

---

## Success Criteria

Week 2 complete when:

- [x] Citations verified working
- [x] ML routing accuracy ≥85%
- [x] 25-query evaluation completed
- [x] Statistical analysis shows significance (p < 0.05)
- [x] WEEK2_COMPLETE.md created
- [x] All code committed to git

---

## Execution Order

1. **Task 1** - Test citations (blocking)
2. **Task 2a** - Add examples
3. **Task 2b** - Retrain
4. **Task 2c** - Check accuracy
5. **Task 3** - Start evaluation (long-running)
6. **Task 5** - Write report (parallel to Task 3)
7. **Task 4** - Analyze results (after Task 3)
8. **Task 6** - Commit everything

---

Ready to execute. Start with Task 1.
