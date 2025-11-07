* What the .pkl actually contains.
* How we inspected it without opening a 348 MB file.
* How we built and fixed the router.
* What the current model behavior is (yes/no per agent).
* What to do next.

---

# Routing Classifier Summary (Today’s Work)

## 1. What the `.pkl` file is

Your file at `/workspaces/multi_agent_workflow/models/routing_classifier.pkl` is not just “a model.” It is a bundle (a dict) with four parts:

```text
{
  "models": { "financial": ..., "leadgen": ..., "market": ..., "operations": ... },
  "agent_labels": ["financial", "leadgen", "market", "operations"],
  "base_model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "training_metrics": { ... }  # the JSON you pasted earlier
}
```

So:

* `models` is the heavy part (per-agent classifiers).
* `agent_labels` is the order.
* `base_model_name` tells you what the run was based on.
* `training_metrics` preserves the run results (timestamp, epochs, per-agent precision/recall, exact match).

This is why the file is ~348 MB.

## 2. What the metrics told us

From `training_metrics` we confirmed:

* Base model: `sentence-transformers/all-MiniLM-L6-v2`
* 4 agents: financial, leadgen, market, operations
* Market was perfect on this split (1.0 across the board)
* Leadgen recall was the lowest (0.714...)
* Exact match accuracy was about 0.77, which means about 23 percent of samples would route to the wrong agent if used as-is

Conclusion: model is usable, but data is the bottleneck, especially for leadgen and operations.

## 3. How we inspected a 348 MB pickle safely

Instead of trying to open it in the editor, we:

1. Wrote a small Python script to `pickle.load(...)` the file.
2. Printed only the top-level `type(...)` and keys.
3. Printed the small pieces (labels, base model, metrics).
4. Did not print the actual model objects.

That gave us visibility without dumping 300+ MB to the terminal.

## 4. Building the router

You wanted to actually use the file, not just look at it. So we wrote a router that:

1. Loads the `.pkl` once.
2. Loops through each label.
3. Calls that label’s model on the input text.
4. Normalizes the output to a float.
5. Picks the best label.

Your sample run:

```python
"Can you follow up with this inbound lead and book a call?"
```

produced:

```python
{'label': 'leadgen',
 'scores': {'financial': 0.0, 'leadgen': 1.0, 'market': 0.0, 'operations': 0.0},
 'base_model': 'sentence-transformers/all-MiniLM-L6-v2'}
```

This shows the router works end to end.

## 5. Fixing the “string output” problem

One of the per-agent models returned strings like `"no"`, not numbers. Your first router tried to cast that to `float(...)` and crashed.

We fixed it by adding `_to_score(...)` that:

* accepts tensors
* accepts lists
* accepts `"yes"/"no"`
* converts everything to a float in `[0, 1]`

After that, the router ran cleanly.

## 6. Making routing deterministic

Because you have 4 separate yes/no classifiers, two of them could return `1.0` for the same text. To avoid random choices, we added a priority list:

```python
PRIORITY = ["leadgen", "operations", "financial", "market"]
```

Now if there is a tie, the router picks the first in that list. This is important for agentic workflows.

## 7. What this tells us about your current setup

* You trained 4 independent binary detectors.
* The router is now the piece that turns those 4 answers into 1 final decision.
* The output format is JSON-friendly, so n8n, FastAPI, or a frontend can use it.
* The weakest part right now is data, not code.

## 8. What you should not commit

The `.pkl` is 348 MB. Do not push it to Git. Add:

```text
models/*.pkl
models/*.pt
models/*.bin
```

to `.gitignore`, and store the big file in Google Drive or similar.

## 9. Next move

Add 15 to 20 near-boundary examples for leadgen and operations, retrain, and overwrite the `.pkl`. That will push your exact-match rate closer to something you can rely on in production.

---