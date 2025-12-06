import pickle
from pathlib import Path
PKL_PATH = Path('/workspaces/multi_agent_workflow/models/routing_classifier.pkl')
with PKL_PATH.open('rb') as f:
    BUNDLE = pickle.load(f)
MODELS = BUNDLE['models']
LABELS = BUNDLE['agent_labels']
PRIORITY = ['leadgen', 'operations', 'financial', 'market']

def _to_score(pred):
    if hasattr(pred, 'item'):
        return float(pred.item())
    if isinstance(pred, (list, tuple)):
        v = pred[0]
        if hasattr(v, 'item'):
            return float(v.item())
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ('yes', 'true', '1'):
                return 1.0
            if v in ('no', 'false', '0'):
                return 0.0
            raise ValueError(f'Cannot convert string prediction to score: {v}')
        return float(v)
    if isinstance(pred, str):
        v = pred.strip().lower()
        if v in ('yes', 'true', '1'):
            return 1.0
        if v in ('no', 'false', '0'):
            return 0.0
        raise ValueError(f'Cannot convert string prediction to score: {v}')
    return float(pred)

def route_text(text: str):
    scores = {}
    for label in LABELS:
        model = MODELS[label]
        if callable(model):
            raw = model([text])
        elif hasattr(model, 'predict'):
            raw = model.predict([text])
        else:
            raise RuntimeError(f'Model for {label} is not callable and has no predict()')
        score = _to_score(raw)
        scores[label] = score
    max_score = max(scores.values())
    candidates = [l for l, s in scores.items() if s == max_score]
    for p in PRIORITY:
        if p in candidates:
            best_label = p
            break
    return {'label': best_label, 'scores': scores, 'base_model': BUNDLE.get('base_model_name')}
if __name__ == '__main__':
    sample = 'Can you follow up with this inbound lead and book a call?'
    print(route_text(sample))