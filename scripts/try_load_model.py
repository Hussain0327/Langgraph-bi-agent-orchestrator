import os, pickle
p = 'models/routing_classifier.pkl'
print('exists:', os.path.exists(p), 'size:', os.path.getsize(p) if os.path.exists(p) else None)
try:
    import torch
    print('torch:', torch.__version__)
except Exception:
    print('torch not available')
try:
    import joblib
    model = joblib.load(p)
    print('loaded with joblib:', type(model))
except Exception as e:
    print('joblib load failed:', repr(e))
    try:
        with open(p, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
            print('loaded with pickle:', type(model))
    except Exception as e2:
        print('pickle load failed:', repr(e2))
        raise
try:
    print('has predict:', hasattr(model, 'predict'))
    print('repr head:', repr(model)[:500])
except Exception:
    pass