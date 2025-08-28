from dataclasses import dataclass
import re
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path
import json

@dataclass
class ReviewFeatures:
    length: int
    exclam: int
    caps_ratio: float
    unique_ratio: float
    digits: int
    words: int

def featurize(text: str) -> ReviewFeatures:
    t = text or ""
    words = re.findall(r"[A-Za-z0-9']+", t)
    words_lower = [w.lower() for w in words]
    unique_ratio = len(set(words_lower)) / (len(words_lower) + 1e-6)
    caps_ratio = sum(1 for c in t if c.isupper()) / (len(t) + 1e-6)
    return ReviewFeatures(
        length=len(t),
        exclam=t.count("!"),
        caps_ratio=float(caps_ratio),
        unique_ratio=float(unique_ratio),
        digits=sum(c.isdigit() for c in t),
        words=len(words),
    )

def vectorize(feats: ReviewFeatures) -> List[float]:
    return [feats.length, feats.exclam, feats.caps_ratio, feats.unique_ratio, feats.digits, feats.words]

def weak_label(text: str) -> int:
    # Heuristic pseudo-label: 1 = fake/suspicious, 0 = genuine
    f = featurize(text)
    if f.length < 25: return 1
    if f.exclam >= 3: return 1
    if f.caps_ratio > 0.35: return 1
    if f.unique_ratio < 0.4 and f.words > 6: return 1
    return 0

def train_and_save(reviews: List[str], pkl_out: Path, metrics_out: Path):
    X, y = [], []
    for r in reviews:
        f = featurize(r)
        X.append(vectorize(f))
        y.append(weak_label(r))
    if not X:
        X = [[10,0,0.0,1.0,0,2]]
        y = [0]
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    # Save model
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_out, "wb") as f:
        pickle.dump(clf, f)

    metrics = {
        "trained": True,
        "n_reviews": int(len(reviews)),
        "pos_rate": float(y.mean()) if len(y) else 0.0,
        "features": ["length", "exclam", "caps_ratio", "unique_ratio", "digits", "words"]
    }
    Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    return metrics

def predict_proba(clf, texts: List[str]) -> List[float]:
    X = np.array([vectorize(featurize(t)) for t in texts], dtype=float)
    return clf.predict_proba(X)[:,1].tolist()
