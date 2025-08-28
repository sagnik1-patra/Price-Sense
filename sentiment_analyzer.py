# Thin wrapper around HuggingFace pipeline (optional at runtime)
from typing import List, Dict, Any

def get_pipeline():
    try:
        from transformers import pipeline
        # Small, commonly cached model
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        return None

def analyze(texts: List[str]) -> List[Dict[str, Any]]:
    nlp = get_pipeline()
    if nlp is None:
        # Fallback: simple rule-based sentiment (positive if 'good'/'great', negative if 'bad'/'poor')
        out = []
        for t in texts:
            tl = (t or "").lower()
            if any(w in tl for w in ["bad", "poor", "worst", "awful"]):
                out.append({"label": "NEGATIVE", "score": 0.99})
            elif any(w in tl for w in ["good", "great", "excellent", "amazing", "love"]):
                out.append({"label": "POSITIVE", "score": 0.99})
            else:
                out.append({"label": "NEUTRAL", "score": 0.51})
        return out
    return nlp(texts, truncation=True)
