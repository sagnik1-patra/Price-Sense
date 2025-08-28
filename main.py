from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from datetime import date, timedelta
from pathlib import Path
import json
import yaml
import numpy as np

from .db import SessionLocal, init_engine, Base, Product, PricePoint, Review
from .utils import read_settings
from .ml.forecast_lstm import forecast_next_k
from .ml.fake_review_model import predict_proba
from .ml.sentiment_analyzer import analyze as sent_analyze
import pickle

app = FastAPI(title="PriceSense API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB engine on import
engine = init_engine()
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/products")
def products(limit: int = 50, q: str = "", db: Session = next(get_db())):
    stmt = select(Product)
    if q:
        stmt = stmt.where(Product.title.ilike(f"%{q}%"))
    stmt = stmt.limit(limit)
    items = db.execute(stmt).scalars().all()
    out = []
    for p in items:
        last_price = db.execute(
            select(PricePoint.price).where(PricePoint.product_id == p.id).order_by(PricePoint.date.desc()).limit(1)
        ).scalar_one_or_none()
        out.append({
            "id": p.id,
            "title": p.title,
            "brand": p.brand,
            "category": p.category,
            "retail_price": p.retail_price,
            "discounted_price": p.discounted_price,
            "rating": p.rating,
            "last_price": last_price,
            "url": p.url
        })
    return out

@app.get("/product/{pid}/price-history")
def price_history(pid: int, days: int = 60, db: Session = next(get_db())):
    stmt = (select(PricePoint.date, PricePoint.price)
            .where(PricePoint.product_id == pid)
            .order_by(PricePoint.date.desc())
            .limit(days))
    rows = db.execute(stmt).all()
    rows = rows[::-1]  # ascending
    if not rows:
        raise HTTPException(404, "No price history")
    return [{"date": str(d), "price": float(p)} for d, p in rows]

@app.get("/product/{pid}/forecast")
def forecast(pid: int, horizon: int = 7, db: Session = next(get_db())):
    settings = read_settings()
    h5_path = Path(settings["artifact_dir"]) / "price_lstm.h5"
    stmt = (select(PricePoint.price)
            .where(PricePoint.product_id == pid)
            .order_by(PricePoint.date.asc()))
    series = [float(x[0]) for x in db.execute(stmt).all()]
    if not series:
        raise HTTPException(404, "No price history")
    preds = forecast_next_k(np.array(series, dtype=float), horizon, int(settings["lstm_sequence_len"]), h5_path)
    future_dates = []
    # compute last date
    last_dt = db.execute(select(PricePoint.date).where(PricePoint.product_id==pid).order_by(PricePoint.date.desc()).limit(1)).scalar_one()
    for i in range(1, horizon+1):
        future_dates.append(str(last_dt + timedelta(days=i)))
    return [{"date": d, "pred_price": float(p)} for d, p in zip(future_dates, preds)]

@app.get("/product/{pid}/reviews/sentiment")
def sentiment(pid: int, db: Session = next(get_db())):
    rows = db.execute(select(Review.text).where(Review.product_id == pid).limit(200)).all()
    texts = [r[0] for r in rows]
    if not texts:
        raise HTTPException(404, "No reviews")
    res = sent_analyze(texts)
    # summarize distribution
    dist = {}
    for r in res:
        label = r.get("label", "NEUTRAL")
        dist[label] = dist.get(label, 0) + 1
    total = sum(dist.values())
    return {"distribution": {k: v for k, v in dist.items()}, "total": total}

@app.get("/product/{pid}/reviews/fake-scan")
def fake_scan(pid: int, db: Session = next(get_db())):
    settings = read_settings()
    pkl = Path(settings["artifact_dir"]) / "fake_review_lr.pkl"
    if not pkl.exists():
        raise HTTPException(500, "Model not trained yet.")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    rows = db.execute(select(Review.text).where(Review.product_id == pid).limit(200)).all()
    texts = [r[0] for r in rows]
    if not texts:
        raise HTTPException(404, "No reviews")
    probs = predict_proba(clf, texts)
    flagged = sum(1 for pr in probs if pr >= 0.5)
    return {"reviews_scanned": len(texts), "flagged_suspicious": flagged, "flag_rate": float(flagged/len(texts))}
