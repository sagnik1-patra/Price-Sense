from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date
from sqlalchemy.orm import Session
from .forecast_lstm import train_lstm_and_save
from .helpers import synthetic_series, daterange
from .fake_review_model import train_and_save as train_fake_review
from ..db import Base, Product, PricePoint, Review, init_engine, SessionLocal
from ..utils import read_settings, dump_json
import yaml

def run_training():
    settings = read_settings()
    data_csv = Path(settings["data_csv"])
    artifact_dir = Path(settings["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV
    df = pd.read_csv(data_csv, encoding="utf-8")
    # Normalize expected columns
    cols = {c.lower(): c for c in df.columns}
    def safe_col(name):
        for k in cols:
            if name in k:
                return cols[k]
        return None
    col_title = safe_col("product_name") or safe_col("title") or list(df.columns)[0]
    col_cat = safe_col("product_category_tree") or safe_col("category")
    col_brand = safe_col("brand")
    col_retail = safe_col("retail_price") or safe_col("mrp") or safe_col("price")
    col_disc = safe_col("discounted_price") or safe_col("discount_price")
    col_pid = safe_col("product_id") or safe_col("pid")
    col_rating = safe_col("product_rating") or safe_col("rating")
    col_desc = safe_col("description") or safe_col("product_description")
    col_url = safe_col("product_url") or safe_col("url")

    # 2) Create DB and seed products + synthetic price history + reviews
    engine = init_engine()
    Base.metadata.create_all(bind=engine)
    sess: Session = SessionLocal()

    # Limit number of products for speed
    df = df.head(int(settings.get("top_n_products", 300)))

    products = []
    for _, r in df.iterrows():
        p = Product(
            pid=str(r.get(col_pid)) if col_pid else None,
            title=str(r.get(col_title)),
            category=str(r.get(col_cat)) if col_cat else None,
            retail_price=float(r.get(col_retail)) if col_retail and not pd.isna(r.get(col_retail)) else None,
            discounted_price=float(r.get(col_disc)) if col_disc and not pd.isna(r.get(col_disc)) else None,
            brand=str(r.get(col_brand)) if col_brand else None,
            rating=float(r.get(col_rating)) if col_rating and not pd.isna(r.get(col_rating)) else None,
            url=str(r.get(col_url)) if col_url else None,
        )
        sess.add(p)
        products.append((p, str(r.get(col_desc)) if col_desc else ""))
    sess.commit()

    # Seed synthetic price history and reviews
    days = int(settings["seed_days"])
    today = date.today()
    all_series = []
    review_corpus = []
    for p, desc in products:
        base = p.discounted_price or p.retail_price or 100.0
        ser = synthetic_series(
            base_price=base,
            days=days,
            noise_frac=float(settings["price_noise_frac"]),
            deal_prob=float(settings["deal_drop_prob"]),
            deal_frac=float(settings["deal_drop_frac"]),
        )
        all_series.append(ser)
        for d, val in zip(daterange(today, days), ser):
            sess.add(PricePoint(product_id=p.id, date=d, price=float(val)))

        # Create 3 pseudo-reviews per product from description slices (for demo)
        desc = desc or p.title or "Good product."
        snippets = [desc[:140], desc[140:280], desc[280:420]]
        for sn in snippets:
            if sn and len(sn.strip()) > 10:
                sess.add(Review(product_id=p.id, author="demo", text=sn.strip(), stars=0.0))
                review_corpus.append(sn.strip())

    sess.commit()

    # 3) Train models and export artifacts
    h5_out = artifact_dir / "price_lstm.h5"
    metrics_out = artifact_dir / "metrics.json"
    fr_pkl = artifact_dir / "fake_review_lr.pkl"
    cfg_out = artifact_dir / "config.yaml"

    # LSTM (with safe fallback handled inside)
    lstm_seq = int(settings["lstm_sequence_len"])
    epochs = int(settings["lstm_epochs"])
    batch = int(settings["lstm_batch_size"])
    train_lstm_and_save(all_series, lstm_seq, epochs, batch, h5_out, metrics_out)

    # Fake-review model (weak-supervision on pseudo reviews)
    train_fake_review(review_corpus, fr_pkl, metrics_out.with_name("fake_review_metrics.json"))

    # Save an explicit YAML config snapshot used for training
    with open(cfg_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)

    # quick training manifest json
    dump_json(artifact_dir / "training_manifest.json", {
        "data_csv": str(data_csv),
        "artifacts": {
            "h5": str(h5_out),
            "json": str(metrics_out),
            "yaml": str(cfg_out),
            "pkl": str(fr_pkl)
        }
    })

    print("Artifacts saved to:", artifact_dir)

if __name__ == "__main__":
    run_training()
