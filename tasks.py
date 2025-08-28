from .celery_worker import app
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import date, timedelta
import numpy as np
from .db import SessionLocal, PricePoint, Product

@app.task
def tick_prices():
    # Simulate a new day of price movement (for demo only)
    db: Session = SessionLocal()
    today = date.today()
    products = db.execute(select(Product)).scalars().all()
    for p in products:
        last = db.execute(
            select(PricePoint).where(PricePoint.product_id==p.id).order_by(PricePoint.date.desc()).limit(1)
        ).scalar_one_or_none()
        base = (last.price if last else (p.discounted_price or p.retail_price or 100.0))
        noise = np.random.uniform(-0.05, 0.05) * base
        new_price = max(1.0, base + noise)
        db.add(PricePoint(product_id=p.id, date=today, price=float(new_price)))
    db.commit()
    db.close()
    return {"updated": len(products)}
