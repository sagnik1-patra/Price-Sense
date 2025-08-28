import numpy as np
import random
from datetime import datetime, timedelta

def synthetic_series(base_price: float, days: int, noise_frac: float = 0.08, deal_prob: float = 0.07, deal_frac: float = 0.12):
    prices = []
    p = base_price
    for d in range(days):
        noise = np.random.uniform(-noise_frac, noise_frac) * p
        p = max(1.0, p + noise)
        if random.random() < deal_prob:
            p = max(1.0, p * (1.0 - deal_frac))
        prices.append(p)
    return np.array(prices, dtype=np.float32)

def daterange(end_date, days):
    for i in range(days):
        yield end_date - timedelta(days=(days - 1 - i))
