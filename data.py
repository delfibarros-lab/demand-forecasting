import numpy as np
import pandas as pd


def generate_sales(start="2022-01-01", days=1095, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    weekly = np.array([0.85, 0.9, 0.95, 1.0, 1.15, 1.35, 1.2])[idx.dayofweek]
    yearly = 1 + 0.25 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365.25)
    trend = 220 + 0.18 * t
    promo = (rng.random(days) < 0.07).astype(int)
    holiday = idx.strftime("%m-%d").isin(["01-01", "05-01", "12-24", "12-25", "12-31"]).astype(int)
    level = trend * weekly * yearly * (1 + 0.30 * promo) * (1 - 0.35 * holiday)
    sales = rng.poisson(level * rng.lognormal(0, 0.08, days))
    return pd.DataFrame({"date": idx, "sales": sales, "promo": promo, "holiday": holiday})
