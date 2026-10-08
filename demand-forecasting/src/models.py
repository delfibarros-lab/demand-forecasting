import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HORIZON = 28
LAGS = [28, 35, 42, 364]


def naive(train, horizon):
    return np.repeat(train["sales"].iloc[-1], horizon).astype(float)


def seasonal_naive(train, horizon, period=7):
    last = train["sales"].to_numpy()[-period:]
    return np.tile(last, horizon // period + 1)[:horizon].astype(float)


def holt_winters(train, horizon):
    fit = ExponentialSmoothing(train["sales"].to_numpy(float), trend="add", seasonal="mul",
                               seasonal_periods=7).fit()
    return fit.forecast(horizon)


def make_features(df):
    out = pd.DataFrame(index=df.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = df["sales"].shift(lag)
    out["roll_mean_28"] = df["sales"].shift(HORIZON).rolling(28).mean()
    out["roll_mean_7"] = df["sales"].shift(HORIZON).rolling(7).mean()
    out["dow"] = df["date"].dt.dayofweek
    out["month"] = df["date"].dt.month
    out["doy"] = df["date"].dt.dayofyear
    out["promo"] = df["promo"]
    out["holiday"] = df["holiday"]
    return out


def _quantile_model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.05, random_state=0)


def gradient_boosting(train, future, alpha=0.2, calib_days=120):
    full = pd.concat([train, future.assign(sales=np.nan)], ignore_index=True)
    X = make_features(full)
    n = len(train)
    X_tr = X.iloc[:n].dropna()
    y_tr = train["sales"].loc[X_tr.index]
    X_fu = X.iloc[n:]

    point = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X_tr, y_tr)

    # conformal: calibra el ancho del intervalo con los últimos días de entrenamiento
    X_fit, y_fit = X_tr.iloc[:-calib_days], y_tr.iloc[:-calib_days]
    X_cal, y_cal = X_tr.iloc[-calib_days:], y_tr.iloc[-calib_days:]
    lo_m, hi_m = _quantile_model(alpha / 2).fit(X_fit, y_fit), _quantile_model(1 - alpha / 2).fit(X_fit, y_fit)
    scores = np.maximum(lo_m.predict(X_cal) - y_cal, y_cal - hi_m.predict(X_cal))
    level = min(1.0, (1 - alpha) * (1 + 1 / len(scores)))
    margin = np.quantile(scores, level)
    bands = {0.1: lo_m.predict(X_fu) - margin, 0.9: hi_m.predict(X_fu) + margin}
    return point.predict(X_fu), bands
