# Pronóstico de demanda con backtesting e intervalos de predicción

Pronóstico de ventas diarias a **28 días** comparando modelos clásicos y de machine learning, con validación por **backtesting de ventana móvil (8 folds)** e **intervalos de predicción calibrados con conformal prediction**.

> **English summary:** 28-day daily sales forecasting on synthetic data with trend, weekly/yearly seasonality, promotions and holidays. Compares naive baselines, Holt-Winters and gradient boosting with lag features using rolling-origin backtesting across 8 folds. Gradient boosting reduces MAE by 32% vs. the seasonal-naive baseline, and conformalized quantile regression yields 80% prediction intervals with 75% empirical coverage.

## Resultados (promedio de 8 folds)

| Modelo | MAE | WAPE |
|--------|----:|-----:|
| Naive | 112.9 | 25.7% |
| Naive estacional | 66.4 | 15.2% |
| Holt-Winters | 50.2 | 11.4% |
| **Gradient Boosting (lags + calendario)** | **45.3** | **10.5%** |

El gradient boosting mejora un **32%** el error frente al baseline estacional. Los intervalos del 80% cubren el **75%** de los valores reales en el backtest; sin la calibración conformal cubrían solo 61%.

![forecast](reports/forecast.png)

## Decisiones de diseño

- **Backtesting de ventana móvil:** cada fold entrena solo con datos anteriores y predice los 28 días siguientes, simulando el uso real.
- **Baselines honestos:** un modelo complejo solo vale si le gana al naive estacional.
- **Sin fuga de información:** los rezagos usan como mínimo 28 días de antigüedad, que es lo que realmente se conoce al momento de pronosticar a ese horizonte. Promociones y feriados sí se usan porque se planifican de antemano.
- **Intervalos calibrados:** regresión cuantílica con ajuste conformal sobre los últimos 120 días de entrenamiento.

## Cómo correrlo

```bash
pip install -r requirements.txt
cd src
python run.py
```

Tarda unos 15 segundos y deja métricas y gráficos en `reports/`.

## Estructura

```
src/data.py     serie sintética de ventas
src/models.py   baselines, Holt-Winters y gradient boosting con intervalos
src/run.py      backtesting, métricas y gráficos
reports/        resultados
```
