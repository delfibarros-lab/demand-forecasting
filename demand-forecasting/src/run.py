import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import generate_sales
from models import HORIZON, gradient_boosting, holt_winters, naive, seasonal_naive

REPORTS = Path(__file__).resolve().parent.parent / "reports"
N_FOLDS = 8


def wape(y, p):
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def mae(y, p):
    return float(np.abs(y - p).mean())


def main():
    REPORTS.mkdir(exist_ok=True)
    df = generate_sales()
    scores = {m: {"mae": [], "wape": []} for m in ["Naive", "Naive estacional", "Holt-Winters", "Gradient Boosting"]}
    coverage, last = [], None

    for k in range(N_FOLDS, 0, -1):
        cut = len(df) - k * HORIZON
        train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:cut + HORIZON].reset_index(drop=True)
        y = test["sales"].to_numpy()
        gb, bands = gradient_boosting(train, test)
        preds = {"Naive": naive(train, HORIZON), "Naive estacional": seasonal_naive(train, HORIZON),
                 "Holt-Winters": holt_winters(train, HORIZON), "Gradient Boosting": gb}
        for name, p in preds.items():
            scores[name]["mae"].append(mae(y, p))
            scores[name]["wape"].append(wape(y, p))
        coverage.append(float(((y >= bands[0.1]) & (y <= bands[0.9])).mean()))
        last = (train, test, preds, bands)

    summary = {
        "horizonte_dias": HORIZON,
        "folds": N_FOLDS,
        "modelos": {m: {"mae": round(float(np.mean(v["mae"])), 2), "wape_pct": round(100 * float(np.mean(v["wape"])), 2)}
                    for m, v in scores.items()},
        "cobertura_intervalo_80pct": round(float(np.mean(coverage)), 3),
    }
    base = summary["modelos"]["Naive estacional"]["mae"]
    best = min(summary["modelos"], key=lambda m: summary["modelos"][m]["mae"])
    summary["mejor_modelo"] = best
    summary["mejora_vs_naive_estacional_pct"] = round(100 * (1 - summary["modelos"][best]["mae"] / base), 1)
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    train, test, preds, bands = last
    hist = train.tail(60)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist["date"], hist["sales"], color="gray", label="Histórico")
    ax.plot(test["date"], test["sales"], color="black", label="Real")
    ax.plot(test["date"], preds["Gradient Boosting"], color="#e45756", lw=2, label="Gradient Boosting")
    ax.plot(test["date"], preds["Holt-Winters"], color="#4c78a8", lw=1.5, ls="--", label="Holt-Winters")
    ax.fill_between(test["date"], bands[0.1], bands[0.9], color="#e45756", alpha=0.2, label="Intervalo 80%")
    ax.set(title="Pronóstico a 28 días (último fold del backtest)", ylabel="Ventas diarias")
    ax.legend(ncol=3, fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(REPORTS / "forecast.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(summary["modelos"])
    ax.bar(names, [summary["modelos"][n]["wape_pct"] for n in names], color="#4c78a8")
    ax.set(ylabel="WAPE (%)", title="Error promedio en el backtest")
    plt.setp(ax.get_xticklabels(), rotation=15); fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png", dpi=150); plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
