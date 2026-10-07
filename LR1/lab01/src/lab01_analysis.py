#!/usr/bin/env python3
"""ЛР 1, вариант 18. Дополнительный анализ (исходные скрипты не изменяются).

1) Сверка с демо преподавателя: отпечатки массивов и PNG для варианта 1.
2) Неопределённость разности средних: SE(Δ) = sqrt(s_b²/n + s_p²/n).
3) Парное сравнение: одинаковые seed дают одинаковые узлы решётки, поэтому
   разности d_i = m_pert(seed_i) − m_base(seed_i) убирают разброс между seed.
Запуск: python src/lab01_analysis.py --out artifacts/analysis
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lab01_experiment as ex  # noqa: E402
import lab01_generator as gen  # noqa: E402

METRICS = ex.METRICS


def series(params: dict, seeds: list[int]) -> list[dict]:
    return [gen.run_once(params, s) for s in seeds]


def analyse(variant: int, n: int, seed: int = 101) -> dict:
    context, factor = ex.variant_to_pair(variant)
    base = {**gen.DEFAULT_PARAMS, **ex.CONTEXTS[context][1]}
    pert = {**base, **ex.FACTORS[factor][1](base)}
    seeds = [seed + i for i in range(n)]
    b, p = series(base, seeds), series(pert, seeds)
    out = {}
    for m in METRICS:
        vb, vp = [r[m] for r in b], [r[m] for r in p]
        sb, sp = statistics.stdev(vb), statistics.stdev(vp)
        delta = statistics.fmean(vp) - statistics.fmean(vb)
        se = math.sqrt(sb ** 2 / n + sp ** 2 / n)
        d = [y - x for x, y in zip(vb, vp)]
        sd = statistics.stdev(d)
        out[m] = {
            "delta": delta, "s_base": sb, "delta_over_s": abs(delta) / sb,
            "se_delta": se, "delta_over_se": abs(delta) / se if se else math.inf,
            "paired_d": d, "paired_mean": statistics.fmean(d), "paired_s": sd,
            "paired_t": statistics.fmean(d) / (sd / math.sqrt(n)) if sd else math.inf,
            "sign_consistent": all(x < 0 for x in d) or all(x > 0 for x in d),
        }
    return {"variant": variant, "context": context, "factor": factor, "n": n, "seeds": seeds, "metrics": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    demo_expected = {"sha256_pixels_prefix": "93fae5913d797284",
                     "metrics_n5": {"mean": 0.4782, "contrast": 0.1311, "edge": 0.0097, "entropy": 6.9969}}
    demo = gen.run_once({**gen.DEFAULT_PARAMS, **ex.CONTEXTS["clouds"][1]}, 101)
    res = {
        "demo_check": {"expected": demo_expected, "observed_sha256_pixels": demo["sha256_pixels"],
                       "observed_sha256_png": demo["sha256_png"],
                       "pixels_match": demo["sha256_pixels"].startswith(demo_expected["sha256_pixels_prefix"])},
        "v18_n5": analyse(18, 5),
        "v18_n20": analyse(18, 20),
    }
    (a.out / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("demo pixels match:", res["demo_check"]["pixels_match"])
    for key in ("v18_n5", "v18_n20"):
        print(f"\n{key}: metric | Δ | |Δ|/s_base | |Δ|/SE(Δ) | paired mean d | paired s_d | t | знак d одинаков")
        for m, t in res[key]["metrics"].items():
            print(f"  {m:8s} {t['delta']:+.6f} {t['delta_over_s']:.2f} {t['delta_over_se']:.2f} "
                  f"{t['paired_mean']:+.6f} {t['paired_s']:.6f} {t['paired_t']:.1f} {t['sign_consistent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
