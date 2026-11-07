# ЛР 9, вариант 18 (контекст 3): оценка девяти ответов языковой модели. Только стандартная библиотека.
# Файлы: answers/ctx3_<zero|few|step>_r<1..3>.txt (ответ целиком, без правок). Результат: печать таблицы и res9_llm_v18.json
import json, os, sys
import lab09_llm as L

CTX = 3
D = sys.argv[1] if len(sys.argv) > 1 else "answers"
NAMES = {"zero": "zero-shot", "few": "few-shot", "step": "пошаговые инструкции"}


def mean(x):
    return sum(x) / len(x)


def main():
    out, missing = {}, []
    for a in NAMES:
        paths = [os.path.join(D, f"ctx{CTX}_{a}_r{i}.txt") for i in (1, 2, 3)]
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            missing += miss
            continue
        rs = [L.score_file(p, CTX) for p in paths]
        out[a] = dict(k1=mean([r["k1"] for r in rs]), k2=mean([r["k2"] for r in rs]), k3=mean([r["k3"] for r in rs]),
                      k4=L.stability(paths), k1_each=[r["k1"] for r in rs], k2_each=[r["k2"] for r in rs],
                      k3_each=[r["k3"] for r in rs], errors=[r["errors"] for r in rs])
    if missing:
        print("нет файлов:", *missing, sep="\n  ")
    print(f"{'приём':22s} {'K1':>5s} {'K2':>6s} {'K3':>5s} {'K4':>6s}")
    for a, r in out.items():
        print(f"{NAMES[a]:22s} {r['k1']:5.2f} {r['k2']:6.3f} {r['k3']:5.2f} {r['k4']:6.3f}")
        for i, e in enumerate(r["errors"], 1):
            if e:
                print(f"   повтор {i}: нарушения:", "; ".join(e))
    json.dump(out, open("res9_llm_v18.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
