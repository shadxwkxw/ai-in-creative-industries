"""Собирает lab09_v18.ipynb для Google Colab (ЛР 9, вариант 18). Запуск: python src/build_notebook.py"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = (ROOT / "src" / "lab09_lib.py").read_text(encoding="utf-8")
LLM = (ROOT / "src" / "lab09_llm.py").read_text(encoding="utf-8")
SCORE = (ROOT / "src" / "score_all.py").read_text(encoding="utf-8")
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# ЛР 9. Промпт-инжиниринг — вариант 18
Контекст 3 «плакат научной выставки» × элемент 6 «отрицание внутри запроса». Серия seed 101…105.
A: `science exhibition poster`; B: `science exhibition poster, no logo, no text`; текст-цель CLIP: `written text and letters`.
Гипотеза и допуск записаны в журнал до запуска (reports/journal.md).

**Запуск:** среда выполнения — обычная (CPU), «Выполнить все». Библиотека рассчитана на процессор (float32, как в МУ); ≈ 10–20 минут, первая загрузка весов ≈ 4 ГБ.
Часть 2 (языковая модель) выполняется руками в чате; в этом блокноте — только оценка загруженных ответов (ячейки 9–10).
""")
md("## Ячейка 1. Установка пакетов и среда")
code("""
!pip -q install "diffusers==0.32.2" "transformers<5" accelerate
import sys, platform, datetime, torch, numpy, diffusers, transformers, os
print("UTC:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
print("python", sys.version.split()[0], "|", platform.platform())
print("torch", torch.__version__, "| diffusers", diffusers.__version__, "| transformers", transformers.__version__, "| numpy", numpy.__version__)
print("CPU потоков (os):", os.cpu_count(), "| CUDA доступна:", torch.cuda.is_available(), "(библиотека всё равно считает на CPU)")
""")
md("## Ячейка 2. lab09_lib.py и lab09_llm.py (приложения А и В МУ, без изменений)")
code("%%writefile lab09_lib.py\n" + LIB.rstrip("\n"))
code("%%writefile lab09_llm.py\n" + LLM.rstrip("\n"))
code("%%writefile score_all.py\n" + SCORE.rstrip("\n"))
md("## Ячейка 3. Сверка библиотек (lab09_lib.py — 7360 байт, SHA-256 cc4d233d…; lab09_llm.py — 18709 байт, 5eabb31f…)")
code("""
import hashlib, pathlib
for fn, ref in (("lab09_lib.py", "cc4d233d"), ("lab09_llm.py", "5eabb31f")):
    b = pathlib.Path(fn).read_bytes()
    for name, x in (("как записан", b), ("с \\\\n в конце", b.rstrip(b"\\\\n") + b"\\\\n")):
        h = hashlib.sha256(x).hexdigest(); print(fn, name, len(x), "байт", h[:16] + "…", "— совпадает с эталоном" if h.startswith(ref) else "")
""")
md("## Ячейка 4. Вариант 18: что сравниваем (шаги 1, 4)")
code("""
import json, time, numpy as np, importlib
import lab09_lib as L
importlib.reload(L)
N = 18
c, k = (N - 1) // 6, (N - 1) % 6
PA, PB = L.prompt_a(c), L.prompt_b(c, k)
TARGET = L.ELEMENTS[c][k][1]
OUT = pathlib.Path("artifacts/v18"); OUT.mkdir(parents=True, exist_ok=True)
print("контекст", c + 1, "| элемент", k + 1, L.FACTORS[k])
print("A:", PA); print("B:", PB); print("текст-цель CLIP:", TARGET)
print("B = A + добавленная фраза:", PB == PA + ", " + L.ELEMENTS[c][k][0])
print("режим:", L.MODEL_ID, L.REVISION[:8], "| CLIP", L.CLIP_ID, L.CLIP_REVISION[:8], "| шагов", L.STEPS, "| guidance", L.GUIDANCE, "| размер", L.SIZE, "| seed", list(L.SEEDS))
""")
md("## Ячейка 5. Загрузка моделей и серия A/B на seed 101…105 (как L.variant_summary, но с сохранением всех десяти изображений)")
code("""
t0 = time.perf_counter(); pipe = L.load_pipe(); clip = L.load_clip(); print(f"модели загружены за {time.perf_counter() - t0:.0f} с")
L.generate(pipe, PA, 0)  # прогрев, как в variant_summary
IMG = {"A": {}, "B": {}}; RA, RB = [], []
t0 = time.perf_counter()
for s in L.SEEDS:
    r, img = L.measure(pipe, clip, c, PA, s); RA.append(r); IMG["A"][s] = img; img.save(OUT / f"A_seed{s}.jpg", quality=92)
    r, img = L.measure(pipe, clip, c, PB, s); RB.append(r); IMG["B"][s] = img; img.save(OUT / f"B_seed{s}.jpg", quality=92)
    print("seed", s, "готов, с начала", f"{time.perf_counter() - t0:.0f} с")
ROWS = L.compare(RA, RB, k)
print("среднее время одного изображения, с:", round(float(np.mean([r["seconds"] for r in RA + RB])), 1))
""")
md("## Ячейка 6. Таблица метрик, признак 2s и допуск (допуски записаны до запуска)")
code("""
TOL = {"clip_target": ("abs", 1.0), "clip_subject": ("abs", 1.0), "edge": ("rel", 0.10), "contrast": ("rel", 0.10)}
def classify(m, a, d):
    kind, t = TOL[m]; x = d if kind == "abs" else d / a
    return "в допуске" if abs(x) <= t else ("рост за допуском" if x > 0 else "падение за допуском")
TABLE = []
print(f"{'метрика':13s} {'A':>9s} {'s':>8s} {'B':>9s} {'Δ':>9s} {'|Δ|/s':>7s}  2s   допуск")
for m, a, s, b, d, q, sig in ROWS:
    cl = classify(m, a, d); TABLE.append(dict(metric=m, A=a, s=s, B=b, d=d, ratio=q, sig2s=sig, tolerance=cl, rel=d / a))
    nd = 4 if m in ("edge", "contrast") else 2
    print(f"{m:13s} {a:9.{nd}f} {s:8.{nd}f} {b:9.{nd}f} {d:+9.{nd}f} {q:7.1f}  {'да ' if sig else 'нет'}  {cl}   (отн. {d / a * 100:+.1f} %)")
""")
md("## Ячейка 7. По каждому seed (диагностика ошибки 1) и знаковая сводка")
code("""
pick = {"clip_target": lambda r: r["sims"][1 + k], "clip_subject": lambda r: r["sims"][0], "edge": lambda r: r["edge"], "contrast": lambda r: r["contrast"]}
PER_SEED = {}
for m in pick:
    A = [pick[m](r) for r in RA]; B = [pick[m](r) for r in RB]; PER_SEED[m] = dict(A=A, B=B, diff=[b - a for a, b in zip(A, B)])
    nd = 4 if m in ("edge", "contrast") else 2
    print(m); print("  seed  ", *[f"{s:>9d}" for s in L.SEEDS]); print("  A     ", *[f"{x:9.{nd}f}" for x in A]); print("  B     ", *[f"{x:9.{nd}f}" for x in B])
    print("  B − A ", *[f"{x:+9.{nd}f}" for x in PER_SEED[m]["diff"]], f"| Δ>0 на {sum(x > 0 for x in PER_SEED[m]['diff'])} из 5 seed")
print("\\nвсе семь сходств CLIP (предмет + шесть целей) на seed 101, A и B:")
print("  A:", [round(x, 2) for x in RA[0]["sims"]]); print("  B:", [round(x, 2) for x in RB[0]["sims"]])
""")
md("## Ячейка 8. Листы сравнения: comparison.png (seed 101) и сетка всех десяти изображений")
code("""
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10.4, 5.6))
for a_, key, ttl in ((ax[0], "A", "A: " + PA + " (seed 101)"), (ax[1], "B", "B: …, no logo, no text (seed 101)")):
    a_.imshow(IMG[key][101]); a_.set_title(ttl, fontsize=10); a_.axis("off")
plt.tight_layout(); plt.savefig(OUT / "comparison_variant18.png", dpi=110); plt.close()
fig, ax = plt.subplots(2, 5, figsize=(17, 7.4))
for j, s in enumerate(L.SEEDS):
    for i, key in enumerate("AB"):
        ax[i, j].imshow(IMG[key][s]); ax[i, j].set_title(f"{key}, seed {s}", fontsize=10); ax[i, j].axis("off")
plt.tight_layout(); plt.savefig(OUT / "grid_all_seeds.png", dpi=100); plt.close()
from IPython.display import display
display(Image.open(OUT / "comparison_variant18.png")); display(Image.open(OUT / "grid_all_seeds.png"))
json.dump(dict(variant=N, prompt_a=PA, prompt_b=PB, target=TARGET, table=TABLE, per_seed=PER_SEED, seconds=[r["seconds"] for r in RA + RB],
               versions=dict(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__)),
          open(OUT / "summary_v18.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
""")
md("""
## Ячейка 9. Часть 2: запросы для языковой модели
Девять запросов (3 приёма × 3 повтора) напечатаны ниже и лежат в `prompts/`. **Каждый — в НОВЫЙ чат**, поиск отключён самим запросом.
Ответ сохраняется ЦЕЛИКОМ и без правок в файл `ctx3_<zero|few|step>_r<1..3>.txt`.
""")
code("""
for a in ("zero", "few", "step"):
    print("=" * 30, a, "=" * 30); print(__import__("lab09_llm").build_prompt(a, 3)); print()
""")
md("## Ячейка 10. Загрузка девяти файлов ответов и оценка K1–K4 (после выполнения части 2)")
code("""
from google.colab import files
os.makedirs("answers", exist_ok=True)
up = files.upload()   # выберите девять файлов ctx3_*_r*.txt
for name, data in up.items(): open(os.path.join("answers", name), "wb").write(data)
!python score_all.py answers
""")
md("## Ячейка 11. Упаковка артефактов")
code("""
import shutil, glob
from google.colab import files
for f in glob.glob("res9_llm_v18.json"): shutil.copy(f, OUT)
if os.path.isdir("answers"): shutil.copytree("answers", OUT / "answers", dirs_exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M")
shutil.make_archive(f"lab09_v18_artifacts_{stamp}", "zip", OUT)
files.download(f"lab09_v18_artifacts_{stamp}.zip")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebook" / "lab09_v18.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", len(cells), "ячеек")
