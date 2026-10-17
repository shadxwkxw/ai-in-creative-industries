"""Собирает lab06_v18.ipynb для Google Colab (ЛР 6, вариант 18). Запуск: python src/build_notebook.py"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB5 = (ROOT / "src" / "lab05_lib.py").read_text(encoding="utf-8")
LIB6 = (ROOT / "src" / "lab06_lib.py").read_text(encoding="utf-8")
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# ЛР 6. Исследование параметров text-to-image — вариант 18
Беспалый Максим Евгеньевич, Т26ИИКТ-МИК11. Бриф 3 «постер научной выставки» × фактор 6 «язык запроса: английский → русский».
Серия seed 101…105. Гипотеза, допуск и диагностические режимы C1–C3 записаны в журнал до запуска.

**Запуск:** «Среда выполнения» → «Сменить среду выполнения» → «Графический процессор T4» → «Сохранить» → «Выполнить все».
""")
md("## Ячейка 1. Среда (шаг 1)")
code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!pip list 2>/dev/null | grep -i -E "^(torch|diffusers|transformers|pillow|scipy|accelerate|safetensors) "
import sys, platform, datetime, torch
print("UTC:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
print("python", sys.version.split()[0], "|", platform.platform())
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "НЕТ GPU — смените среду выполнения на T4")
""")
md("## Ячейка 2. lab05_lib.py (из ЛР 5, без изменений)")
code("%%writefile lab05_lib.py\n" + LIB5.rstrip("\n"))
md("## Ячейка 3. lab06_lib.py (приложение А МУ, без изменений)")
code("%%writefile lab06_lib.py\n" + LIB6.rstrip("\n"))
md("## Ячейка 4. Сверка библиотек с эталоном")
code("""
import hashlib, pathlib
for f, ref, size in (("lab05_lib.py", "4eb17650", 6447), ("lab06_lib.py", "c6d8473c", 6580)):
    b = pathlib.Path(f).read_bytes()
    for name, x in (("как записан", b), ("с \\\\n в конце", b.rstrip(b"\\n") + b"\\n")):
        h = hashlib.sha256(x).hexdigest()
        print(f"{f} {name}: {len(x)} байт, SHA-256 {h[:16]}…", "— совпадает с эталоном" if h.startswith(ref) else "")
""")
md("## Ячейка 5. Загрузка модели и CLIP; режимы A и B (шаги 3–4)")
code("""
import json, time, importlib, numpy as np, torch, diffusers, transformers, PIL, scipy
import lab05_lib as L, lab06_lib as M
importlib.reload(L); importlib.reload(M)
N = 18
c, k = (N - 1) // 6, (N - 1) % 6
OUT = pathlib.Path("artifacts/v18"); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter(); pipe = L.load_pipe(); t_pipe = time.perf_counter() - t0
t0 = time.perf_counter(); clip = M.load_clip(); t_clip = time.perf_counter() - t0
print(f"загрузка: SD Turbo {t_pipe:.1f} с, CLIP {t_clip:.1f} с")
a_cfg, b_cfg = M.settings(M.FACTORS[k], c)
print("фактор:", M.FACTORS[k], "| бриф", c + 1)
print("A:", a_cfg); print("B:", b_cfg)
print("различаются поля:", [key for key in a_cfg if a_cfg[key] != b_cfg[key]])
ENV = dict(device=torch.cuda.get_device_name(0), python=sys.version.split()[0], platform=platform.platform(),
           cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
           packages=dict(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
                         pillow=PIL.__version__, scipy=scipy.__version__),
           sd_turbo=f"{L.MODEL_ID}@{L.REVISION}", clip=f"{M.CLIP_ID}@{M.CLIP_REVISION}", load_seconds=[round(t_pipe, 1), round(t_clip, 1)])
""")
md("## Ячейка 6. Воспроизведение демонстрационного варианта 1 (продвинутый уровень)")
code("""
demo = M.variant_summary(pipe, clip, 1)
print("Вариант", demo["variant"], "фактор", demo["factor"], "(МУ: CLIP 30,796/1,394→30,292; границы 0,0457→0,0920; контраст 0,1923→0,2778)")
for m, a, s, b, d, ratio, sig in demo["rows"]:
    print(f"{m:9s} A={a:.4f} s={s:.4f} B={b:.4f} d={d:+.4f} |d|/s={ratio:.2f} 2s={'да' if sig else 'нет'}")
""")
md("## Ячейка 7. Основной эксперимент: variant_summary(pipe, clip, 18) и классификация по допуску (шаги 5, 7)")
code("""
res = M.variant_summary(pipe, clip, N)
TOL = {"clip": ("abs", 1.0), "edge": ("rel", 0.10), "contrast": ("rel", 0.10), "seconds": ("rel", 0.20)}
def classify(m, a, d):
    kind, t = TOL[m]
    x = d if kind == "abs" else d / a
    if abs(x) <= t: return "нейтрально"
    if m == "seconds": return "ухудшение (медленнее)" if x > 0 else "улучшение (быстрее)"
    if m == "clip": return "улучшение" if x > 0 else "ухудшение"
    return "рост" if x > 0 else "падение"
print("Вариант", res["variant"], "фактор", res["factor"])
print(f"{'метрика':9s} {'A':>9s} {'s':>8s} {'B':>9s} {'Δ':>9s} {'Δ%':>8s} {'|Δ|/s':>7s} 2s   допуск")
ROWS = []
for m, a, s, b, d, ratio, sig in res["rows"]:
    cl = classify(m, a, d); ROWS.append(dict(metric=m, A=a, s=s, B=b, d=d, rel=d / a, ratio=ratio, sig2s=sig, tolerance=cl))
    print(f"{m:9s} {a:9.4f} {s:8.4f} {b:9.4f} {d:+9.4f} {100 * d / a:+7.1f}% {ratio:7.2f} {'да ' if sig else 'нет'}  {cl}")
print("\\nКритерий 2s и допуск расходятся:", [r["metric"] for r in ROWS if r["sig2s"] != (r["tolerance"] != "нейтрально")])

# Значения по каждому seed (для отчёта и проверки разброса); изображения сохраняются
PER = {"A": [], "B": []}; IMGS = {"A": {}, "B": {}}
for s in M.SEEDS:
    for mode, cfg in (("A", a_cfg), ("B", b_cfg)):
        mm, img = M.measure(pipe, clip, cfg, c, s); PER[mode].append(dict(seed=s, **mm)); IMGS[mode][s] = img
        img.save(OUT / f"{mode}_seed{s}.png")
for s, ra, rb in zip(M.SEEDS, PER["A"], PER["B"]):
    print(f"seed {s}: CLIP {ra['clip']:.2f}→{rb['clip']:.2f}  границы {ra['edge']:.4f}→{rb['edge']:.4f}  контраст {ra['contrast']:.4f}→{rb['contrast']:.4f}")
print("CLIP (повторное измерение) совпадает с variant_summary:",
      np.isclose(np.mean([r["clip"] for r in PER["A"]]), ROWS[0]["A"]), np.isclose(np.mean([r["clip"] for r in PER["B"]]), ROWS[0]["B"]))
""")
md("## Ячейка 8. Сравнительный лист comparison.png (шаг 6) и лист всей серии")
code("""
from PIL import Image, ImageDraw
pair = [L.generate(pipe, cfg["prompt"], 101, steps=cfg["steps"], guidance=cfg["guidance"], size=cfg["size"])[0].resize((384, 384)) for cfg in (a_cfg, b_cfg)]
sheet = Image.new("RGB", (768, 384)); sheet.paste(pair[0], (0, 0)); sheet.paste(pair[1], (384, 0))
sheet.save(OUT / "comparison.png"); display(sheet)
full = Image.new("RGB", (5 * 256, 2 * 256 + 24), "white"); dr = ImageDraw.Draw(full)
for j, s in enumerate(M.SEEDS):
    full.paste(IMGS["A"][s].resize((256, 256)), (j * 256, 24)); full.paste(IMGS["B"][s].resize((256, 256)), (j * 256, 280))
    dr.text((j * 256 + 4, 4), f"seed {s}: A (EN) сверху, B (RU) снизу", fill="black")
full.save(OUT / "comparison_series.png"); display(full.resize((5 * 160, 2 * 160 + 15)))
""")
md("""
## Ячейка 9. Диагностика причины: длина запроса в токенах и режимы C1–C3
Режимы заданы в журнале до запуска. CLIP всегда считается относительно полного английского брифа.
""")
code("""
MODES = {
    "A_EN": L.BRIEFS[c],
    "B_RU": M.RU[c],
    "C1_EN_trunc": "poster concept for a science exhibition, abstract constellation of connected nodes",
    "C2_RU_compact": "постер выставки, созвездие узлов, тёмный фон, голубые линии, без текста",
    "C3_EN_compact": "exhibition poster, constellation of nodes, dark background, cyan lines, no text",
}
tok = pipe.tokenizer
TOK = {}
for name, p in MODES.items():
    n = len(tok(p).input_ids); TOK[name] = n
    kept = tok.decode(tok(p, truncation=True, max_length=77).input_ids[1:-1])
    print(f"{name:14s} токенов {n:3d}{' — УСЕЧЁН до 77' if n > 77 else ''}" + (f"\\n{'':14s} модель видит: «{kept}»" if n > 77 else ""))
DIAG = {}
for name, p in MODES.items():
    vals = {"clip": [], "edge": [], "contrast": []}
    for s in M.SEEDS:
        mm, img = M.measure(pipe, clip, dict(prompt=p, steps=1, guidance=0.0, size=512), c, s)
        for key in vals: vals[key].append(mm[key])
        if s == 101: img.save(OUT / f"{name}_seed101.png")
    DIAG[name] = {key: (float(np.mean(v)), float(np.std(v, ddof=1))) for key, v in vals.items()}
    print(f"{name:14s} CLIP {DIAG[name]['clip'][0]:6.2f} ± {DIAG[name]['clip'][1]:.2f} | границы {DIAG[name]['edge'][0]:.4f} | контраст {DIAG[name]['contrast'][0]:.4f}")
print("\\nэффект только усечения (C1 − A), CLIP:", round(DIAG["C1_EN_trunc"]["clip"][0] - DIAG["A_EN"]["clip"][0], 2))
print("эффект только языка (C2 − C3), CLIP:", round(DIAG["C2_RU_compact"]["clip"][0] - DIAG["C3_EN_compact"]["clip"][0], 2))
print("полный эффект (B − A), CLIP:", round(DIAG["B_RU"]["clip"][0] - DIAG["A_EN"]["clip"][0], 2))

# Справочно: как сама метрика CLIP ViT-B/32 относится к русскому тексту (не заменяет эталон!)
@torch.no_grad()
def text_emb(t):
    mdl, proc = clip; x = proc(text=[t], return_tensors="pt", padding=True, truncation=True).to("cuda")
    e = mdl.get_text_features(**x)
    if not torch.is_tensor(e):  # в transformers 5 может вернуться объект вывода модели
        e = e.text_embeds if getattr(e, "text_embeds", None) is not None else e.pooler_output
    return e / e.norm(dim=-1, keepdim=True)
TT = float((text_emb(L.BRIEFS[c]) * text_emb(M.RU[c])).sum()) * 100
RUSIM = float(np.mean([M.clip_sim(clip, IMGS["B"][s], M.RU[c]) for s in M.SEEDS]))
print(f"\\nсправочно: сходство текстов EN и RU брифа в CLIP ×100 = {TT:.1f}; сходство изображений B с русским текстом = {RUSIM:.2f}")
strip = Image.new("RGB", (5 * 256, 256))
for j, name in enumerate(MODES): strip.paste(Image.open(OUT / f"{name}_seed101.png").resize((256, 256)), (j * 256, 0))
strip.save(OUT / "modes_seed101.png"); display(strip.resize((5 * 160, 160)))
print("слева направо:", list(MODES))
""")
md("## Ячейка 10. Намеренная ошибка на своём брифе: guidance_scale = 1,0 (шаг 8)")
code("""
p = L.BRIEFS[c]
x0 = L.generate(pipe, p, 101, guidance=0.0)[0]
x1 = L.generate(pipe, p, 101, guidance=1.0)[0]; cfg1 = pipe.do_classifier_free_guidance
x3 = L.generate(pipe, p, 101, guidance=3.0)[0]; cfg3 = pipe.do_classifier_free_guidance
ERR = dict(sha_g0=L.sha(L.png_bytes(x0)), sha_g1=L.sha(L.png_bytes(x1)), g1_equals_g0=L.sha(L.png_bytes(x0)) == L.sha(L.png_bytes(x1)),
           cfg_active_g1=cfg1, cfg_active_g3=cfg3, psnr_g3_vs_g0=round(L.psnr_db(x0, x3), 2))
x3.save(OUT / "error_guidance3_seed101.png")
print(json.dumps(ERR, ensure_ascii=False, indent=2))
""")
md("## Ячейка 11. Сводка и скачивание")
code("""
import uuid, shutil
SESSION = uuid.uuid4().hex[:8]
SUMMARY = dict(session=SESSION, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), env=ENV, variant=N, factor=M.FACTORS[k],
               settings=dict(A=a_cfg, B=b_cfg), rows=ROWS, per_seed=PER, demo_variant1=demo["rows"], tokens=TOK,
               diagnostics=DIAG, text_text_clip=TT, clip_B_vs_ru_text=RUSIM, intentional_error=ERR)
(OUT / f"summary_{SESSION}.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
shutil.make_archive(f"lab06_v18_artifacts_{SESSION}", "zip", "artifacts")
print("Сеанс", SESSION)
try:
    from google.colab import files; files.download(f"lab06_v18_artifacts_{SESSION}.zip")
except Exception as e: print("Скачайте zip из панели «Файлы»:", e)
""")
md("""
## После выполнения
«Файл» → «Скачать» → «.ipynb» (блокнот с выводом). По желанию: «Отключиться и удалить среду выполнения» → «Выполнить все»
ещё раз и сравнить таблицу ячейки 7 (CLIP, границы, контраст должны совпасть; время будет другим).
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "lab06_v18.ipynb"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebook" / "lab06_v18.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"); print(out)
