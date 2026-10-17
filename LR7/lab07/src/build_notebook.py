"""Собирает lab07_v18.ipynb для Google Colab (ЛР 7, вариант 18). Запуск: python src/build_notebook.py"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = (ROOT / "src" / "lab07_lib.py").read_text(encoding="utf-8")
DEMO = (ROOT / "src" / "lab07_demo.py").read_text(encoding="utf-8")
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# ЛР 7. Построение автокодировщика — вариант 18
Беспалый Максим Евгеньевич, Т26ИИКТ-МИК11. Набор 3 «disc_noise» × фактор 6 «4096 → 512 сцен (20 → 160 эпох, 640 шагов)».
Серия seed 101…105. Гипотеза, допуск и дополнительные проверки E1–E3 записаны в журнал до запуска.

**Запуск:** «Среда выполнения» → «Сменить среду выполнения» → «Графический процессор T4» → «Сохранить» → «Выполнить все» (≈ 4–6 минут).
""")
md("## Ячейка 1. Среда (шаг 1)")
code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import sys, platform, datetime, torch, numpy
print("UTC:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
print("python", sys.version.split()[0], "|", platform.platform(), "| numpy", numpy.__version__)
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "НЕТ GPU — смените среду на T4")
""")
md("## Ячейка 2. lab07_lib.py (приложение А МУ, без изменений)")
code("%%writefile lab07_lib.py\n" + LIB.rstrip("\n"))
md("## Ячейка 3. Сверка библиотеки (8679 байт, SHA-256 d84b83a8…)")
code("""
import hashlib, pathlib
b = pathlib.Path("lab07_lib.py").read_bytes()
for name, x in (("как записан", b), ("с \\\\n в конце", b.rstrip(b"\\n") + b"\\n")):
    h = hashlib.sha256(x).hexdigest(); print(f"{name}: {len(x)} байт, SHA-256 {h[:16]}…", "— совпадает с эталоном" if h.startswith("d84b83a8") else "")
""")
md("## Ячейка 4. Демонстрационный вариант 1 — как в МУ (воспроизведение, продвинутый уровень)")
code(DEMO.rstrip("\n") + '\nimport shutil; shutil.move("comparison.png", "comparison_variant01.png")\nDEMO_ROWS = result["rows"]')
md("## Ячейка 5. Вариант 18: режимы A и B, variant_summary(18), классификация по допуску (шаги 3, 4, 7)")
code("""
import json, time, numpy as np, torch, importlib
import lab07_lib as L
importlib.reload(L)
N = 18
c, k = (N - 1) // 6, (N - 1) % 6
KIND = L.KINDS[c]
OUT = pathlib.Path("artifacts/v18"); OUT.mkdir(parents=True, exist_ok=True)
a_cfg, b_cfg = L.settings(L.FACTORS[k])
print("набор:", KIND, "| фактор:", L.FACTORS[k])
print("A:", a_cfg); print("B:", b_cfg)
print("различаются поля:", [key for key in a_cfg if a_cfg[key] != b_cfg[key]],
      "| шагов обучения A, B:", a_cfg["n_train"] // 128 * a_cfg["epochs"], b_cfg["n_train"] // 128 * b_cfg["epochs"])
t0 = time.perf_counter(); res = L.variant_summary(N); print(f"серия: {time.perf_counter() - t0:.0f} с")
TOL = {"psnr": ("abs", 1.0), "gap": ("abs", 0.1), "params": ("abs", 0.0), "seconds": ("rel", 0.20)}
def classify(m, a, d):
    kind, t = TOL[m]
    if m == "params": return "без изменений" if d == 0 else "изменилось"
    x = d if kind == "abs" else d / a
    if abs(x) <= t: return "нейтрально"
    return {"psnr": ("улучшение" if x > 0 else "ухудшение"), "gap": ("переобучение сильнее" if x > 0 else "переобучение слабее"),
            "seconds": ("медленнее" if x > 0 else "быстрее")}[m]
ROWS = []
print(f"\\n{'метрика':8s} {'A':>9s} {'s':>7s} {'B':>9s} {'Δ':>9s} {'|Δ|/s':>8s} 2s   допуск")
for name, a, s, b, d, q, sig in res["rows"]:
    cl = classify(name, a, d); ROWS.append(dict(metric=name, A=a, s=s, B=b, d=d, ratio=q, sig2s=sig, tolerance=cl))
    print(f"{name:8s} {a:9.3f} {s:7.3f} {b:9.3f} {d:+9.3f} {q:8.2f} {'да ' if sig else 'нет'}  {cl}")
""")
md("## Ячейка 6. Эталон «средняя картинка» для своего набора (шаг 5)")
code("""
Xtr, Xte, Ftr, Fte = L.dataset(KIND)
mean_image = Xtr.mean(0, keepdim=True).expand_as(Xte)
BASELINE = L.psnr_images(mean_image, Xte)
print("эталон «средняя картинка», PSNR на тесте:", round(BASELINE, 3), "дБ")
print("выигрыш A и B над эталоном, дБ:", round(ROWS[0]["A"] - BASELINE, 2), round(ROWS[0]["B"] - BASELINE, 2))
""")
md("## Ячейка 7. Лист сравнения comparison.png (шаг 6) и расширенный лист с чистыми сценами")
code("""
from PIL import Image
# чистые сцены: те же факторы и тот же генератор, что в L.dataset, но без шума (проверка совпадения ниже)
rng = np.random.default_rng(7)
f_all = rng.random((4096 + 1024, L.N_FACTORS[KIND]))
clean_np = np.stack([L.render(KIND, fi) for fi in f_all])
noisy_np = np.clip(clean_np + 0.05 * rng.standard_normal(clean_np.shape).astype(np.float32), 0, 1)
assert np.allclose(noisy_np[4096:], Xte.cpu().numpy()), "реконструкция данных не совпала с L.dataset"
CLEAN_TE = torch.tensor(clean_np[4096:], device=L.DEVICE); CLEAN_TR = torch.tensor(clean_np[:4096], device=L.DEVICE)
print("чистые сцены восстановлены, совпадение зашумлённых с L.dataset: да")

def to_row(t): return np.concatenate(list(t.permute(0, 2, 3, 1).cpu().numpy()), axis=1)
MODELS101 = {}
rows = [Xte[:8]]
for name, cfg in (("A", a_cfg), ("B", b_cfg)):
    model = L.run(KIND, cfg, 101)[1]; MODELS101[name] = model
    rows.append(L.reconstruct(model, Xte[:8], cfg["eval_noise"]))
grid = np.concatenate([to_row(r) for r in rows], axis=0)
sheet = Image.fromarray((grid * 255).round().astype("uint8")).resize((8 * 32 * 4, 3 * 32 * 4), Image.NEAREST)
sheet.save(OUT / "comparison.png"); display(sheet)
grid2 = np.concatenate([to_row(CLEAN_TE[:8])] + [to_row(r) for r in rows], axis=0)
Image.fromarray((grid2 * 255).round().astype("uint8")).resize((8 * 32 * 4, 4 * 32 * 4), Image.NEAREST).save(OUT / "comparison_with_clean.png")
diff = np.concatenate([to_row((r - CLEAN_TE[:8]).abs() * 5) for r in rows[1:]], axis=0).clip(0, 1)
Image.fromarray((diff * 255).round().astype("uint8")).resize((8 * 32 * 4, 2 * 32 * 4), Image.NEAREST).save(OUT / "abs_error_vs_clean_x5.png")
print("сохранены: comparison.png (оригинал | A | B), comparison_with_clean.png (чистая | зашумлённая | A | B), abs_error_vs_clean_x5.png")
""")
md("## Ячейка 8. Диагностика по размеру латентного вектора для своего набора (шаг 8)")
code("""
DSWEEP = {}
for d in (1, 2, 3, 4, 8):
    vals = [L.run(KIND, dict(L.BASE, d=d), s)[0]["psnr"] for s in (101, 102, 103)]
    DSWEEP[d] = vals
    print(f"d={d}: PSNR = {np.mean(vals):.2f} дБ (по seed: {', '.join(f'{v:.2f}' for v in vals)})")
""")
md("## Ячейка 9. E1: потолок шума и качество относительно чистых сцен; значения по seed")
code("""
CEIL = L.psnr_images(CLEAN_TE, Xte)
print(f"потолок: PSNR чистых сцен относительно зашумлённых тестовых эталонов = {CEIL:.3f} дБ")
PER = {"A": [], "B": []}
for s in L.SEEDS:
    for name, cfg in (("A", a_cfg), ("B", b_cfg)):
        m, model = L.run(KIND, cfg, s)
        rec_te = L.reconstruct(model, Xte); rec_tr = L.reconstruct(model, Xtr[:cfg["n_train"]][:512])
        PER[name].append(dict(seed=s, psnr_noisy=m["psnr"], gap=m["gap"], seconds=m["seconds"],
                              psnr_clean=L.psnr_images(rec_te, CLEAN_TE),
                              train_clean=L.psnr_images(rec_tr, CLEAN_TR[:cfg["n_train"]][:512]),
                              train_noisy=L.psnr_images(rec_tr, Xtr[:cfg["n_train"]][:512])))
for name in "AB":
    for r in PER[name]:
        print(f"{name} seed {r['seed']}: тест к зашумл. {r['psnr_noisy']:.2f} | тест к чистым {r['psnr_clean']:.2f} | "
              f"train к зашумл. {r['train_noisy']:.2f} | train к чистым {r['train_clean']:.2f} | gap {r['gap']:+.3f}")
for name in "AB":
    for key in ("psnr_noisy", "psnr_clean", "gap"):
        v = [r[key] for r in PER[name]]; print(f"{name} {key:10s}: {np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
""")
md("## Ячейка 10. E2: почему нужно 160 эпох — 512 сцен при 20 эпохах (80 шагов)")
code("""
E2 = [L.run(KIND, dict(b_cfg, epochs=20), s)[0] for s in L.SEEDS]
print("512 сцен, 20 эпох: PSNR на тесте", [round(m["psnr"], 2) for m in E2], "| среднее", round(np.mean([m["psnr"] for m in E2]), 2),
      "| gap", round(np.mean([m["gap"] for m in E2]), 3))
""")
md("## Ячейка 11. E3: динамика переобучения — 512 сцен, 20…320 эпох (seed 101)")
code("""
E3 = {}
for ep in (20, 40, 80, 160, 320):
    m, model = L.run(KIND, dict(b_cfg, epochs=ep), 101)
    E3[ep] = dict(test=m["psnr"], gap=m["gap"], test_clean=L.psnr_images(L.reconstruct(model, Xte), CLEAN_TE))
    print(f"эпох {ep:3d} (шагов {ep * 4:4d}): тест {m['psnr']:.2f} дБ | к чистым {E3[ep]['test_clean']:.2f} | gap {m['gap']:+.3f}")
""")
md("## Ячейка 12. Сводка и скачивание")
code("""
import uuid, shutil
SESSION = uuid.uuid4().hex[:8]
ENV = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__, cuda=torch.version.cuda,
           python=sys.version.split()[0], numpy=numpy.__version__, platform=platform.platform())
SUMMARY = dict(session=SESSION, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), env=ENV, variant=N, kind=KIND,
               factor=L.FACTORS[k], settings=dict(A=a_cfg, B=b_cfg), rows=ROWS, baseline=BASELINE, demo_variant1=DEMO_ROWS,
               d_sweep=DSWEEP, noise_ceiling=CEIL, per_seed=PER, e2_512_20ep=E2, e3_epochs=E3)
(OUT / f"summary_{SESSION}.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
shutil.copy("comparison_variant01.png", OUT / "comparison_variant01.png")
shutil.make_archive(f"lab07_v18_artifacts_{SESSION}", "zip", "artifacts")
print("Сеанс", SESSION)
try:
    from google.colab import files; files.download(f"lab07_v18_artifacts_{SESSION}.zip")
except Exception as e: print("Скачайте zip из панели «Файлы»:", e)
""")
md("""
## После выполнения
«Файл» → «Скачать» → «.ipynb». Обучение на GPU не побитово детерминировано: при повторе значения совпадут в пределах разброса между seed,
а не до последнего знака (это отмечено и в МУ).
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "lab07_v18.ipynb"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebook" / "lab07_v18.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"); print(out)
