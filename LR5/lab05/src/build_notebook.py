"""Собирает lab05_v18.ipynb для Google Colab (ЛР 5, вариант 18). Запуск: python src/build_notebook.py"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = (ROOT / "src" / "lab05_lib.py").read_text(encoding="utf-8")

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# ЛР 5. Запуск открытой модели text-to-image — вариант 18
Беспалый Максим Евгеньевич, Т26ИИКТ-МИК11.
Бриф 3 «постер научной выставки» × условие 6 «одиночный запуск и первый элемент пакета из четырёх»; seed 1018.

**Перед запуском:** «Среда выполнения» → «Сменить среду выполнения» → «Графический процессор T4» → «Сохранить».
Затем «Среда выполнения» → «Выполнить все». Гипотеза и допуск записаны в журнал до запуска (reports/journal.md).
""")
md("## Ячейка 1. Среда: GPU и версии (шаг 1)")
code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!pip list 2>/dev/null | grep -i -E "^(torch|diffusers|transformers|pillow|accelerate|safetensors) "
import sys, platform, datetime, torch
print("UTC:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
print("python", sys.version.split()[0], "|", platform.platform())
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "НЕТ GPU — смените среду выполнения на T4")
""")
md("## Ячейка 2. Библиотека lab05_lib.py (приложение А МУ, без изменений)")
code("%%writefile lab05_lib.py\n" + LIB.rstrip("\n"))
md("## Ячейка 3. Сверка библиотеки с эталоном (6447 байт, SHA-256 4eb17650…)")
code("""
import hashlib, pathlib
raw = pathlib.Path("lab05_lib.py").read_bytes()
norm = raw.rstrip(b"\\n") + b"\\n"   # %%writefile может не дописать последний перевод строки
for name, b in (("как записан", raw), ("с одним \\\\n в конце", norm)):
    h = hashlib.sha256(b).hexdigest()
    print(f"{name}: {len(b)} байт, SHA-256 {h}", "— совпадает с эталоном" if h.startswith("4eb17650") else "")
""")
md("## Ячейка 4. Загрузка модели, базовый запуск A и пакет B; result.png и manifest.json (шаги 3–4)")
code("""
import json, pathlib, datetime, platform, sys, time, uuid, numpy as np, torch, diffusers, transformers, PIL
import importlib, lab05_lib as L
importlib.reload(L)

N, BRIEF = 18, 2                      # вариант 18: бриф 3 (индекс 2), условие 6
SEED, STEPS, GUIDANCE, SIZE = 1000 + N, 1, 0.0, 512
PROMPT = L.BRIEFS[BRIEF]
OUT = pathlib.Path("artifacts/v18"); OUT.mkdir(parents=True, exist_ok=True)
SESSION = uuid.uuid4().hex[:8]

def pkg(name):
    try: return importlib.import_module(name).__version__
    except Exception: return "?"

ENV = dict(device=torch.cuda.get_device_name(0), platform=platform.platform(), python=sys.version.split()[0],
           cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
           packages={n: pkg(n) for n in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "PIL")})

t0 = time.perf_counter(); pipe = L.load_pipe(); load_s = time.perf_counter() - t0
L.generate(pipe, PROMPT, 1)           # прогрев
L.generate(pipe, PROMPT, 1, batch=4)  # прогрев для пакета

def save_run(name, img, batch_size, batch_index, gen_s):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    data = L.png_bytes(img); (d / "result.png").write_bytes(data)
    m = dict(variant=N, condition="batch", run=name, model_id=L.MODEL_ID, revision=L.REVISION, prompt=PROMPT,
             seed=SEED, steps=STEPS, guidance_scale=GUIDANCE, size=SIZE, generator_device="cpu", dtype="float16",
             batch_size=batch_size, batch_index=batch_index, **ENV, load_seconds=round(load_s, 1),
             generation_seconds=round(gen_s, 3), image_size=list(img.size), image_mode=img.mode,
             sha256=L.sha(data), session=SESSION,
             created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m

torch.cuda.synchronize(); t0 = time.perf_counter()
A = L.generate(pipe, PROMPT, SEED)[0]; torch.cuda.synchronize(); tA = time.perf_counter() - t0
t0 = time.perf_counter()
batch = L.generate(pipe, PROMPT, SEED, batch=4); torch.cuda.synchronize(); tB = time.perf_counter() - t0
B = batch[0]
mA = save_run("A_single", A, 1, 0, tA)
mB = save_run("B_batch4_elem0", B, 4, 0, tB)
for i in range(1, 4): batch[i].save(OUT / f"B_batch4_elem{i}_seed{SEED + i}.png")
print(json.dumps(mA, ensure_ascii=False, indent=2))
print("\\nB:", mB["sha256"], f"| пакет из 4 за {tB:.3f} с")

def compare(a, b):
    p = L.psnr_db(a, b)
    return dict(sha_a=L.sha(L.png_bytes(a)), sha_b=L.sha(L.png_bytes(b)),
                pixel_equal=bool(np.array_equal(L.to_arr(a), L.to_arr(b))), max_diff=L.max_abs_diff(a, b),
                psnr=p, verdict=L.classify(p))
CMP = compare(A, B)
print("\\nТАБЛИЦА СРАВНЕНИЯ A (одиночный) и B (элемент 0 пакета из 4):")
for k, v in CMP.items(): print(f"  {k:12s} {v}")
""")
md("## Ячейка 5. variant_summary(pipe, 18) — проверка библиотекой (шаг 5)")
code("""
vs = L.variant_summary(pipe, N)
print(json.dumps(vs, ensure_ascii=False, indent=2))
print("SHA A совпадает с ячейкой 4:", vs["sha_a"] == CMP["sha_a"], "| SHA B совпадает с ячейкой 4:", vs["sha_b"] == CMP["sha_b"])
""")
md("""
## Ячейка 6. Проверка причины различия отдельными экспериментами
E1 — повтор пакета из 4: побитово ли? E2 — первый элемент при размерах пакета 1, 2, 3, 4, 8.
E3 — **одинаковый явный шум** во всех элементах пакета: различие из-за шума или из-за вычислений?
E4 — элементы 1–3 пакета против одиночных запусков с seed+1…+3. E5 — то же в float32.
""")
code("""
EXP = {}
# E1: повтор пакета той же формы
b2 = L.generate(pipe, PROMPT, SEED, batch=4)
EXP["E1_repeat_batch4"] = [L.sha(L.png_bytes(x)) == L.sha(L.png_bytes(y)) for x, y in zip(batch, b2)]
print("E1 повтор пакета из 4, совпадение SHA по элементам:", EXP["E1_repeat_batch4"])

# E2: первый элемент при разных размерах пакета
EXP["E2_first_elem_vs_single"] = {}
for bs in (1, 2, 3, 4, 8):
    x = L.generate(pipe, PROMPT, SEED, batch=bs)[0]; c = compare(A, x)
    EXP["E2_first_elem_vs_single"][bs] = dict(pixel_equal=c["pixel_equal"], max_diff=c["max_diff"], psnr=c["psnr"])
    print(f"E2 пакет {bs}: pixel_equal={c['pixel_equal']}, max_diff={c['max_diff']}, PSNR={c['psnr']:.2f}")

# E3: явный шум. Как в конвейере: randn на CPU в float16 своим генератором (seed + i).
def noise(seed):
    return torch.randn((1, 4, SIZE // 8, SIZE // 8), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float16)
def run_latents(lat):
    return pipe(prompt=[PROMPT] * lat.shape[0], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                height=SIZE, width=SIZE, latents=lat.to("cuda")).images
A_lat = run_latents(noise(SEED))[0]
EXP["E3_explicit_noise_equals_A"] = L.sha(L.png_bytes(A_lat)) == CMP["sha_a"]
print("E3 явный шум (1 элемент) воспроизводит A побитово:", EXP["E3_explicit_noise_equals_A"])
same4 = run_latents(torch.cat([noise(SEED)] * 4))
EXP["E3_same_noise_x4"] = [dict(vs_A=compare(A, x)["psnr"], equal_A=compare(A, x)["pixel_equal"]) for x in same4]
EXP["E3_elements_equal_each_other"] = len({L.sha(L.png_bytes(x)) for x in same4}) == 1
print("E3 один и тот же шум ×4: PSNR к A по элементам:", [round(e["vs_A"], 2) for e in EXP["E3_same_noise_x4"]],
      "| элементы пакета равны между собой:", EXP["E3_elements_equal_each_other"])

# E4: остальные элементы пакета против одиночных запусков с seed + i
EXP["E4_other_elems"] = {}
for i in range(1, 4):
    c = compare(L.generate(pipe, PROMPT, SEED + i)[0], batch[i])
    EXP["E4_other_elems"][i] = dict(pixel_equal=c["pixel_equal"], max_diff=c["max_diff"], psnr=c["psnr"])
    print(f"E4 элемент {i} (seed {SEED + i}) против одиночного: max_diff={c['max_diff']}, PSNR={c['psnr']:.2f}")

# E5: float32
pipe.to(torch.float32)
A32 = L.generate(pipe, PROMPT, SEED)[0]; B32 = L.generate(pipe, PROMPT, SEED, batch=4)[0]
c32 = compare(A32, B32)
EXP["E5_fp32_single_vs_batch"] = dict(pixel_equal=c32["pixel_equal"], max_diff=c32["max_diff"], psnr=c32["psnr"])
pipe.to(torch.float16)
print("E5 float32: одиночный против элемента 0 пакета:", EXP["E5_fp32_single_vs_batch"])
""")
md("## Ячейка 7. Намеренная ошибка на своём брифе: один генератор на два запуска (шаг 7)")
code("""
g = torch.Generator(device="cpu").manual_seed(SEED)
e1 = pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=g).images[0]
e2 = pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=g).images[0]
ERR = dict(sha_e1=L.sha(L.png_bytes(e1)), sha_e2=L.sha(L.png_bytes(e2)),
           e1_equals_A=L.sha(L.png_bytes(e1)) == CMP["sha_a"], e2_equals_A=L.sha(L.png_bytes(e2)) == CMP["sha_a"],
           psnr_e1_e2=round(L.psnr_db(e1, e2), 2))
e2.save(OUT / "error_shared_generator_run2.png")
print(json.dumps(ERR, ensure_ascii=False, indent=2))
fixed = L.generate(pipe, PROMPT, SEED)[0]
print("Исправление (свежий генератор в каждом запуске) снова даёт A:", L.sha(L.png_bytes(fixed)) == CMP["sha_a"])
""")
md("## Ячейка 8. Изображения для визуальной оценки, сводка и скачивание")
code("""
from PIL import Image, ImageChops
diff = np.clip(np.abs(L.to_arr(A) - L.to_arr(B)) * 10, 0, 255).astype(np.uint8)
Image.fromarray(diff).save(OUT / "diff_A_B_x10.png")
grid = Image.new("RGB", (SIZE * 3 + 20, SIZE), "white")
for k, im in enumerate((A, B, Image.fromarray(diff))): grid.paste(im, (k * (SIZE + 10), 0))
grid.save(OUT / "grid_A_B_diffx10.png")
display(grid.resize((SIZE * 3 // 2 + 10, SIZE // 2)))

SUMMARY = dict(session=SESSION, env=ENV, seed=SEED, prompt=PROMPT, load_seconds=round(load_s, 1),
               compare_A_B=CMP, variant_summary=vs, experiments=EXP, intentional_error=ERR,
               sha_A=CMP["sha_a"], sha_B=CMP["sha_b"])
(OUT / f"summary_session_{SESSION}.json").write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("Сеанс", SESSION, "| SHA A:", CMP["sha_a"], "| SHA B:", CMP["sha_b"])

import shutil
shutil.make_archive(f"lab05_v18_artifacts_{SESSION}", "zip", "artifacts")
try:
    from google.colab import files; files.download(f"lab05_v18_artifacts_{SESSION}.zip")
except Exception as e: print("Скачайте zip из панели «Файлы»:", e)
""")
md("""
## Ячейка 9. Проверка в новом сеансе (продвинутый уровень)
1. Сохраните блокнот: «Файл» → «Сохранить» (Google Диск) и «Файл» → «Скачать» → «.ipynb».
2. «Среда выполнения» → «Отключиться и удалить среду выполнения».
3. Снова «Выполнить все». Сравните SHA A и SHA B в ячейке 8 со значениями первого сеанса (они в summary_session_*.json).
4. Скачайте второй zip и сохраните блокнот ещё раз (с выводом второго сеанса).
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "lab05_v18.ipynb"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebook" / "lab05_v18.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)
