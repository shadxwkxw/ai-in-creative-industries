"""ЛР 5, вариант 18 — локальная проверка альтернативного объяснения на CPU (float32).
Вопрос: различие «одиночный запуск vs элемент 0 пакета» — свойство только GPU или и CPU тоже?
Использует закреплённую ревизию из кэша ЛР 4 (HF_HOME), без сети. Не заменяет основной запуск в Colab.
"""
import hashlib, io, json, os, platform, sys, time
import numpy as np, torch, diffusers
from diffusers import AutoPipelineForText2Image

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.find_spec("lab05_lib")
BRIEF = "poster concept for a science exhibition, abstract constellation of connected nodes, dark background with white and cyan lines, balanced composition, no text, no logo"
MODEL_ID, REVISION, SEED = "stabilityai/sd-turbo", "b261bac6fd2cf515557d5d0707481eafa0485ec2", 1018
OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "cpu_check")
os.makedirs(OUT, exist_ok=True)

def png(img): b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()
def arr(img): return np.asarray(img.convert("RGB"), dtype=np.float64)
def psnr(a, b):
    m = np.mean((arr(a) - arr(b)) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)
def gen(pipe, batch):
    gens = [torch.Generator("cpu").manual_seed(SEED + i) for i in range(batch)]
    return pipe(prompt=[BRIEF] * batch, num_inference_steps=1, guidance_scale=0.0, height=512, width=512, generator=gens).images

torch.manual_seed(0)
t0 = time.perf_counter()
pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float32, use_safetensors=True).to("cpu")
pipe.set_progress_bar_config(disable=True)
print(f"загрузка {time.perf_counter() - t0:.1f} с; torch {torch.__version__}, diffusers {diffusers.__version__}, потоков {torch.get_num_threads()}")
res = {"platform": platform.platform(), "torch": torch.__version__, "diffusers": diffusers.__version__, "threads": torch.get_num_threads(), "dtype": "float32", "device": "cpu"}
t0 = time.perf_counter(); A = gen(pipe, 1)[0]; res["t_single"] = round(time.perf_counter() - t0, 1)
A2 = gen(pipe, 1)[0]
t0 = time.perf_counter(); B = gen(pipe, 4); res["t_batch4"] = round(time.perf_counter() - t0, 1)
A.save(os.path.join(OUT, "cpu_A_single.png")); B[0].save(os.path.join(OUT, "cpu_B_batch4_elem0.png"))
res["repeat_single_equal"] = hashlib.sha256(png(A)).hexdigest() == hashlib.sha256(png(A2)).hexdigest()
res["sha_A"], res["sha_B"] = hashlib.sha256(png(A)).hexdigest(), hashlib.sha256(png(B[0])).hexdigest()
res["pixel_equal"] = bool(np.array_equal(arr(A), arr(B[0])))
res["max_diff"] = int(np.max(np.abs(arr(A) - arr(B[0]))))
res["psnr"] = psnr(A, B[0])
B1 = gen(pipe, 1)  # контроль: элемент 1 пакета против одиночного seed+1
gens = [torch.Generator("cpu").manual_seed(SEED + 1)]
S1 = pipe(prompt=[BRIEF], num_inference_steps=1, guidance_scale=0.0, height=512, width=512, generator=gens).images[0]
res["elem1_vs_single_seed+1_psnr"] = psnr(S1, B[1])
print(json.dumps(res, ensure_ascii=False, indent=2))
json.dump(res, open(os.path.join(OUT, "cpu_check.json"), "w"), ensure_ascii=False, indent=2)
