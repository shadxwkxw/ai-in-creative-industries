# ЛР 8. Вариант 18: набор «диск с шумом» (disc_noise), фактор «разброс сэмплов tau: 1 → 2».
# Структура как в lab08_demo.py (раздел 6.2 МУ); lab07_lib.py и lab08_lib.py — без изменений.
# Запуск из папки с результатами: PYTHONPATH=<src> python lab08_v18.py  (пишет comparison.png, листы и results_v18.json)
import json, platform, sys, time
import numpy as np
import torch
from PIL import Image
import lab07_lib as L
import lab08_lib as V

N = 18
c, k = (N - 1) // 6, (N - 1) % 6
kind = L.KINDS[c]
print(f"Python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}, устройство {L.DEVICE}, "
      f"потоков {torch.get_num_threads()}, {platform.platform()}")
print(f"Вариант {N}: набор {c + 1} {kind}, фактор {k + 1} {V.FACTORS[k]}")
a_cfg, b_cfg = V.settings(V.FACTORS[k])
print("A:", a_cfg)
print("B:", b_cfg)
print("различаются полями:", [f for f in a_cfg if a_cfg[f] != b_cfg[f]])
out = dict(variant=N, kind=kind, factor=V.FACTORS[k], A=a_cfg, B=b_cfg)

# Шаг 4. Серия из пяти seed: четыре показателя режимов A и B
t0 = time.time()
result = V.variant_summary(N)
out["time_variant_s"] = round(time.time() - t0, 1)
print(f"\nvariant_summary({N}) — {out['time_variant_s']} с (5 моделей; режим B использует те же модели)")
for name, a, s, b, d, q, sig in result["rows"]:
    print(f"{name:7s} A={a:7.3f} s={s:6.3f} B={b:7.3f} d={d:+7.3f} |d|/s={q:7.2f} 2s={'да' if sig else 'нет'}")
out["rows"] = result["rows"]

# Значения по каждому seed (модели уже в кэше)
per_seed = {}
print("\nПо seed:            " + "  ".join(f"{s:>6d}" for s in L.SEEDS))
for tag, cfg in (("A", a_cfg), ("B", b_cfg)):
    ms = [V.run(kind, cfg, s) for s in L.SEEDS]
    for m in ("recon", "interp", "sample", "r2"):
        per_seed[f"{tag}_{m}"] = [round(x[m], 3) for x in ms]
        print(f"{tag} {m:7s}" + " " * 10 + "  ".join(f"{x[m]:6.2f}" for x in ms))
out["per_seed"] = per_seed

# Время обучения одной модели (журнал, раздел 4, шаг 5)
Xtr, Xte, Ftr, Fte = L.dataset(kind)
torch.manual_seed(999)
t0 = time.time()
V.train_vae(V.ConvVAE(8).to(L.DEVICE), Xtr, 1.0, 30, seed=999)
out["time_one_model_s"] = round(time.time() - t0, 1)
print(f"\nобучение одной модели (VAE d=8, 30 эпох): {out['time_one_model_s']} с")


def nn_psnr(model, z, ref):
    """PSNR ближайшей сцены из ref для изображений D(z) — как в V.metrics (по каждому сэмплу, затем среднее)."""
    with torch.no_grad():
        xs = model.decode(z).flatten(1)
    mse = torch.cdist(xs, ref.flatten(1)).pow(2) / xs.shape[1]
    return 10 * torch.log10(1.0 / mse.min(1).values.clamp_min(1e-10))


def eps(seed, d=8, n=V.N_SAMPLES):
    """Те же 𝜀, что в V.metrics: z = tau * eps."""
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    return torch.randn((n, d), generator=g).to(L.DEVICE)


def hf(x):
    """Высокочастотная энергия: std остатка после усреднения 3×3 (внутренние пиксели)."""
    xp = torch.nn.functional.avg_pool2d(x, 3, stride=1)
    return float((x[:, :, 1:-1, 1:-1] - xp).std())


# Шаг 5 (D1). Диагностика: разброс mu, KL по осям, активные оси; нормы ‖mu‖ и ‖z‖
print("\nДиагностика (A и B — одна модель, поэтому совпадают):")
diag = []
for s in L.SEEDS:
    model = V.get_model(kind, a_cfg, s)
    with torch.no_grad():
        mu, lv = model.encode_dist(Xte)
        kl_axis = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
    act = kl_axis > 0.05
    nmu = mu.norm(dim=1)
    p99 = float(nmu.quantile(0.99))
    e = eps(s)
    row = dict(seed=s, mu_std=mu.std(0).cpu().numpy().astype(float).round(2).tolist(), kl_axis=kl_axis.cpu().numpy().astype(float).round(2).tolist(),
               active=int(act.sum()), mu_norm_median=round(float(nmu.median()), 2), mu_norm_p99=round(p99, 2))
    mx = mu[:, act].abs().max(0).values
    for tau in (1.0, 2.0):
        z = tau * e
        row[f"z_norm_median_t{tau:g}"] = round(float(z.norm(dim=1).median()), 2)
        row[f"frac_norm_gt_p99_t{tau:g}"] = round(float((z.norm(dim=1) > p99).float().mean()), 3)
        row[f"frac_active_out_t{tau:g}"] = round(float((z[:, act].abs() > mx).any(1).float().mean()), 3)
    diag.append(row)
    print(f"seed {s}: разброс mu {row['mu_std']}; KL по осям {row['kl_axis']}; активных осей {row['active']}")
    print(f"   ‖mu‖ медиана {row['mu_norm_median']}, p99 {row['mu_norm_p99']}; ‖z‖ медиана tau=1 {row['z_norm_median_t1']}, "
          f"tau=2 {row['z_norm_median_t2']}; доля ‖z‖ > p99(‖mu‖): {row['frac_norm_gt_p99_t1']} → {row['frac_norm_gt_p99_t2']}; "
          f"доля с |z_j| > max|mu_j| на активной оси: {row['frac_active_out_t1']} → {row['frac_active_out_t2']}")
out["diag"] = diag

# D2. Развёртка tau по пяти seed
taus = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
sweep = {}
print("\nРазвёртка tau (PSNR сэмплов, среднее ± s по seed 101–105):")
for tau in taus:
    v = [float(nn_psnr(V.get_model(kind, a_cfg, s), tau * eps(s), Xte).mean()) for s in L.SEEDS]
    sweep[tau] = [round(x, 3) for x in v]
    print(f"tau={tau:3.1f}: {np.mean(v):6.2f} ± {np.std(v, ddof=1):4.2f}   {np.round(v, 2).tolist()}")
out["tau_sweep"] = {str(t): v for t, v in sweep.items()}

# D3, D4. Шум в сэмплах и «правдоподобные» сцены: сравнение с чистыми сценами (тот же генератор, без шума)
rng = np.random.default_rng(99)
clean_bank = torch.tensor(np.stack([L.render(kind, f) for f in rng.random((10000, L.N_FACTORS[kind]))]), device=L.DEVICE)
rng7 = np.random.default_rng(7)
f_all = rng7.random((4096 + 1024, L.N_FACTORS[kind]))
Xte_clean = torch.tensor(np.stack([L.render(kind, f) for f in f_all[4096:]]), device=L.DEVICE)
assert torch.allclose(Xte_clean, Fte.new_tensor(np.stack([L.render(kind, f) for f in Fte.cpu().numpy()])))
print(f"\nПотолок шума: PSNR чистых тестовых сцен к зашумлённым {L.psnr_images(Xte_clean, Xte):.2f} дБ")
print(f"Высокочастотная энергия: зашумлённые тест {hf(Xte):.4f}; чистые тест {hf(Xte_clean):.4f}")
noise_rows = []
for s in L.SEEDS:
    model = V.get_model(kind, a_cfg, s)
    r = dict(seed=s)
    for tau in (1.0, 2.0):
        with torch.no_grad():
            xs = model.decode(tau * eps(s))
        r[f"hf_t{tau:g}"] = round(hf(xs), 4)
        pc = nn_psnr(model, tau * eps(s), clean_bank)
        r[f"psnr_clean_t{tau:g}"] = round(float(pc.mean()), 2)
        r[f"frac_plausible_t{tau:g}"] = round(float((pc >= 25).float().mean()), 3)
        r[f"psnr_noisy_t{tau:g}"] = round(float(nn_psnr(model, tau * eps(s), Xte).mean()), 2)
    noise_rows.append(r)
    print(f"seed {s}: HF сэмплов tau=1 {r['hf_t1']}, tau=2 {r['hf_t2']}; PSNR к ближайшей чистой сцене (10 000) "
          f"{r['psnr_clean_t1']} → {r['psnr_clean_t2']} дБ; доля ≥ 25 дБ {r['frac_plausible_t1']} → {r['frac_plausible_t2']}")
out["noise_check"] = noise_rows

# Шаг 8. Обычный автокодировщик (beta = 0) на своём наборе: сэмплы из N(0, I) и из распределения mu на обучающей выборке
print("\nОбычный автокодировщик (beta = 0) против VAE (beta = 1), сэмплы, PSNR ближайшей тестовой сцены:")
ae_rows = []
for s in L.SEEDS:
    ae = V.get_model(kind, dict(V.BASE, beta=0.0), s)
    with torch.no_grad():
        mu_tr, mu_te = ae.encode(Xtr), ae.encode(Xte)
    m = V.metrics(ae, kind, "lin", 1.0, s)
    mean, cov = mu_tr.mean(0), torch.cov(mu_tr.T)
    chol = torch.linalg.cholesky(cov + 1e-6 * torch.eye(8, device=L.DEVICE))
    z_fit = mean + eps(s) @ chol.T
    r = dict(seed=s, recon=round(m["recon"], 2), interp=round(m["interp"], 2), r2=round(m["r2"], 3), sample_N01=round(m["sample"], 2),
             sample_fit=round(float(nn_psnr(ae, z_fit, Xte).mean()), 2), mu_std=mu_te.std(0).cpu().numpy().astype(float).round(2).tolist(),
             vae_sample_t1=round(V.run(kind, a_cfg, s)["sample"], 2))
    ae_rows.append(r)
    print(f"seed {s}: AE разброс mu {r['mu_std']}; AE сэмплы N(0,I) {r['sample_N01']} дБ, N(mean,cov mu) {r['sample_fit']} дБ; "
          f"VAE tau=1 {r['vae_sample_t1']} дБ; AE реконструкция {r['recon']}, интерполяция {r['interp']}")
for key in ("sample_N01", "sample_fit", "vae_sample_t1", "recon", "interp"):
    v = [r[key] for r in ae_rows]
    print(f"   {key}: {np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}")
out["ae_check"] = ae_rows


# Шаг 6. Лист сравнения comparison.png (как в демо: строки по 8 изображений 32×32, увеличение ×4)
def strip(imgs):
    return np.concatenate(list(imgs.permute(0, 2, 3, 1).cpu().numpy()), axis=1)


def save_sheet(rows, path):
    a = np.concatenate(rows, axis=0)
    Image.fromarray((a * 255).round().astype("uint8")).resize((a.shape[1] * 4, a.shape[0] * 4), Image.NEAREST).save(path)


g = torch.Generator(device="cpu").manual_seed(5)
z = torch.randn((8, 8), generator=g).to(L.DEVICE)
vae = V.get_model(kind, a_cfg, 101)
ae = V.get_model(kind, dict(V.BASE, beta=0.0), 101)
with torch.no_grad():
    x1, x2 = vae.decode(1.0 * z), vae.decode(2.0 * z)
    nn2 = torch.cdist(x2.flatten(1), Xte.flatten(1)).argmin(1)
    # строки: тестовые сцены (с шумом) | VAE tau=1 | VAE tau=2 (те же 𝜀, умноженные на 2) | ближайшие тестовые к tau=2
    save_sheet([strip(Xte[:8]), strip(x1), strip(x2), strip(Xte[nn2])], "comparison.png")
    # развёртка tau для одного 𝜀 на строку: столбцы tau = 0; 0,5; 1; 1,5; 2; 2,5; 3; 4
    tt = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    save_sheet([strip(vae.decode(torch.stack([t * z[i] for t in tt]))) for i in range(4)], "tau_path.png")
    # сетки 8×8 сэмплов tau=1 и tau=2 (те же 𝜀 из V.metrics, seed 101)
    e = eps(101, n=64)
    for tau in (1.0, 2.0):
        xs = vae.decode(tau * e).permute(0, 2, 3, 1).cpu().numpy()
        save_sheet([np.concatenate(list(xs[r * 8:(r + 1) * 8]), axis=1) for r in range(8)], f"samples_grid_tau{tau:g}.png")
    # обычный автокодировщик: N(0, I) и N(mean, cov mu)
    mu_tr = ae.encode(Xtr)
    chol = torch.linalg.cholesky(torch.cov(mu_tr.T) + 1e-6 * torch.eye(8, device=L.DEVICE))
    save_sheet([strip(ae.decode(z)), strip(ae.decode(mu_tr.mean(0) + z @ chol.T)), strip(x1)], "comparison_ae.png")
print("\nСохранены: comparison.png, tau_path.png, samples_grid_tau1.png, samples_grid_tau2.png, comparison_ae.png")

json.dump(out, open("results_v18.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("results_v18.json записан")
