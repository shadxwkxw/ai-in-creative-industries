# ЛР 3, вариант 18: сцена 3 «тёмная» × воздействие 6 «сохранение в PNG 8 бит и чтение».
# Запуск в Engee: загрузить lab03_lib.jl и этот файл в папку LR03, затем в ячейке:
#     cd("/user/<папка курса>/LR03"); include("LR03_18.jl")
# Каждый блок ниже можно также вставить в отдельную ячейку .ngscript (шаги МУ 3–9).
# Все файлы результатов пишутся в подпапку OUT рядом с рабочим каталогом.

include("lab03_lib.jl")
using Plots, Printf, Dates, SHA

const N_VARIANT = 18
const SCENE, FACTOR = divrem(N_VARIANT - 1, 6) .+ 1        # (3, 6)
const P = scene_preset(SCENE)                              # (lo=0.02, hi=0.30, noise=0.02, checker=0)
const SEEDS = 101:105
const OUT = joinpath(pwd(), "artifacts_v18")
mkpath(OUT)

# Практический допуск — тот же, что записан в журнал ДО запуска (не менять после просмотра результатов)
const TOL_REL = 0.01          # среднее и контраст: относительное изменение > 1 %
const TOL_ENTROPY = 0.05      # энтропия: |Δ| > 0,05 бит
const TOL_PSNR = 40.0         # PSNR < 40 дБ — практически значимая потеря

println("=== Среда ===")
println("Дата (UTC): ", Dates.format(now(UTC), "yyyy-mm-dd HH:MM:SS"))
println("Julia: ", VERSION)
println("Рабочий каталог: ", pwd())
lib = read("lab03_lib.jl")
println("lab03_lib.jl: ", length(lib), " байт, SHA-256 ", bytes2hex(sha256(lib)))
println("Вариант ", N_VARIANT, ": сцена ", SCENE, " ", P, ", воздействие ", FACTOR)
for pkg in ("Images", "Plots", "SHA")
    v = try string(pkgversion(getfield(Main, Symbol(pkg)))) catch; "?" end
    println("  ", pkg, " ", v)
end

println("\n=== Шаг 3: сцена и её свойства (seed 101) ===")
img = make_scene(101; P...)
d = describe_image(img)
println(d)
@printf("Проверка объёма: 128·128·3·8 = %d байт; describe_image: %d\n", 128 * 128 * 3 * 8, d.bytes)

println("\n=== Шаг 4: яркость двумя способами ===")
g_manual = luma_manual(img)
g = luma_images(img)
maxdiff = maximum(abs.(g_manual .- g))
println("max |luma_manual − luma_images| = ", maxdiff)
println("isapprox (atol = 1e-6): ", isapprox(g_manual, g; atol=1e-6))
println("SHA manual[1:16] = ", array_sha(g_manual)[1:16], "   SHA Gray[1:16] = ", array_sha(g)[1:16],
        "   совпадают: ", array_sha(g_manual) == array_sha(g))
@printf("Диапазон яркости: min = %.4f, max = %.4f, среднее = %.4f\n", minimum(g), maximum(g), mean(g))

println("\n=== Шаг 5: контрольная сумма и её воспроизведение ===")
sha1 = array_sha(luma_images(make_scene(101; P...)))
sha2 = array_sha(luma_images(make_scene(101; P...)))
println("SHA-256 (seed 101), запуск 1: ", sha1)
println("SHA-256 (seed 101), запуск 2: ", sha2)
println("совпадают: ", sha1 == sha2)
println("SHA-256 (seed 102)[1:16] = ", array_sha(luma_images(make_scene(102; P...)))[1:16], "  — другой seed, другая сумма")

println("\n=== Шаги 6–7: воздействие, гистограммы, изображения ===")
h = apply_factor(g, FACTOR)                  # png_roundtrip: Gray{N0f8} → PNG → чтение
println("После воздействия: тип ", eltype(h), ", уровней яркости в сцене: до ", length(unique(g)),
        ", после ", length(unique(h)))
@printf("max |g − h| = %.6f (половина шага 8 бит = %.6f)\n", maximum(abs.(g .- h)), 0.5 / 255)
@printf("PSNR (seed 101) = %.4f дБ\n", psnr_db(g, h))
cb, ca = hist_counts(g, 32), hist_counts(h, 32)
println("Гистограмма до   (32 инт.): ", cb)
println("Гистограмма после(32 инт.): ", ca)
println("Отсчётов, сменивших интервал: ", sum(abs.(ca .- cb)) ÷ 2, " из ", length(g))

save(joinpath(OUT, "scene_rgb_seed101.png"), img)
# Внимание: сохранение «до» в PNG само квантует до 8 бит, поэтому файл «до» побитово равен файлу «после».
# Это и есть иллюстрация воздействия; для честного «до» сохраняем ещё 16-битную версию.
save(joinpath(OUT, "luma_before_seed101.png"), Gray.(g))
save(joinpath(OUT, "luma_before_16bit_seed101.png"), Gray{N0f16}.(g))
save(joinpath(OUT, "luma_after_png_seed101.png"), Gray.(h))
# Разность ×50 для наглядности (иначе изображение чёрное)
save(joinpath(OUT, "diff_x50_seed101.png"), Gray.(clamp.(0.5 .+ 50 .* (h .- g), 0, 1)))

edges = range(0, 1; length=33); mids = (edges[1:end-1] .+ edges[2:end]) ./ 2
p1 = bar(mids, cb; bar_width=1/32, label="до", xlabel="яркость", ylabel="число отсчётов",
         title="Гистограмма до (seed 101)", xlims=(0, 1), color=:steelblue)
p2 = bar(mids, ca; bar_width=1/32, label="после PNG 8 бит", xlabel="яркость", ylabel="число отсчётов",
         title="Гистограмма после PNG 8 бит (seed 101)", xlims=(0, 1), color=:darkorange)
savefig(p1, joinpath(OUT, "hist_before.png"))
savefig(p2, joinpath(OUT, "hist_after.png"))
w = 1 / 32 / 2.2
p3 = bar(mids[1:12] .- w / 2, cb[1:12]; bar_width=w, label="до", color=:steelblue,
         xlabel="яркость (интервалы 1–12 из 32)", ylabel="число отсчётов", title="До и после PNG 8 бит (seed 101)")
bar!(p3, mids[1:12] .+ w / 2, ca[1:12]; bar_width=w, label="после", color=:darkorange)
savefig(p3, joinpath(OUT, "hist_before_after_zoom.png"))
println("Файлы: ", join(readdir(OUT), ", "))

println("\n=== Шаг 8: variant_summary(18), серия seed 101…105 ===")
vs = variant_summary(N_VARIANT; seeds=SEEDS)
@printf("Средний PSNR = %.4f дБ\n", vs.psnr)
println("метрика | база | s | после | Δ | |Δ|/s | |Δ|>2s | отн. изм. | практически значимо")
rows = []
for (m, b, s, a, dd, ratio, sig) in vs.rows
    rel = dd / b
    prac = m == :entropy ? abs(dd) > TOL_ENTROPY : abs(rel) > TOL_REL
    push!(rows, (m, b, s, a, dd, ratio, sig, rel, prac))
    @printf("%-8s | %.6f | %.3e | %.6f | %+.3e | %.3f | %s | %+.4f %% | %s\n",
            m, b, s, a, dd, ratio, sig ? "да" : "нет", 100rel, prac ? "да" : "нет")
end
println("PSNR практически значимая потеря (< ", TOL_PSNR, " дБ): ", vs.psnr < TOL_PSNR ? "да" : "нет")

println("\n=== Значения по каждому seed (для проверки разброса) ===")
for s in SEEDS
    gg = luma_images(make_scene(s; P...)); hh = apply_factor(gg, FACTOR)
    m0, m1 = metrics3(gg), metrics3(hh)
    @printf("seed %d: mean %.6f→%.6f  std %.6f→%.6f  H %.4f→%.4f  PSNR %.3f\n",
            s, m0.mean, m1.mean, m0.contrast, m1.contrast, m0.entropy, m1.entropy, psnr_db(gg, hh))
end

println("\n=== Чувствительность вывода к допуску ===")
for tol in (0.001, 0.0001, 0.00001)
    print("допуск ", 100tol, " %: ")
    println(join(["$(r[1])=$(abs(r[8]) > tol ? "значимо" : "нет")" for r in rows if r[1] != :entropy], ", "))
end
for tol in (0.05, 0.01, 0.001)
    r = rows[3]; println("энтропия, допуск ", tol, " бит: ", abs(r[5]) > tol ? "значимо" : "нет")
end
for t in (40.0, 50.0, 60.0)
    println("PSNR-порог ", t, " дБ: потеря ", vs.psnr < t ? "значима" : "не значима")
end

println("\n=== Сверка с демо преподавателя (вариант 1) ===")
d1 = describe_image(make_scene(101))
println("describe_image(make_scene(101)) = ", d1)
println("maxdiff luma (сцена 1) = ", maximum(abs.(luma_manual(make_scene(101)) .- luma_images(make_scene(101)))))
println("sha256(сцена 1, seed 101)[1:16] = ", array_sha(luma_images(make_scene(101)))[1:16], "  (демо: 50567c6b9e5e4ac1)")
v1 = variant_summary(1)
println("variant_summary(1): psnr = ", v1.psnr, "  (демо: 34.594244830067446)")
for r in v1.rows
    println("  ", r[1], ": base ", r[2], ", s ", r[3], ", after ", r[4], ", d ", r[5], ", ratio ", round(r[6]; digits=3))
end

println("\n=== Паспорт (поля для записи) ===")
println("Вариант 18; сцена 3 ", P, "; воздействие 6 (PNG 8 бит, Gray{N0f8}); n = 128; seeds ", SEEDS,
        "; интервалов гистограммы 32; критерий |Δ| > 2s; допуск: 1 % / 0,05 бит / 40 дБ")
println("SHA-256 яркости (seed 101): ", sha1)
