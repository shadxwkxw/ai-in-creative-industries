# ЛР 8 — Генерация и интерполяция в латентном пространстве. Вариант 18

Набор «disc_noise» × разброс сэмплов τ: 1 → 2 (одна и та же модель VAE d = 8, β = 1); seed 101…105.

```
src/lab07_lib.py, lab08_lib.py, lab08_demo.py — из МУ без изменений (d84b83a8…, f414c49b…, 8ab781bf…)
src/lab08_v18.py — скрипт варианта 18 (шаги 3–8 + проверки D1–D4); src/console_png.py — снимок консоли в PNG
reports/journal.md — журнал (гипотеза, допуск и D1–D4 записаны до запуска)
reports/passport_v18.md — паспорт
artifacts/v18/ — вывод консоли (txt, png), comparison.png, листы сэмплов, results_v18.json
artifacts/demo/ — воспроизведение демо (вариант 1): вывод консоли и comparison.png
```

Запуск (из artifacts/v18): `PYTHONPATH=../../src python ../../src/lab08_v18.py`
