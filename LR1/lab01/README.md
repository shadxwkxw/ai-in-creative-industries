# ЛР 1 — Подготовка среды и паспорт эксперимента. Вариант 18

Контекст: мраморный узор (обложка). Фактор: линейная интерполяция вместо сглаженной.

## Структура

```
src/        lab01_generator.py, lab01_experiment.py — скрипты из МУ, не изменены
            lab01_analysis.py — доп. анализ (SE(Δ), парное сравнение, сверка с демо)
            lab01_pair_image.py — пара изображений seed 101 в одном масштабе
configs/    variant18.json — параметры варианта
artifacts/  v18/            — основной эксперимент (n = 5): паспорт, results.json, PNG, пара изображений
            v18_n20/        — повтор при n = 20
            demo_v1_check/  — воспроизведение демо преподавателя (вариант 1)
            analysis/       — analysis.json
            error_demo/     — типовая ошибка: запуск без seed и с seed 101
reports/    journal.md — журнал (гипотеза записана до запуска), passport_v18.md — паспорт,
            error_demo.log
requirements.txt — пуст: сторонние пакеты не используются
```

## Запуск (macOS / Linux)

```bash
python3 -m venv .venv
.venv/bin/python -m pip freeze > requirements.txt
.venv/bin/python src/lab01_experiment.py --variant 18 --out artifacts/v18
```

Windows PowerShell: `python -m venv .venv`, затем `.venv\Scripts\python.exe` вместо `.venv/bin/python`.

`.venv` в архив не включается.
