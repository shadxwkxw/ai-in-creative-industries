# ЛР 3 — Подготовка изображений и визуализация в Engee. Вариант 18

Сцена 3 «тёмная» × воздействие 6 «сохранение в PNG 8 бит и чтение».

```
src/        lab03_lib.jl (из МУ, без изменений), LR03_18.jl — скрипт варианта (шаги 3–9)
artifacts_engee/  PNG из основного запуска в Engee (2026-10-07 14:31 UTC)
artifacts/  локальная проверка: PNG сцены, яркости до (8 и 16 бит) / после, разности ×50; гистограммы до, после, совмещённая
reports/    engee_output.txt — вывод Engee; screenshots/ — скриншоты Engee; engee_artifacts_v18.zip — архив из Engee
            journal.md — журнал (гипотеза и допуск записаны до запуска), passport_v18.md, run_local*.log
Project.toml, Manifest.toml — точные версии пакетов локальной проверки
ENGEE_STEPS.md — порядок запуска в Engee
```

Локальная проверка (Julia 1.12.4, как в Engee):

```bash
cd src && julia +1.12.4 --project=.. LR03_18.jl
```
