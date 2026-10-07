# ЛР 2 — Карточка модели, лицензии, происхождение данных и анализ рисков. Вариант 18

Сценарий: открытый учебный набор университета (некоммерческий).
Профиль дефектов: пробелы в происхождении (источник, дата, дубликаты).

## Структура

```
src/        5 скриптов из МУ, не изменены
artifacts/  v18/              manifest.csv, policy.json, variant.json, audit_report.*, data_card_draft.md,
                              data_card.md — заполненный паспорт набора (7 разделов)
            model_card_raw.md — карточка до устранения пробелов; model_card.md — итоговая (9 разделов + дополнения)
            risk_register.csv, risk_ranking.md — реестр рисков (RESULT: PASS)
            risk_register_bad.csv — намеренная ошибка (RESULT: FAIL)
            v18_repeat/, demo_v1_check/ — проверки воспроизводимости
reports/    journal.md — журнал (ожидания записаны до аудита), decisions.md — решения по находкам
```

## Команды (из папки lab02; среда — из ЛР 1, сторонние пакеты не нужны)

```bash
PY=../../LR1/lab01/.venv/bin/python     # или python3 >= 3.10
$PY src/lab02_make_variant.py --variant 18 --out artifacts/v18
$PY src/lab02_audit.py --manifest artifacts/v18/manifest.csv --policy artifacts/v18/policy.json --out artifacts/v18
$PY src/lab02_model_card.py --passport ../../LR1/lab01/artifacts/v18/passport.json --owner "<ФИО>" --out artifacts/model_card.md
$PY src/lab02_risks.py --csv artifacts/risk_register.csv --out artifacts/risk_ranking.md
```

Windows: при сравнении текстовых файлов с этой работой учитывайте переводы строк (CRLF и LF), см. журнал.
