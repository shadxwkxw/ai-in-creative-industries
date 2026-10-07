# ЛР №1 — Вариант 18: постер студенческого хакатона

Динамичная композиция из абстрактных геометрических форм и световых следов,
без логотипов платформ и брендов. Критерий результата — динамика считывается
через форму и свет, а не через текст или знакомую символику.

## Структура

```
data/      prompt.md — описание входа (авторский prompt и его структура)
configs/   run_config.json — модель, revision, prompt, seed, шаги, размер
src/       run_reproducibility.py — два запуска + сравнение SHA-256
           error_demo.py — безопасная копия с намеренной ошибкой Generator
artifacts/ run_001/, run_002/ — result.png + manifest.json
reports/   environment.txt, run_00X.log, sha256_comparison.json, error_demo.log
```

## Установка окружения

Проверено на macOS 15.5 (arm64), Python 3.14.2, только CPU.

```bash
cd v18
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Сначала PyTorch под своё устройство: https://pytorch.org/get-started/locally/
python -m pip install "diffusers==0.40.0" "transformers>=5,<6" \
  "accelerate>=1.2,<2" "safetensors>=0.5,<1" "Pillow>=11,<13"

python -m pip freeze > reports/environment.txt
```

Точные версии всех пакетов — в `reports/environment.txt`
(`pip install -r reports/environment.txt` восстанавливает среду на той же платформе).

> **Отклонение от методички.** Методичка указывает `transformers>=4.51,<5`, но
> эта связка несовместима с `diffusers==0.40.0`: diffusers 0.40.0 требует
> `huggingface-hub>=1.23,<2`, а transformers 4.x — `huggingface-hub<1`.
> pip завершается с `ResolutionImpossible`. Использован `transformers==5.17.0`.
> То же расхождение зафиксировано в протоколе демо-примера
> (`../LR01_reproducibility_protocol.md`, раздел 5).
>
> Также методичка рекомендует Python 3.11; фактически использован 3.14.2.

## Проверка среды до загрузки весов

```bash
python -c "import torch, diffusers; print(torch.__version__); print(diffusers.__version__); print(torch.cuda.is_available())"
```

## Запуск

```bash
export HF_HOME="$PWD/cache/huggingface"
python src/run_reproducibility.py 2>&1 | tee reports/run_console.log
```

Скрипт выполняет **два** запуска подряд с одинаковым seed, сохраняет
`artifacts/run_001` и `artifacts/run_002`, затем сравнивает SHA-256 и пишет
вердикт в `reports/sha256_comparison.json`. При несовпадении хешей скрипт
завершается с кодом 2.

## Проверка артефактов

```bash
python -c "from PIL import Image; im=Image.open('artifacts/run_001/result.png'); print(im.size, im.mode)"
python -m json.tool artifacts/run_001/manifest.json
shasum -a 256 artifacts/*/result.png      # Linux: sha256sum
```

## Типовая ошибка

```bash
python src/error_demo.py 2>&1 | tee reports/error_demo.log
```

На CPU-машине ожидается перехваченный `RuntimeError` при создании
`torch.Generator(device="cuda")`. Фактический вывод — в `reports/error_demo.log`.

## Критерий воспроизводимости

`reports/sha256_comparison.json` → `"exact_sha256_match": true` подтверждает
побитовое совпадение двух запусков **внутри одной фиксированной среды**.
Совпадение не гарантируется между CPU и GPU, разными версиями библиотек или
разными ОС.

## Что не входит в архив

`venv/` и кэш весов модели (`cache/huggingface`, ~4.8 ГБ) — исключить перед
сдачей (оба уже в `.gitignore`).
