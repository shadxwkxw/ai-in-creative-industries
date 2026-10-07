"""
Демонстрация типовой ошибки из методических указаний (раздел
"Намеренно введённая типовая ошибка") на безопасной копии кода.

Ошибка: генератор случайных чисел создаётся на устройстве "cuda", хотя
фактический инференс выполняется на CPU. Diffusers/PyTorch не позволяют
использовать CUDA-generator для операций на CPU-тензорах.

Запуск на CPU-машине:
    python src/error_demo.py

Ожидаемый результат: перехваченное исключение с диагностикой, а не
"тихий" неверный результат.
"""

import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "run_config.json").read_text(encoding="utf-8"))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    pipeline = AutoPipelineForText2Image.from_pretrained(
        CONFIG["model_id"],
        revision=CONFIG["revision"],
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).to(device)

    try:
        # --- НАМЕРЕННАЯ ОШИБКА ---
        # Создаём генератор на "cuda", даже если фактическое устройство — CPU.
        # На машине без CUDA это может упасть уже здесь, при создании Generator.
        broken_generator = torch.Generator(device="cuda").manual_seed(CONFIG["seed"])

        pipeline(
            prompt=CONFIG["prompt"],
            num_inference_steps=CONFIG["num_inference_steps"],
            guidance_scale=CONFIG["guidance_scale"],
            height=CONFIG["height"],
            width=CONFIG["width"],
            generator=broken_generator,
        )
    except Exception as exc:  # диагностика, а не сокрытие ошибки
        print("\n--- ПОЙМАННОЕ ИСКЛЮЧЕНИЕ (ожидаемо) ---")
        print(f"{type(exc).__name__}: {exc}")
        print("\nДиагноз: torch.Generator(device=\"cuda\") создан на машине без "
              "доступной или используемой CUDA, а инференс выполняется на "
              f"устройстве \"{device}\".")
        print("Исправление: torch.Generator(device=\"cpu\").manual_seed(seed) "
              "— как в src/run_reproducibility.py — либо выбор device для "
              "Generator только после проверки torch.cuda.is_available().")
        return

    print("Ошибка не воспроизвелась (например, есть доступная CUDA). "
          "Для демонстрации запускайте на CPU-машине без доступной CUDA.")


if __name__ == "__main__":
    main()
