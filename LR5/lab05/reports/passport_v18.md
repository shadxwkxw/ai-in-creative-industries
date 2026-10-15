# Паспорт эксперимента — ЛР 5, вариант 18

Исполнитель: Беспалый Максим Евгеньевич, Т26ИИКТ-МИК11. Дата: 2026-10-15.

## 1. Цель и критерий
Проверить, совпадает ли одиночная генерация с первым элементом пакета из четырёх при одном seed, и объяснить причину.
Допуск (до запуска): identical — побитово; close — PSNR ≥ 40 дБ; different — PSNR < 40 дБ.

## 2. Данные
Текст запроса — бриф 3 из BRIEFS (lab05_lib.py): «poster concept for a science exhibition, abstract constellation of connected
nodes, dark background with white and cyan lines, balanced composition, no text, no logo». Реальных людей, логотипов и закрытых
материалов нет.

## 3. Модель
stabilityai/sd-turbo, ревизия b261bac6fd2cf515557d5d0707481eafa0485ec2, файлы fp16 (variant="fp16"), safetensors.
Лицензия: Stability AI Community License Agreement (5 июля 2024 г.), LICENSE.md на этой ревизии; использование учебное,
некоммерческое. Safety checker в конфигурации модели отсутствует.

## 4. Параметры
| Параметр | Значение |
|---|---|
| seed | 1018 (вариант 18: 1000 + N) |
| steps / guidance_scale / size | 1 / 0,0 / 512×512 |
| генератор | CPU, свой Generator на каждый элемент пакета (seed + i) |
| тип вычислений | float16 (E5 — float32) |
| A | batch_size 1 |
| B | batch_size 4, элемент 0 |

## 5. Среда
Google Colab, Tesla T4 (15360 МиБ, драйвер 580.82.07); Python 3.13.15; torch 2.11.0+cu130 (CUDA 13.0, cuDNN 9.27);
diffusers 0.40.0; transformers 5.18.0; accelerate 1.15.0; safetensors 0.8.0; Pillow 11.3.0; Linux-6.6.122+ x86_64.
lab05_lib.py — 6447 байт, SHA-256 4eb176506585ffcda1702605a767309fb451808229c3622f80ce882b7fa2ce1a.
Отличия от МУ: cu130 вместо cu128, transformers 5.18.0 вместо 5.16.1.
Доп. проверка: macOS 15.5 arm64, CPU, float32, torch 2.14.0, diffusers 0.40.0.

## 6. Процедура
Блокнот notebook/lab05_v18.ipynb → «Выполнить все» (T4); сеанс 2 — после «Отключиться и удалить среду выполнения».

## 7. Результаты
| | Значение |
|---|---|
| SHA-256 A | 38b03cfb71fd6fca99957c922915fd64f7377bf3e803600a2f44e83163bb9207 |
| SHA-256 B | f76faa5876e03320d3839d01330cb406e2f58dda161fdf59829291e8ee091217 |
| pixel_equal / max_diff / PSNR | false / 10 / 51,24 дБ |
| вердикт | close — практически совпадают |
| сеанс 2 | все SHA и метрики совпали; все PNG побитово идентичны |
| повтор пакета (E1) | побитово |
| одинаковый шум ×4 (E3) | 51,24 дБ к A — причина в вычислениях |
| float32 (E5) | max_diff 1, 79,76 дБ |
| CPU float32 (локально) | max_diff 1, 71,69 дБ |
| ошибка: общий генератор | 2-й запуск ≠ A, PSNR 12,7 дБ |

## 8. Ограничения
Один бриф, один seed, одна модель GPU (T4) и одна сборка torch/CUDA. На другой видеокарте или версии CUDA/cuDNN
SHA-256 может измениться. PSNR — численная мера, не оценка восприятия. Допуск 40 дБ условный.
