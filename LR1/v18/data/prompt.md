# Входные данные — вариант 18

Единственный вход работы — авторский текстовый prompt. Внешние изображения,
брифы заказчика, персональные данные, товарные знаки и реальные лица
не используются. Машиночитаемая копия prompt и всех параметров —
`configs/run_config.json`; при расхождении источником истины считается конфиг.

## Prompt

```
dynamic poster concept for a student hackathon, diagonal composition of abstract
geometric shapes suggesting motion and speed, sharp light trails, energetic
contrast between dark background and bright cyan-magenta accents, clean layout
with empty space for a headline, no text, no logos, no brand marks, no people
```

## Структура prompt

| Элемент | Фрагмент | Зачем |
| --- | --- | --- |
| Предмет | dynamic poster concept for a student hackathon | контекст варианта |
| Форма (динамика) | diagonal composition of abstract geometric shapes suggesting motion and speed | динамика через форму |
| Свет (динамика) | sharp light trails, energetic contrast between dark background and bright … accents | динамика через свет |
| Палитра | cyan-magenta accents на тёмном фоне | контраст |
| Композиция | clean layout with empty space for a headline | место под заголовок, добавляемый в редакторе |
| Ограничения | no text, no logos, no brand marks, no people | ограничение варианта: без логотипов платформ |

## Замечание о negative-ограничениях

SD-Turbo работает с `guidance_scale=0.0`, поэтому отрицательный prompt и
classifier-free guidance не применяются. Фразы `no text, no logos` — это часть
обычного prompt, а не жёсткий фильтр. Отсутствие логотипов и текста
подтверждается только визуальным осмотром результата (см. отчёт, раздел 6).
