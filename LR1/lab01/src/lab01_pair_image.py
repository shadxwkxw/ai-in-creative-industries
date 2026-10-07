#!/usr/bin/env python3
"""ЛР 1, вариант 18. Пара изображений seed 101 (база | фактор) в одном масштабе.

Пиксели берутся из генератора без изменений и увеличиваются ближайшим соседом ×SCALE,
между изображениями — светлая полоса 8 px. Только стандартная библиотека.
Запуск: python src/lab01_pair_image.py --out artifacts/v18/pair_seed101_x4.png
"""
from __future__ import annotations

import argparse
import sys
import zlib
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lab01_experiment as ex  # noqa: E402
import lab01_generator as gen  # noqa: E402

SCALE, GAP = 4, 8


def upscale(pixels: bytes, size: int) -> list[bytes]:
    rows = []
    for y in range(size):
        row = bytes(v for v in pixels[y * size:(y + 1) * size] for _ in range(SCALE))
        rows.extend([row] * SCALE)
    return rows


def write_png(path: Path, rows: list[bytes]) -> None:
    w, h = len(rows[0]), len(rows)
    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=18)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    context, factor = ex.variant_to_pair(a.variant)
    base = {**gen.DEFAULT_PARAMS, **ex.CONTEXTS[context][1]}
    pert = {**base, **ex.FACTORS[factor][1](base)}
    size = int(base["size"])
    left, right = upscale(gen.generate(base, a.seed), size), upscale(gen.generate(pert, a.seed), size)
    gap = bytes([255]) * GAP
    write_png(a.out, [l + gap + r for l, r in zip(left, right)])
    print(f"{a.out}: слева база ({base['interp']}), справа фактор ({pert['interp']}), seed {a.seed}, ×{SCALE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
