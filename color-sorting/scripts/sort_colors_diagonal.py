#!/usr/bin/env python3
"""
Градиент по диагонали: верхний левый (Black) → нижний правый (Ivory White).
Яркость растёт вдоль диагоналей (row + col).
"""

from PIL import Image
import math
import random


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_oklab(rgb):
    r, g, b = [srgb_to_linear(c / 255.0) for c in rgb]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = l ** (1/3) if l >= 0 else -((-l) ** (1/3))
    m_ = m ** (1/3) if m >= 0 else -((-m) ** (1/3))
    s_ = s ** (1/3) if s >= 0 else -((-s) ** (1/3))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_val = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return (L, a, b_val)


OKLAB = {}
NAMES = {}


def precompute(cells, names):
    for i in range(len(cells)):
        OKLAB[i] = rgb_to_oklab(cells[i][1])
        NAMES[i] = names[i]


def L(i):
    return OKLAB[i][0]


def oklab_dist(i, j):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(OKLAB[i], OKLAB[j])))


def get_diagonal_cells(rows, cols):
    """
    Возвращает список диагоналей. Каждая диагональ — список (row, col).
    Диагональ d содержит все ячейки где row + col = d.
    """
    diagonals = {}
    for r in range(rows):
        for c in range(cols):
            d = r + c
            if d not in diagonals:
                diagonals[d] = []
            diagonals[d].append((r, c))

    # Сортируем ключи и возвращаем в порядке
    max_diag = rows - 1 + cols - 1
    return [diagonals.get(d, []) for d in range(max_diag + 1)]


def assign_colors_to_diagonals(n_colors, diagonals, black_idx, ivory_idx):
    """
    Распределяет цвета по диагоналям пропорционально их размеру.
    Black — диагональ 0, Ivory White — последняя диагональ.
    """
    # Сортируем все цвета по яркости
    sorted_by_L = sorted(range(n_colors), key=lambda i: L(i))

    # Убираем Black и Ivory из общего списка
    sorted_by_L.remove(black_idx)
    sorted_by_L.remove(ivory_idx)

    # Считаем сколько ячеек на каждой диагонали
    diag_sizes = [len(d) for d in diagonals]
    total_cells = sum(diag_sizes)

    # Black на диагонали 0, Ivory на последней
    # Остальные 78 цветов распределяем по остальным ячейкам

    # Резервируем 1 место на диагонали 0 для Black и 1 на последней для Ivory
    assignments = {i: [] for i in range(len(diagonals))}
    assignments[0].append(black_idx)
    assignments[len(diagonals) - 1].append(ivory_idx)

    # Сколько ещё нужно на каждую диагональ
    remaining = diag_sizes.copy()
    remaining[0] -= 1
    remaining[-1] -= 1

    # Распределяем оставшиеся 78 цветов
    color_idx = 0
    for diag_idx in range(len(diagonals)):
        while remaining[diag_idx] > 0 and color_idx < len(sorted_by_L):
            assignments[diag_idx].append(sorted_by_L[color_idx])
            color_idx += 1
            remaining[diag_idx] -= 1

    return assignments


def optimize_within_diagonal(color_indices, prev_diag_colors, next_diag_colors):
    """
    Оптимизирует порядок цветов внутри диагонали для плавности
    с соседними диагоналями.
    """
    if len(color_indices) <= 1:
        return color_indices

    # Используем nearest neighbor относительно предыдущей диагонали
    if prev_diag_colors:
        # Начинаем с цвета, ближайшего к последнему из предыдущей диагонали
        last_prev = prev_diag_colors[-1]
        start = min(color_indices, key=lambda i: oklab_dist(i, last_prev))
    else:
        # Первая диагональ — начинаем с самого тёмного
        start = min(color_indices, key=lambda i: L(i))

    result = [start]
    remaining = set(color_indices) - {start}

    current = start
    while remaining:
        # Выбираем ближайший
        next_color = min(remaining, key=lambda i: oklab_dist(current, i))
        result.append(next_color)
        remaining.remove(next_color)
        current = next_color

    return result


def build_diagonal_grid(cells, names, rows, cols, black_idx, ivory_idx):
    """
    Строит сетку с градиентом по диагонали.
    """
    diagonals = get_diagonal_cells(rows, cols)

    # Распределяем цвета по диагоналям
    assignments = assign_colors_to_diagonals(len(cells), diagonals, black_idx, ivory_idx)

    # Оптимизируем порядок внутри каждой диагонали
    optimized = {}
    prev_colors = []

    for diag_idx in range(len(diagonals)):
        colors = assignments[diag_idx]
        next_colors = assignments.get(diag_idx + 1, [])
        optimized[diag_idx] = optimize_within_diagonal(colors, prev_colors, next_colors)
        prev_colors = optimized[diag_idx]

    # Создаём сетку
    grid = [[None] * cols for _ in range(rows)]

    for diag_idx, diag_cells in enumerate(diagonals):
        colors_for_diag = optimized[diag_idx]

        # Сортируем позиции в диагонали для консистентности
        # Идём от верхнего правого к нижнему левому внутри диагонали
        sorted_positions = sorted(diag_cells, key=lambda x: (x[0], x[1]))

        for pos_idx, (r, c) in enumerate(sorted_positions):
            if pos_idx < len(colors_for_diag):
                grid[r][c] = colors_for_diag[pos_idx]

    return grid


def extract_color(cell):
    width, height = cell.size
    cx, cy = width // 2, height // 2 - 10
    samples = []
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            px, py = cx + dx, cy + dy
            if 0 <= px < width and 0 <= py < height:
                samples.append(cell.getpixel((px, py)))
    if samples:
        return tuple(sum(s[i] for s in samples) // len(samples) for i in range(3))
    return (128, 128, 128)


COLOR_NAMES = [
    "Black", "Electric Purple", "Lavender", "Cyberpunk", "Electric Indigo",
    "Neon Blue", "Navy Blue", "Sapphire", "Sky Blue", "Azure Blue",
    "Pacific Cyan", "Aquamarine", "Pacific Green", "Emerald", "Mint Green",
    "Malachite", "Shamrock Green", "Lemongrass", "Light Olive", "Satin Gold",
    "Pure Gold", "Amber", "Caramel", "Orange", "Carrot Juice",
    "Coral Red", "Persimmon", "Strawberry", "Raspberry", "Mystic Pearl",
    "Fandango", "Dark Lilac", "English Violet", "Moonstone", "Pine Green",
    "Hunter Green", "Pistachio", "Khaki Green", "Desert Sand", "Cappuccino",
    "Rosewood", "Ivory White", "Platinum", "Roman Silver", "Steel Grey",
    "Silver Blue", "Burgundy", "Indigo Dye", "Midnight Blue", "Onyx Black",
    "Battleship Grey", "Purple", "Grape", "Cobalt Blue", "French Blue",
    "Turquoise", "Jade Green", "Copper", "Chestnut", "Chocolate",
    "Marine Blue", "Tactical Pine", "Gunship Green", "Dark Green", "Seal Brown",
    "Rifle Green", "Ranger Green", "Camo Green", "Feldgrau", "Gunmetal",
    "Deep Cyan", "Mexican Pink", "Tomato", "Fire Engine", "Celtic Blue",
    "Old Gold", "Burnt Sienna", "Carmine", "Mustard", "French Violet",
]


def main():
    input_image = "../input/photo_2025-12-19_12-09-33.jpg"
    output_image = "../output/sorted_colors_diagonal.png"

    print("=" * 60)
    print("Градиент по ДИАГОНАЛИ")
    print("Верхний левый (Black) → Нижний правый (Ivory White)")
    print("=" * 60)

    img = Image.open(input_image).convert("RGB")
    width, height = img.size
    cols, rows = 10, 8
    cell_width = width // cols
    cell_height = height // rows

    print("\nИзвлекаю ячейки...")
    cells = []
    names = []
    for row in range(rows):
        for col in range(cols):
            x1, y1 = col * cell_width, row * cell_height
            cell = img.crop((x1, y1, x1 + cell_width, y1 + cell_height))
            cells.append((cell, extract_color(cell)))
            names.append(COLOR_NAMES[row * cols + col])

    print("Вычисляю OKLab...")
    precompute(cells, names)

    black_idx = names.index("Black")
    ivory_idx = names.index("Ivory White")

    print(f"\n🔵 Угол (0,0): Black (L={L(black_idx):.3f})")
    print(f"⚪ Угол (7,9): Ivory White (L={L(ivory_idx):.3f})")

    print("\nСтрою диагональный градиент...")
    grid = build_diagonal_grid(cells, names, rows, cols, black_idx, ivory_idx)

    # Проверка
    assert grid[0][0] == black_idx, "Black должен быть в (0,0)!"
    assert grid[rows-1][cols-1] == ivory_idx, "Ivory White должен быть в (7,9)!"

    print(f"\n✅ Black в позиции (0,0)")
    print(f"✅ Ivory White в позиции ({rows-1},{cols-1})")

    # Сборка изображения
    print("\nСобираю изображение...")
    new_img = Image.new("RGB", (width, height))

    for r in range(rows):
        for c in range(cols):
            color_idx = grid[r][c]
            if color_idx is not None:
                cell, _ = cells[color_idx]
                new_img.paste(cell, (c * cell_width, r * cell_height))

    new_img.save(output_image, quality=95)
    print(f"Сохранено: {output_image}")

    # Статистика по диагоналям
    print(f"\n{'='*60}")
    print("📊 Яркость по диагоналям")
    print(f"{'='*60}")

    diagonals = get_diagonal_cells(rows, cols)
    for d_idx, diag_cells in enumerate(diagonals):
        if not diag_cells:
            continue
        lums = [L(grid[r][c]) for r, c in diag_cells if grid[r][c] is not None]
        if lums:
            avg_L = sum(lums) / len(lums)
            bar = "█" * int(avg_L * 35)
            print(f"  Диаг {d_idx:2}: L={avg_L:.3f} {bar}")


if __name__ == "__main__":
    main()
