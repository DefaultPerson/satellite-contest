#!/usr/bin/env python3
"""
Диагональный градиент v3:
- OKLab + Simulated Annealing
- Штраф за насыщенность на светлом конце (серые → к белому)
- Фиксированные "якоря": Black, Ivory White, Platinum, Roman Silver
"""

from PIL import Image
import math
import random
from copy import deepcopy


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
ROWS, COLS = 8, 10
L_MIN, L_MAX = 0.0, 1.0
MAX_DIAG = ROWS - 1 + COLS - 1  # = 16


def precompute(cells, names):
    global L_MIN, L_MAX
    for i in range(len(cells)):
        OKLAB[i] = rgb_to_oklab(cells[i][1])
        NAMES[i] = names[i]
    all_L = [OKLAB[i][0] for i in range(len(cells))]
    L_MIN = min(all_L)
    L_MAX = max(all_L)


def L(i):
    return OKLAB[i][0]


def chroma(i):
    """Насыщенность в OKLab: C = sqrt(a² + b²)"""
    _, a, b = OKLAB[i]
    return math.sqrt(a**2 + b**2)


def oklab_delta_e(i, j):
    if i is None or j is None:
        return 0
    lab1, lab2 = OKLAB[i], OKLAB[j]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def diag_param(r, c):
    """Параметр диагонали t ∈ [0, 1]. 0 = верхний левый, 1 = нижний правый."""
    return (r + c) / MAX_DIAG


def target_lightness(r, c):
    return L_MIN + (L_MAX - L_MIN) * diag_param(r, c)


def get_neighbors(r, c):
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            neighbors.append((nr, nc))
    return neighbors


def calculate_cost(grid, lambda_L=3.0, gamma_chroma=1.5, chroma_power=3):
    """
    Стоимость = ΔE(соседи) + λ·(L - target_L)² + γ·C²·t^p

    Последний член: штраф за насыщенность на светлом конце.
    Чем ближе к правому-нижнему углу (t→1), тем сильнее штраф за chroma.
    Это заставляет серые/серебристые цвета группироваться у белого.
    """
    neighbor_cost = 0
    lightness_cost = 0
    chroma_cost = 0

    for r in range(ROWS):
        for c in range(COLS):
            color_idx = grid[r][c]
            if color_idx is None:
                continue

            t = diag_param(r, c)

            # Штраф за отклонение светлоты
            target_L = target_lightness(r, c)
            actual_L = L(color_idx)
            lightness_cost += (actual_L - target_L) ** 2

            # Штраф за насыщенность на светлом конце
            # На тёмном конце (t≈0) штраф минимальный
            # На светлом конце (t≈1) штраф максимальный
            C = chroma(color_idx)
            chroma_cost += C**2 * (t ** chroma_power)

            # ΔE с соседями
            for nr, nc in [(r, c+1), (r+1, c)]:
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    neighbor_idx = grid[nr][nc]
                    if neighbor_idx is not None:
                        neighbor_cost += oklab_delta_e(color_idx, neighbor_idx)

    return neighbor_cost + lambda_L * lightness_cost + gamma_chroma * chroma_cost


def create_initial_grid(n_colors, anchors):
    """
    Начальное размещение с учётом якорей.
    anchors = {(r, c): color_idx, ...}
    """
    sorted_colors = sorted(range(n_colors), key=lambda i: L(i))

    # Убираем якоря из общего списка
    anchor_colors = set(anchors.values())
    sorted_colors = [c for c in sorted_colors if c not in anchor_colors]

    # Позиции по диагонали
    positions = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in anchors:
                positions.append((r, c, r + c))
    positions.sort(key=lambda x: (x[2], x[0]))

    grid = [[None] * COLS for _ in range(ROWS)]

    # Ставим якоря
    for (r, c), color_idx in anchors.items():
        grid[r][c] = color_idx

    # Заполняем остальное
    for i, (r, c, _) in enumerate(positions):
        if i < len(sorted_colors):
            grid[r][c] = sorted_colors[i]

    return grid


def simulated_annealing(grid, anchors, iterations=250000,
                        lambda_L=3.0, gamma_chroma=1.5, chroma_power=3):
    """SA с фиксированными якорями."""
    current = deepcopy(grid)
    current_cost = calculate_cost(current, lambda_L, gamma_chroma, chroma_power)

    best = deepcopy(current)
    best_cost = current_cost

    temp = 1.5
    cooling = 0.99997

    # Позиции без якорей
    anchor_positions = set(anchors.keys())
    swappable = [(r, c) for r in range(ROWS) for c in range(COLS)
                 if (r, c) not in anchor_positions]

    for iteration in range(iterations):
        pos1 = random.choice(swappable)
        pos2 = random.choice(swappable)
        if pos1 == pos2:
            continue

        r1, c1 = pos1
        r2, c2 = pos2

        new_grid = deepcopy(current)
        new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]

        new_cost = calculate_cost(new_grid, lambda_L, gamma_chroma, chroma_power)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_grid
            current_cost = new_cost
            if current_cost < best_cost:
                best = deepcopy(current)
                best_cost = current_cost

        temp *= cooling

        if iteration % 50000 == 0:
            print(f"      Итерация {iteration}: cost={current_cost:.4f}, best={best_cost:.4f}")

    return best, best_cost


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
    output_image = "../output/sorted_colors_diagonal_v3.png"

    print("=" * 60)
    print("Диагональный градиент v3")
    print("+ Штраф за насыщенность на светлом конце")
    print("+ Якоря: Black, Ivory White, Platinum, Roman Silver")
    print("=" * 60)

    img = Image.open(input_image).convert("RGB")
    width, height = img.size
    cell_width = width // COLS
    cell_height = height // ROWS

    print("\nИзвлекаю ячейки...")
    cells = []
    names = []
    for row in range(ROWS):
        for col in range(COLS):
            x1, y1 = col * cell_width, row * cell_height
            cell = img.crop((x1, y1, x1 + cell_width, y1 + cell_height))
            cells.append((cell, extract_color(cell)))
            names.append(COLOR_NAMES[row * COLS + col])

    print("Вычисляю OKLab...")
    precompute(cells, names)

    # Находим индексы ключевых цветов
    black_idx = names.index("Black")
    ivory_idx = names.index("Ivory White")
    platinum_idx = names.index("Platinum")
    roman_silver_idx = names.index("Roman Silver")
    steel_grey_idx = names.index("Steel Grey")

    print(f"\n📊 Ключевые цвета (L, Chroma):")
    for name, idx in [("Black", black_idx), ("Ivory White", ivory_idx),
                      ("Platinum", platinum_idx), ("Roman Silver", roman_silver_idx),
                      ("Steel Grey", steel_grey_idx)]:
        print(f"   {name:15} L={L(idx):.3f}, C={chroma(idx):.4f}")

    # Якоря: фиксируем позиции
    # Black в (0,0), Ivory White в (7,9)
    # Platinum, Roman Silver, Steel Grey рядом с Ivory White
    anchors = {
        (0, 0): black_idx,           # Black — верхний левый
        (7, 9): ivory_idx,           # Ivory White — нижний правый
        (7, 8): platinum_idx,        # Platinum — рядом с Ivory
        (6, 9): roman_silver_idx,    # Roman Silver — рядом
        (7, 7): steel_grey_idx,      # Steel Grey — рядом
    }

    print(f"\n🔒 Якоря:")
    for (r, c), idx in anchors.items():
        print(f"   ({r},{c}): {NAMES[idx]}")

    print("\n[1/2] Создаю начальное размещение...")
    grid = create_initial_grid(len(cells), anchors)
    initial_cost = calculate_cost(grid)
    print(f"      Начальная стоимость: {initial_cost:.4f}")

    print("\n[2/2] Simulated Annealing (250K итераций)...")
    grid, final_cost = simulated_annealing(grid, anchors, iterations=250000)
    print(f"\n      Финальная стоимость: {final_cost:.4f}")
    print(f"      Улучшение: {100*(initial_cost - final_cost)/initial_cost:.1f}%")

    # Проверка якорей
    print(f"\n✅ Проверка якорей:")
    for (r, c), idx in anchors.items():
        actual = grid[r][c]
        status = "✓" if actual == idx else "✗"
        print(f"   {status} ({r},{c}): {NAMES[actual]}")

    # Сборка изображения
    print("\nСобираю изображение...")
    new_img = Image.new("RGB", (width, height))
    for r in range(ROWS):
        for c in range(COLS):
            color_idx = grid[r][c]
            if color_idx is not None:
                cell, _ = cells[color_idx]
                new_img.paste(cell, (c * cell_width, r * cell_height))

    new_img.save(output_image, quality=95)
    print(f"Сохранено: {output_image}")

    # Статистика
    print(f"\n{'='*60}")
    print("📊 Статистика")
    print(f"{'='*60}")

    neighbor_deltas = []
    for r in range(ROWS):
        for c in range(COLS):
            for nr, nc in get_neighbors(r, c):
                if nr > r or (nr == r and nc > c):
                    delta = oklab_delta_e(grid[r][c], grid[nr][nc])
                    neighbor_deltas.append(delta)

    print(f"ΔE между соседями: avg={sum(neighbor_deltas)/len(neighbor_deltas):.4f}, max={max(neighbor_deltas):.4f}")

    print(f"\nЯркость по диагоналям:")
    for d in range(MAX_DIAG + 1):
        diag_L = [L(grid[r][c]) for r in range(ROWS) for c in range(COLS) if r + c == d]
        if diag_L:
            avg = sum(diag_L) / len(diag_L)
            bar = "█" * int(avg * 35)
            print(f"  Диаг {d:2}: L={avg:.3f} {bar}")


if __name__ == "__main__":
    main()
