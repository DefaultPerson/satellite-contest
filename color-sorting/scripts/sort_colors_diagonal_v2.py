#!/usr/bin/env python3
"""
Диагональный градиент v2: OKLab + Simulated Annealing
Минимизирует: ΔE между соседями + штраф за отклонение от целевой светлоты.
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


# Глобальные данные
OKLAB = {}
NAMES = {}
ROWS, COLS = 8, 10
L_MIN, L_MAX = 0.0, 1.0  # Будут установлены после загрузки


def precompute(cells, names):
    global L_MIN, L_MAX
    for i in range(len(cells)):
        OKLAB[i] = rgb_to_oklab(cells[i][1])
        NAMES[i] = names[i]

    # Находим диапазон яркости
    all_L = [OKLAB[i][0] for i in range(len(cells))]
    L_MIN = min(all_L)
    L_MAX = max(all_L)


def L(i):
    return OKLAB[i][0]


def oklab_delta_e(i, j):
    """ΔE в OKLab пространстве."""
    if i is None or j is None:
        return 0
    lab1, lab2 = OKLAB[i], OKLAB[j]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def target_lightness(r, c):
    """
    Целевая светлота для позиции (r, c) по диагонали.
    Диагональ = r + c, от 0 до (ROWS-1 + COLS-1) = 16
    """
    max_diag = ROWS - 1 + COLS - 1
    diag = r + c
    # Линейная интерполяция от L_MIN до L_MAX
    return L_MIN + (L_MAX - L_MIN) * (diag / max_diag)


def get_neighbors(r, c):
    """Возвращает список соседних позиций (4-связность)."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            neighbors.append((nr, nc))
    return neighbors


def calculate_cost(grid, lambda_L=2.0):
    """
    Общая стоимость размещения:
    - Сумма ΔE между всеми соседями
    - Штраф за отклонение от целевой светлоты
    """
    neighbor_cost = 0
    lightness_cost = 0

    for r in range(ROWS):
        for c in range(COLS):
            color_idx = grid[r][c]
            if color_idx is None:
                continue

            # Штраф за отклонение от целевой светлоты
            target_L = target_lightness(r, c)
            actual_L = L(color_idx)
            lightness_cost += (actual_L - target_L) ** 2

            # ΔE с соседями (считаем каждую пару один раз)
            for nr, nc in [(r, c+1), (r+1, c)]:  # Только право и низ
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    neighbor_idx = grid[nr][nc]
                    if neighbor_idx is not None:
                        neighbor_cost += oklab_delta_e(color_idx, neighbor_idx)

    return neighbor_cost + lambda_L * lightness_cost


def create_initial_grid(n_colors, black_idx, ivory_idx):
    """
    Начальное размещение: сортировка по яркости с учётом диагонали.
    """
    # Сортируем по яркости
    sorted_colors = sorted(range(n_colors), key=lambda i: L(i))

    # Создаём список позиций, отсортированных по диагонали
    positions = []
    for r in range(ROWS):
        for c in range(COLS):
            positions.append((r, c, r + c))  # (row, col, diagonal)

    positions.sort(key=lambda x: (x[2], x[0]))  # Сортируем по диагонали, потом по строке

    # Назначаем цвета позициям
    grid = [[None] * COLS for _ in range(ROWS)]

    for i, (r, c, _) in enumerate(positions):
        if i < len(sorted_colors):
            grid[r][c] = sorted_colors[i]

    # Убеждаемся что Black в (0,0) и Ivory в (7,9)
    # Находим где сейчас Black и Ivory
    black_pos = None
    ivory_pos = None
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == black_idx:
                black_pos = (r, c)
            if grid[r][c] == ivory_idx:
                ivory_pos = (r, c)

    # Swap Black to (0,0)
    if black_pos != (0, 0):
        old_color = grid[0][0]
        grid[0][0] = black_idx
        if black_pos:
            grid[black_pos[0]][black_pos[1]] = old_color

    # Swap Ivory to (7,9)
    if ivory_pos != (ROWS-1, COLS-1):
        old_color = grid[ROWS-1][COLS-1]
        grid[ROWS-1][COLS-1] = ivory_idx
        if ivory_pos and ivory_pos != (0, 0):  # Не перезаписываем Black
            grid[ivory_pos[0]][ivory_pos[1]] = old_color

    return grid


def simulated_annealing(grid, black_idx, ivory_idx, iterations=200000, lambda_L=2.0):
    """
    Simulated Annealing для оптимизации размещения.
    Фиксирует Black в (0,0) и Ivory White в (7,9).
    """
    current = deepcopy(grid)
    current_cost = calculate_cost(current, lambda_L)

    best = deepcopy(current)
    best_cost = current_cost

    temp = 1.0
    cooling = 0.99997

    # Позиции, которые можно менять (не углы)
    swappable = [(r, c) for r in range(ROWS) for c in range(COLS)
                 if (r, c) != (0, 0) and (r, c) != (ROWS-1, COLS-1)]

    for iteration in range(iterations):
        # Выбираем две случайные позиции для обмена
        pos1 = random.choice(swappable)
        pos2 = random.choice(swappable)

        if pos1 == pos2:
            continue

        # Делаем swap
        r1, c1 = pos1
        r2, c2 = pos2

        new_grid = deepcopy(current)
        new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]

        new_cost = calculate_cost(new_grid, lambda_L)
        delta = new_cost - current_cost

        # Принимаем или отклоняем
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_grid
            current_cost = new_cost

            if current_cost < best_cost:
                best = deepcopy(current)
                best_cost = current_cost

        temp *= cooling

        # Прогресс
        if iteration % 50000 == 0:
            print(f"      Итерация {iteration}: cost={current_cost:.4f}, best={best_cost:.4f}, T={temp:.6f}")

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
    output_image = "../output/sorted_colors_diagonal_v2.png"

    print("=" * 60)
    print("Диагональный градиент v2")
    print("OKLab + Simulated Annealing")
    print("Минимизация: ΔE(соседи) + λ·(L - target_L)²")
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

    black_idx = names.index("Black")
    ivory_idx = names.index("Ivory White")

    print(f"\n🔵 Black: L={L(black_idx):.3f}")
    print(f"⚪ Ivory White: L={L(ivory_idx):.3f}")
    print(f"📊 Диапазон L: [{L_MIN:.3f}, {L_MAX:.3f}]")

    print("\n[1/2] Создаю начальное размещение...")
    grid = create_initial_grid(len(cells), black_idx, ivory_idx)
    initial_cost = calculate_cost(grid, lambda_L=2.0)
    print(f"      Начальная стоимость: {initial_cost:.4f}")

    print("\n[2/2] Simulated Annealing (200K итераций)...")
    grid, final_cost = simulated_annealing(grid, black_idx, ivory_idx,
                                            iterations=200000, lambda_L=2.0)
    print(f"\n      Финальная стоимость: {final_cost:.4f}")
    print(f"      Улучшение: {100*(initial_cost - final_cost)/initial_cost:.1f}%")

    # Проверка
    assert grid[0][0] == black_idx, "Black должен быть в (0,0)!"
    assert grid[ROWS-1][COLS-1] == ivory_idx, "Ivory должен быть в (7,9)!"

    print(f"\n✅ Black в (0,0): {NAMES[grid[0][0]]}")
    print(f"✅ Ivory в (7,9): {NAMES[grid[ROWS-1][COLS-1]]}")

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

    # ΔE между соседями
    neighbor_deltas = []
    for r in range(ROWS):
        for c in range(COLS):
            for nr, nc in get_neighbors(r, c):
                if nr > r or (nr == r and nc > c):  # Избегаем дублей
                    delta = oklab_delta_e(grid[r][c], grid[nr][nc])
                    neighbor_deltas.append(delta)

    print(f"ΔE между соседями:")
    print(f"  Среднее: {sum(neighbor_deltas)/len(neighbor_deltas):.4f}")
    print(f"  Макс:    {max(neighbor_deltas):.4f}")
    print(f"  Мин:     {min(neighbor_deltas):.4f}")

    # Яркость по диагоналям
    print(f"\nЯркость по диагоналям:")
    for d in range(ROWS + COLS - 1):
        diag_L = []
        for r in range(ROWS):
            for c in range(COLS):
                if r + c == d:
                    diag_L.append(L(grid[r][c]))
        if diag_L:
            avg = sum(diag_L) / len(diag_L)
            target = target_lightness(d // 2, d - d // 2)  # Примерная целевая
            bar = "█" * int(avg * 35)
            print(f"  Диаг {d:2}: L={avg:.3f} {bar}")


if __name__ == "__main__":
    main()
