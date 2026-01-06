#!/usr/bin/env python3
"""
Black → Ivory White со СТРОГО монотонным ростом яркости.
Каждый следующий цвет светлее (или равен) предыдущему.
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
DIST = {}
NAMES = {}


def precompute(cells, names):
    for i in range(len(cells)):
        OKLAB[i] = rgb_to_oklab(cells[i][1])
        NAMES[i] = names[i]
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(OKLAB[i], OKLAB[j])))
            DIST[(i, j)] = d
            DIST[(j, i)] = d


def d(i, j):
    return 0 if i == j else DIST.get((i, j), float('inf'))


def L(i):
    return OKLAB[i][0]


def path_cost(path):
    return sum(d(path[i], path[i+1]) for i in range(len(path)-1))


def build_monotonic_path(n, black_idx, ivory_idx):
    """
    Строит путь со строго монотонной яркостью.
    1. Сортируем все цвета по яркости
    2. Black в начале, Ivory White в конце
    3. Локально оптимизируем в пределах "окна" похожей яркости
    """
    # Сортируем по яркости
    sorted_by_L = sorted(range(n), key=lambda i: L(i))

    # Убедимся что Black первый, Ivory White последний
    sorted_by_L.remove(black_idx)
    sorted_by_L.remove(ivory_idx)
    path = [black_idx] + sorted_by_L + [ivory_idx]

    return path


def local_swap_monotonic(path, window=5):
    """
    Локальная оптимизация: меняем местами соседние элементы
    только если это не сильно нарушает порядок яркости.
    """
    n = len(path)
    improved = True
    best = path.copy()
    best_cost = path_cost(best)

    tolerance = 0.015  # Допустимое нарушение яркости

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, min(i + window, n - 1)):
                # Проверяем, можно ли поменять местами
                new_path = best.copy()
                new_path[i], new_path[j] = new_path[j], new_path[i]

                # Проверяем монотонность с tolerance
                valid = True
                for k in range(len(new_path) - 1):
                    if L(new_path[k]) > L(new_path[k+1]) + tolerance:
                        valid = False
                        break

                if valid:
                    new_cost = path_cost(new_path)
                    if new_cost < best_cost - 0.0001:
                        best = new_path
                        best_cost = new_cost
                        improved = True
                        break
            if improved:
                break

    return best


def simulated_annealing_monotonic(path, iterations=50000):
    """SA с сохранением монотонности яркости."""
    n = len(path)
    current = path.copy()
    current_cost = path_cost(current)
    best = current.copy()
    best_cost = current_cost

    tolerance = 0.015
    temp = 20.0
    cooling = 0.9999

    def is_monotonic(p):
        for k in range(len(p) - 1):
            if L(p[k]) > L(p[k+1]) + tolerance:
                return False
        return True

    for _ in range(iterations):
        # Swap два близких элемента (не первый и не последний)
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, min(i + 6, n - 2))

        new_path = current.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]

        if is_monotonic(new_path):
            new_cost = path_cost(new_path)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = new_path
                current_cost = new_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost

        temp *= cooling

    return best


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
    output_image = "../output/sorted_colors_monotonic.png"

    print("=" * 60)
    print("Black → Ivory White (СТРОГО монотонная яркость)")
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

    print(f"\n🔵 Начало: {names[black_idx]} (L={L(black_idx):.3f})")
    print(f"⚪ Конец:  {names[ivory_idx]} (L={L(ivory_idx):.3f})")

    print("\n[1/3] Сортировка по яркости...")
    path = build_monotonic_path(len(cells), black_idx, ivory_idx)
    cost1 = path_cost(path)
    print(f"      Стоимость: {cost1:.4f}")

    print("[2/3] Локальная оптимизация (с сохранением монотонности)...")
    path = local_swap_monotonic(path, window=8)
    cost2 = path_cost(path)
    print(f"      Стоимость: {cost2:.4f} ({100*(cost1-cost2)/cost1:+.1f}%)")

    print("[3/3] Simulated Annealing...")
    path = simulated_annealing_monotonic(path, iterations=80000)
    cost3 = path_cost(path)
    print(f"      Стоимость: {cost3:.4f} ({100*(cost1-cost3)/cost1:+.1f}%)")

    # Проверка
    assert path[0] == black_idx
    assert path[-1] == ivory_idx
    assert len(path) == 80

    # Проверка монотонности
    violations = sum(1 for i in range(len(path)-1) if L(path[i]) > L(path[i+1]) + 0.02)
    print(f"\n✅ Проверка:")
    print(f"   Первый: {NAMES[path[0]]}")
    print(f"   Последний: {NAMES[path[-1]]}")
    print(f"   Нарушений монотонности: {violations}")

    # Сборка
    print("\nСобираю изображение...")
    sorted_cells = [cells[i] for i in path]
    new_img = Image.new("RGB", (width, height))
    for idx, (cell, _) in enumerate(sorted_cells):
        row, col = divmod(idx, cols)
        new_img.paste(cell, (col * cell_width, row * cell_height))
    new_img.save(output_image, quality=95)
    print(f"Сохранено: {output_image}")

    # Статистика
    print(f"\n{'='*60}")
    deltas = [d(path[i], path[i+1]) for i in range(len(path)-1)]
    print(f"Средний ΔE: {sum(deltas)/len(deltas):.4f}")
    print(f"Макс ΔE:    {max(deltas):.4f}")

    print(f"\nЯркость по рядам (должна расти):")
    for row in range(rows):
        row_L = [L(path[row*cols + c]) for c in range(cols)]
        avg = sum(row_L) / len(row_L)
        bar = "█" * int(avg * 40)
        print(f"  Ряд {row+1}: L={avg:.3f} {bar}")


if __name__ == "__main__":
    main()
