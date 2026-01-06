#!/usr/bin/env python3
"""
Финальная сортировка: Black → ... → Ivory White
Все 80 цветов, максимально плавные переходы в OKLab.
"""

from PIL import Image, ImageDraw, ImageFont
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


# Глобальные кэши
OKLAB = {}
DIST = {}
NAMES = {}


def precompute(cells, names):
    global OKLAB, DIST, NAMES
    n = len(cells)

    for i in range(n):
        OKLAB[i] = rgb_to_oklab(cells[i][1])
        NAMES[i] = names[i]

    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(OKLAB[i], OKLAB[j])))
            DIST[(i, j)] = d
            DIST[(j, i)] = d


def d(i, j):
    return 0 if i == j else DIST.get((i, j), float('inf'))


def L(i):
    return OKLAB[i][0]


def path_cost(path):
    return sum(d(path[i], path[i+1]) for i in range(len(path)-1))


def find_path_fixed_ends(n, start, end):
    """
    Находит путь от start к end через все n точек.
    Использует nearest neighbor с приоритетом роста яркости,
    но резервирует end для последнего шага.
    """
    visited = [False] * n
    path = [start]
    visited[start] = True
    visited[end] = True  # Резервируем конец

    current = start
    remaining = n - 2  # Без start и end

    for _ in range(remaining):
        # Кандидаты: непосещённые, не end
        candidates = [j for j in range(n) if not visited[j]]

        if not candidates:
            break

        # Выбираем ближайшего с бонусом за рост яркости
        current_L = L(current)
        target_L = L(end)

        def score(j):
            distance = d(current, j)
            # Бонус за движение к целевой яркости
            progress = (L(j) - current_L) / (target_L - current_L + 0.001)
            return distance - progress * 0.02

        best = min(candidates, key=score)
        visited[best] = True
        path.append(best)
        current = best

    # Добавляем конечную точку
    path.append(end)

    return path


def two_opt_fixed_ends(path):
    """2-opt с фиксированными концами."""
    n = len(path)
    improved = True
    best = path.copy()
    best_cost = path_cost(best)

    while improved:
        improved = False
        # Не трогаем первый и последний элементы
        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):  # n-1, не n, чтобы не трогать конец
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = path_cost(new_path)
                if new_cost < best_cost - 0.0001:
                    best = new_path
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return best


def sa_fixed_ends(path, iterations=100000):
    """Simulated Annealing с фиксированными концами."""
    n = len(path)
    current = path.copy()
    current_cost = path_cost(current)
    best = current.copy()
    best_cost = current_cost

    temp = 50.0
    cooling = 0.99995

    for _ in range(iterations):
        # Выбираем два индекса (не первый и не последний)
        i = random.randint(1, n - 3)
        j = random.randint(i + 1, n - 2)

        # Реверсируем сегмент
        new_path = current[:i] + current[i:j+1][::-1] + current[j+1:]
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


def or_opt(path):
    """Or-opt: перемещаем сегменты из 1-3 элементов."""
    n = len(path)
    improved = True
    best = path.copy()
    best_cost = path_cost(best)

    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(1, n - seg_len - 1):
                segment = best[i:i + seg_len]
                rest = best[:i] + best[i + seg_len:]

                for j in range(1, len(rest)):
                    new_path = rest[:j] + segment + rest[j:]
                    new_cost = path_cost(new_path)
                    if new_cost < best_cost - 0.0001:
                        best = new_path
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best


def extract_color_and_name(cell, img, col, row, cell_width, cell_height):
    """Извлекает цвет из центра ячейки."""
    width, height = cell.size
    cx, cy = width // 2, height // 2 - 10

    samples = []
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            px, py = cx + dx, cy + dy
            if 0 <= px < width and 0 <= py < height:
                samples.append(cell.getpixel((px, py)))

    if samples:
        color = tuple(sum(s[i] for s in samples) // len(samples) for i in range(3))
    else:
        color = (128, 128, 128)

    return color


# Названия цветов (порядок как на оригинале)
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
    output_image = "../output/sorted_colors_final.png"

    print("=" * 60)
    print("Black → Ivory White (все 80 цветов)")
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
            x2, y2 = x1 + cell_width, y1 + cell_height
            cell = img.crop((x1, y1, x2, y2))
            color = extract_color_and_name(cell, img, col, row, cell_width, cell_height)
            cells.append((cell, color))
            names.append(COLOR_NAMES[row * cols + col])

    print(f"Всего: {len(cells)} цветов")

    print("Вычисляю OKLab расстояния...")
    precompute(cells, names)

    # Находим Black и Ivory White
    black_idx = names.index("Black")
    ivory_idx = names.index("Ivory White")

    print(f"\n🔵 Начало: {names[black_idx]} (L={L(black_idx):.3f})")
    print(f"⚪ Конец:  {names[ivory_idx]} (L={L(ivory_idx):.3f})")

    # Строим путь
    print("\n[1/4] Nearest neighbor с фиксированными концами...")
    path = find_path_fixed_ends(len(cells), black_idx, ivory_idx)
    cost1 = path_cost(path)
    print(f"      Стоимость: {cost1:.4f}")

    print("[2/4] 2-opt оптимизация...")
    path = two_opt_fixed_ends(path)
    cost2 = path_cost(path)
    print(f"      Стоимость: {cost2:.4f} ({100*(cost1-cost2)/cost1:+.1f}%)")

    print("[3/4] Or-opt оптимизация...")
    path = or_opt(path)
    cost3 = path_cost(path)
    print(f"      Стоимость: {cost3:.4f} ({100*(cost1-cost3)/cost1:+.1f}%)")

    print("[4/4] Simulated Annealing (100K итераций)...")
    path = sa_fixed_ends(path, iterations=100000)
    cost4 = path_cost(path)
    print(f"      Стоимость: {cost4:.4f} ({100*(cost1-cost4)/cost1:+.1f}%)")

    # Проверяем
    assert path[0] == black_idx, "Путь должен начинаться с Black!"
    assert path[-1] == ivory_idx, "Путь должен заканчиваться на Ivory White!"
    assert len(path) == len(cells), "Все цвета должны быть в пути!"
    assert len(set(path)) == len(path), "Каждый цвет только один раз!"

    print(f"\n✅ Проверка пройдена:")
    print(f"   Первый: {NAMES[path[0]]}")
    print(f"   Последний: {NAMES[path[-1]]}")
    print(f"   Всего цветов: {len(path)}")

    # Собираем изображение
    print("\nСобираю изображение...")
    sorted_cells = [cells[i] for i in path]

    new_img = Image.new("RGB", (width, height))
    for idx, (cell, _) in enumerate(sorted_cells):
        row, col = divmod(idx, cols)
        x, y = col * cell_width, row * cell_height
        new_img.paste(cell, (x, y))

    new_img.save(output_image, quality=95)
    print(f"Сохранено: {output_image}")

    # Статистика
    print(f"\n{'='*60}")
    print("📊 Финальная статистика")
    print(f"{'='*60}")

    deltas = [d(path[i], path[i+1]) for i in range(len(path)-1)]
    print(f"Средний ΔE (OKLab): {sum(deltas)/len(deltas):.4f}")
    print(f"Макс ΔE: {max(deltas):.4f}")
    print(f"Мин ΔE: {min(deltas):.4f}")

    # Тренд яркости
    print(f"\nЯркость по позициям:")
    print(f"  Позиция 1:  {NAMES[path[0]]:20} L={L(path[0]):.3f}")
    print(f"  Позиция 40: {NAMES[path[39]]:20} L={L(path[39]):.3f}")
    print(f"  Позиция 80: {NAMES[path[79]]:20} L={L(path[79]):.3f}")

    # Средняя яркость по рядам
    print(f"\nСредняя яркость по рядам:")
    for row in range(rows):
        row_indices = path[row*cols : (row+1)*cols]
        avg_L = sum(L(i) for i in row_indices) / len(row_indices)
        bar = "█" * int(avg_L * 40)
        print(f"  Ряд {row+1}: L={avg_L:.3f} {bar}")


if __name__ == "__main__":
    main()
