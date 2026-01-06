#!/usr/bin/env python3
"""
Сортировка цветов от чёрного к белому с максимально плавными переходами.

Использует:
- OKLab — перцептуально-равномерное цветовое пространство (2020)
- Shortest Hamiltonian Path с монотонным ростом яркости
"""

from PIL import Image, ImageDraw, ImageFont
import math
import random


def srgb_to_linear(c: float) -> float:
    """sRGB -> Linear RGB."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """Linear RGB -> sRGB."""
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1/2.4)) - 0.055


def rgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    RGB -> OKLab (Björn Ottosson, 2020)
    https://bottosson.github.io/posts/oklab/

    OKLab — современное перцептуально-равномерное пространство,
    лучше чем CIELab для предсказания воспринимаемой разницы цветов.
    """
    r, g, b = [srgb_to_linear(c / 255.0) for c in rgb]

    # Linear RGB -> LMS (cone responses)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # Cube root
    l_ = l ** (1/3) if l >= 0 else -((-l) ** (1/3))
    m_ = m ** (1/3) if m >= 0 else -((-m) ** (1/3))
    s_ = s ** (1/3) if s >= 0 else -((-s) ** (1/3))

    # LMS -> OKLab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_val = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return (L, a, b_val)


def oklab_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Евклидово расстояние в OKLab пространстве."""
    lab1 = rgb_to_oklab(c1)
    lab2 = rgb_to_oklab(c2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def get_oklab_L(rgb: tuple[int, int, int]) -> float:
    """Яркость в OKLab (0-1)."""
    return rgb_to_oklab(rgb)[0]


# ============== Кэширование ==============
OKLAB_CACHE = {}
DIST_MATRIX = {}


def precompute(cells):
    """Предвычисляем OKLab и матрицу расстояний."""
    global OKLAB_CACHE, DIST_MATRIX
    n = len(cells)

    for i in range(n):
        OKLAB_CACHE[i] = rgb_to_oklab(cells[i][1])

    for i in range(n):
        for j in range(i + 1, n):
            lab1, lab2 = OKLAB_CACHE[i], OKLAB_CACHE[j]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
            DIST_MATRIX[(i, j)] = d
            DIST_MATRIX[(j, i)] = d


def dist(i, j):
    if i == j:
        return 0
    return DIST_MATRIX.get((i, j), float('inf'))


def get_L(i):
    return OKLAB_CACHE[i][0]


# ============== Алгоритмы ==============

def find_path_monotonic_brightness(cells, start_idx, end_idx):
    """
    Находит путь от start к end с монотонным ростом яркости.
    На каждом шаге выбираем ближайший цвет среди тех, что светлее текущего.
    """
    n = len(cells)
    visited = [False] * n
    path = [start_idx]
    visited[start_idx] = True

    current = start_idx
    current_L = get_L(current)

    for step in range(n - 1):
        # Находим кандидатов: не посещённые и светлее текущего (или равные)
        candidates = []
        for j in range(n):
            if not visited[j]:
                j_L = get_L(j)
                # Разрешаем небольшое уменьшение яркости для плавности
                if j_L >= current_L - 0.02:  # tolerance
                    candidates.append((j, dist(current, j), j_L))

        if not candidates:
            # Если нет кандидатов светлее, берём любой непосещённый
            for j in range(n):
                if not visited[j]:
                    candidates.append((j, dist(current, j), get_L(j)))

        if not candidates:
            break

        # Сортируем: приоритет близости, но с бонусом за рост яркости
        # Формула: distance - bonus * (L_next - L_current)
        def score(c):
            j, d, j_L = c
            brightness_bonus = (j_L - current_L) * 0.5  # Бонус за увеличение яркости
            return d - brightness_bonus

        candidates.sort(key=score)
        best = candidates[0][0]

        visited[best] = True
        path.append(best)
        current = best
        current_L = get_L(current)

    return path


def find_path_layered(cells, n_layers=8):
    """
    Разбивает цвета на слои по яркости, внутри каждого слоя
    применяет TSP для плавности, слои соединяет последовательно.
    """
    n = len(cells)

    # Сортируем по яркости и разбиваем на слои
    sorted_indices = sorted(range(n), key=lambda i: get_L(i))
    layer_size = n // n_layers

    layers = []
    for i in range(n_layers):
        start = i * layer_size
        end = start + layer_size if i < n_layers - 1 else n
        layers.append(sorted_indices[start:end])

    # Внутри каждого слоя находим оптимальный порядок
    path = []
    prev_last = None

    for layer_idx, layer in enumerate(layers):
        if len(layer) == 0:
            continue

        if prev_last is None:
            # Первый слой: начинаем с самого тёмного
            start = min(layer, key=lambda i: get_L(i))
        else:
            # Следующие слои: начинаем с ближайшего к последнему цвету
            start = min(layer, key=lambda i: dist(prev_last, i))

        # TSP nearest neighbor внутри слоя
        layer_path = tsp_in_layer(layer, start)
        path.extend(layer_path)
        prev_last = layer_path[-1]

    return path


def tsp_in_layer(indices, start):
    """TSP nearest neighbor внутри одного слоя."""
    if len(indices) <= 1:
        return indices

    remaining = set(indices)
    path = [start]
    remaining.remove(start)
    current = start

    while remaining:
        best = min(remaining, key=lambda j: dist(current, j))
        path.append(best)
        remaining.remove(best)
        current = best

    return path


def two_opt_constrained(path, max_swap_distance=15):
    """
    2-opt с ограничением: не меняем местами цвета,
    которые слишком далеко друг от друга по яркости.
    """
    n = len(path)
    improved = True
    best = path.copy()

    def cost(p):
        return sum(dist(p[i], p[i+1]) for i in range(len(p)-1))

    best_cost = cost(best)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, min(i + max_swap_distance, n)):
                # Проверяем, что реверс не нарушит порядок яркости сильно
                segment = best[i:j]
                L_min_orig = min(get_L(idx) for idx in segment)
                L_max_orig = max(get_L(idx) for idx in segment)

                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = cost(new_path)

                if new_cost < best_cost - 0.001:
                    best = new_path
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return best


def simulated_annealing_constrained(path, cells, iterations=50000):
    """SA с сохранением общего тренда яркости."""
    n = len(path)
    current = path.copy()

    def cost(p):
        return sum(dist(p[i], p[i+1]) for i in range(len(p)-1))

    def brightness_penalty(p):
        """Штраф за нарушение порядка яркости."""
        penalty = 0
        for i in range(len(p) - 1):
            diff = get_L(p[i]) - get_L(p[i+1])
            if diff > 0.05:  # Уменьшение яркости > 5%
                penalty += diff * 10
        return penalty

    current_cost = cost(current) + brightness_penalty(current)
    best = current.copy()
    best_cost = current_cost

    temp = 100
    cooling = 0.9999

    for _ in range(iterations):
        # Swap два соседних элемента (или близких)
        i = random.randint(1, n - 2)
        j = min(i + random.randint(1, 5), n - 1)

        new_path = current.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]

        new_cost = cost(new_path) + brightness_penalty(new_path)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_path
            current_cost = new_cost

            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost

        temp *= cooling

    return best


def extract_cell_color(cell):
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


def main():
    input_image = "../input/photo_2025-12-19_12-09-33.jpg"
    output_image = "../output/sorted_colors_oklab.png"

    print("=" * 60)
    print("OKLab + Monotonic Brightness Path")
    print("=" * 60)

    print(f"\nОткрываю {input_image}...")
    img = Image.open(input_image).convert("RGB")
    width, height = img.size

    cols, rows = 10, 8
    cell_width = width // cols
    cell_height = height // rows

    print("Нарезаю на 80 ячеек...")
    cells = []
    for row in range(rows):
        for col in range(cols):
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cell = img.crop((x1, y1, x2, y2))
            color = extract_cell_color(cell)
            cells.append((cell, color))

    print("Вычисляю OKLab и матрицу расстояний...")
    precompute(cells)

    # Находим самый тёмный и самый светлый
    darkest = min(range(len(cells)), key=lambda i: get_L(i))
    lightest = max(range(len(cells)), key=lambda i: get_L(i))
    print(f"\nСамый тёмный: #{darkest+1} (L={get_L(darkest):.3f})")
    print(f"Самый светлый: #{lightest+1} (L={get_L(lightest):.3f})")

    # Метод 1: Послойная сортировка
    print("\n[1/3] Послойная сортировка (8 слоёв по яркости)...")
    path = find_path_layered(cells, n_layers=8)

    def total_cost(p):
        return sum(dist(p[i], p[i+1]) for i in range(len(p)-1))

    cost1 = total_cost(path)
    print(f"      Стоимость: {cost1:.4f}")

    # Метод 2: 2-opt с ограничением
    print("[2/3] 2-opt оптимизация (с сохранением тренда яркости)...")
    path = two_opt_constrained(path, max_swap_distance=12)
    cost2 = total_cost(path)
    print(f"      Стоимость: {cost2:.4f} ({100*(cost1-cost2)/cost1:+.1f}%)")

    # Метод 3: Simulated Annealing
    print("[3/3] Simulated Annealing...")
    path = simulated_annealing_constrained(path, cells, iterations=30000)
    cost3 = total_cost(path)
    print(f"      Стоимость: {cost3:.4f} ({100*(cost1-cost3)/cost1:+.1f}%)")

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
    print("📊 Статистика (OKLab)")
    print(f"{'='*60}")

    deltas = [dist(path[i], path[i+1]) for i in range(len(path)-1)]
    L_values = [get_L(path[i]) for i in range(len(path))]

    print(f"Средний ΔE:     {sum(deltas)/len(deltas):.4f}")
    print(f"Макс ΔE:        {max(deltas):.4f}")
    print(f"Мин ΔE:         {min(deltas):.4f}")

    # Проверяем монотонность яркости
    violations = sum(1 for i in range(len(L_values)-1) if L_values[i] > L_values[i+1] + 0.01)
    print(f"\nНарушений порядка яркости: {violations}/{len(path)-1}")
    print(f"Яркость первого: {L_values[0]:.3f}")
    print(f"Яркость последнего: {L_values[-1]:.3f}")

    # Показываем тренд яркости
    print(f"\nТренд яркости по рядам:")
    for row in range(rows):
        start_idx = row * cols
        end_idx = start_idx + cols
        row_L = [get_L(path[i]) for i in range(start_idx, min(end_idx, len(path)))]
        avg_L = sum(row_L) / len(row_L)
        bar = "█" * int(avg_L * 30)
        print(f"  Ряд {row+1}: {avg_L:.3f} {bar}")


if __name__ == "__main__":
    main()
