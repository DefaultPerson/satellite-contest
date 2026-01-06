#!/usr/bin/env python3
"""
Полный градиент через все 80 цветов — один непрерывный путь
слева направо, сверху вниз.

Использует Simulated Annealing для глобальной оптимизации.
"""

from PIL import Image, ImageDraw, ImageFont
import math
import random


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """RGB -> LAB."""
    r, g, b = [x / 255.0 for x in rgb]

    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16/116)
    y = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16/116)
    z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16/116)

    L = (116 * y) - 16
    a = 500 * (x - y)
    b_val = 200 * (y - z)

    return (L, a, b_val)


def ciede2000(lab1, lab2) -> float:
    """CIEDE2000 — перцептивное расстояние."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    L_bar = (L1 + L2) / 2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            delta_h_prime = diff
        elif diff > 180:
            delta_h_prime = diff - 360
        else:
            delta_h_prime = diff + 360

    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))

    if C1_prime * C2_prime == 0:
        H_bar_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            H_bar_prime = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            H_bar_prime = (h1_prime + h2_prime + 360) / 2
        else:
            H_bar_prime = (h1_prime + h2_prime - 360) / 2

    T = (1 - 0.17 * math.cos(math.radians(H_bar_prime - 30))
         + 0.24 * math.cos(math.radians(2 * H_bar_prime))
         + 0.32 * math.cos(math.radians(3 * H_bar_prime + 6))
         - 0.20 * math.cos(math.radians(4 * H_bar_prime - 63)))

    S_L = 1 + (0.015 * (L_bar - 50)**2) / math.sqrt(20 + (L_bar - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    delta_theta = 30 * math.exp(-((H_bar_prime - 275) / 25)**2)
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))

    delta_E = math.sqrt(
        (delta_L_prime / S_L)**2 +
        (delta_C_prime / S_C)**2 +
        (delta_H_prime / S_H)**2 +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )

    return delta_E


# Предвычисляем LAB и матрицу расстояний для скорости
LAB_CACHE = {}
DIST_CACHE = {}


def precompute_distances(cells):
    """Предвычисляет все попарные расстояния."""
    global LAB_CACHE, DIST_CACHE
    n = len(cells)

    for i in range(n):
        LAB_CACHE[i] = rgb_to_lab(cells[i][1])

    for i in range(n):
        for j in range(i + 1, n):
            d = ciede2000(LAB_CACHE[i], LAB_CACHE[j])
            DIST_CACHE[(i, j)] = d
            DIST_CACHE[(j, i)] = d


def dist(i, j):
    """Быстрый доступ к расстоянию."""
    if i == j:
        return 0
    return DIST_CACHE.get((i, j), 0)


def path_cost(path):
    """Стоимость пути."""
    return sum(dist(path[i], path[i+1]) for i in range(len(path) - 1))


def get_luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


def nearest_neighbor(cells):
    """TSP nearest neighbor от самого тёмного."""
    n = len(cells)
    visited = [False] * n
    path = []

    current = min(range(n), key=lambda i: get_luminance(cells[i][1]))
    visited[current] = True
    path.append(current)

    for _ in range(n - 1):
        best = None
        best_d = float('inf')
        for j in range(n):
            if not visited[j]:
                d = dist(current, j)
                if d < best_d:
                    best_d = d
                    best = j
        if best is not None:
            visited[best] = True
            path.append(best)
            current = best

    return path


def two_opt(path):
    """2-opt локальная оптимизация."""
    n = len(path)
    improved = True
    best = path.copy()
    best_cost = path_cost(best)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n):
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = path_cost(new_path)
                if new_cost < best_cost:
                    best = new_path
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return best


def simulated_annealing(path, initial_temp=1000, cooling_rate=0.9995, min_temp=0.1):
    """Simulated Annealing для глобальной оптимизации."""
    current = path.copy()
    current_cost = path_cost(current)
    best = current.copy()
    best_cost = current_cost

    temp = initial_temp
    n = len(path)
    iterations = 0

    while temp > min_temp:
        # Случайная перестановка (2-opt move)
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)

        new_path = current[:i] + current[i:j+1][::-1] + current[j+1:]
        new_cost = path_cost(new_path)

        delta = new_cost - current_cost

        # Принимаем улучшение или с вероятностью exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_path
            current_cost = new_cost

            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost

        temp *= cooling_rate
        iterations += 1

    return best, iterations


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
        return (
            sum(s[0] for s in samples) // len(samples),
            sum(s[1] for s in samples) // len(samples),
            sum(s[2] for s in samples) // len(samples),
        )
    return (128, 128, 128)


def main():
    input_image = "../input/photo_2025-12-19_12-09-33.jpg"
    output_image = "../output/sorted_colors_full.png"

    print(f"Открываю {input_image}...")
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

    print("Предвычисляю матрицу расстояний (CIEDE2000)...")
    precompute_distances(cells)

    print("\n[1/3] TSP Nearest Neighbor...")
    path = nearest_neighbor(cells)
    cost1 = path_cost(path)
    print(f"      Стоимость: {cost1:.2f}")

    print("[2/3] 2-opt оптимизация...")
    path = two_opt(path)
    cost2 = path_cost(path)
    print(f"      Стоимость: {cost2:.2f} (-{100*(cost1-cost2)/cost1:.1f}%)")

    print("[3/3] Simulated Annealing (глобальная оптимизация)...")
    path, iters = simulated_annealing(path, initial_temp=500, cooling_rate=0.9997, min_temp=0.01)
    cost3 = path_cost(path)
    print(f"      Стоимость: {cost3:.2f} (-{100*(cost1-cost3)/cost1:.1f}% от начала)")
    print(f"      Итераций SA: {iters}")

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

    # Добавляем нумерацию для наглядности
    output_numbered = "../output/sorted_colors_full_numbered.png"
    img_numbered = new_img.copy()
    draw = ImageDraw.Draw(img_numbered)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for idx in range(len(sorted_cells)):
        row, col = divmod(idx, cols)
        x = col * cell_width + 5
        y = row * cell_height + 5
        # Номер в кружке
        draw.text((x, y), str(idx + 1), fill=(255, 255, 0), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))

    img_numbered.save(output_numbered, quality=95)
    print(f"С нумерацией: {output_numbered}")

    # Статистика
    print(f"\n📊 Статистика:")
    avg_delta = cost3 / (len(path) - 1)
    deltas = [dist(path[i], path[i+1]) for i in range(len(path) - 1)]
    max_delta = max(deltas)
    min_delta = min(deltas)

    print(f"   Средний ΔE: {avg_delta:.2f}")
    print(f"   Мин ΔE:     {min_delta:.2f}")
    print(f"   Макс ΔE:    {max_delta:.2f}")

    # Показываем самые большие скачки
    print(f"\n🔴 Самые большие переходы:")
    jumps = [(i, deltas[i]) for i in range(len(deltas))]
    jumps.sort(key=lambda x: -x[1])
    for pos, delta in jumps[:5]:
        print(f"   {pos+1}→{pos+2}: ΔE={delta:.2f}")


if __name__ == "__main__":
    main()
