#!/usr/bin/env python3
"""
Сортировка цветов для максимально плавного перехода
с точки зрения человеческого восприятия.

Использует:
- CIEDE2000 — стандарт CIE для перцептивного различия цветов
- TSP nearest neighbor + 2-opt оптимизация
"""

from PIL import Image
import math


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """RGB -> LAB (CIE L*a*b*)."""
    r, g, b = [x / 255.0 for x in rgb]

    # sRGB -> linear RGB
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # linear RGB -> XYZ (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ -> LAB
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


def ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """
    CIEDE2000 — стандарт CIE для измерения перцептивного различия цветов.
    Учитывает особенности человеческого зрения.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Средние значения
    L_bar = (L1 + L2) / 2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    # G factor
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))

    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2

    # Hue angles
    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    # Delta values
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Delta h'
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

    # H_bar_prime
    if C1_prime * C2_prime == 0:
        H_bar_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            H_bar_prime = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            H_bar_prime = (h1_prime + h2_prime + 360) / 2
        else:
            H_bar_prime = (h1_prime + h2_prime - 360) / 2

    T = (1
         - 0.17 * math.cos(math.radians(H_bar_prime - 30))
         + 0.24 * math.cos(math.radians(2 * H_bar_prime))
         + 0.32 * math.cos(math.radians(3 * H_bar_prime + 6))
         - 0.20 * math.cos(math.radians(4 * H_bar_prime - 63)))

    # Weighting functions
    S_L = 1 + (0.015 * (L_bar - 50)**2) / math.sqrt(20 + (L_bar - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    # Rotation function
    delta_theta = 30 * math.exp(-((H_bar_prime - 275) / 25)**2)
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))

    # Final calculation (kL = kC = kH = 1 for standard conditions)
    delta_E = math.sqrt(
        (delta_L_prime / S_L)**2 +
        (delta_C_prime / S_C)**2 +
        (delta_H_prime / S_H)**2 +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )

    return delta_E


def perceptual_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Перцептивное расстояние между двумя RGB цветами (CIEDE2000)."""
    return ciede2000(rgb_to_lab(c1), rgb_to_lab(c2))


def get_luminance(rgb: tuple[int, int, int]) -> float:
    """Яркость для определения начальной точки."""
    r, g, b = [x / 255.0 for x in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


def tsp_nearest_neighbor(cells: list) -> list:
    """TSP nearest neighbor, начиная с самого тёмного цвета."""
    if not cells:
        return []

    n = len(cells)
    visited = [False] * n
    path_indices = []

    # Начинаем с самого тёмного
    current = min(range(n), key=lambda i: get_luminance(cells[i][1]))
    visited[current] = True
    path_indices.append(current)

    for _ in range(n - 1):
        best_next = None
        best_dist = float('inf')

        for j in range(n):
            if not visited[j]:
                dist = perceptual_distance(cells[current][1], cells[j][1])
                if dist < best_dist:
                    best_dist = dist
                    best_next = j

        if best_next is not None:
            visited[best_next] = True
            path_indices.append(best_next)
            current = best_next

    return path_indices


def calculate_path_cost(cells: list, path: list) -> float:
    """Общая перцептивная стоимость пути."""
    total = 0
    for i in range(len(path) - 1):
        total += perceptual_distance(cells[path[i]][1], cells[path[i+1]][1])
    return total


def two_opt(cells: list, path: list, max_iterations: int = 1000) -> list:
    """
    2-opt локальная оптимизация TSP.
    Итеративно меняет местами рёбра для уменьшения общей длины пути.
    """
    n = len(path)
    improved = True
    iteration = 0

    best_path = path.copy()
    best_cost = calculate_path_cost(cells, best_path)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue

                # Пробуем реверсировать сегмент [i:j]
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                new_cost = calculate_path_cost(cells, new_path)

                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
                    improved = True

    return best_path


def three_opt_segment(cells: list, path: list) -> list:
    """
    Упрощённая 3-opt оптимизация для дополнительного улучшения.
    """
    n = len(path)
    best_path = path.copy()
    best_cost = calculate_path_cost(cells, best_path)

    for i in range(1, n - 4):
        for j in range(i + 2, n - 2):
            for k in range(j + 2, n):
                # Пробуем различные комбинации реверсирования
                segments = [
                    best_path[:i] + best_path[i:j][::-1] + best_path[j:k][::-1] + best_path[k:],
                    best_path[:i] + best_path[j:k] + best_path[i:j] + best_path[k:],
                ]

                for new_path in segments:
                    new_cost = calculate_path_cost(cells, new_path)
                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost

    return best_path


def extract_cell_color(cell) -> tuple[int, int, int]:
    """Извлекает средний цвет из центра ячейки."""
    width, height = cell.size
    cx, cy = width // 2, height // 2 - 10

    samples = []
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            px, py = cx + dx, cy + dy
            if 0 <= px < width and 0 <= py < height:
                samples.append(cell.getpixel((px, py)))

    if samples:
        avg_r = sum(s[0] for s in samples) // len(samples)
        avg_g = sum(s[1] for s in samples) // len(samples)
        avg_b = sum(s[2] for s in samples) // len(samples)
        return (avg_r, avg_g, avg_b)

    return (128, 128, 128)


def main():
    input_image = "../input/photo_2025-12-19_12-09-33.jpg"
    output_image = "../output/sorted_colors_human.png"

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
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            cell = img.crop((x1, y1, x2, y2))
            color = extract_cell_color(cell)
            cells.append((cell, color))

    print(f"Извлечено {len(cells)} ячеек")

    # Шаг 1: TSP nearest neighbor
    print("\n[1/3] TSP nearest neighbor (CIEDE2000)...")
    path = tsp_nearest_neighbor(cells)
    cost1 = calculate_path_cost(cells, path)
    print(f"      Начальная стоимость: {cost1:.2f}")

    # Шаг 2: 2-opt оптимизация
    print("[2/3] 2-opt оптимизация...")
    path = two_opt(cells, path, max_iterations=500)
    cost2 = calculate_path_cost(cells, path)
    print(f"      После 2-opt: {cost2:.2f} ({100*(cost1-cost2)/cost1:.1f}% улучшение)")

    # Шаг 3: 3-opt для финальной полировки
    print("[3/3] 3-opt финальная оптимизация...")
    path = three_opt_segment(cells, path)
    cost3 = calculate_path_cost(cells, path)
    print(f"      Финальная стоимость: {cost3:.2f} ({100*(cost1-cost3)/cost1:.1f}% общее улучшение)")

    # Собираем результат
    print("\nСобираю изображение...")
    sorted_cells = [cells[i] for i in path]

    new_img = Image.new("RGB", (width, height))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            x = col * cell_width
            y = row * cell_height
            new_img.paste(sorted_cells[idx][0], (x, y))
            idx += 1

    new_img.save(output_image, quality=95)
    print(f"\nСохранено: {output_image}")

    # Статистика
    avg_delta = cost3 / (len(path) - 1)
    max_delta = max(perceptual_distance(sorted_cells[i][1], sorted_cells[i+1][1])
                    for i in range(len(sorted_cells) - 1))

    print(f"\n📊 Статистика (CIEDE2000):")
    print(f"   Средний ΔE: {avg_delta:.2f}")
    print(f"   Макс ΔE:    {max_delta:.2f}")
    print(f"   (ΔE < 1: незаметно, < 2: едва заметно, < 3.5: заметно при сравнении)")


if __name__ == "__main__":
    main()
