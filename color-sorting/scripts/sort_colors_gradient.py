#!/usr/bin/env python3
"""
Сортировка цветов для идеального градиента с помощью TSP (nearest neighbor)
в LAB цветовом пространстве для перцептивно плавных переходов.
"""

from PIL import Image
import colorsys
import math


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Конвертирует RGB в LAB цветовое пространство.
    LAB лучше отражает человеческое восприятие цвета.
    """
    # RGB -> XYZ
    r, g, b = [x / 255.0 for x in rgb]

    # sRGB gamma correction
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # Observer: 2°, Illuminant: D65
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
    b = 200 * (y - z)

    return (L, a, b)


def color_distance_lab(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Евклидово расстояние в LAB пространстве (Delta E)."""
    lab1 = rgb_to_lab(c1)
    lab2 = rgb_to_lab(c2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


def color_distance_weighted(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """
    Взвешенное расстояние: LAB + штраф за большой скачок яркости.
    Это помогает сохранить общий тренд от тёмного к светлому.
    """
    lab1 = rgb_to_lab(c1)
    lab2 = rgb_to_lab(c2)

    # Базовое расстояние в LAB
    delta_e = math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))

    return delta_e


def get_luminance(rgb: tuple[int, int, int]) -> float:
    """Яркость для определения начальной и конечной точки."""
    r, g, b = [x / 255.0 for x in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


def tsp_nearest_neighbor(cells: list, start_idx: int = None) -> list:
    """
    TSP с жадным алгоритмом nearest neighbor.
    Находит путь с минимальными перцептивными скачками между цветами.
    """
    if not cells:
        return []

    n = len(cells)
    visited = [False] * n
    path = []

    # Начинаем с самого тёмного цвета
    if start_idx is None:
        start_idx = min(range(n), key=lambda i: get_luminance(cells[i][1]))

    current = start_idx
    visited[current] = True
    path.append(cells[current])

    for _ in range(n - 1):
        # Находим ближайший непосещённый цвет
        best_next = None
        best_dist = float('inf')

        for j in range(n):
            if not visited[j]:
                dist = color_distance_lab(cells[current][1], cells[j][1])
                if dist < best_dist:
                    best_dist = dist
                    best_next = j

        if best_next is not None:
            visited[best_next] = True
            path.append(cells[best_next])
            current = best_next

    return path


def tsp_with_luminance_bias(cells: list) -> list:
    """
    TSP с учётом общего тренда яркости.
    Комбинирует плавность переходов с движением от тёмного к светлому.
    """
    if not cells:
        return []

    n = len(cells)

    # Сначала сортируем по яркости для определения "полос"
    sorted_by_lum = sorted(enumerate(cells), key=lambda x: get_luminance(x[1][1]))

    # Разбиваем на 8 групп по яркости (по рядам)
    groups = [[] for _ in range(8)]
    for i, (orig_idx, cell) in enumerate(sorted_by_lum):
        group_idx = min(i // 10, 7)  # 10 цветов на ряд
        groups[group_idx].append(cell)

    # Внутри каждой группы применяем TSP для плавности
    result = []
    prev_color = None

    for group in groups:
        if not group:
            continue

        if prev_color is None:
            # Первая группа: начинаем с самого тёмного
            start_idx = min(range(len(group)), key=lambda i: get_luminance(group[i][1]))
        else:
            # Следующие группы: начинаем с ближайшего к последнему цвету
            start_idx = min(range(len(group)),
                          key=lambda i: color_distance_lab(prev_color, group[i][1]))

        # TSP внутри группы
        group_path = tsp_nearest_neighbor(group, start_idx)
        result.extend(group_path)

        if group_path:
            prev_color = group_path[-1][1]

    return result


def extract_cell_color(cell: Image.Image) -> tuple[int, int, int]:
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
    output_tsp = "../output/sorted_colors_gradient.png"
    output_lum = "../output/sorted_colors_gradient_lum.png"

    print(f"Открываю {input_image}...")
    img = Image.open(input_image).convert("RGB")
    width, height = img.size

    cols, rows = 10, 8
    cell_width = width // cols
    cell_height = height // rows

    # Нарезаем на ячейки
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

    # === Вариант 1: Чистый TSP (максимальная плавность) ===
    print("\n[1] Чистый TSP (максимальная плавность)...")
    sorted_tsp = tsp_nearest_neighbor(cells)

    new_img = Image.new("RGB", (width, height))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            x = col * cell_width
            y = row * cell_height
            new_img.paste(sorted_tsp[idx][0], (x, y))
            idx += 1
    new_img.save(output_tsp, quality=95)

    total_dist = sum(color_distance_lab(sorted_tsp[i][1], sorted_tsp[i+1][1])
                     for i in range(len(sorted_tsp) - 1))
    avg_dist = total_dist / (len(sorted_tsp) - 1)
    print(f"    Сохранено: {output_tsp}")
    print(f"    Средний Delta E: {avg_dist:.2f}")

    # === Вариант 2: TSP с группировкой по яркости ===
    print("\n[2] TSP с группировкой по яркости (тёмное→светлое)...")
    sorted_lum = tsp_with_luminance_bias(cells)

    new_img2 = Image.new("RGB", (width, height))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            x = col * cell_width
            y = row * cell_height
            new_img2.paste(sorted_lum[idx][0], (x, y))
            idx += 1
    new_img2.save(output_lum, quality=95)

    total_dist2 = sum(color_distance_lab(sorted_lum[i][1], sorted_lum[i+1][1])
                      for i in range(len(sorted_lum) - 1))
    avg_dist2 = total_dist2 / (len(sorted_lum) - 1)
    print(f"    Сохранено: {output_lum}")
    print(f"    Средний Delta E: {avg_dist2:.2f}")

    print("\n(Меньше Delta E = плавнее переход)")


if __name__ == "__main__":
    main()
