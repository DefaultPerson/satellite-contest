#!/usr/bin/env python3
"""
Нарезает оригинальное изображение на 80 ячеек и пересобирает
в порядке от чёрного к белому.
"""

from PIL import Image
import colorsys


def get_luminance(rgb: tuple[int, int, int]) -> float:
    """Относительная яркость (0-1)."""
    r, g, b = [x / 255.0 for x in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_hue(rgb: tuple[int, int, int]) -> float:
    """Оттенок (0-1)."""
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h


def extract_cell_color(cell: Image.Image) -> tuple[int, int, int]:
    """Извлекает средний цвет из центра ячейки (где кружок)."""
    width, height = cell.size

    # Центр кружка (немного выше центра ячейки, т.к. под ним текст)
    cx, cy = width // 2, height // 2 - 10

    # Семплируем область
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
    output_image = "../output/sorted_colors_original.png"

    print(f"Открываю {input_image}...")
    img = Image.open(input_image).convert("RGB")
    width, height = img.size
    print(f"Размер: {width}x{height}")

    cols, rows = 10, 8
    cell_width = width // cols
    cell_height = height // rows
    print(f"Размер ячейки: {cell_width}x{cell_height}")

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

    # Сортируем по яркости, затем по оттенку
    print("Сортирую от тёмного к светлому...")
    sorted_cells = sorted(cells, key=lambda x: (get_luminance(x[1]), get_hue(x[1])))

    # Собираем новое изображение
    print("Собираю новое изображение...")
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
    print(f"Размер: {width}x{height} px")

    # Выводим порядок яркости
    print("\nПорядок (первые и последние 5):")
    for i in [0, 1, 2, 3, 4]:
        lum = get_luminance(sorted_cells[i][1])
        print(f"  {i+1}. L={lum:.3f}")
    print("  ...")
    for i in [75, 76, 77, 78, 79]:
        lum = get_luminance(sorted_cells[i][1])
        print(f"  {i+1}. L={lum:.3f}")


if __name__ == "__main__":
    main()
