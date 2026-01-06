#!/usr/bin/env python3
"""
Создаёт изображение с отсортированными цветами в формате исходной картинки:
сетка 10×8 кружков с названиями под ними, от чёрного к белому.
"""

from PIL import Image, ImageDraw, ImageFont
import colorsys
import math


# Названия цветов из изображения (8 рядов × 10 колонок = 80 цветов)
COLOR_NAMES = [
    # Row 1
    "Black", "Electric Purple", "Lavender", "Cyberpunk", "Electric Indigo",
    "Neon Blue", "Navy Blue", "Sapphire", "Sky Blue", "Azure Blue",
    # Row 2
    "Pacific Cyan", "Aquamarine", "Pacific Green", "Emerald", "Mint Green",
    "Malachite", "Shamrock Green", "Lemongrass", "Light Olive", "Satin Gold",
    # Row 3
    "Pure Gold", "Amber", "Caramel", "Orange", "Carrot Juice",
    "Coral Red", "Persimmon", "Strawberry", "Raspberry", "Mystic Pearl",
    # Row 4
    "Fandango", "Dark Lilac", "English Violet", "Moonstone", "Pine Green",
    "Hunter Green", "Pistachio", "Khaki Green", "Desert Sand", "Cappuccino",
    # Row 5
    "Rosewood", "Ivory White", "Platinum", "Roman Silver", "Steel Grey",
    "Silver Blue", "Burgundy", "Indigo Dye", "Midnight Blue", "Onyx Black",
    # Row 6
    "Battleship Grey", "Purple", "Grape", "Cobalt Blue", "French Blue",
    "Turquoise", "Jade Green", "Copper", "Chestnut", "Chocolate",
    # Row 7
    "Marine Blue", "Tactical Pine", "Gunship Green", "Dark Green", "Seal Brown",
    "Rifle Green", "Ranger Green", "Camo Green", "Feldgrau", "Gunmetal",
    # Row 8
    "Deep Cyan", "Mexican Pink", "Tomato", "Fire Engine", "Celtic Blue",
    "Old Gold", "Burnt Sienna", "Carmine", "Mustard", "French Violet",
]


def extract_colors_from_image(image_path: str) -> dict[str, tuple[int, int, int]]:
    """Извлекает RGB цвета из центров кружков на изображении."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    cols, rows = 10, 8
    cell_width = width / cols
    cell_height = height / rows

    colors = {}
    idx = 0

    for row in range(rows):
        for col in range(cols):
            if idx >= len(COLOR_NAMES):
                break

            # Центр кружка (смещение вверх к центру круга, т.к. под ним текст)
            cx = int(cell_width * (col + 0.5))
            cy = int(cell_height * (row + 0.5)) - 10

            # Семплируем область для усреднения
            samples = []
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < width and 0 <= py < height:
                        samples.append(img.getpixel((px, py)))

            if samples:
                avg_r = sum(s[0] for s in samples) // len(samples)
                avg_g = sum(s[1] for s in samples) // len(samples)
                avg_b = sum(s[2] for s in samples) // len(samples)
                colors[COLOR_NAMES[idx]] = (avg_r, avg_g, avg_b)

            idx += 1

    return colors


def get_luminance(rgb: tuple[int, int, int]) -> float:
    """Относительная яркость (0-1)."""
    r, g, b = [x / 255.0 for x in rgb]
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """RGB -> HSL."""
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h, s, l)


def sort_colors(colors: dict[str, tuple[int, int, int]]) -> list[tuple[str, tuple[int, int, int]]]:
    """Сортировка от тёмного к светлому с группировкой по оттенку."""
    def sort_key(item):
        name, rgb = item
        h, s, l = get_hsl(rgb)
        luminance = get_luminance(rgb)
        return (luminance, h)

    return sorted(colors.items(), key=sort_key)


def draw_gradient_circle(draw, cx, cy, radius, base_color):
    """Рисует кружок с градиентом (имитация 3D эффекта как на оригинале)."""
    r, g, b = base_color

    # Рисуем круг слоями от внешнего к внутреннему
    for i in range(radius, 0, -1):
        # Коэффициент от 0 (край) до 1 (центр)
        t = 1 - (i / radius)
        # Осветляем к центру для 3D эффекта
        factor = 1 + t * 0.3
        cr = min(255, int(r * factor))
        cg = min(255, int(g * factor))
        cb = min(255, int(b * factor))

        draw.ellipse(
            [cx - i, cy - i, cx + i, cy + i],
            fill=(cr, cg, cb)
        )


def create_grid_visualization(
    sorted_colors: list[tuple[str, tuple[int, int, int]]],
    output_path: str,
    cols: int = 10,
    rows: int = 8,
) -> None:
    """Создаёт изображение в формате сетки с кружками и подписями."""

    # Размеры ячейки
    cell_width = 130
    cell_height = 100
    circle_radius = 35

    # Общие размеры
    width = cell_width * cols
    height = cell_height * rows

    # Тёмный фон как на оригинале
    bg_color = (40, 42, 45)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Загружаем шрифт
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 11)
        except OSError:
            font = ImageFont.load_default()

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= len(sorted_colors):
                break

            name, rgb = sorted_colors[idx]

            # Центр ячейки
            cx = int(cell_width * (col + 0.5))
            cy = int(cell_height * (row + 0.5)) - 10  # Смещение вверх для текста

            # Рисуем кружок с градиентом
            draw_gradient_circle(draw, cx, cy, circle_radius, rgb)

            # Подпись под кружком
            text_y = cy + circle_radius + 5

            # Центрируем текст
            bbox = draw.textbbox((0, 0), name, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = cx - text_width // 2

            # Белый текст
            draw.text((text_x, text_y), name, fill=(255, 255, 255), font=font)

            idx += 1

    img.save(output_path, quality=95)
    print(f"Сохранено: {output_path}")
    print(f"Размер: {width}x{height} px")


def main():
    input_image = "../input/photo_2025-12-19_12-09-33.jpg"
    output_image = "../output/sorted_colors_grid.png"

    print(f"Извлекаю цвета из {input_image}...")
    colors = extract_colors_from_image(input_image)
    print(f"Извлечено {len(colors)} цветов")

    print("Сортирую от чёрного к белому...")
    sorted_colors = sort_colors(colors)

    print("Создаю сетку...")
    create_grid_visualization(sorted_colors, output_image)

    # Выводим порядок
    print("\nПорядок цветов (слева направо, сверху вниз):")
    for i, (name, rgb) in enumerate(sorted_colors, 1):
        lum = get_luminance(rgb)
        row = (i - 1) // 10 + 1
        col = (i - 1) % 10 + 1
        print(f"{i:2}. [{row},{col}] {name:20} L={lum:.3f}")


if __name__ == "__main__":
    main()
