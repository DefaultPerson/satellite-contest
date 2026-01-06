#!/usr/bin/env python3
"""
Сортировка цветов от чёрного к белому с плавными переходами.
Извлекает цвета из изображения, сортирует по HSL, создаёт PNG визуализацию.
"""

from PIL import Image, ImageDraw, ImageFont
import colorsys


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

    # Сетка: 10 колонок × 8 рядов
    cols, rows = 10, 8

    # Вычисляем шаг сетки и начальные отступы
    # Анализируя изображение: кружки расположены равномерно
    cell_width = width / cols
    cell_height = height / rows

    colors = {}
    idx = 0

    for row in range(rows):
        for col in range(cols):
            if idx >= len(COLOR_NAMES):
                break

            # Центр кружка
            cx = int(cell_width * (col + 0.5))
            cy = int(cell_height * (row + 0.5)) - 10  # Смещение вверх к центру кружка

            # Семплируем несколько пикселей для усреднения
            samples = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < width and 0 <= py < height:
                        samples.append(img.getpixel((px, py)))

            # Усредняем цвет
            if samples:
                avg_r = sum(s[0] for s in samples) // len(samples)
                avg_g = sum(s[1] for s in samples) // len(samples)
                avg_b = sum(s[2] for s in samples) // len(samples)
                colors[COLOR_NAMES[idx]] = (avg_r, avg_g, avg_b)

            idx += 1

    return colors


def get_luminance(rgb: tuple[int, int, int]) -> float:
    """Вычисляет яркость цвета (0-1)."""
    r, g, b = [x / 255.0 for x in rgb]
    # Используем формулу относительной яркости
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Конвертирует RGB в HSL."""
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h, s, l)


def sort_colors(colors: dict[str, tuple[int, int, int]]) -> list[tuple[str, tuple[int, int, int]]]:
    """
    Сортирует цвета от тёмного к светлому с группировкой по оттенку.
    Использует HSL: сначала по яркости (L), затем по оттенку (H).
    """
    def sort_key(item):
        name, rgb = item
        h, s, l = get_hsl(rgb)
        luminance = get_luminance(rgb)
        # Основной критерий - яркость, вторичный - оттенок
        return (luminance, h)

    return sorted(colors.items(), key=sort_key)


def create_visualization(
    sorted_colors: list[tuple[str, tuple[int, int, int]]],
    output_path: str,
    bar_height: int = 30,
    bar_width: int = 200,
    text_width: int = 180,
) -> None:
    """Создаёт PNG с отсортированными цветами и подписями."""
    total_height = bar_height * len(sorted_colors)
    total_width = bar_width + text_width

    img = Image.new("RGB", (total_width, total_height), (40, 40, 45))
    draw = ImageDraw.Draw(img)

    # Пытаемся загрузить шрифт
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

    for i, (name, rgb) in enumerate(sorted_colors):
        y = i * bar_height

        # Рисуем цветную полосу
        draw.rectangle([0, y, bar_width, y + bar_height], fill=rgb)

        # Определяем цвет текста (белый на тёмном, чёрный на светлом)
        luminance = get_luminance(rgb)
        text_color = (255, 255, 255) if luminance < 0.5 else (0, 0, 0)

        # Рисуем название на цветной полосе
        text_x = 10
        text_y = y + (bar_height - 14) // 2
        draw.text((text_x, text_y), name, fill=text_color, font=font)

        # RGB значение справа (на тёмном фоне)
        rgb_text = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        draw.text((bar_width + 10, text_y), rgb_text, fill=(180, 180, 180), font=font)

    img.save(output_path)
    print(f"Сохранено: {output_path}")
    print(f"Размер: {total_width}x{total_height} px")
    print(f"Цветов: {len(sorted_colors)}")


def main():
    input_image = "../input/photo_2025-12-19_12-09-33.jpg"
    output_image = "../output/sorted_colors.png"

    print(f"Извлекаю цвета из {input_image}...")
    colors = extract_colors_from_image(input_image)
    print(f"Извлечено {len(colors)} цветов")

    print("Сортирую по яркости...")
    sorted_colors = sort_colors(colors)

    print("Создаю визуализацию...")
    create_visualization(sorted_colors, output_image)

    # Выводим список для проверки
    print("\nПорядок (от тёмного к светлому):")
    for i, (name, rgb) in enumerate(sorted_colors, 1):
        lum = get_luminance(rgb)
        print(f"{i:2}. {name:20} RGB{rgb} L={lum:.3f}")


if __name__ == "__main__":
    main()
