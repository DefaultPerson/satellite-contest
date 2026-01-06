#!/usr/bin/env python3
"""
Generate gift collections as 3 separate images.
Each image contains 6 groups with larger icons and fonts.
"""

import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

# Configuration
COLLECTIONS_DIR = "../icons"
BACKGROUND_PATH = "../background.png"
OUTPUT_PREFIX = "../output/gift_groups_part"
GROUPS_JSON = "../data/collection_groups.json"

# Canvas size (match background)
CANVAS_WIDTH = 2048
CANVAS_HEIGHT = 1280

# Layout - 2 columns per image (6 groups = 2 cols x 3 rows)
NUM_COLUMNS = 2
MARGIN_X = 80
MARGIN_TOP = 260
COLUMN_GAP = 60

# Card settings (LARGER but fitting)
ICON_SIZE = 90
CARD_PADDING = 8
CARD_SIZE = ICON_SIZE + CARD_PADDING * 2
CARD_RADIUS = 12
CARD_GAP = 10
CARDS_PER_ROW_IN_GROUP = 7

# Group settings
GROUP_TITLE_HEIGHT = 50
GROUP_PADDING = 10
GROUP_GAP_Y = 20

# Border
BORDER_WIDTH = 3

# Colors
CARD_BG = (30, 25, 55, 200)
TITLE_COLOR = (255, 255, 255)
GRADIENT_START = (138, 43, 226)  # Purple
GRADIENT_END = (65, 105, 225)  # Royal Blue


def load_groups():
    """Load collection groups from JSON."""
    with open(GROUPS_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_font(size, bold=True):
    """Get Montserrat font with Cyrillic support."""
    font_paths = [
        "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Bold.otf",
        "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-SemiBold.otf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    if not bold:
        font_paths = [
            p.replace("-Bold", "-Regular").replace("-SemiBold", "-Regular")
            for p in font_paths
        ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def create_gradient_border(size, radius, border_width):
    """Create a rounded rectangle with gradient border."""
    w, h = size
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    outer_mask = Image.new("L", (w, h), 0)
    outer_draw = ImageDraw.Draw(outer_mask)
    outer_draw.rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255)

    inner_mask = Image.new("L", (w, h), 0)
    inner_draw = ImageDraw.Draw(inner_mask)
    bw = border_width
    inner_draw.rounded_rectangle(
        [bw, bw, w - 1 - bw, h - 1 - bw], max(0, radius - bw), fill=255
    )

    for y in range(h):
        ratio = y / h
        r = int(GRADIENT_START[0] * (1 - ratio) + GRADIENT_END[0] * ratio)
        g = int(GRADIENT_START[1] * (1 - ratio) + GRADIENT_END[1] * ratio)
        b = int(GRADIENT_START[2] * (1 - ratio) + GRADIENT_END[2] * ratio)
        for x in range(w):
            if outer_mask.getpixel((x, y)) > 0 and inner_mask.getpixel((x, y)) == 0:
                result.putpixel((x, y), (r, g, b, 255))

    return result


def create_card(icon_path):
    """Create a single gift card with icon."""
    card = Image.new("RGBA", (CARD_SIZE, CARD_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)

    draw.rounded_rectangle(
        [0, 0, CARD_SIZE - 1, CARD_SIZE - 1], CARD_RADIUS, fill=CARD_BG
    )

    border = create_gradient_border((CARD_SIZE, CARD_SIZE), CARD_RADIUS, BORDER_WIDTH)
    card.paste(border, (0, 0), border)

    if os.path.exists(icon_path):
        try:
            icon = Image.open(icon_path).convert("RGBA")
            icon = icon.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
            icon_x = (CARD_SIZE - ICON_SIZE) // 2
            icon_y = (CARD_SIZE - ICON_SIZE) // 2
            card.paste(icon, (icon_x, icon_y), icon)
        except Exception as e:
            print(f"Error loading {icon_path}: {e}")

    return card


def calculate_group_height(num_items):
    """Calculate height needed for a group."""
    num_rows = math.ceil(num_items / CARDS_PER_ROW_IN_GROUP)
    cards_height = num_rows * CARD_SIZE + (num_rows - 1) * CARD_GAP
    return GROUP_TITLE_HEIGHT + GROUP_PADDING + cards_height + GROUP_PADDING


def generate_image(groups_subset, bg_image, title_font, part_num):
    """Generate a single image with given groups."""
    canvas = bg_image.copy()

    column_width = (CANVAS_WIDTH - 2 * MARGIN_X - COLUMN_GAP) // NUM_COLUMNS

    # Distribute groups into 2 columns, balancing by height
    groups_list = list(groups_subset)

    # Calculate heights and distribute to balance columns
    col_heights = [0, 0]
    columns = [[], []]

    for group_key, group_info in groups_list:
        num_items = len(group_info["collections"])
        num_rows = math.ceil(num_items / CARDS_PER_ROW_IN_GROUP)
        height = GROUP_TITLE_HEIGHT + num_rows * (CARD_SIZE + CARD_GAP) + GROUP_GAP_Y

        # Add to shorter column
        shorter_col = 0 if col_heights[0] <= col_heights[1] else 1
        columns[shorter_col].append((group_key, group_info))
        col_heights[shorter_col] += height

    for col_idx in range(NUM_COLUMNS):
        col_x = MARGIN_X + col_idx * (column_width + COLUMN_GAP)
        current_y = MARGIN_TOP

        for group_key, group_info in columns[col_idx]:
            group_name = group_info["name"]
            collections = group_info["collections"]

            print(f"  Drawing: {group_name} ({len(collections)} items)")

            # Draw title
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (col_x + GROUP_PADDING, current_y),
                group_name,
                font=title_font,
                fill=TITLE_COLOR
            )

            # Draw cards
            cards_start_y = current_y + GROUP_TITLE_HEIGHT

            for i, collection in enumerate(collections):
                row = i // CARDS_PER_ROW_IN_GROUP
                col = i % CARDS_PER_ROW_IN_GROUP

                card_x = col_x + GROUP_PADDING + col * (CARD_SIZE + CARD_GAP)
                card_y = cards_start_y + row * (CARD_SIZE + CARD_GAP)

                icon_name = collection["name"] + ".png"
                icon_path = os.path.join(COLLECTIONS_DIR, icon_name)

                card = create_card(icon_path)
                canvas.paste(card, (card_x, card_y), card)

            # Move to next group
            num_rows = math.ceil(len(collections) / CARDS_PER_ROW_IN_GROUP)
            group_height = GROUP_TITLE_HEIGHT + num_rows * (CARD_SIZE + CARD_GAP) + GROUP_GAP_Y
            current_y += group_height

    output_path = f"{OUTPUT_PREFIX}_{part_num}.png"
    canvas.save(output_path, "PNG")
    print(f"  Saved: {output_path}")
    return output_path


def balance_groups_into_parts(groups_data, num_parts=3):
    """
    Distribute groups into parts with balanced total item counts.
    Uses greedy algorithm: add group to part with fewest items.
    """
    all_groups = list(groups_data["groups"].items())

    # Calculate items per group
    groups_with_counts = [
        (key, info, len(info["collections"]))
        for key, info in all_groups
    ]

    # Sort by count descending for better bin-packing
    groups_with_counts.sort(key=lambda x: -x[2])

    # Distribute using greedy algorithm
    parts = [[] for _ in range(num_parts)]
    part_totals = [0] * num_parts

    for key, info, count in groups_with_counts:
        # Add to part with smallest total
        min_part = min(range(num_parts), key=lambda i: part_totals[i])
        parts[min_part].append((key, info))
        part_totals[min_part] += count

    # Sort groups within each part by original order for visual consistency
    original_order = {k: i for i, (k, _) in enumerate(all_groups)}
    for part in parts:
        part.sort(key=lambda x: original_order.get(x[0], 999))

    return parts, part_totals


def main():
    print("Loading groups...")
    groups_data = load_groups()

    print("Loading background...")
    bg = Image.open(BACKGROUND_PATH).convert("RGBA")

    # Larger font for split version
    title_font = get_font(40, bold=True)

    # Balance groups across 3 images by item count
    parts, part_totals = balance_groups_into_parts(groups_data, 3)

    total_items = sum(part_totals)
    print(f"\nTotal items: {total_items}")
    print(f"Distribution: {part_totals} (target: ~{total_items // 3} each)")

    output_files = []
    for part_num, subset in enumerate(parts):
        group_names = [info["name"] for _, info in subset]
        item_count = sum(len(info["collections"]) for _, info in subset)

        print(f"\n=== Part {part_num + 1} ({item_count} items, {len(subset)} groups) ===")
        print(f"    Groups: {', '.join(group_names)}")

        output = generate_image(subset, bg, title_font, part_num + 1)
        output_files.append(output)

    print(f"\n✅ Generated {len(output_files)} images:")
    for f in output_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
