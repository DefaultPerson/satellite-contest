#!/usr/bin/env python3
"""
Generate gift collections grouped image - compact single-page version.
All 107 gifts on one 2048x1280 background.
"""

import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

# Configuration
COLLECTIONS_DIR = "../icons"
BACKGROUND_PATH = "../background.png"
OUTPUT_PATH = "../output/gift_groups.png"
GROUPS_JSON = "../data/collection_groups.json"

# Canvas size (match background)
CANVAS_WIDTH = 2048
CANVAS_HEIGHT = 1280

# Layout - 3 columns of groups
NUM_COLUMNS = 3
MARGIN_X = 40
MARGIN_TOP = 200  # Space for logo
MARGIN_BOTTOM = 15
COLUMN_GAP = 25

# Card settings (compact)
ICON_SIZE = 55
CARD_PADDING = 6
CARD_SIZE = ICON_SIZE + CARD_PADDING * 2
CARD_RADIUS = 10
CARD_GAP = 6
CARDS_PER_ROW_IN_GROUP = 7  # Per group column

# Group settings
GROUP_TITLE_HEIGHT = 32
GROUP_PADDING = 8
GROUP_GAP_Y = 10

# Border
BORDER_WIDTH = 2

# Colors
CARD_BG = (30, 25, 55, 200)
GROUP_BG = (20, 15, 45, 150)
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
            except:
                continue
    return ImageFont.load_default()


def create_gradient_border(size, radius, border_width):
    """Create a rounded rectangle with gradient border."""
    w, h = size
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Create outer rounded rect
    outer_mask = Image.new("L", (w, h), 0)
    outer_draw = ImageDraw.Draw(outer_mask)
    outer_draw.rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255)

    # Create inner rounded rect
    inner_mask = Image.new("L", (w, h), 0)
    inner_draw = ImageDraw.Draw(inner_mask)
    bw = border_width
    inner_draw.rounded_rectangle(
        [bw, bw, w - 1 - bw, h - 1 - bw], max(0, radius - bw), fill=255
    )

    # Apply gradient only to border area
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

    # Background
    draw.rounded_rectangle(
        [0, 0, CARD_SIZE - 1, CARD_SIZE - 1], CARD_RADIUS, fill=CARD_BG
    )

    # Border
    border = create_gradient_border((CARD_SIZE, CARD_SIZE), CARD_RADIUS, BORDER_WIDTH)
    card.paste(border, (0, 0), border)

    # Icon
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


def layout_groups(groups_data):
    """
    Distribute groups into 3 columns with balanced heights.
    Uses greedy algorithm: always add to shortest column.
    """
    groups = list(groups_data["groups"].items())

    # Calculate heights
    group_with_heights = [
        (k, v, calculate_group_height(len(v["collections"]))) for k, v in groups
    ]

    # Sort by height descending for better bin-packing
    group_with_heights.sort(key=lambda x: -x[2])

    # Greedy: add each group to currently shortest column
    columns = [[] for _ in range(NUM_COLUMNS)]
    column_heights = [0] * NUM_COLUMNS

    for group_key, group_info, height in group_with_heights:
        shortest = min(range(NUM_COLUMNS), key=lambda i: column_heights[i])
        columns[shortest].append((group_key, group_info, height))
        column_heights[shortest] += height + GROUP_GAP_Y

    # Sort groups within each column by original order for visual consistency
    original_order = {k: i for i, (k, _) in enumerate(groups)}
    for col in columns:
        col.sort(key=lambda x: original_order.get(x[0], 999))

    # Calculate final y positions
    result = []
    for col_groups in columns:
        y = 0
        col_result = []
        for group_key, group_info, height in col_groups:
            col_result.append((group_key, group_info, y))
            y += height + GROUP_GAP_Y
        result.append(col_result)

    return result


def main():
    print("Loading groups...")
    groups_data = load_groups()

    print("Loading background...")
    bg = Image.open(BACKGROUND_PATH).convert("RGBA")
    canvas = bg.copy()

    # Get fonts
    title_font = get_font(26, bold=True)

    # Calculate column width
    available_width = CANVAS_WIDTH - 2 * MARGIN_X - (NUM_COLUMNS - 1) * COLUMN_GAP
    column_width = available_width // NUM_COLUMNS

    # Layout groups into columns
    columns = layout_groups(groups_data)

    # Draw each column
    for col_idx, column in enumerate(columns):
        col_x = MARGIN_X + col_idx * (column_width + COLUMN_GAP)

        for group_key, group_info, y_offset in column:
            current_y = MARGIN_TOP + y_offset
            group_name = group_info["name"]
            collections = group_info["collections"]

            print(f"Drawing: {group_name} ({len(collections)} items)")

            # Calculate group dimensions
            num_rows = math.ceil(len(collections) / CARDS_PER_ROW_IN_GROUP)
            cards_height = num_rows * CARD_SIZE + (num_rows - 1) * CARD_GAP
            group_height = (
                GROUP_TITLE_HEIGHT + GROUP_PADDING + cards_height + GROUP_PADDING
            )

            # Draw title (no group background)
            draw = ImageDraw.Draw(canvas)
            title_x = col_x + GROUP_PADDING
            title_y = current_y + 6
            draw.text((title_x, title_y), group_name, font=title_font, fill=TITLE_COLOR)

            # Draw cards
            cards_start_y = current_y + GROUP_TITLE_HEIGHT + GROUP_PADDING

            for i, collection in enumerate(collections):
                row = i // CARDS_PER_ROW_IN_GROUP
                col = i % CARDS_PER_ROW_IN_GROUP

                card_x = col_x + GROUP_PADDING + col * (CARD_SIZE + CARD_GAP)
                card_y = cards_start_y + row * (CARD_SIZE + CARD_GAP)

                # Load icon
                icon_name = collection["name"] + ".png"
                icon_path = os.path.join(COLLECTIONS_DIR, icon_name)

                card = create_card(icon_path)
                canvas.paste(card, (card_x, card_y), card)

    print(f"Saving to {OUTPUT_PATH}...")
    canvas.save(OUTPUT_PATH, "PNG")
    print("Done!")
    print(f"Canvas size: {canvas.size}")


if __name__ == "__main__":
    main()
