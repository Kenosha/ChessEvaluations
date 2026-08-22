#!/usr/bin/env python3
"""Refresh the reaction and depth fallback posters from the live exhibit data."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web" / "assets" / "reddit-comments"
POSTERS = ROOT / "web" / "posters"
W, H = 1180, 760
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

COLORS = {
    "night": "#080b0b",
    "panel": "#0d1212",
    "line": "#33403d",
    "line_strong": "#65736e",
    "amber": "#f2b84b",
    "cyan": "#67d5d0",
    "paper": "#e5dfcf",
    "muted": "#87918d",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(MONO_BOLD if bold else MONO, size)


def base(page: str, page_no: str, caption: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), COLORS["night"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, W - 17, H - 17), outline=COLORS["line_strong"], width=1)
    draw.line((16, 51, W - 17, 51), fill=COLORS["line"])
    draw.text((31, 29), "●  CHESS EVALUATIONS   CE / 2022", fill=COLORS["cyan"], font=font(8, True))
    draw.text((995, 29), f"ACTIVE MODULE  {page}", fill=COLORS["amber"], font=font(8, True))
    draw.line((16, 658, W - 17, 658), fill=COLORS["line_strong"])
    draw.text((82, 685), f"{page_no} / {page}", fill=COLORS["cyan"], font=font(9, True))
    draw.text((82, 706), caption, fill=COLORS["muted"], font=font(9))
    draw.text((1056, 695), f"{page_no} / 04", fill=COLORS["paper"], font=font(9))
    return image, draw


def paste_crop(
    canvas: Image.Image,
    comment_id: str,
    width: int,
    x: int,
    y: int,
    angle: float,
) -> None:
    crop = Image.open(ASSETS / f"{comment_id}.png").convert("RGBA")
    height = round(crop.height * width / crop.width)
    crop = crop.resize((width, height), Image.Resampling.LANCZOS)
    crop = crop.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
    shadow = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    alpha = crop.getchannel("A")
    shadow.putalpha(alpha.point(lambda value: min(105, value)))
    canvas.paste(shadow, (x + 8, y + 10), shadow)
    canvas.paste(crop, (x, y), crop)


def reaction() -> None:
    image, draw = base(
        "REACTION",
        "03",
        "The public review, preserved as archived comment captures.",
    )
    draw.rectangle((35, 65, 1144, 642), fill="#141715", outline=COLORS["line"])
    paste_crop(image, "iqdmicr", 330, 35, 78, 3.1)
    paste_crop(image, "iqcupl9", 330, 823, 103, -2.2)
    paste_crop(image, "iqcte6x", 430, 45, 252, -1.2)
    paste_crop(image, "iqcr7g7", 380, 756, 265, 2.3)
    paste_crop(image, "iqg3gk3", 430, 350, 386, -1.4)
    paste_crop(image, "iqcscdb", 350, 37, 475, 2.1)
    paste_crop(image, "iqfxgp3", 360, 794, 485, -2.6)
    paste_crop(image, "iqf81cx", 310, 520, 516, 3.0)
    paste_crop(image, "iqhogfm", 465, 405, 75, 0.55)
    draw.rectangle((43, 591, 302, 632), fill="#080b0b", outline="#7f642d")
    draw.text((55, 599), "89  ARCHIVED THREAD", fill=COLORS["amber"], font=font(12, True))
    draw.text((55, 617), "comments · 52 score · 82% upvoted", fill=COLORS["paper"], font=font(8))
    image.save(POSTERS / "reaction.png", optimize=True)


def depth() -> None:
    image, draw = base(
        "DEPTH",
        "04",
        "The initiating comment remains; the Stockfish 7 rank table appears beneath it.",
    )
    draw.rectangle((35, 65, 1144, 642), fill=COLORS["panel"], outline=COLORS["line"])
    draw.rectangle((45, 76, 149, 105), fill=COLORS["cyan"])
    draw.text((58, 86), "STOCKFISH 7", fill=COLORS["night"], font=font(9, True))
    draw.text((47, 125), "GAME", fill=COLORS["muted"], font=font(7, True))
    draw.text((47, 141), "MATTHIEU CORNETTE — HANS NIEMANN · 2020", fill=COLORS["paper"], font=font(8, True))
    paste_crop(image, "iqhogfm", 465, 410, 72, 0.55)

    x0, y0, x1, y1 = 45, 280, 1135, 573
    label_w = 148
    row_h = (y1 - y0) / 5
    col_w = (x1 - x0 - label_w) / 6
    draw.rectangle((x0, y0, x1, y1), outline=COLORS["line_strong"])
    for row in range(1, 5):
        y = round(y0 + row_h * row)
        draw.line((x0, y, x1, y), fill=COLORS["line"])
    draw.line((x0 + label_w, y0, x0 + label_w, y1), fill=COLORS["line_strong"])
    for col in range(1, 6):
        x = round(x0 + label_w + col_w * col)
        draw.line((x, y0, x, y1), fill=COLORS["line"])

    headers = ["PLAYED MOVE", "BEST", "2ND", "3RD", "4TH", "5TH", "6TH"]
    centers = [x0 + label_w / 2] + [x0 + label_w + col_w * (i + 0.5) for i in range(6)]
    for label, center in zip(headers, centers):
        box = draw.textbbox((0, 0), label, font=font(9, True))
        draw.text((center - (box[2] - box[0]) / 2, y0 + 25), label, fill=COLORS["cyan"] if label == "BEST" else COLORS["paper"], font=font(9, True))

    moves = [("MOVE 13", "h3", 2), ("MOVE 15", "e5", 1), ("MOVE 18", "Rfc1", 2), ("MOVE 22", "Nd6+", 2)]
    for index, (move_no, move, rank) in enumerate(moves):
        cy = y0 + row_h * (index + 1.5)
        draw.text((80, cy - 5), move_no, fill=COLORS["cyan"], font=font(9, True))
        cx = x0 + label_w + col_w * (rank - 0.5)
        fill = COLORS["cyan"] if rank == 1 else COLORS["amber"]
        draw.rectangle((cx - 41, cy - 19, cx + 41, cy + 19), fill=fill)
        box = draw.textbbox((0, 0), move, font=font(11, True))
        draw.text((cx - (box[2] - box[0]) / 2, cy - 7), move, fill=COLORS["night"], font=font(11, True))

    draw.rectangle((35, 584, 1144, 642), fill="#181d1b", outline=COLORS["line_strong"])
    draw.text((54, 603), "SEARCH DEPTH  17", fill=COLORS["paper"], font=font(10, True))
    draw.line((235, 614, 1067, 614), fill=COLORS["line_strong"], width=4)
    draw.line((235, 614, 623, 614), fill=COLORS["amber"], width=4)
    draw.rectangle((618, 603, 628, 625), fill=COLORS["amber"])
    image.save(POSTERS / "depth.png", optimize=True)


reaction()
depth()
print("Updated reaction.png and depth.png")
