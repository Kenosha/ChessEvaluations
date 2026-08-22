#!/usr/bin/env python3
"""Render exact archived Reddit comments as reusable, tightly cropped PNGs."""

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "assets" / "reddit-comments"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = 2
WIDTH = 660
PAD = 18
VOTE_GUTTER = 46
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

COMMENTS = [
    {
        "id": "iqdmicr",
        "author": "learnedhand91",
        "score": 2,
        "body": "Doing God’s work. I’m firmly in the Magnus camp but greatly appreciate your excellent post.",
    },
    {
        "id": "iqcupl9",
        "author": "DatChemDawg",
        "score": 35,
        "body": "It’s useful in the sense that it shows that the previous analysis with many engines was even more useless.",
    },
    {
        "id": "iqcte6x",
        "author": "AmazedCoder",
        "score": 29,
        "body": "Whatever metric or engine is used for correlation, the absolute number doesn't matter. What matters is how other players of similar caliber/age compare to him, against similar opposition. Perhaps other players of similar rating from the same tournaments.",
    },
    {
        "id": "iqcr7g7",
        "author": "throwaway_7_3_7",
        "score": 22,
        "body": "So you checked the games with an engine that wasn't even released when the games were played.\n\nSorry for losing your time, but this is useless.",
    },
    {
        "id": "iqg3gk3",
        "author": "Mothrahlurker",
        "score": 1,
        "body": "If you consistently use the same set of engines at the same depth with the same amount of CPUs that is better, but still not good.\n\nYou need to set up beforehand what is considered a suspicious outlier and what not. Else you're really setting yourself up to p-hacking anyway. Just like when people decided that #90%+ is the way to go, because 80%+ and 100% wouldn't support their conclusion, so obviously they can be ignored.",
    },
    {
        "id": "iqcscdb",
        "author": "Bakanyanter",
        "score": 18,
        "body": "Gambitman's stockfish 7 was pretty suspicious, is it possible to check what your configuration stockfish 7 says?\n\nP.S. I appreciate your work, at least you were very open about what engines you used and how you did your analysis.",
    },
    {
        "id": "iqfxgp3",
        "author": "musicnoviceoscar",
        "score": 1,
        "body": "This is so useless that it doesn't even successfully discredit Yosha's evidence, which has already been discredited. Can't get more useless than that.",
    },
    {
        "id": "iqf81cx",
        "author": "IMJorose",
        "score": 3,
        "body": "18.5 Sounds on the low end for theory, especially for Caruana. Eg. the base starting position in the Marshall Attack (so after c6) occurs after 22 half moves.",
    },
    {
        "id": "iqhogfm",
        "author": "Sure_Tradition",
        "score": 1,
        "body": "It is fine. Could you please check with Stockfish 7, at which depth it recommends 13. h3, 15. e5, 18. Rfc1, and 22. Nd6+ in the Cornette vs Niemann game. Those moves were only recommended by Stockfish 7/gambit-man. If you can not reproduce his results, something fishy are happening.",
    },
]


def logical_lines(text: str, width: int = 52) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
        else:
            lines.extend(wrap(paragraph, width=width, break_long_words=False))
    return lines


def render(comment: dict[str, object]) -> None:
    body_lines = logical_lines(str(comment["body"]))
    line_height = 30
    body_top = 54
    height = body_top + len(body_lines) * line_height + 38
    image = Image.new("RGB", (WIDTH * SCALE, height * SCALE), "#1a1a1b")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype(FONT, 22 * SCALE)
    small = ImageFont.truetype(FONT, 15 * SCALE)
    action = ImageFont.truetype(BOLD, 13 * SCALE)
    bold = ImageFont.truetype(BOLD, 18 * SCALE)

    def xy(x: int, y: int) -> tuple[int, int]:
        return x * SCALE, y * SCALE

    draw.rectangle((0, 0, WIDTH * SCALE - 1, height * SCALE - 1), outline="#343536", width=1 * SCALE)
    draw.line((*xy(VOTE_GUTTER, 10), *xy(VOTE_GUTTER, height - 10)), fill="#343536", width=1 * SCALE)

    arrow_x = 22 * SCALE
    draw.polygon(
        [(arrow_x, 14 * SCALE), (12 * SCALE, 27 * SCALE), (32 * SCALE, 27 * SCALE)],
        fill="#818384",
    )
    score = str(comment["score"])
    score_box = draw.textbbox((0, 0), score, font=bold)
    draw.text(((22 * SCALE) - (score_box[2] - score_box[0]) / 2, 35 * SCALE), score, fill="#d7dadc", font=bold)
    draw.polygon(
        [(12 * SCALE, 69 * SCALE), (32 * SCALE, 69 * SCALE), (arrow_x, 82 * SCALE)],
        fill="#818384",
    )

    x = VOTE_GUTTER + PAD
    draw.text(xy(x, 15), str(comment["author"]), fill="#4fbcff", font=bold)
    author_width = draw.textlength(str(comment["author"]), font=bold) / SCALE
    points = "point" if comment["score"] == 1 else "points"
    draw.text(
        xy(int(x + author_width + 9), 18),
        f"{comment['score']} {points} · 3 years ago",
        fill="#818384",
        font=small,
    )

    y = body_top
    for line in body_lines:
        if line:
            draw.text(xy(x, y), line, fill="#d7dadc", font=regular)
        y += line_height

    draw.text(
        xy(x, height - 25),
        "permalink   embed   save   report   reply",
        fill="#818384",
        font=action,
    )

    image.save(OUT / f"{comment['id']}.png", optimize=True)


for item in COMMENTS:
    render(item)

print(f"Rendered {len(COMMENTS)} archived comment crops to {OUT}")
