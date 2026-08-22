#!/usr/bin/env python3
"""Build tidy CSVs from the recovered 2022 spreadsheet, images, and comments."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RECOVERED = ROOT / "recovered_2022"
TABLES = RECOVERED / "tables"
XLSX = RECOVERED / "gambitman" / "gambitman_analysis.xlsx"
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def column_number(cell: str) -> int:
    letters = re.match(r"[A-Z]+", cell).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value


def read_gambitman() -> list[dict[str, object]]:
    with ZipFile(XLSX) as archive:
        ns = {"m": SHEET_NS}
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t"))
            for item in shared_root.findall("m:si", ns)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: dict[int, dict[int, object]] = {}
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            row_number = int(row.attrib["r"])
            cells: dict[int, object] = {}
            for cell in row.findall("m:c", ns):
                value_node = cell.find("m:v", ns)
                if value_node is None:
                    continue
                value: object = value_node.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                cells[column_number(cell.attrib["r"])] = value
            rows[row_number] = cells

        rel_root = ET.fromstring(
            archive.read("xl/worksheets/_rels/sheet1.xml.rels")
        )
        relationships = {
            node.attrib["Id"]: node.attrib.get("Target", "") for node in rel_root
        }
        links: dict[str, str] = {}
        for node in sheet.findall(".//m:hyperlinks/m:hyperlink", ns):
            relation = node.attrib.get(f"{{{REL_NS}}}id")
            links[node.attrib["ref"]] = relationships.get(relation, "")

    output: list[dict[str, object]] = []
    for row_number in range(2, 55):
        cells = rows.get(row_number, {})
        tournament = str(cells.get(1, ""))
        if not tournament:
            continue
        for round_number in range(1, 25):
            raw = cells.get(round_number + 4, "")
            if raw == "":
                continue
            try:
                percent = round(float(raw))
            except (TypeError, ValueError):
                continue
            output.append(
                {
                    "dataset": "lets_check",
                    "tournament": tournament,
                    "round": round_number,
                    "percent": percent,
                    "games_link": links.get(f"A{row_number}", ""),
                    "tournament_link": links.get(f"C{row_number}", ""),
                }
            )
    return output


SF15_NIEMANN = {
    "National Open 2021": [74, 83, 75, 72, 78, 50, 59, 59, 38],
    "14th Philadelphia Int": {1: 80, 2: 66, 3: 54, 4: 75, 5: 63, 6: 64, 7: 59, 9: 82},
    "49th Annual World Open": {1: 79, 2: 79, 3: 70, 4: 88, 5: 64, 7: 82, 8: 57, 9: 56, 10: 59},
    "ch-USA Junior 2021": [77, 82, 83, 68, 78, 87, 84, 70, 72],
    "121st US Open 2021": {4: 80, 5: 81, 7: 57, 8: 77, 9: 69},
    "2nd Tras-os-Montes Open": {2: 56, 4: 86, 5: 70, 7: 91, 8: 75, 9: 78},
}

SF7_NIEMANN = {
    "National Open 2021": [78, 70, 70, 65, 64, 45, 41, 55, 33],
    "14th Philadelphia Int": {1: 76, 2: 56, 3: 58, 4: 75, 5: 78, 6: 64, 7: 77, 9: 82},
    "49th Annual World Open": {1: 71, 2: 55, 3: 55, 4: 63, 5: 66, 7: 82, 8: 68, 9: 56, 10: 39},
    "ch-USA Junior 2021": [77, 72, 83, 73, 74, 83, 75, 55, 64],
    "121st US Open 2021": {4: 85, 5: 68, 7: 52, 8: 86, 9: 72},
    "2nd Tras-os-Montes Open": {2: 61, 4: 80, 5: 74, 7: 65, 8: 74, 9: 69},
}

SF15_CARUANA = {
    2014: [89, 94, 75, 84, 83, 79, 79, 58, 76, 57],
    2015: [68, 62, 81, 69, 76, 64, 45, 68, 51],
    2016: [62, 59, 55, 65, 69, 69, 73, 87, 72],
    2017: [82, 69, 65, 58, 54, 57, 38, 59, 74],
    2018: [55, 62, 70, 72, 79, 64, 71, 67, 70, 88, 59],
    2019: [67, 64, 72, 79, 94, 87, 79, 69, 74, 88, 59],
    2021: [79, 76, 79, 60, 50, 84, 76, 74, 66],
}

DEPTH_RANKS = {
    10: (3, 1, 4, 5), 11: (3, 1, 3, 2), 12: (1, 1, 3, 1),
    13: (2, 1, 4, 2), 14: (4, 1, 6, 3), 15: (4, 1, 5, 3),
    16: (3, 2, 3, 4), 17: (2, 1, 2, 2), 18: (5, 3, 2, 1),
    19: (4, 1, 1, 2), 20: (2, 1, 5, 2), 21: (4, 1, 3, 2),
    22: (2, 1, 2, 2), 23: (1, 1, 2, 3), 24: (1, 1, 2, 2),
    25: (1, 1, 4, 2),
}


def normalized_rows(dataset: str, values: dict) -> list[dict[str, object]]:
    rows = []
    for tournament, rounds in values.items():
        if isinstance(rounds, list):
            rounds = dict(enumerate(rounds, start=1))
        for game_round, percent in rounds.items():
            rows.append(
                {
                    "dataset": dataset,
                    "tournament": tournament,
                    "round": game_round,
                    "percent": percent,
                }
            )
    return rows


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = TABLES / name
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path.relative_to(ROOT))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    lets_check = read_gambitman()
    write_csv("gambitman_lets_check_all.csv", lets_check)
    selected = [
        row
        for row in lets_check
        if row["tournament"]
        in {
            "National Open 2021",
            "14th Philadelphia Int",
            "49th Annual World Open",
            "ch-USA Junior 2021",
            "121st US Open 2021",
            "2nd Tras-os-Montes Open",
        }
    ]
    published = [
        {key: row[key] for key in ("dataset", "tournament", "round", "percent")}
        for row in selected
    ]
    published += normalized_rows("stockfish_15", SF15_NIEMANN)
    published += normalized_rows("stockfish_7", SF7_NIEMANN)
    write_csv("niemann_published_correlations.csv", published)

    triplet_map: dict[tuple[str, object], dict[str, object]] = {}
    for row in published:
        key = (str(row["tournament"]), row["round"])
        triplet_map.setdefault(
            key, {"tournament": row["tournament"], "round": row["round"]}
        )[str(row["dataset"])] = row["percent"]
    triplets = [
        row
        for _, row in sorted(triplet_map.items())
        if all(name in row for name in ("lets_check", "stockfish_15", "stockfish_7"))
    ]
    write_csv("niemann_published_triplets.csv", triplets)

    caruana = []
    for year, percentages in SF15_CARUANA.items():
        for game_round, percent in enumerate(percentages, start=1):
            caruana.append({"year": year, "round": game_round, "percent": percent})
    write_csv("caruana_stockfish_15.csv", caruana)

    moves = ((24, "13.h3"), (28, "15.e5"), (34, "18.Rfc1"), (42, "22.Nd6+"))
    depth_rows = []
    for depth, ranks in DEPTH_RANKS.items():
        for (ply, move), rank in zip(moves, ranks):
            depth_rows.append({"depth": depth, "ply": ply, "move": move, "rank": rank})
    write_csv("cornette_niemann_sf7_depth_ranks.csv", depth_rows)


if __name__ == "__main__":
    main()
