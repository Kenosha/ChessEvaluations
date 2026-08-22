#!/usr/bin/env python3
"""Re-run the recovered Niemann–Cornette Stockfish 7 depth experiment."""

from __future__ import annotations

import asyncio
import argparse
import csv
from pathlib import Path

import chess
import chess.engine
import chess.pgn


ROOT = Path(__file__).resolve().parents[1]
ENGINE = (
    ROOT
    / "recovered_2022"
    / "external_inputs"
    / "Stockfish-sf_7"
    / "src"
    / "stockfish"
)
PGN = ROOT / "recovered_2022" / "depth_experiment" / "niemann_cornette_2020.pgn"
PUBLISHED = ROOT / "recovered_2022" / "tables" / "cornette_niemann_sf7_depth_ranks.csv"
OUTPUT = ROOT / "recovered_2022" / "depth_experiment" / "sf7_depth_ranks_reproduced.csv"
TARGET_PLIES = (24, 28, 34, 42)


def positions() -> dict[int, tuple[chess.Board, chess.Move]]:
    with PGN.open() as source:
        game = chess.pgn.read_game(source)
    board = game.board()
    result = {}
    for move in game.mainline_moves():
        if board.ply() in TARGET_PLIES:
            result[board.ply()] = (board.copy(), move)
        board.push(move)
    return result


def published_ranks() -> dict[tuple[int, int], int]:
    with PUBLISHED.open() as source:
        return {
            (int(row["depth"]), int(row["ply"])): int(row["rank"])
            for row in csv.DictReader(source)
        }


def display_score(score: chess.engine.PovScore, turn: chess.Color) -> str:
    relative = score.pov(turn)
    if relative.is_mate():
        return f"M{relative.mate()}"
    return f"{relative.score() / 100:+.2f}"


def write_rows(rows: list[dict[str, object]]) -> None:
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-depth", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument(
        "--keep-hash",
        action="store_true",
        help="Do not clear Stockfish's transposition table between positions",
    )
    args = parser.parse_args()
    expected = published_ranks()
    target_positions = positions()
    _, engine = await chess.engine.popen_uci(str(ENGINE))
    rows = []
    try:
        for depth in range(args.min_depth, args.max_depth + 1):
            for ply in TARGET_PLIES:
                board, played_move = target_positions[ply]
                if not args.keep_hash:
                    await engine.configure({"Clear Hash": None})
                infos = await engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=min(10, board.legal_moves.count()),
                )
                candidates = [info["pv"][0] for info in infos]
                reproduced = candidates.index(played_move) + 1 if played_move in candidates else ">10"
                row = {
                    "depth": depth,
                    "ply": ply,
                    "played_move": board.san(played_move),
                    "published_rank": expected[(depth, ply)],
                    "reproduced_rank": reproduced,
                    "rank_matches": reproduced == expected[(depth, ply)],
                }
                for index, info in enumerate(infos[:6], start=1):
                    row[f"move_{index}"] = board.san(info["pv"][0])
                    row[f"eval_{index}"] = display_score(info["score"], board.turn)
                rows.append(row)
                write_rows(rows)
                print(
                    f"depth {depth:2} ply {ply:2}: published {row['published_rank']} "
                    f"reproduced {reproduced}",
                    flush=True,
                )
    finally:
        await engine.quit()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    matches = sum(row["rank_matches"] for row in rows)
    print(f"Matched {matches}/{len(rows)} recovered ranks")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    asyncio.run(main())
