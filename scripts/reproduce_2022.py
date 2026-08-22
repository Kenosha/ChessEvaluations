#!/usr/bin/env python3
"""Re-run the September 2022 engine-correlation experiment.

This is a portable replacement for ``engine_correlation_analysis.py``.  Its
default settings preserve the published method: all Codekiddy book positions
are excluded, ten principal variations are requested, and only the first
engine choice counts as a match.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import chess.polyglot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK = (
    ROOT
    / "recovered_2022"
    / "external_inputs"
    / "codekiddy"
    / "codekiddy.bin"
)
ENGINES = {
    "sf15": (
        ROOT
        / "recovered_2022"
        / "external_inputs"
        / "Stockfish-sf_15"
        / "src"
        / "stockfish"
    ),
    "sf7": (
        ROOT
        / "recovered_2022"
        / "external_inputs"
        / "Stockfish-sf_7"
        / "src"
        / "stockfish"
    ),
}
EVENT_FILTERS = (
    "Philadelphia",
    "Annual World Open",
    "Junior 2021",
    "US Open",
    "Tras-os-Montes",
    "National Open",
)


@dataclass(frozen=True)
class GameRecord:
    index: int
    game: chess.pgn.Game


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_selected_games(pgn_path: Path) -> list[GameRecord]:
    selected: list[GameRecord] = []
    with pgn_path.open(encoding="utf-8", errors="replace") as pgn:
        index = 0
        while game := chess.pgn.read_game(pgn):
            event = game.headers.get("Event", "")
            if any(fragment in event for fragment in EVENT_FILTERS):
                selected.append(GameRecord(index=index, game=game))
            index += 1
    return selected


def make_limit(mode: str, value: float) -> chess.engine.Limit:
    if mode == "time":
        return chess.engine.Limit(time=value)
    if mode == "depth":
        return chess.engine.Limit(depth=int(value))
    if mode == "nodes":
        return chess.engine.Limit(nodes=int(value))
    raise ValueError(f"Unknown limit mode: {mode}")


def expectation(score: chess.engine.PovScore) -> float:
    win, draw, loss = score.wdl()
    return (win + 0.5 * draw) / (win + draw + loss)


async def analyse_game(
    record: GameRecord,
    *,
    engine_path: Path,
    book_path: Path,
    limit_mode: str,
    limit_value: float,
    multipv: int,
) -> list[dict[str, Any]]:
    game = record.game
    board = game.board()
    rows: list[dict[str, Any]] = []
    _, engine = await chess.engine.popen_uci(str(engine_path))
    try:
        with chess.polyglot.open_reader(book_path) as book:
            for move in game.mainline_moves():
                mover = game.headers.get("White" if board.turn else "Black", "?")
                color = "white" if board.turn else "black"
                san = board.san(move)
                book_moves = {entry.move for entry in book.find_all(board)}

                row: dict[str, Any] = {
                    "game_index": record.index,
                    "event": game.headers.get("Event", "?"),
                    "date": game.headers.get("Date", "?"),
                    "round": game.headers.get("Round", "?"),
                    "white": game.headers.get("White", "?"),
                    "black": game.headers.get("Black", "?"),
                    "result": game.headers.get("Result", "?"),
                    "ply": board.ply(),
                    "color": color,
                    "player": mover,
                    "move": san,
                    "rank": "B" if move in book_moves else ">10",
                    "best_move": "",
                    "chosen_expectation": "",
                    "best_expectation": "",
                    "legacy_code_match": "",
                    "depth_best": "",
                    "depth_played": "",
                    "depth_min": "",
                    "depth_max": "",
                }

                if move not in book_moves:
                    infos = await engine.analyse(
                        board,
                        make_limit(limit_mode, limit_value),
                        multipv=min(multipv, board.legal_moves.count()),
                    )
                    if isinstance(infos, dict):
                        infos = [infos]
                    candidates = [info["pv"][0] for info in infos]
                    row["best_move"] = board.san(candidates[0])
                    row["best_expectation"] = expectation(infos[0]["score"])
                    row["depth_best"] = infos[0].get("depth", "")
                    depths = [info.get("depth") for info in infos if "depth" in info]
                    if depths:
                        row["depth_min"] = min(depths)
                        row["depth_max"] = max(depths)
                    if move in candidates:
                        candidate_index = candidates.index(move)
                        row["rank"] = candidate_index + 1
                        row["chosen_expectation"] = expectation(
                            infos[candidate_index]["score"]
                        )
                        row["depth_played"] = infos[candidate_index].get("depth", "")
                        row["legacy_code_match"] = (
                            row["chosen_expectation"] >= row["best_expectation"]
                        )
                    else:
                        # The old code replaces missing chosen-move expectation with 0.
                        row["chosen_expectation"] = 0.0
                        row["legacy_code_match"] = 0.0 >= row["best_expectation"]

                rows.append(row)
                board.push(move)
    finally:
        await engine.quit()
    return rows


def round_number(raw_round: str) -> str:
    return raw_round.split(".", 1)[0]


def aggregate(
    rows: list[dict[str, Any]], player: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["player"] == player and row["rank"] != "B":
            key = (str(row["event"]), round_number(str(row["round"])))
            groups.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for (event, game_round), group in sorted(groups.items()):
        matches = sum(row["rank"] == 1 for row in group)
        legacy_matches = sum(row["legacy_code_match"] is True for row in group)
        total = len(group)
        result.append(
            {
                "event": event,
                "round": game_round,
                "matches": matches,
                "legacy_code_matches": legacy_matches,
                "non_book_moves": total,
                "correlation": matches / total if total else "",
                "percent_rounded": round(100 * matches / total) if total else "",
                "legacy_code_correlation": legacy_matches / total if total else "",
                "legacy_code_percent_rounded": (
                    round(100 * legacy_matches / total) if total else ""
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=ENGINES, default="sf15")
    parser.add_argument("--engine-path", type=Path)
    parser.add_argument("--book", type=Path, default=DEFAULT_BOOK)
    parser.add_argument("--pgn", type=Path, default=ROOT / "data" / "Niemann.pgn")
    parser.add_argument("--player", default="Niemann, Hans Moke")
    parser.add_argument("--limit-mode", choices=("time", "depth", "nodes"), default="time")
    parser.add_argument("--limit-value", type=float, default=0.004)
    parser.add_argument("--multipv", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--event", help="Keep selected games whose event contains this text")
    parser.add_argument("--round", help="Keep games whose round number begins with this value")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "recovered_2022" / "reruns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_path = (args.engine_path or ENGINES[args.engine]).resolve()
    book_path = args.book.resolve()
    pgn_path = args.pgn.resolve()
    for required in (engine_path, book_path, pgn_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    games = read_selected_games(pgn_path)
    if args.event:
        games = [game for game in games if args.event in game.game.headers.get("Event", "")]
    if args.round:
        games = [
            game
            for game in games
            if round_number(game.game.headers.get("Round", "?")) == args.round
        ]
    if args.max_games is not None:
        games = games[: args.max_games]
    print(f"Selected {len(games)} games; analysing with {args.engine} ...", flush=True)

    started = time.monotonic()
    worker = lambda record: asyncio.run(
        analyse_game(
            record,
            engine_path=engine_path,
            book_path=book_path,
            limit_mode=args.limit_mode,
            limit_value=args.limit_value,
            multipv=args.multipv,
        )
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, record): i for i, record in enumerate(games)}
        nested_rows: list[list[dict[str, Any]] | None] = [None] * len(games)
        for completed, future in enumerate(as_completed(futures), start=1):
            nested_rows[futures[future]] = future.result()
            print(f"Completed {completed}/{len(games)} games", flush=True)
    rows = [row for game_rows in nested_rows if game_rows for row in game_rows]
    aggregate_rows = aggregate(rows, args.player)
    elapsed = time.monotonic() - started

    run_name = f"{args.engine}_{args.limit_mode}_{args.limit_value:g}"
    raw_path = args.output_dir / f"{run_name}_raw.csv"
    aggregate_path = args.output_dir / f"{run_name}_aggregate.csv"
    metadata_path = args.output_dir / f"{run_name}_metadata.json"
    write_csv(raw_path, rows)
    write_csv(aggregate_path, aggregate_rows)

    depths = [int(row["depth_best"]) for row in rows if row["depth_best"] != ""]
    played_depths = [
        int(row["depth_played"]) for row in rows if row["depth_played"] != ""
    ]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine_label": args.engine,
        "engine_path": str(engine_path.relative_to(ROOT)),
        "engine_sha256": sha256(engine_path),
        "book_path": str(book_path.relative_to(ROOT)),
        "book_sha256": sha256(book_path),
        "pgn_path": str(pgn_path.relative_to(ROOT)),
        "pgn_sha256": sha256(pgn_path),
        "event_filters": EVENT_FILTERS,
        "selected_game_records": len(games),
        "analysed_plies": len(rows),
        "limit_mode": args.limit_mode,
        "limit_value": args.limit_value,
        "multipv": args.multipv,
        "workers": args.workers,
        "average_best_line_depth": statistics.mean(depths) if depths else None,
        "average_played_line_depth_when_in_multipv": (
            statistics.mean(played_depths) if played_depths else None
        ),
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "python_chess": chess.__version__,
        "platform": platform.platform(),
        "processor_count": os.cpu_count(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {display_path(raw_path)}", flush=True)
    print(f"Wrote {display_path(aggregate_path)}", flush=True)
    print(f"Average best-line depth: {metadata['average_best_line_depth']:.2f}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
