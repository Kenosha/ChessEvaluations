# Old computer recovery checklist

> Historical checklist only. The old folder is now known to be lost; see
> [RECOVERY_STATUS.md](RECOVERY_STATUS.md) for what was recovered publicly,
> rebuilt, or remains irrecoverable.

This checklist covers the historical material needed to replace mock data in the website visualization and make the 2022 analysis reproducible. Copy recovered files into a separate folder first; do not rename or edit the originals.

## 1. Published analysis outputs — highest priority

Search the old `ChessEvaluations` project and nearby Documents/Desktop/Downloads folders for:

- `outputs_raw/`
- `outputs_aggregated/`
- `outputs_edited/`
- `results/`
- CSV, XLSX, XLS, PNG, JPG, or PDF files containing `Niemann`, `Caruana`, `Stockfish`, `correlation`, `accuracy`, `Let’s Check`, `Gambit`, or a timestamp.

Please preserve:

- Raw move-level Stockfish 15 output.
- Aggregated per-game Stockfish 15 correlation results.
- Raw move-level Stockfish 7 output.
- Aggregated per-game Stockfish 7 correlation results.
- Any Caruana comparison output.
- The exact tables uploaded to Reddit.
- The workbook, notebook, or script used to color and export those tables.

Minimum useful substitute: a screenshot or spreadsheet giving tournament, round, opponent, and correlation percentage for every game under Stockfish 15 and Stockfish 7.

These files unlock the real paired scatterplot, the cooling table, exact summary counts, and trustworthy game-to-game transitions.

## 2. Depth and MultiPV evidence — highest priority

Find the screenshot/table in which candidate moves change rank as engine depth changes. Also search for CSVs, notebooks, console captures, or spreadsheets containing columns such as:

- `depth`
- `multipv`
- `rank`
- `best_move`
- `best_eval`
- `pv`
- `score`
- `nodes`

For every displayed position, the ideal recovered record contains:

- Game, tournament, and round.
- Move number.
- FEN before the move.
- Played move.
- Side to move.
- Engine name and version.
- Search depth.
- Nodes or time limit.
- First four candidate moves at each depth.
- Evaluation and principal variation for each candidate.

Minimum useful substitute: the original screenshot plus enough game information to reconstruct the positions.

These files unlock the depth slider where moves visibly jump between Best, 2nd, 3rd, and 4th.

## 3. Original Reddit media and discussion captures

Search for:

- The PNG/JPG files embedded in the Reddit post.
- Screenshots taken before or after publishing.
- The original Excel tables.
- Draft post text or Markdown.
- Browser downloads with `reddit`, `chess`, `niemann`, `gambit`, or `stockfish` in their names.
- The screenshot of the FIDE Master reply.

For the FIDE Master reply, please preserve the full screenshot so the username, title flair, wording, score, and reply context can be verified.

Also collect any specific positive, critical, or funny comments that should appear in the final newspaper-style reaction page.

## 4. Engine and opening-book inputs

Search for the ignored project directories:

- `engine/`
- `engine/executables/`
- `openings/`

Expected or likely files include:

- `stockfish 7 x64 bmi2.exe`
- A Stockfish 15 executable.
- NNUE network files, if stored separately.
- `codekiddy.bin`

For every engine file, retain its original filename. If possible, also record:

- File size.
- SHA-256 checksum.
- Download URL or archive.
- UCI settings: Threads, Hash, MultiPV, Syzygy path, and NNUE options.

The website does not need to ship these binaries. They are needed to identify or reproduce the historical calculation. A future reproducible runner should normally download a known engine build and verify its checksum.

## 5. Software environment

Look for:

- `requirements.txt`
- `environment.yml`
- Conda environment exports.
- `pip freeze` output.
- Jupyter notebooks.
- IDE project metadata.
- Python virtual environments, if the original computer is still operational.

Record if possible:

- Python version.
- `python-chess` version.
- pandas version.
- Operating system.
- CPU model and core count.
- Approximate RAM.

Also note whether Stockfish was run with 12 independent processes and whether any engine-level thread count was changed.

## 6. Source data and selection notes

Preserve any alternate PGNs or database exports used in the published run, especially:

- The exact Niemann PGN analyzed for Reddit.
- The exact Caruana PGN.
- CaissaBase source files or filtered exports.
- Scid vs PC filter files or notes.
- Lists of included/excluded tournaments.
- Time-control classification notes.

This matters because the 47 values in Gambit-Man’s public spreadsheet do not distribute across tournaments exactly like the event matches in the surviving PGN/script. The original output is needed to establish a valid round-by-round pairing.

## 7. Code that may not have reached GitHub

Search for copies of:

- `funcs.py`
- `engine_correlation_analysis.py`
- Other `.py`, `.ipynb`, `.R`, `.do`, `.m`, or spreadsheet macro files in the project.
- Files containing `get_accuracy_frame`, `get_move_evals`, `ENGINE_LIMIT`, `Stockfish 15`, or `codekiddy`.

Keep files even if they appear duplicated. Modification timestamps may identify the version that produced the published tables.

## 8. Recommended transfer structure

Copy recovered material without altering it into:

```text
recovered_2022/
  original_outputs/
  original_tables/
  reddit_media/
  depth_experiment/
  engines_metadata/
  opening_book/
  environment/
  alternate_code/
  alternate_pgn/
  notes/
```

Do not commit engine binaries, opening books, private data, or large database files until their licenses and repository size implications have been checked.

## 9. If time is limited

Recover these five things first:

1. Stockfish 15 per-game correlations.
2. Stockfish 7 per-game correlations.
3. Original Reddit table images/workbook.
4. Depth-ranking screenshot or source table.
5. FIDE Master comment screenshot.

Those five evidence categories were sufficient to remove every placeholder from the website exhibit. The needed published values, ranks, and comment text have now been recovered from public artifacts; the lost private files matter only for exact historical reproducibility.
