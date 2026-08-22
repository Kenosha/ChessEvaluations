# 2022 analysis recovery status

Updated 2026-08-20 after the original untracked folder was confirmed lost.

## Short answer

The website story is safe. The published numerical results, the full
GambitMan/Let's Check spreadsheet, the Reddit discussion, the depth-rank
experiment, the opening book, the two engine versions, and most game records
have all been recovered or reconstructed from public sources.

What is permanently missing is mostly provenance at the rawest level: the
original move-by-move output CSVs, the exact Windows executables and their
checksums, and a frozen record of the 2022 Python/Windows/CPU environment.
Those losses prevent a byte-for-byte rerun, but not a transparent modern
replication or the visualizations.

## Recovered exactly

- The original Reddit post text and 89-comment archive, from two independent
  public Reddit archives (`reddit_archive/`).
- All four result images uploaded to Reddit, at their original displayed
  resolution (`reddit_media/`).
- GambitMan's original Google workbook, including all tournament-by-round
  Let's Check values and its original links (`gambitman/`).
- The published Stockfish 15 Niemann, Stockfish 7 Niemann, and Stockfish 15
  Caruana values. These were transcribed from the recovered source images into
  tidy CSVs (`tables/`).
- The exact Stockfish 7 depth 10–25 ranks for `13.h3`, `15.e5`, `18.Rfc1`, and
  `22.Nd6+` in Cornette–Niemann, recovered from Reddit comment `iqoansf`.
- The verified-master reply by `IMJorose`, whose archived flair reads
  `FM · Verified Master · FIDE 2300`.
- The Codekiddy Polyglot archive from the original SourceForge location and its
  sole member, `codekiddy.bin`. The archive matches the publisher's 2015 MD5
  file exactly (`015469a86eb9d5770a7146f7b721880b`).
- The official `sf_15` and `sf_7` Stockfish source tags. Both have been compiled
  locally for x86-64 BMI2; Stockfish 15 includes its required official NNUE net
  `nn-6877cd24400e.nnue`.
- The TWIC issues linked by the spreadsheet for the National Open,
  Philadelphia/World Open, US Junior, and US Open game data.

## Ready-to-use website data

- `tables/niemann_published_correlations.csv`: all recovered long-form values.
- `tables/niemann_published_triplets.csv`: 43 games with all three published
  values (Let's Check, Stockfish 15, and Stockfish 7). This can directly feed
  the triplet scatterplot.
- `tables/caruana_stockfish_15.csv`: the full Caruana comparison table.
- `tables/cornette_niemann_sf7_depth_ranks.csv`: the real depth-slider ranks.
- `tables/gambitman_lets_check_all.csv`: all 416 populated game cells in the
  recovered GambitMan workbook, not just the six selected tournaments.

The 43 triplets are the honest intersection of the published tables. The
Let's Check table contains 47 cells, while the two Stockfish tables contain 46
cells each. A naive scatterplot that invents values for the non-overlapping
rounds would be wrong; retaining the unmatched cells as faint single/double
points is fine.

## Recreated, but not byte-identical to 2022

- Stockfish binaries: the algorithms/tags are exact, but these are Linux
  binaries compiled with the current GCC toolchain, not the lost Windows BMI2
  executables.
- Python environment: `requirements-reproduction.txt` pins `chess==1.9.3`, the
  newest release available on publication day. The original Python,
  `python-chess`, and pandas versions were never recorded, so this remains a
  best-evidence reconstruction rather than proof of the installed version.
- Source PGNs: the repo's original Niemann and Caruana PGNs survive. Public TWIC
  archives can fill some games absent from the surviving Niemann file, but the
  published Stockfish table itself deliberately covers a different round set
  than GambitMan's table in the US Open and a few other cases.
- Rerun outputs: `scripts/reproduce_2022.py` records engine/book/PGN checksums,
  platform information, search settings, per-move ranks, depths, and aggregate
  percentages. It also supports fixed depth or nodes for a more reproducible
  modern sensitivity analysis.

## Irrecoverable unless another private backup appears

- Original raw move-level Stockfish 15 and Stockfish 7 CSVs.
- Original aggregate CSV/XLSX files used before exporting the Reddit images.
  The values themselves are safe in the images and reconstructed CSVs.
- The exact Stockfish `.exe` files and hashes used in 2022.
- Exact engine UCI settings if they differed from defaults (Threads, Hash,
  tablebases, and so on).
- The precise 2022 CPU model, OS build, Python package versions, and concurrent
  system load.
- Full MultiPV candidate move names, evaluations, and principal variations for
  the Cornette–Niemann depth experiment. The played moves' exact ranks survive,
  which is enough for the planned jumping-rank visualization, but not for an
  exact reconstruction of every other candidate slot.
- Original browser screenshots of selected Reddit comments. The archive keeps
  the exact text, author, score, flair, IDs, and timestamps, so faithful new
  newspaper-style excerpts can still be designed.

## Important discrepancy in the committed code

The published post reports an average search depth of **22.07**. The committed
script says `ENGINE_LIMIT = 0.004` seconds per position. On the reconstructed
Stockfish 15 build, that literal limit averages only about depth 5. Four seconds
per position, not four milliseconds, is consistent with the reported search
depth. This strongly suggests that the public constant is a typo or reflects a
different later test.

For that reason, reruns should preserve both facts:

1. A literal-code run at 0.004 seconds, labelled as such.
2. A published-method reconstruction at 4 seconds, checked against the reported
   average depth and source image values.

Neither should be silently substituted for the other.

## Other surviving-code quirks to preserve in the audit

- The post says “12 processes”; the script uses a 12-worker Python thread pool,
  with each worker spawning a separate Stockfish process for each game.
- Although the post describes an exact-best-move count, the aggregator compares
  integer-rounded WDL expectations at `LENIENCY = 0`. A second-ranked move can
  therefore count in the legacy code if its rounded expectation equals the
  best move's. The new runner reports literal top-move correlation, matching the
  prose definition; a legacy-compatible comparison should also be added before
  claiming numerical equivalence.
- Book positions are excluded wherever they occur in the Polyglot book, not
  necessarily as one continuous opening prefix. The old TODO explicitly notes
  that the book can contain “holes.” The reported average book length uses the
  last excluded ply, which is not the same statistic as a continuous opening
  cutoff.
- `Niemann.pgn` contains an exact duplicate of National Open round 5. The old
  group-by calculation combines the duplicate, but because the game is
  identical it does not alter that cell's percentage. There are 47 selected PGN
  records but 46 displayed Stockfish cells.

## Replication results so far

The first complete best-effort Stockfish 15 reconstruction used the exact
published engine tag and book, 4 seconds per position, MultiPV 10, 12 workers,
and all 47 selected PGN records. It took 1,455.8 seconds and produced 4,183
move rows and 46 aggregate cells.

- Average best-line depth was 19.52, below the published 22.07.
- A literal top-move-only aggregation averaged 12.87 percentage points below
  the published cells (mean signed difference -12.30; 4/46 cells exact).
- A targeted National Open round 2 rerun produced 68% literal top-move
  correlation but 80% under the surviving legacy WDL-equality aggregator. The
  published cell is 83%.
- On the complete first run, simply counting rank 1 or 2 reduces the mean
  absolute error to 7.80 points but overshoots by 5.63 points on average. This
  brackets the legacy behavior: it accepts some, but not all, lower-ranked
  lines whose rounded WDL expectation ties the first line.

This does not numerically reproduce the old output, but it successfully
diagnoses why the literal prose definition does not: the published cells are
consistent with the legacy WDL-equality code, plus a somewhat deeper and
machine-dependent time-limited search. The exact published CSV remains the
correct source for the website. The reruns stay in the recovery archive and do
not appear in the website story.

The Cornette–Niemann fixed-depth check was also rerun for depths 10–18 with
Stockfish 7, MultiPV 10, and a cleared hash before each position. Only 9 of 36
published played-move ranks matched exactly; none of the nine `18.Rfc1` ranks
matched. The output preserves the reconstructed top six moves and evaluations
at every tested position. That mismatch does not invalidate the archived
table—it shows that build/compiler details, hash state, and the precise analysis
sequence must also be known before “Stockfish 7 at depth N” is reproducible.

## Checksums for recovered external inputs

```text
Codekiddy archive:
3f0d42a475d8e75422bcb7a6d0aec1b7c40e5ed87fda8a2dc918f8e840270d86

codekiddy.bin:
46e8ad19a960bbc491a64d5b54ff3648b9ac2c10666cf0fc29dcfc08e249e79c

Stockfish sf_15 source archive:
0553fe53ea57ce6641048049d1a17d4807db67eecd3531a3749401362a27c983

Stockfish sf_7 source archive:
a2aada4dd070fcc870bf6743bb6219519475771fb2c3269c4bb5128f20026add
```

Compiled-binary hashes and every rerun's input hashes are written into the
corresponding `recovered_2022/reruns/*_metadata.json` file.
