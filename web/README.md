# Chess Evaluations exhibit

This directory contains a self-contained four-page exhibit for the Work viewer on the personal website.

Run it from the repository root:

```bash
python3 -m http.server 4173 --directory web
```

Open `http://localhost:4173`.

Open `http://localhost:4173/?page=incident&ply=24` to hold the replay on the postgame-interview checkpoint for capture or review.

## Story

1. Watch the Sinquefield Cup replay pause at five annotated positions: the disputed preparation claim, the clearly marked `13.Qh4?` interview line, Carlsen's decisive error, Niemann's conversion, and the resignation. Then play the three-ply online resignation without interruption.
2. Move 43 matched games between the three result tables published in 2022.
3. Explore eight reactions preserved in the archived Reddit thread. The comment that requested the depth check remains on screen during the transition.
4. Follow that request into the Cornette–Niemann experiment and move four played moves through their published Stockfish 7 ranks at depths 10–25.

The long explanation sits below the visual frame. The personal website can place that copy in its Description field and keep the interactive pages in the dominant asset field.

## Evidence

The exhibit has no generated measurements or invented quotes.

- `data/evidence.js` is the portable browser data boundary.
- `../recovered_2022/tables/niemann_published_triplets.csv` is the canonical 43-game comparison table.
- `../recovered_2022/tables/cornette_niemann_sf7_depth_ranks.csv` is the canonical depth table.
- `../recovered_2022/reddit_archive/` contains the archived post and comments.
- `assets/reddit-comments/` contains reproducible crops rendered from the exact archived comment text and metadata.
- `../scripts/render_reddit_crops.py` regenerates those comment assets.
- `../scripts/render_reddit_depth_posters.py` refreshes the reaction and depth fallback posters from the same assets.
- `../recovered_2022/depth_experiment/niemann_cornette_2020.pgn` contains the depth-experiment game.
- `posters/` contains four rendered fallbacks for previews and error states.

The incident annotations are editorial summaries of the [original postgame interview](https://www.youtube.com/watch?v=DCeJrItfQqw), [Chess.com's Hans Niemann report](https://www.chess.com/blog/CHESScom/hans-niemann-report), [ChessBase's round-three game report](https://en.chessbase.com/post/sinquefield-cup-2022-r3), and [FIDE's disciplinary decision](https://www.fide.com/decision-on-the-magnus-carlsen-hans-niemann-case/). The cyan `13.Qh4?` queen is explicitly an unplayed interview line, never a move from the game score.

The game boards and all charts use plain HTML, CSS, JavaScript, and SVG. The exhibit has no runtime dependency and no large media asset.

See `PERSONAL_SITE_HANDOFF.md` for the React integration boundary.
