# Personal website handoff

## Intended location

Create the exhibit at:

```text
src/exhibits/chess-evaluations/ChessEvaluationsExhibit.tsx
```

Copy the evidence into a strict TypeScript file in the same directory. Do not import from this adjacent checkout at build time.

## Page contract

The component needs one prop:

```ts
type ChessEvaluationsPage = "incident" | "measure" | "reaction" | "depth";

interface ChessEvaluationsExhibitProps {
  view: ChessEvaluationsPage;
}
```

The host owns project and page navigation. The exhibit owns only local state:

- `incident`: selected game, current ply, checkpoint, and playback state.
- `measure`: selected analysis source.
- `reaction`: no local state.
- `depth`: selected depth from 10 through 25.

## Files to take

- `index.html`: semantic structure and public copy.
- `styles.css`: exhibit-only styles. Translate selectors to a CSS Module.
- `app.js`: board, chart, heatmap, and interaction logic.
- `data/evidence.js`: exact display data.
- `personal-site-manifest.json`: provenance and page metadata.
- `posters/`: small PNG fallbacks rendered from each live page.

## Integration rules

- Keep the visualization inside the host's dominant asset field.
- Move each `data-caption` string into the host Description field.
- Remove this prototype's header, footer, and outer project copy after the host supplies them.
- Use the host's previous and next chevrons. Do not nest a second project paginator.
- Preserve the 43-game matched set. Do not combine the unmatched cells from the three source images in one moving-point chart.
- Keep the three inactive values faintly visible in the scatterplot.
- Keep the four depth chips limited to the played moves. The archive does not preserve the other candidate moves or their evaluations.
- Guard timers for pre-rendering and stop playback when the page unmounts.
- Use the matching file in `posters/` if the lazy exhibit cannot start.
- Respect reduced motion. Replace the fast game autoplay with a settled annotated position when motion is reduced.
- Start the Sinquefield replay automatically, pause at plies 23, 24, 55, 64, and 113, and resume only after the visitor presses play. Ply 23 isolates Niemann's disputed “ridiculous miracle” preparation claim. At ply 24, render `13.Qh4?` as a translucent unplayed interview line from c4 to h4, mark the bishop on g5 as hanging, and clear the entire trace before the actual game continues. Frame `28.g4?` as the principal counterweight: Carlsen made the decisive error. Do not pause the three-ply resignation replay.
- Keep controls reachable by keyboard and use at least 44-by-44-pixel touch targets on small screens.

## Suggested catalog pages

| Page       | Title              | Description source        |
| ---------- | ------------------ | ------------------------- |
| `incident` | The incident       | `#incident[data-caption]` |
| `measure`  | Engine correlation | `#measure[data-caption]`  |
| `reaction` | Public review      | `#reaction[data-caption]` |
| `depth`    | Search depth       | `#depth[data-caption]`    |

The visual prototype is framework-neutral on purpose. The personal-site implementation should use React state and authored SVG. It does not need a chart library, chess library, image asset, iframe, or backend.
