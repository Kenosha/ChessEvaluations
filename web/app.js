const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const PIECES = {
  K: "♔",
  Q: "♕",
  R: "♖",
  B: "♗",
  N: "♘",
  P: "♙",
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
};

const SINQUEFIELD_UCI = `
  d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 g2g3 e8g8 f1g2 d7d5 a2a3 b4c3 b2c3 d5c4
  g1f3 c7c5 e1g1 c5d4 d1d4 b8c6 d4c4 e6e5 c1g5 h7h6 f1d1 c8e6 d1d8 e6c4
  d8a8 f8a8 g5f6 g7f6 g1f1 a8d8 f1e1 c6a5 a1d1 d8c8 f3d2 c4e6 c3c4 e6c4
  d2c4 c8c4 d1d8 g8g7 g2d5 c4c7 d8a8 a7a6 a8b8 f6f5 b8e8 e5e4 g3g4 c7c5
  d5a2 a5c4 a3a4 c4d6 e8e7 f5g4 e7d7 e4e3 f2e3 d6e4 e1f1 c5c1 f1g2 c1c2
  a2f7 c2e2 g2g1 e2e1 g1g2 e1e2 g2g1 g7f6 f7d5 e2d2 d7f7 f6g6 f7d7 e4g5
  d5f7 g6f5 d7d2 g5f3 g1g2 f3d2 a4a5 f5e5 g2g3 d2f1 g3f2 f1h2 e3e4 e5e4
  f7e6 e4f4 e6c8 h2f3 c8b7 f3e5 b7a6 e5c6 a6b7 c6a5 b7d5 h6h5 d5f7 h5h4
  f7d5
`
  .trim()
  .split(/\s+/);

const SINQUEFIELD_SAN = `
  d4 Nf6 c4 e6 Nc3 Bb4 g3 O-O Bg2 d5 a3 Bxc3+ bxc3 dxc4 Nf3 c5 O-O cxd4 Qxd4 Nc6
  Qxc4 e5 Bg5 h6 Rfd1 Be6 Rxd8 Bxc4 Rxa8 Rxa8 Bxf6 gxf6 Kf1 Rd8 Ke1 Na5 Rd1 Rc8
  Nd2 Be6 c4 Bxc4 Nxc4 Rxc4 Rd8+ Kg7 Bd5 Rc7 Ra8 a6 Rb8 f5 Re8 e4 g4 Rc5 Ba2 Nc4
  a4 Nd6 Re7 fxg4 Rd7 e3 fxe3 Ne4 Kf1 Rc1+ Kg2 Rc2 Bxf7 Rxe2+ Kg1 Re1+ Kg2 Re2+
  Kg1 Kf6 Bd5 Rd2 Rf7+ Kg6 Rd7 Ng5 Bf7+ Kf5 Rxd2 Nf3+ Kg2 Nxd2 a5 Ke5 Kg3 Nf1+
  Kf2 Nxh2 e4 Kxe4 Be6 Kf4 Bc8 Nf3 Bxb7 Ne5 Bxa6 Nc6 Bb7 Na5 Bd5 h5 Bf7 h4 Bd5
`
  .trim()
  .split(/\s+/);

const GAMES = {
  sinquefield: {
    white: "Magnus Carlsen",
    whiteRating: "2861",
    black: "Hans Niemann",
    blackRating: "2688",
    result: "0–1",
    moves: SINQUEFIELD_UCI,
    san: SINQUEFIELD_SAN,
    start: 0,
    label: "Sinquefield Cup · Round 3",
  },
  resignation: {
    white: "Hans Niemann",
    whiteRating: "2699",
    black: "Magnus Carlsen",
    blackRating: "2861",
    result: "1–0",
    moves: ["d2d4", "g8f6", "c2c4"],
    san: ["d4", "Nf6", "c4"],
    start: 0,
    label: "Julius Baer Generation Cup · Round 6",
  },
};

const MOMENTS = {
  sinquefield: [
    {
      ply: 23,
      kicker: "THE ‘RIDICULOUS MIRACLE’",
      text: "Niemann said he had checked this rare line that same morning, but could not remember why. The reference game he later named did not match exactly.",
      context: "POSTGAME PREPARATION CLAIM · DISPUTED",
    },
    {
      ply: 24,
      kicker: "13.Qh4? · POSTGAME DOUBTS",
      text: "Niemann then proposed this move, hanging his bishop. Critics doubted the level of his analysis; a similarly disputed Firouzja recap intensified the suspicion.",
      context: "UNPLAYED LINE · INTERVIEW CRITICISM, NOT PROOF",
      tone: "disputed",
      ghost: { from: "c4", to: "h4", piece: "Q", risk: "g5" },
    },
    {
      ply: 55,
      kicker: "28.g4? · THE COUNTERWEIGHT",
      text: "The decisive mistake comes from Carlsen. Contemporary analysis judged the position lost from here: elite but ordinary play could convert it.",
      context: "PLAYED MOVE · EVIDENCE CUTS BOTH WAYS",
      tone: "counterpoint",
    },
    {
      ply: 64,
      kicker: "32…e3 · CONVERSION, NOT MIRACLE",
      text: "This forcing pawn push looks dramatic, but follows from the advantage after 28.g4? Contemporary reports praised it without identifying it as suspicious.",
      context: "PLAYED MOVE · NIEMANN",
      tone: "counterpoint",
    },
    {
      ply: 113,
      kicker: "0–1 · THE ACCUSATION",
      text: "Carlsen resigns, then leaves the tournament. His later complaint concerned Niemann’s demeanor in ‘critical positions’, not any named move.",
      context: "FIDE LATER FOUND NO EVIDENCE OF OTB CHEATING",
    },
  ],
  resignation: [
    {
      ply: 3,
      kicker: "2.c4",
      text: "Carlsen resigns after three half-moves. The game becomes a public statement.",
      context: "SEPTEMBER 19, 2022 · ONLINE",
    },
  ],
};

const {
  methods: METHOD_META,
  events: SOURCE_EVENTS,
  depthExperiment,
} = window.CHESS_EVIDENCE;

const games = window.CHESS_EVIDENCE.correlations.map(
  ([eventKey, round, letscheck, sf15, sf7]) => {
    const eventIndex = SOURCE_EVENTS.findIndex(
      (event) => event.key === eventKey,
    );
    const event = SOURCE_EVENTS[eventIndex];
    return {
      id: `${eventKey}-${round}`,
      event: eventKey,
      eventName: event.name,
      eventShort: event.short,
      eventIndex,
      round,
      letscheck,
      sf15,
      sf7,
    };
  },
);

let activeGame = "sinquefield";
let activePly = GAMES.sinquefield.start;
let gameTimer = null;
let autoplayStartTimer = null;
let gameResumeTimer = null;
let activeMethod = "letscheck";
let activeDepth = 17;
const METHOD_ANNOTATIONS = {
  letscheck: {
    eyebrow: "LET’S CHECK · POOLED ENGINES",
    headline: "The viral outliers sit at the ceiling.",
    copy: "GambitMan and Yosha presented the 97–100% games as suspicious. The red marks locate that original claim.",
  },
  sf15: {
    eyebrow: "STOCKFISH 15 · FIXED · SINGLE · REPRODUCIBLE",
    headline: "The suspicious games cool sharply.",
    copy: "My cleaner rerun uses one fixed engine across the same games. But Stockfish 15 postdates them; Niemann could not have used it then.",
  },
  sf7: {
    eyebrow: "STOCKFISH 7 · PERIOD-AVAILABLE ENGINE",
    headline: "A historically plausible engine cools them again.",
    copy: "Stockfish 7 was available during the period. The same formerly ‘perfect’ games no longer cluster at 100%.",
  },
};
const prefersReducedMotion =
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
if (prefersReducedMotion) activePly = MOMENTS.sinquefield[0].ply;
const requestedPlyValue = new URLSearchParams(window.location.search).get(
  "ply",
);
const requestedPly = Number(requestedPlyValue);
const hasRequestedPly =
  requestedPlyValue !== null &&
  Number.isInteger(requestedPly) &&
  requestedPly >= 0 &&
  requestedPly <= GAMES.sinquefield.moves.length;
if (hasRequestedPly) activePly = requestedPly;

const boardElement = document.querySelector("#chessboard");
const scrubber = document.querySelector("#game-scrubber");
const playButton = document.querySelector("#play-game");
const boardCallout = document.querySelector("#board-callout");

function initialBoard() {
  const board = {};
  const backWhite = ["R", "N", "B", "Q", "K", "B", "N", "R"];
  const backBlack = ["r", "n", "b", "q", "k", "b", "n", "r"];
  FILES.forEach((file, index) => {
    board[`${file}1`] = backWhite[index];
    board[`${file}2`] = "P";
    board[`${file}7`] = "p";
    board[`${file}8`] = backBlack[index];
  });
  return board;
}

function applyMove(board, move) {
  const from = move.slice(0, 2);
  const to = move.slice(2, 4);
  const promotion = move.slice(4, 5);
  const piece = board[from];

  if (!piece) return;

  if (
    (piece === "K" || piece === "k") &&
    Math.abs(FILES.indexOf(from[0]) - FILES.indexOf(to[0])) === 2
  ) {
    const rank = from[1];
    const kingSide = to[0] === "g";
    const rookFrom = `${kingSide ? "h" : "a"}${rank}`;
    const rookTo = `${kingSide ? "f" : "d"}${rank}`;
    board[rookTo] = board[rookFrom];
    delete board[rookFrom];
  }

  if ((piece === "P" || piece === "p") && from[0] !== to[0] && !board[to]) {
    const capturedRank =
      piece === "P" ? String(Number(to[1]) - 1) : String(Number(to[1]) + 1);
    delete board[`${to[0]}${capturedRank}`];
  }

  board[to] = promotion
    ? piece === piece.toUpperCase()
      ? promotion.toUpperCase()
      : promotion.toLowerCase()
    : piece;
  delete board[from];
}

function boardAtPly(game, ply) {
  const board = initialBoard();
  game.moves.slice(0, ply).forEach((move) => applyMove(board, move));
  return board;
}

function renderBoard() {
  const game = GAMES[activeGame];
  const board = boardAtPly(game, activePly);
  const lastMove = activePly ? game.moves[activePly - 1] : "";
  const moment = nearestMoment();
  const ghost = moment?.ghost || null;
  const fragment = document.createDocumentFragment();

  for (let rank = 8; rank >= 1; rank -= 1) {
    FILES.forEach((file, fileIndex) => {
      const squareName = `${file}${rank}`;
      const square = document.createElement("div");
      const dark = (fileIndex + rank) % 2 === 1;
      square.className = `square${dark ? " is-dark" : ""}`;
      square.dataset.square = squareName;

      if (lastMove.startsWith(squareName)) square.classList.add("is-from");
      if (lastMove.slice(2, 4) === squareName) square.classList.add("is-to");
      if (ghost?.from === squareName) square.classList.add("is-ghost-origin");
      if (ghost?.to === squareName) square.classList.add("is-ghost-target");
      if (ghost?.risk === squareName) square.classList.add("is-ghost-risk");

      if (file === "a") {
        const coordinate = document.createElement("span");
        coordinate.className = "coordinate rank";
        coordinate.textContent = String(rank);
        square.appendChild(coordinate);
      }
      if (rank === 1) {
        const coordinate = document.createElement("span");
        coordinate.className = "coordinate file";
        coordinate.textContent = file;
        square.appendChild(coordinate);
      }

      if (board[squareName]) {
        const piece = document.createElement("span");
        piece.className = `piece${board[squareName] === board[squareName].toLowerCase() ? " is-black" : ""}`;
        piece.textContent = PIECES[board[squareName]];
        square.appendChild(piece);
      }

      if (ghost?.to === squareName) {
        const ghostPiece = document.createElement("span");
        ghostPiece.className = "piece ghost-piece";
        ghostPiece.setAttribute("aria-hidden", "true");
        ghostPiece.textContent = PIECES[ghost.piece];

        const badge = document.createElement("span");
        badge.className = "ghost-badge";
        badge.setAttribute("aria-hidden", "true");
        badge.textContent = "?";
        square.append(ghostPiece, badge);
      }

      if (ghost?.risk === squareName) {
        const riskBadge = document.createElement("span");
        riskBadge.className = "ghost-risk-badge";
        riskBadge.setAttribute("aria-hidden", "true");
        riskBadge.textContent = "HANGS";
        square.appendChild(riskBadge);
      }
      fragment.appendChild(square);
    });
  }

  if (ghost) fragment.appendChild(buildGhostTrace(ghost));

  boardElement.replaceChildren(fragment);
  boardElement.classList.toggle("has-ghost", Boolean(ghost));
  boardElement.setAttribute(
    "aria-label",
    ghost
      ? `Chess position with an unplayed analysis line from ${ghost.from} to ${ghost.to}`
      : "Animated chess position",
  );
  updateGameInterface();
}

function buildGhostTrace(ghost) {
  const ns = "http://www.w3.org/2000/svg";
  const layer = document.createElementNS(ns, "svg");
  const path = document.createElementNS(ns, "path");
  const fileIndex = (square) => FILES.indexOf(square[0]);
  const rankIndex = (square) => 8 - Number(square[1]);
  const startX = fileIndex(ghost.from) + 0.72;
  const startY = rankIndex(ghost.from) + 0.5;
  const endX = fileIndex(ghost.to) + 0.22;
  const endY = rankIndex(ghost.to) + 0.5;

  layer.setAttribute("class", "ghost-layer");
  layer.setAttribute("viewBox", "0 0 8 8");
  layer.setAttribute("aria-hidden", "true");
  path.setAttribute("class", "ghost-path");
  path.setAttribute("d", `M ${startX} ${startY} L ${endX} ${endY}`);
  path.setAttribute("pathLength", "1");
  layer.appendChild(path);
  return layer;
}

function getMoveLabel(game, ply) {
  if (!ply) return "Start position";
  const moveNumber = Math.ceil(ply / 2);
  const dots = ply % 2 === 0 ? "…" : ".";
  return `${moveNumber}${dots} ${game.san[ply - 1] || game.moves[ply - 1]}`;
}

function nearestMoment() {
  const moments = MOMENTS[activeGame];
  return moments.find((moment) => moment.ply === activePly) || null;
}

function updateGameInterface() {
  const game = GAMES[activeGame];
  scrubber.max = String(game.moves.length);
  scrubber.value = String(activePly);
  scrubber.style.setProperty(
    "--range-progress",
    `${(activePly / Math.max(game.moves.length, 1)) * 100}%`,
  );

  document.querySelector("#white-player").textContent = game.white;
  document.querySelector("#white-rating").textContent = game.whiteRating;
  document.querySelector("#black-player").textContent = game.black;
  document.querySelector("#black-rating").textContent = game.blackRating;
  document.querySelector("#game-result").textContent = game.result;
  document.querySelector("#move-label").textContent = getMoveLabel(
    game,
    activePly,
  );
  document.querySelector("#move-progress").textContent =
    `${activePly} / ${game.moves.length}`;
  document.querySelector("#evidence-label").textContent = game.label;

  const moment = nearestMoment();
  if (moment) {
    boardCallout.querySelector(".callout-kicker").textContent = moment.kicker;
    boardCallout.querySelector("p").textContent = moment.text;
    boardCallout.querySelector(".callout-context").textContent =
      moment.context || "";
    boardCallout.dataset.tone = moment.tone || "neutral";
    boardCallout.classList.remove("is-hidden");
    boardCallout.classList.remove("is-revealed");
    void boardCallout.offsetWidth;
    boardCallout.classList.add("is-revealed");
  } else {
    boardCallout.dataset.tone = "neutral";
    boardCallout.classList.add("is-hidden");
    boardCallout.classList.remove("is-revealed");
  }
}

function stopGame(label = "Play game") {
  window.clearInterval(gameTimer);
  window.clearTimeout(autoplayStartTimer);
  window.clearTimeout(gameResumeTimer);
  gameTimer = null;
  autoplayStartTimer = null;
  gameResumeTimer = null;
  boardElement.classList.remove("is-clearing-ghost");
  playButton.classList.remove("is-playing");
  playButton.setAttribute("aria-label", label);
}

function scheduleGamePlayback(delay = 350) {
  if (prefersReducedMotion) return;
  window.clearTimeout(autoplayStartTimer);
  autoplayStartTimer = window.setTimeout(() => {
    autoplayStartTimer = null;
    playGame();
  }, delay);
}

function playGame() {
  window.clearTimeout(autoplayStartTimer);
  autoplayStartTimer = null;

  if (gameTimer || gameResumeTimer) {
    stopGame();
    return;
  }

  const game = GAMES[activeGame];
  if (activePly >= game.moves.length) {
    if (activeGame === "sinquefield") {
      setGame("resignation", true);
      return;
    }
    activePly = 0;
    renderBoard();
  }
  playButton.classList.add("is-playing");
  playButton.setAttribute("aria-label", "Pause game");

  if (nearestMoment()?.ghost && !prefersReducedMotion) {
    boardElement.classList.add("is-clearing-ghost");
    gameResumeTimer = window.setTimeout(() => {
      gameResumeTimer = null;
      boardElement.classList.remove("is-clearing-ghost");
      startGameTimer(game);
    }, 280);
    return;
  }

  startGameTimer(game);
}

function startGameTimer(game) {
  gameTimer = window.setInterval(
    () => {
      activePly += 1;
      renderBoard();

      if (activeGame === "sinquefield" && nearestMoment()) {
        const finalPosition = activePly >= game.moves.length;
        stopGame(
          finalPosition ? "Continue to resignation game" : "Continue game",
        );
        return;
      }

      if (activePly >= game.moves.length) {
        stopGame();
      }
    },
    activeGame === "resignation" ? 850 : 180,
  );
}

function setGame(gameKey, autoplay = false) {
  stopGame();
  activeGame = gameKey;
  activePly = autoplay
    ? prefersReducedMotion
      ? (MOMENTS[gameKey][0]?.ply ?? GAMES[gameKey].moves.length)
      : 0
    : GAMES[gameKey].start;
  document.querySelectorAll("[data-game]").forEach((button) => {
    const active = button.dataset.game === gameKey;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  renderBoard();
  if (autoplay) scheduleGamePlayback();
}

document.querySelectorAll("[data-game]").forEach((button) => {
  button.addEventListener("click", () => setGame(button.dataset.game, true));
});

playButton.addEventListener("click", playGame);
scrubber.addEventListener("input", (event) => {
  stopGame();
  activePly = Number(event.target.value);
  renderBoard();
});

function correlationY(value) {
  const top = 36;
  const bottom = 448;
  return bottom - ((value - 20) / 80) * (bottom - top);
}

function buildChart() {
  const svg = document.querySelector("#correlation-chart");
  const ns = "http://www.w3.org/2000/svg";
  const left = 62;
  const right = 22;
  const chartWidth = 1120 - left - right;
  const groupWidth = chartWidth / SOURCE_EVENTS.length;

  [20, 40, 60, 80, 100].forEach((value) => {
    const y = correlationY(value);
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", String(left));
    line.setAttribute("x2", String(1120 - right));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("class", "grid-line");
    svg.appendChild(line);

    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", "48");
    label.setAttribute("y", String(y + 4));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "grid-label");
    label.textContent = `${value}%`;
    svg.appendChild(label);
  });

  SOURCE_EVENTS.forEach((event, eventIndex) => {
    const center = left + groupWidth * eventIndex + groupWidth / 2;
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", String(center));
    label.setAttribute("y", "490");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "event-label");
    label.textContent = event.short;
    svg.appendChild(label);
  });

  games.forEach((game) => {
    const eventGames = games.filter((item) => item.event === game.event);
    const localIndex = eventGames.findIndex((item) => item.id === game.id);
    const center = left + groupWidth * game.eventIndex + groupWidth / 2;
    const spread = Math.min(groupWidth * 0.72, 112);
    const x =
      center +
      (localIndex - (eventGames.length - 1) / 2) *
        (spread / Math.max(eventGames.length - 1, 1));

    const connector = document.createElementNS(ns, "line");
    const allValues = [game.letscheck, game.sf15, game.sf7];
    connector.setAttribute("x1", String(x));
    connector.setAttribute("x2", String(x));
    connector.setAttribute("y1", String(correlationY(Math.max(...allValues))));
    connector.setAttribute("y2", String(correlationY(Math.min(...allValues))));
    connector.setAttribute("class", "game-connector");
    svg.appendChild(connector);

    Object.keys(METHOD_META).forEach((method) => {
      const point = document.createElementNS(ns, "circle");
      point.setAttribute("cx", String(x));
      point.setAttribute("cy", String(correlationY(game[method])));
      point.setAttribute("r", "3");
      point.setAttribute("fill", heatColor(game[method]));
      point.setAttribute("class", "game-point is-inactive ghost-point");
      point.dataset.game = game.id;
      point.dataset.method = method;
      point.dataset.x = String(x);
      attachTooltip(point, game, method);
      svg.appendChild(point);
    });

    const activePoint = document.createElementNS(ns, "circle");
    activePoint.setAttribute("cx", String(x));
    activePoint.setAttribute("cy", String(correlationY(game.letscheck)));
    activePoint.setAttribute("r", "4.4");
    activePoint.setAttribute("fill", heatColor(game.letscheck));
    activePoint.setAttribute(
      "class",
      `game-point is-active active-point${game.letscheck >= 95 ? " was-flagged" : ""}`,
    );
    activePoint.dataset.game = game.id;
    activePoint.dataset.x = String(x);
    attachTooltip(activePoint, game, "active");
    svg.appendChild(activePoint);
  });

  renderLegend();
}

function attachTooltip(element, game, method) {
  const tooltip = document.querySelector("#chart-tooltip");
  element.addEventListener("mouseenter", () => {
    const selectedMethod = method === "active" ? activeMethod : method;
    tooltip.innerHTML = `<strong>${game.eventName} · Round ${game.round}</strong><span>${METHOD_META[selectedMethod].name}: ${game[selectedMethod]}%</span>`;
    const chartShell = document.querySelector(".chart-shell");
    const shellRect = chartShell.getBoundingClientRect();
    const pointRect = element.getBoundingClientRect();
    tooltip.style.left = `${pointRect.left - shellRect.left + pointRect.width / 2}px`;
    tooltip.style.top = `${pointRect.top - shellRect.top}px`;
    tooltip.classList.add("is-visible");
  });
  element.addEventListener("mouseleave", () =>
    tooltip.classList.remove("is-visible"),
  );
}

function renderLegend() {
  const legend = document.querySelector("#chart-legend");
  legend.replaceChildren();
  [
    ["95–100", 97],
    ["80–94", 85],
    ["65–79", 70],
    ["<65", 55],
  ].forEach(([label, value]) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.style.color = heatColor(value);
    item.innerHTML = `<i class="legend-swatch"></i>${label}`;
    legend.appendChild(item);
  });
}

function heatColor(value) {
  if (value >= 95) return "#ef665d";
  if (value >= 80) return "#f2b84b";
  if (value >= 65) return "#8a713d";
  if (value >= 50) return "#3e625e";
  return "#243b39";
}

function heatTextColor(value) {
  return value >= 80 ? "#080b0b" : "#e5dfcf";
}

function buildHeatmap() {
  const heatmap = document.querySelector("#heatmap");
  heatmap.replaceChildren();

  const header = document.createElement("div");
  header.className = "heatmap-row heatmap-header";
  const headerLabel = document.createElement("div");
  headerLabel.className = "heatmap-label";
  headerLabel.textContent = "ROUND";
  header.appendChild(headerLabel);
  for (let round = 1; round <= 10; round += 1) {
    const roundLabel = document.createElement("div");
    roundLabel.className = "heatmap-round";
    roundLabel.textContent = String(round);
    header.appendChild(roundLabel);
  }
  heatmap.appendChild(header);

  SOURCE_EVENTS.forEach((event) => {
    const row = document.createElement("div");
    row.className = "heatmap-row";
    const label = document.createElement("div");
    label.className = "heatmap-label";
    label.textContent = event.short;
    row.appendChild(label);

    for (let round = 1; round <= 10; round += 1) {
      const game = games.find(
        (item) => item.event === event.key && item.round === round,
      );
      const cell = document.createElement("div");
      cell.className = `heatmap-cell${game ? "" : " is-empty"}`;
      cell.dataset.game = game ? game.id : "";
      cell.textContent = game ? "" : "—";
      if (game) {
        cell.dataset.event = event.name;
        cell.dataset.round = String(round);
      }
      row.appendChild(cell);
    }
    heatmap.appendChild(row);
  });
}

function updateHeatmap(method) {
  document.querySelectorAll(".heatmap-cell[data-game]:not(.is-empty)").forEach(
    (cell, index) => {
      const game = games.find((item) => item.id === cell.dataset.game);
      const value = game[method];
      cell.style.transitionDelay = `${Math.min(index * 24, 520)}ms`;
      cell.style.backgroundColor = heatColor(value);
      cell.style.color = heatTextColor(value);
      cell.style.borderColor =
        game.letscheck >= 95 ? "rgba(239, 102, 93, 0.5)" : "rgba(229, 223, 207, 0.12)";
      cell.textContent = `${value}%`;
      cell.title = `${cell.dataset.event}, round ${cell.dataset.round}: ${value}%`;
    },
  );
}

function updateMethodAnnotation(method) {
  const annotation = document.querySelector("#method-annotation");
  const copy = METHOD_ANNOTATIONS[method];
  annotation.classList.add("is-changing");
  window.setTimeout(() => {
    annotation.dataset.method = method;
    document.querySelector("#method-eyebrow").textContent = copy.eyebrow;
    document.querySelector("#method-headline").textContent = copy.headline;
    document.querySelector("#method-copy").textContent = copy.copy;
    annotation.classList.remove("is-changing");
  }, prefersReducedMotion ? 0 : 170);
}

function updateMethod(method) {
  activeMethod = method;
  const methods = Object.keys(METHOD_META);
  const methodIndex = methods.indexOf(method);

  document.querySelectorAll(".method-stop").forEach((button) => {
    const active = button.dataset.method === method;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-checked", String(active));
  });

  document.querySelector("#track-fill").style.width = `${methodIndex * 50}%`;
  document.querySelector(".measure-page").dataset.method = method;
  updateMethodAnnotation(method);

  document.querySelectorAll(".active-point").forEach((point, index) => {
    const game = games.find((item) => item.id === point.dataset.game);
    point.style.transitionDelay = `${Math.min(index * 13, 420)}ms`;
    point.setAttribute("cy", String(correlationY(game[method])));
    point.setAttribute("fill", heatColor(game[method]));
  });

  document.querySelectorAll(".ghost-point").forEach((point) => {
    point.classList.toggle("is-active", point.dataset.method === method);
    point.classList.toggle("is-inactive", point.dataset.method !== method);
  });

  updateHeatmap(method);
}

document.querySelectorAll(".method-stop").forEach((button) => {
  button.addEventListener("click", () => updateMethod(button.dataset.method));
});

function rankLabel(rank) {
  if (rank === 1) return "BEST";
  if (rank === 2) return "2ND";
  if (rank === 3) return "3RD";
  return `${rank}TH`;
}

function buildRankTable() {
  document.querySelectorAll(".rank-row").forEach((row) => {
    const positionIndex = Number(row.dataset.position);
    const position = depthExperiment.positions[positionIndex];
    const slots = row.querySelector(".rank-slots");
    const chip = document.createElement("div");
    chip.className = "rank-chip";
    chip.innerHTML = `<strong>${position.move.replace(/^\d+\./, "")}</strong><small></small>`;
    slots.appendChild(chip);
  });
  updateDepthTable();
}

function updateDepthTable() {
  document.querySelector("#depth-value").textContent = String(activeDepth);
  const slider = document.querySelector("#depth-slider");
  slider.style.setProperty(
    "--range-progress",
    `${((activeDepth - 10) / 15) * 100}%`,
  );

  document.querySelectorAll(".rank-row").forEach((row) => {
    const positionIndex = Number(row.dataset.position);
    const rank = depthExperiment.ranks[activeDepth][positionIndex];
    const chip = row.querySelector(".rank-chip");
    chip.dataset.rank = String(rank);
    chip.style.setProperty("--rank-left", `${((rank - 0.5) / 6) * 100}%`);
    chip.querySelector("small").textContent = rankLabel(rank);
  });

  const observation = document.querySelector("#rank-observation");
  if (activeDepth === 17) {
    observation.innerHTML = `<span>Depth 17 → 18</span><p><strong>15.e5</strong> falls from first to third; <strong>13.h3</strong> falls from second to fifth.</p>`;
  } else if (activeDepth === 18) {
    observation.innerHTML = `<span>One ply later</span><p><strong>22.Nd6+</strong> rises to best while <strong>15.e5</strong> drops two ranks.</p>`;
  } else {
    const ranks = depthExperiment.ranks[activeDepth];
    const summary = depthExperiment.positions
      .map(
        (position, index) =>
          `${position.move} ${rankLabel(ranks[index]).toLowerCase()}`,
      )
      .join(" · ");
    observation.innerHTML = `<span>Stockfish 7 · depth ${activeDepth}</span><p><strong>${summary}</strong></p>`;
  }
}

document.querySelector("#depth-slider").addEventListener("input", (event) => {
  activeDepth = Number(event.target.value);
  updateDepthTable();
});

function buildReactions() {
  const { thread } = window.CHESS_EVIDENCE;
  const field = document.querySelector(".clipping-field");
  const signal = field.querySelector(".thread-signal");
  const positions = ["a", "b", "c", "d", "e", "f", "g", "h"];

  thread.reactions.forEach((reaction, index) => {
    const crop = document.createElement("a");
    crop.className = `reddit-crop crop-${positions[index]}`;
    crop.href = `https://www.reddit.com/r/chess/comments/xr51fn/comment/${reaction.commentId}/`;
    crop.target = "_blank";
    crop.rel = "noreferrer";
    crop.setAttribute(
      "aria-label",
      `Open archived Reddit comment by ${reaction.author}`,
    );

    const image = document.createElement("img");
    image.src = `assets/reddit-comments/${reaction.commentId}.png`;
    image.alt = `Archived Reddit comment by ${reaction.author}: ${reaction.quote}`;
    crop.appendChild(image);
    field.insertBefore(crop, signal);
  });

  document.querySelector("#thread-comments").textContent = String(
    thread.comments,
  );
  document.querySelector("#thread-metrics").textContent =
    `comments · ${thread.score} score · ${thread.upvoteRatio}% upvoted`;
  document.querySelector(".thread-link").href = thread.url;
}

const exhibitPages = [...document.querySelectorAll(".exhibit-page")];
let activeExhibitPage = 0;

function showExhibitPage(nextPage) {
  const normalizedPage = (nextPage + exhibitPages.length) % exhibitPages.length;
  if (normalizedPage === activeExhibitPage) return;

  const previousIndex = activeExhibitPage;
  const pageDeck = document.querySelector(".page-deck");
  const isDepthHandoff = previousIndex === 2 && normalizedPage === 3;
  pageDeck.classList.toggle("is-depth-handoff", isDepthHandoff);

  const previousPage = exhibitPages[activeExhibitPage];
  previousPage.classList.remove("is-active");
  previousPage.classList.add("is-leaving-left");
  previousPage.setAttribute("aria-hidden", "true");
  window.setTimeout(
    () => previousPage.classList.remove("is-leaving-left"),
    320,
  );
  window.setTimeout(
    () => pageDeck.classList.remove("is-depth-handoff"),
    760,
  );

  activeExhibitPage = normalizedPage;
  pageDeck.dataset.activePage = String(activeExhibitPage);
  const activePage = exhibitPages[activeExhibitPage];
  activePage.classList.add("is-active");
  activePage.setAttribute("aria-hidden", "false");
  document.querySelector("#active-module").textContent =
    activePage.dataset.title;
  document.querySelector("#page-description-title").textContent =
    activePage.dataset.captionTitle;
  document.querySelector("#page-description-copy").textContent =
    activePage.dataset.caption;
  document.querySelector("#page-current").textContent = String(
    activeExhibitPage + 1,
  ).padStart(2, "0");

  document.querySelectorAll(".page-dot").forEach((dot, index) => {
    const active = index === activeExhibitPage;
    dot.classList.toggle("is-active", active);
    dot.setAttribute("aria-selected", String(active));
  });

  if (activeExhibitPage !== 0) {
    stopGame();
  } else if (activePly === 0) {
    scheduleGamePlayback(650);
  }
}

document
  .querySelector("#page-prev")
  .addEventListener("click", () => showExhibitPage(activeExhibitPage - 1));
document
  .querySelector("#page-next")
  .addEventListener("click", () => showExhibitPage(activeExhibitPage + 1));
document.querySelectorAll(".page-dot").forEach((dot) => {
  dot.addEventListener("click", () =>
    showExhibitPage(Number(dot.dataset.page)),
  );
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, button, a")) return;
  if (event.key === "ArrowLeft") showExhibitPage(activeExhibitPage - 1);
  if (event.key === "ArrowRight") showExhibitPage(activeExhibitPage + 1);
});

if (SINQUEFIELD_UCI.length !== SINQUEFIELD_SAN.length) {
  console.warn(
    "Game notation mismatch",
    SINQUEFIELD_UCI.length,
    SINQUEFIELD_SAN.length,
  );
}

renderBoard();
buildChart();
buildHeatmap();
buildRankTable();
buildReactions();
updateMethod("letscheck");

const requestedPageId =
  new URLSearchParams(window.location.search).get("page") ||
  window.location.hash.slice(1);
const requestedPage = exhibitPages.findIndex(
  (page) => page.id === requestedPageId,
);
if (requestedPage > 0) {
  document.documentElement.classList.add("is-direct-load");
  showExhibitPage(requestedPage);
  window.setTimeout(
    () => document.documentElement.classList.remove("is-direct-load"),
    20,
  );
} else {
  if (!hasRequestedPly) scheduleGamePlayback(650);
}
