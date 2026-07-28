function colorFor(blockId, paletteById) {
  return paletteById.get(blockId)?.color ?? "#5d6b7a";
}

function labelFor(blockId, paletteById) {
  return paletteById.get(blockId)?.label ?? blockId;
}

class Swap2DBoardView {
  constructor(container, { interactive = false } = {}) {
    this.container = container;
    this.interactive = Boolean(interactive);
    this.gridRows = 3;
    this.gridCols = 3;
    this.paletteById = new Map();
    this.cells = [];

    Object.assign(this.container.style, {
      display: "grid",
      placeItems: "center",
      padding: "24px",
      background: "#ffffff"
    });

    this.board = document.createElement("div");
    Object.assign(this.board.style, {
      display: "grid",
      gap: "6px",
      width: "min(92%, 680px)",
      height: "min(88%, 360px)",
      maxHeight: "430px",
      alignContent: "stretch",
      justifyContent: "stretch",
      padding: "0",
      borderRadius: "18px",
      border: "0",
      background: "#ffffff",
      boxShadow: "none"
    });
    this.container.appendChild(this.board);
  }

  setup({ gridRows, gridCols, blockPalette }) {
    this.gridRows = gridRows;
    this.gridCols = gridCols;
    this.paletteById = new Map((blockPalette || []).map((block) => [block.id, block]));
    this.cells = [];
    this.board.textContent = "";
    this.board.style.gridTemplateColumns = `repeat(${gridCols}, minmax(72px, 1fr))`;
    this.board.style.gridTemplateRows = `repeat(${gridRows}, minmax(58px, 1fr))`;

    const cellCount = gridRows * gridCols;
    for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
      const row = Math.floor(cellIndex / gridCols);
      const col = cellIndex % gridCols;
      const cell = document.createElement("div");
      const indexBadge = document.createElement("div");
      const coordBadge = document.createElement("div");
      const blockLabel = document.createElement("div");

      cell.dataset.cellIndex = String(cellIndex);
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      Object.assign(cell.style, {
        position: "relative",
        minWidth: "72px",
        minHeight: "58px",
        display: "grid",
        placeItems: "center",
        borderRadius: "12px",
        border: "2px solid transparent",
        background: "#f8fbff",
        overflow: "hidden",
        boxShadow: "0 8px 16px rgba(40, 62, 84, 0.08)",
        transition: "transform 120ms ease, box-shadow 120ms ease"
      });

      indexBadge.textContent = String(cellIndex);
      Object.assign(indexBadge.style, {
        position: "absolute",
        top: "6px",
        left: "7px",
        minWidth: "24px",
        padding: "2px 6px",
        borderRadius: "999px",
        background: "rgba(28, 45, 64, 0.86)",
        color: "#ffffff",
        fontFamily: "\"Avenir Next\", \"Trebuchet MS\", sans-serif",
        fontSize: "12px",
        fontWeight: "700",
        lineHeight: "1.2",
        textAlign: "center"
      });

      coordBadge.textContent = `r${row} c${col}`;
      Object.assign(coordBadge.style, {
        position: "absolute",
        right: "7px",
        bottom: "6px",
        color: "rgba(32, 49, 66, 0.62)",
        fontFamily: "\"Avenir Next\", \"Trebuchet MS\", sans-serif",
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "0"
      });

      Object.assign(blockLabel.style, {
        color: "#ffffff",
        fontFamily: "\"Avenir Next\", \"Trebuchet MS\", sans-serif",
        fontSize: "clamp(24px, 4.5vw, 42px)",
        fontWeight: "800",
        lineHeight: "1",
        textShadow: "0 1px 2px rgba(0, 0, 0, 0.25)"
      });

      cell.appendChild(indexBadge);
      cell.appendChild(blockLabel);
      cell.appendChild(coordBadge);
      this.board.appendChild(cell);
      this.cells.push({ cell, blockLabel });
    }
  }

  syncArrangement(arrangement, lastMove = null) {
    for (let cellIndex = 0; cellIndex < this.cells.length; cellIndex += 1) {
      const entry = this.cells[cellIndex];
      const blockId = arrangement[cellIndex];
      const isBlank = blockId === "_";
      const movedHere = lastMove && lastMove.blankIndexBefore === cellIndex;
      const movedFrom = lastMove && lastMove.blankIndexAfter === cellIndex;
      entry.cell.style.background = isBlank ? "#ffffff" : colorFor(blockId, this.paletteById);
      entry.cell.style.borderStyle = isBlank ? "dashed" : "solid";
      entry.cell.style.borderColor = isBlank ? "rgba(66, 91, 118, 0.28)" : "transparent";
      entry.cell.style.cursor = this.interactive && !isBlank ? "pointer" : "default";
      entry.cell.tabIndex = this.interactive && !isBlank ? 0 : -1;
      entry.cell.style.transform = movedHere || movedFrom ? "translateY(-1px)" : "none";
      entry.cell.style.boxShadow = movedHere || movedFrom
        ? "0 12px 22px rgba(36, 64, 92, 0.16)"
        : "0 8px 16px rgba(40, 62, 84, 0.08)";
      entry.blockLabel.textContent = isBlank ? "" : labelFor(blockId, this.paletteById);
    }
  }
}

export class Swap2DPuzzleRenderer {
  constructor({ currentRoot, goalRoot }) {
    this.currentView = new Swap2DBoardView(currentRoot, { interactive: true });
    this.goalView = new Swap2DBoardView(goalRoot);
    this.animationsEnabled = true;
  }

  setAnimationEnabled(enabled) {
    this.animationsEnabled = Boolean(enabled);
  }

  setup(config) {
    const blockPalette = config.selectedBlockIds.map((blockId) => ({ ...config.blockById.get(blockId) }));
    const payload = {
      gridRows: config.gridRows,
      gridCols: config.gridCols,
      blockPalette
    };
    this.currentView.setup(payload);
    this.goalView.setup(payload);
  }

  async animateSwap(_move) {
    if (!this.animationsEnabled) {
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 90));
  }

  syncState(state) {
    this.currentView.syncArrangement(state.currentArrangement, state.lastMove);
    this.goalView.syncArrangement(state.goalArrangement, null);
  }
}
