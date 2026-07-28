import * as THREE from "https://unpkg.com/three@0.161.0/build/three.module.js";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

class LabelOverlay {
  constructor(container) {
    this.container = container;
    this.root = document.createElement("div");
    Object.assign(this.root.style, {
      position: "absolute",
      inset: "0",
      pointerEvents: "none",
      zIndex: "2"
    });
    this.container.appendChild(this.root);
    this.indexLabels = new Map();

    this.catLabel = document.createElement("div");
    this.catLabel.textContent = "CAT";
    Object.assign(this.catLabel.style, {
      position: "absolute",
      padding: "3px 10px",
      borderRadius: "999px",
      background: "rgba(255, 166, 38, 0.94)",
      color: "#ffffff",
      fontFamily: "\"Avenir Next\", \"Trebuchet MS\", sans-serif",
      fontWeight: "800",
      fontSize: "15px",
      letterSpacing: "0.06em",
      transform: "translate(-50%, -50%)",
      boxShadow: "0 8px 20px rgba(214, 116, 0, 0.26)"
    });
    this.root.appendChild(this.catLabel);
  }

  ensureIndexLabel(index) {
    if (this.indexLabels.has(index)) {
      return this.indexLabels.get(index);
    }
    const label = document.createElement("div");
    label.textContent = String(index);
    Object.assign(label.style, {
      position: "absolute",
      width: "52px",
      height: "52px",
      borderRadius: "999px",
      transform: "translate(-50%, -50%)",
      display: "grid",
      placeItems: "center",
      fontFamily: "\"Avenir Next\", \"Trebuchet MS\", sans-serif",
      fontWeight: "700",
      fontSize: "24px",
      boxShadow: "0 4px 14px rgba(20, 28, 36, 0.08)"
    });
    this.root.appendChild(label);
    this.indexLabels.set(index, label);
    return label;
  }

  setIndexLabel(index, screenPosition, tone) {
    const label = this.ensureIndexLabel(index);
    label.style.left = `${screenPosition.x}px`;
    label.style.top = `${screenPosition.y}px`;
    if (tone === "blocked") {
      label.style.background = "rgba(255, 255, 255, 0.18)";
      label.style.border = "1px solid rgba(255, 255, 255, 0.18)";
      label.style.color = "#f7fafc";
    } else if (tone === "cat") {
      label.style.background = "rgba(255, 255, 255, 0.92)";
      label.style.border = "1px solid rgba(255, 166, 38, 0.32)";
      label.style.color = "#91500c";
    } else {
      label.style.background = "rgba(255, 255, 255, 0.96)";
      label.style.border = "1px solid rgba(33, 48, 59, 0.08)";
      label.style.color = "#314556";
    }
  }

  setCatLabel(screenPosition) {
    this.catLabel.style.display = "block";
    this.catLabel.style.left = `${screenPosition.x}px`;
    this.catLabel.style.top = `${screenPosition.y + 42}px`;
  }

  clearIndexLabels() {
    for (const label of this.indexLabels.values()) {
      label.remove();
    }
    this.indexLabels.clear();
  }
}

export class ChatNoirRenderer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#faf8f2");

    this.camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 100);
    this.camera.position.set(0, 0, 20);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(clamp(window.devicePixelRatio || 1, 1, 2));
    this.renderer.setSize(Math.max(container.clientWidth, 1), Math.max(container.clientHeight, 1));
    container.appendChild(this.renderer.domElement);

    this.labels = new LabelOverlay(container);
    this.cellEntries = new Map();
    this.cellOrder = [];
    this._projection = new THREE.Vector3();
    this._targetLabelMap = new Map();

    const ambient = new THREE.AmbientLight("#ffffff", 1.0);
    this.scene.add(ambient);

    this.boardGroup = new THREE.Group();
    this.scene.add(this.boardGroup);

    this.haloMesh = new THREE.Mesh(
      new THREE.RingGeometry(0.76, 0.98, 40),
      new THREE.MeshBasicMaterial({ color: "#f59a2a", transparent: true, opacity: 0.0 })
    );
    this.scene.add(this.haloMesh);

    this._boundResize = () => this.resize();
    window.addEventListener("resize", this._boundResize);

    this.resize();
    this.renderNow();
  }

  setup(cells) {
    while (this.boardGroup.children.length) {
      const child = this.boardGroup.children.pop();
      if (!child) {
        continue;
      }
      if (child.geometry) {
        child.geometry.dispose();
      }
      if (Array.isArray(child.material)) {
        for (const material of child.material) {
          material.dispose();
        }
      } else if (child.material) {
        child.material.dispose();
      }
      this.boardGroup.remove(child);
    }

    this.cellEntries.clear();
    this.cellOrder = cells.slice();
    this._targetLabelMap.clear();
    this.labels.clearIndexLabels();

    const cardMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(17.2, 14.2),
      new THREE.MeshBasicMaterial({ color: "#faf8f2" })
    );
    cardMesh.position.set(0, 0, -0.8);
    this.boardGroup.add(cardMesh);

    for (const cell of cells) {
      const ring = new THREE.Mesh(
        new THREE.CircleGeometry(cell.radius * 1.08, 48),
        new THREE.MeshBasicMaterial({
          color: cell.isBoundary ? "#d9e4f3" : "#e8e1d5"
        })
      );
      ring.position.set(cell.x, cell.y, -0.05);
      this.boardGroup.add(ring);

      const disk = new THREE.Mesh(
        new THREE.CircleGeometry(cell.radius, 48),
        new THREE.MeshBasicMaterial({ color: "#ece8df" })
      );
      disk.position.set(cell.x, cell.y, 0);
      this.boardGroup.add(disk);

      this.cellEntries.set(cell.index, { disk, ring, cell });
    }

    this._fitCamera();
    this.renderNow();
  }

  syncState(state) {
    const blocked = new Set(state.blockedIndices || []);
    const catIndex = state.catIndex;
    const lastBlocked = state.lastTurn?.blockedCellIndex ?? null;

    for (const cell of this.cellOrder) {
      const entry = this.cellEntries.get(cell.index);
      if (!entry) {
        continue;
      }
      const isBlocked = blocked.has(cell.index);
      const isCat = cell.index === catIndex;

      let fill = "#ece8df";
      let ring = cell.isBoundary ? "#cad8e9" : "#dfd7ca";
      if (isBlocked) {
        fill = "#242629";
        ring = "#50555a";
      }
      if (isCat) {
        fill = "#f4af32";
        ring = "#e28f12";
      }
      if (lastBlocked === cell.index && !isCat) {
        ring = "#db5d48";
      }

      entry.disk.material.color.set(fill);
      entry.ring.material.color.set(ring);
      this._targetLabelMap.set(cell.index, isCat ? "cat" : isBlocked ? "blocked" : "open");
    }

    if (catIndex !== null && catIndex !== undefined && this.cellEntries.has(catIndex)) {
      const entry = this.cellEntries.get(catIndex);
      this.haloMesh.position.set(entry.cell.x, entry.cell.y, 0.04);
      this.haloMesh.material.opacity = 0.92;
    } else {
      this.haloMesh.material.opacity = 0.0;
    }

    this._updateLabels(state.catIndex);
    this.renderNow();
  }

  _fitCamera() {
    if (!this.cellOrder.length) {
      return;
    }
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const cell of this.cellOrder) {
      minX = Math.min(minX, cell.x - cell.radius * 1.4);
      maxX = Math.max(maxX, cell.x + cell.radius * 1.4);
      minY = Math.min(minY, cell.y - cell.radius * 1.5);
      maxY = Math.max(maxY, cell.y + cell.radius * 1.5);
    }
    const width = maxX - minX;
    const height = maxY - minY;
    const aspect = Math.max(this.container.clientWidth, 1) / Math.max(this.container.clientHeight, 1);
    const halfWidth = Math.max(width * 0.6, (height * aspect) * 0.58, 7.6);
    const halfHeight = Math.max(height * 0.6, (width / aspect) * 0.58, 6.4);

    this.camera.left = -halfWidth;
    this.camera.right = halfWidth;
    this.camera.top = halfHeight;
    this.camera.bottom = -halfHeight;
    this.camera.position.set(0, 0, 20);
    this.camera.updateProjectionMatrix();
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(clamp(window.devicePixelRatio || 1, 1, 2));
    this._fitCamera();
    this._updateLabels(null);
    this.renderNow();
  }

  _updateLabels(catIndex) {
    const bounds = this.renderer.domElement.getBoundingClientRect();
    const width = Math.max(bounds.width, 1);
    const height = Math.max(bounds.height, 1);

    for (const cell of this.cellOrder) {
      this._projection.set(cell.x, cell.y, 0.1).project(this.camera);
      const screenPosition = {
        x: ((this._projection.x + 1) / 2) * width,
        y: ((1 - this._projection.y) / 2) * height
      };
      const tone = this._targetLabelMap.get(cell.index) || "open";
      this.labels.setIndexLabel(cell.index, screenPosition, tone);
      if (cell.index === catIndex) {
        this.labels.setCatLabel(screenPosition);
      }
    }
  }

  renderNow() {
    this.renderer.render(this.scene, this.camera);
  }
}
