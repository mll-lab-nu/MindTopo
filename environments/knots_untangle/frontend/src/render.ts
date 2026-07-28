import * as THREE from "three";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export class UntangleRenderer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xfcefe6);

    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    const aspect = width / height;
    this.viewSize = 13;
    this.camera = new THREE.OrthographicCamera(
      (-this.viewSize * aspect) / 2,
      (this.viewSize * aspect) / 2,
      this.viewSize / 2,
      -this.viewSize / 2,
      0.1,
      100,
    );
    this.camera.position.set(0, 20, 0);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(clamp(window.devicePixelRatio || 1, 1, 2));
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(5, 15, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    this.scene.add(key);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(30, 30),
      new THREE.MeshStandardMaterial({ color: 0xeaddcf, roughness: 0.8 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    ground.receiveShadow = true;
    this.scene.add(ground);

    this.boardGroup = new THREE.Group();
    this.scene.add(this.boardGroup);

    this.ropeMeshes = [];
    this.plugMeshes = [];
    this.curves = [];
    this.ropeSegments = 16;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();

    this.animationsEnabled = true;
    this.game = null;

    this._bindResize = () => this.resize();
    window.addEventListener("resize", this._bindResize);
    this._startRenderLoop();
  }

  setAnimationEnabled(enabled) {
    const next = Boolean(enabled);
    if (this.animationsEnabled === next) {
      if (!next) this.renderOnce();
      return;
    }
    this.animationsEnabled = next;
    if (next) {
      this._startRenderLoop();
    } else {
      this._stopRenderLoop();
      this.renderOnce();
    }
  }

  setup(game) {
    this.game = game;
    this.ropeSegments = game.config.ropeSegments;
    const boardSpan = (game.gridSize - 1) * game.config.holeSpacing;
    this.viewSize = Math.max(14, boardSpan + 6);
    this.resize();
    this._clearBoard();
    this._buildHoles(game);
    this._buildRopes(game);
    this.renderOnce();
  }

  syncState() {
    if (!this.game) return;
    this._updateRopeMeshes();
    this.renderOnce();
  }

  pickEndpoint(clientX, clientY) {
    this._updatePointer(clientX, clientY);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.plugMeshes, false);
    if (hits.length) return hits[0].object.userData?.endpoint || null;
    return (
      this._pickEndpointByWorldDistance(clientX, clientY) ||
      this._pickEndpointByScreenDistance(clientX, clientY)
    );
  }

  projectPointerToPlane(clientX, clientY, y) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const width = Math.max(rect.width, 1);
    const height = Math.max(rect.height, 1);
    const nx = clamp((clientX - rect.left) / width, 0, 1);
    const ny = clamp((clientY - rect.top) / height, 0, 1);
    const x = this.camera.left + nx * (this.camera.right - this.camera.left);
    const z = this.camera.bottom + ny * (this.camera.top - this.camera.bottom);
    return new THREE.Vector3(x, y, z);
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    const aspect = width / height;
    this.camera.left = (-this.viewSize * aspect) / 2;
    this.camera.right = (this.viewSize * aspect) / 2;
    this.camera.top = this.viewSize / 2;
    this.camera.bottom = -this.viewSize / 2;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.renderOnce();
  }

  dispose() {
    window.removeEventListener("resize", this._bindResize);
    this._stopRenderLoop();
    this._clearBoard();
    this.renderer.dispose();
  }

  renderOnce() {
    this.renderer.render(this.scene, this.camera);
  }

  _startRenderLoop() {
    if (this._loopId) return;
    this._loopId = requestAnimationFrame(() => this._renderLoop());
  }

  _stopRenderLoop() {
    if (!this._loopId) return;
    cancelAnimationFrame(this._loopId);
    this._loopId = null;
  }

  _renderLoop() {
    this._loopId = null;
    if (!this.animationsEnabled) return;
    this._startRenderLoop();
    if (this.game && this.animationsEnabled) {
      this.game.tickPhysics(1 / 60);
      this._updateRopeMeshes();
    }
    this.renderOnce();
  }

  _clearBoard() {
    while (this.boardGroup.children.length) {
      const child = this.boardGroup.children.pop();
      if (child.geometry) child.geometry.dispose();
      if (Array.isArray(child.material)) {
        for (const m of child.material) {
          if (m.map) m.map.dispose();
          m.dispose();
        }
      } else if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
      this.boardGroup.remove(child);
    }
    for (const m of this.ropeMeshes) {
      if (m.geometry) m.geometry.dispose();
      if (m.material) m.material.dispose();
      this.scene.remove(m);
    }
    for (const p of this.plugMeshes) {
      if (p.geometry) p.geometry.dispose();
      if (p.material) p.material.dispose();
      this.scene.remove(p);
    }
    this.ropeMeshes = [];
    this.plugMeshes = [];
    this.curves = [];
  }

  _buildHoles(game) {
    const rimGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.1, 32);
    const rimMat = new THREE.MeshBasicMaterial({ color: 0xd7ccc8 });
    const innerGeo = new THREE.CircleGeometry(0.3, 32);
    const innerMat = new THREE.MeshBasicMaterial({ color: 0x8d6e63 });
    for (const hole of game.holes) {
      const rim = new THREE.Mesh(rimGeo, rimMat);
      rim.position.copy(hole.pos);
      this.boardGroup.add(rim);

      const inner = new THREE.Mesh(innerGeo, innerMat);
      inner.rotation.x = -Math.PI / 2;
      inner.position.set(hole.pos.x, 0.06, hole.pos.z);
      this.boardGroup.add(inner);
    }
    this._buildAxisLabels(game);
  }

  _buildAxisLabels(game) {
    const spacing = game.config.holeSpacing;
    const offset = ((game.gridSize - 1) * spacing) / 2;
    const labelGap = spacing * 0.7;
    const labelSize = Math.max(spacing * 0.55, 1.0);
    for (let i = 0; i < game.gridSize; i += 1) {
      const coord = i * spacing - offset;
      const rowLabel = this._makeLabelSprite(String(i), labelSize);
      rowLabel.position.set(coord, 0.2, -offset - labelGap);
      this.boardGroup.add(rowLabel);

      const colLabel = this._makeLabelSprite(String(i), labelSize);
      colLabel.position.set(-offset - labelGap, 0.2, coord);
      this.boardGroup.add(colLabel);
    }
    const rowAxisName = this._makeLabelSprite("row", labelSize * 0.8);
    rowAxisName.position.set(offset + labelGap, 0.2, -offset - labelGap);
    this.boardGroup.add(rowAxisName);
    const colAxisName = this._makeLabelSprite("col", labelSize * 0.8);
    colAxisName.position.set(-offset - labelGap, 0.2, offset + labelGap);
    this.boardGroup.add(colAxisName);
  }

  _makeLabelSprite(text, worldHeight) {
    const fontPx = 96;
    const padding = 12;
    const measureCanvas = document.createElement("canvas");
    const measureCtx = measureCanvas.getContext("2d");
    measureCtx.font = `bold ${fontPx}px sans-serif`;
    const textWidth = Math.ceil(measureCtx.measureText(text).width);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(textWidth + padding * 2, fontPx + padding * 2);
    canvas.height = fontPx + padding * 2;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = `bold ${fontPx}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(252, 239, 230, 0.95)";
    ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
    ctx.fillStyle = "#3e2723";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    const aspect = canvas.width / canvas.height;
    sprite.scale.set(worldHeight * aspect, worldHeight, 1);
    sprite.renderOrder = 10;
    return sprite;
  }

  _buildRopes(game) {
    for (let ropeIdx = 0; ropeIdx < game.ropes.length; ropeIdx += 1) {
      const rope = game.ropes[ropeIdx];
      const curve = new THREE.CatmullRomCurve3(rope.visualPositions);
      curve.tension = 0.5;
      this.curves.push(curve);

      const tubeGeo = new THREE.TubeGeometry(
        curve,
        this.ropeSegments * 3,
        game.config.ropeRadius,
        8,
        false,
      );
      const mat = new THREE.MeshStandardMaterial({
        color: rope.color,
        roughness: 0.5,
        metalness: 0.1,
      });
      const mesh = new THREE.Mesh(tubeGeo, mat);
      mesh.castShadow = true;
      this.scene.add(mesh);
      this.ropeMeshes.push(mesh);

      const plugGeo = new THREE.CylinderGeometry(0.3, 0.35, 0.5, 16);
      const plugMatStart = new THREE.MeshStandardMaterial({
        color: rope.color,
        roughness: 0.4,
      });
      const plugMatEnd = new THREE.MeshStandardMaterial({
        color: rope.color,
        roughness: 0.4,
      });
      const startPlug = new THREE.Mesh(plugGeo, plugMatStart);
      const endPlug = new THREE.Mesh(plugGeo, plugMatEnd);
      startPlug.userData.endpoint = { ropeIdx, isStartEnd: true };
      endPlug.userData.endpoint = { ropeIdx, isStartEnd: false };
      startPlug.userData.rope = rope;
      endPlug.userData.rope = rope;
      startPlug.userData.isStart = true;
      endPlug.userData.isStart = false;
      startPlug.position.copy(rope.particles[0].pos);
      endPlug.position.copy(rope.particles[rope.particles.length - 1].pos);
      this.scene.add(startPlug);
      this.scene.add(endPlug);
      this.plugMeshes.push(startPlug, endPlug);
    }
  }

  _updateRopeMeshes() {
    for (let i = 0; i < this.game.ropes.length; i += 1) {
      const rope = this.game.ropes[i];
      const mesh = this.ropeMeshes[i];
      const curve = this.curves[i];
      if (!mesh || !curve) continue;
      curve.points = rope.visualPositions;
      if (mesh.geometry) mesh.geometry.dispose();
      mesh.geometry = new THREE.TubeGeometry(
        curve,
        this.ropeSegments * 3,
        this.game.config.ropeRadius,
        8,
        false,
      );

      const startPlug = this.plugMeshes[i * 2];
      const endPlug = this.plugMeshes[i * 2 + 1];
      if (startPlug) startPlug.position.copy(rope.particles[0].pos);
      if (endPlug) {
        endPlug.position.copy(rope.particles[rope.particles.length - 1].pos);
      }
    }
  }

  _updatePointer(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const width = Math.max(rect.width, 1);
    const height = Math.max(rect.height, 1);
    this.pointer.x = ((clientX - rect.left) / width) * 2 - 1;
    this.pointer.y = -(((clientY - rect.top) / height) * 2 - 1);
  }

  _pickEndpointByScreenDistance(clientX, clientY) {
    if (!this.game) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const hitRadiusPx = 30;
    let best = null;
    let bestDistSq = hitRadiusPx * hitRadiusPx;

    for (let ropeIdx = 0; ropeIdx < this.game.ropes.length; ropeIdx += 1) {
      const rope = this.game.ropes[ropeIdx];
      const endpointPairs = [
        { isStartEnd: true, pos: rope.particles[0].pos },
        { isStartEnd: false, pos: rope.particles[rope.particles.length - 1].pos },
      ];

      for (const endpoint of endpointPairs) {
        const projected = endpoint.pos.clone().project(this.camera);
        const screenX = rect.left + ((projected.x + 1) * rect.width) / 2;
        const screenY = rect.top + ((1 - projected.y) * rect.height) / 2;
        const dx = clientX - screenX;
        const dy = clientY - screenY;
        const distSq = dx * dx + dy * dy;
        if (distSq <= bestDistSq) {
          bestDistSq = distSq;
          best = { ropeIdx, isStartEnd: endpoint.isStartEnd };
        }
      }
    }

    return best;
  }

  _pickEndpointByWorldDistance(clientX, clientY) {
    if (!this.game) return null;
    const groundY = this.game.config.ropeRadius;
    const point = this.projectPointerToPlane(clientX, clientY, groundY);
    if (!point) return null;

    const hitRadius = Math.max(this.game.config.ropeRadius * 2.4, this.game.config.holeSpacing * 0.18);
    let best = null;
    let bestDistSq = hitRadius * hitRadius;
    for (let ropeIdx = 0; ropeIdx < this.game.ropes.length; ropeIdx += 1) {
      const rope = this.game.ropes[ropeIdx];
      const endpointPairs = [
        { isStartEnd: true, pos: rope.particles[0].pos },
        { isStartEnd: false, pos: rope.particles[rope.particles.length - 1].pos },
      ];
      for (const endpoint of endpointPairs) {
        const dx = endpoint.pos.x - point.x;
        const dz = endpoint.pos.z - point.z;
        const distSq = dx * dx + dz * dz;
        if (distSq <= bestDistSq) {
          bestDistSq = distSq;
          best = { ropeIdx, isStartEnd: endpoint.isStartEnd };
        }
      }
    }
    return best;
  }
}
