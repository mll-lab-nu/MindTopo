/**
 * sheep-placer.js
 * Sheep placement with inside/outside labeling and 3D sheep model generation
 */

import * as THREE from 'three';
import { RegionAnalyzer } from './region-analyzer.js';

export const SHEEP_COLORS = {
  RED: 0xff3333,
  BLUE: 0x3366ff,
  GREEN: 0x33cc33,
  YELLOW: 0xffcc00,
  WHITE: 0xffffff,
  PURPLE: 0x9933ff,
  ORANGE: 0xff8800
};

export class SheepPlacer {

  /**
   * Generate sheep positions
   */
  static placeSheep(options = {}) {
    const {
      numSheep = 5,
      fenceLayers = [],
      sceneRadius = 10,
      minDistance = 2.0,
      minFenceDist = 1.5,
      insideRatio = 0.5,
      colors = ['WHITE']
    } = options;

    const sheep = [];
    const numInside = Math.round(numSheep * insideRatio);
    const maxAttempts = 800;
    let nextId = 1;

    // Round-robin the target polygon across nesting layers so the inner
    // rings actually get sheep. Uniform sampling in fenceLayers[0] biases
    // by area, leaving small inner rings consistently empty.
    for (let i = 0; i < numInside; i++) {
      const target = fenceLayers[i % fenceLayers.length] || [];
      let placed = false;
      for (let attempt = 0; attempt < maxAttempts && !placed; attempt++) {
        const pos = this._randomPointInPolygon(target, sceneRadius);
        if (pos &&
            this._minDistanceOk(pos, sheep, minDistance) &&
            this._minDistanceToFenceOk(pos, fenceLayers, minFenceDist)) {
          const nestingDepth = RegionAnalyzer.nestingDepth(pos, fenceLayers);
          if (nestingDepth > 0) {
            sheep.push({
              ...pos,
              isInside: true,
              nestingDepth,
              regionLabel: RegionAnalyzer.regionLabel(pos, fenceLayers),
              color: colors[sheep.length % colors.length],
              id: nextId++
            });
            placed = true;
          }
        }
      }
      // Fallback widens to the outermost polygon with relaxed spacing — if
      // the targeted inner ring was too tight, we'd rather place the sheep
      // somewhere in the outer band than drop it entirely.
      if (!placed && fenceLayers.length > 0) {
        const outermost = fenceLayers[0];
        let fallbackPos = null;
        for (let attempt = 0; attempt < 400; attempt++) {
          const candidate = this._randomPointInPolygon(outermost, sceneRadius);
          if (candidate &&
              this._minDistanceOk(candidate, sheep, minDistance * 0.7) &&
              this._minDistanceToFenceOk(candidate, fenceLayers, minFenceDist * 0.95)) {
            fallbackPos = candidate;
            break;
          }
        }
        if (fallbackPos) {
          sheep.push({
            x: fallbackPos.x,
            z: fallbackPos.z,
            isInside: true,
            nestingDepth: RegionAnalyzer.nestingDepth(fallbackPos, fenceLayers),
            regionLabel: RegionAnalyzer.regionLabel(fallbackPos, fenceLayers),
            color: colors[sheep.length % colors.length],
            id: nextId++
          });
        }
        // else: skip — scene cannot fit this sheep without overlap
      }
    }

    // Place sheep outside fences — keep them close to the outermost fence
    const outerFence = fenceLayers[0] || [];
    let fMinX = -5, fMaxX = 5, fMinZ = -5, fMaxZ = 5;
    if (outerFence.length > 0) {
      fMinX = Infinity; fMaxX = -Infinity; fMinZ = Infinity; fMaxZ = -Infinity;
      for (const v of outerFence) {
        fMinX = Math.min(fMinX, v.x); fMaxX = Math.max(fMaxX, v.x);
        fMinZ = Math.min(fMinZ, v.z); fMaxZ = Math.max(fMaxZ, v.z);
      }
    }
    const outsideMargin = 3.5;

    for (let i = numInside; i < numSheep; i++) {
      let placed = false;
      for (let attempt = 0; attempt < maxAttempts && !placed; attempt++) {
        const pos = {
          x: fMinX - outsideMargin + Math.random() * (fMaxX - fMinX + outsideMargin * 2),
          z: fMinZ - outsideMargin + Math.random() * (fMaxZ - fMinZ + outsideMargin * 2)
        };
        const nestingDepth = RegionAnalyzer.nestingDepth(pos, fenceLayers);
        if (nestingDepth === 0 &&
            this._minDistanceOk(pos, sheep, minDistance) &&
            this._minDistanceToFenceOk(pos, fenceLayers, minFenceDist)) {
          sheep.push({
            ...pos,
            isInside: false,
            nestingDepth: 0,
            regionLabel: RegionAnalyzer.regionLabel(pos, fenceLayers),
            color: colors[sheep.length % colors.length],
            id: nextId++
          });
          placed = true;
        }
      }
      // Fallback: search just outside fence perimeter with fence distance check
      if (!placed) {
        let fallbackPos = null;
        for (let attempt = 0; attempt < 400; attempt++) {
          const angle = Math.random() * Math.PI * 2;
          const fenceExtent = Math.max(fMaxX - fMinX, fMaxZ - fMinZ) / 2;
          const dist = fenceExtent + minFenceDist + Math.random() * 2.0;
          const candidate = {
            x: Math.cos(angle) * dist,
            z: Math.sin(angle) * dist
          };
          if (RegionAnalyzer.nestingDepth(candidate, fenceLayers) === 0 &&
              this._minDistanceOk(candidate, sheep, minDistance * 0.7) &&
              this._minDistanceToFenceOk(candidate, fenceLayers, minFenceDist * 0.95)) {
            fallbackPos = candidate;
            break;
          }
        }
        if (!fallbackPos) {
          // Last resort: place at safe distance and push away from fences
          const fenceExtent = Math.max(fMaxX - fMinX, fMaxZ - fMinZ) / 2;
          const angle = i * 2.4 + 0.7;
          fallbackPos = {
            x: (fenceExtent + minFenceDist + 1.0) * Math.cos(angle),
            z: (fenceExtent + minFenceDist + 1.0) * Math.sin(angle)
          };
          fallbackPos = this._pushAwayFromFences(fallbackPos, fenceLayers, minFenceDist);
        }
        if (
          fallbackPos &&
          RegionAnalyzer.nestingDepth(fallbackPos, fenceLayers) === 0 &&
          this._minDistanceOk(fallbackPos, sheep, minDistance * 0.85) &&
          this._minDistanceToFenceOk(fallbackPos, fenceLayers, minFenceDist * 0.95)
        ) {
          sheep.push({
            ...fallbackPos,
            isInside: false,
            nestingDepth: 0,
            regionLabel: '0'.repeat(fenceLayers.length),
            color: colors[sheep.length % colors.length],
            id: nextId++
          });
        }
      }
    }

    return sheep;
  }

  /**
   * Create a low-poly 3D sheep mesh
   */
  static createSheepMesh(colorName = 'WHITE', id = 0) {
    // All sheep render identically white — color is unused as a visual cue
    // so counting / identification tasks rely solely on the numeric ID badge.
    colorName = 'WHITE';
    const group = new THREE.Group();
    const bodyColor = 0xf0ece0;
    const headColor = 0x333333;
    const legColor = 0x222222;
    const markerColor = SHEEP_COLORS.WHITE;

    // Body - fluffy ellipsoid
    const bodyGeom = new THREE.SphereGeometry(0.35, 12, 10);
    bodyGeom.scale(1.3, 1, 1);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: bodyColor,
      roughness: 1.0,
      metalness: 0
    });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    body.position.y = 0.45;
    body.castShadow = true;
    group.add(body);

    // Fluffy bumps on body
    const bumpGeom = new THREE.SphereGeometry(0.15, 8, 6);
    const bumpMat = new THREE.MeshStandardMaterial({ color: bodyColor, roughness: 1.0 });
    const bumpPositions = [
      [0.2, 0.6, 0.15], [-0.15, 0.65, -0.1], [0.05, 0.55, -0.2],
      [-0.25, 0.5, 0.1], [0.3, 0.5, -0.05], [0, 0.7, 0]
    ];
    for (const [bx, by, bz] of bumpPositions) {
      const bump = new THREE.Mesh(bumpGeom, bumpMat);
      bump.position.set(bx, by, bz);
      bump.castShadow = true;
      group.add(bump);
    }

    // Head
    const headGeom = new THREE.SphereGeometry(0.18, 10, 8);
    headGeom.scale(1, 1.1, 1.2);
    const headMat = new THREE.MeshStandardMaterial({ color: headColor, roughness: 0.8 });
    const head = new THREE.Mesh(headGeom, headMat);
    head.position.set(0.45, 0.55, 0);
    head.castShadow = true;
    group.add(head);

    // Ears
    const earGeom = new THREE.SphereGeometry(0.06, 6, 4);
    earGeom.scale(1, 0.5, 1.5);
    const earMat = new THREE.MeshStandardMaterial({ color: headColor, roughness: 0.8 });
    for (const side of [-1, 1]) {
      const ear = new THREE.Mesh(earGeom, earMat);
      ear.position.set(0.48, 0.65, side * 0.15);
      group.add(ear);
    }

    // Eyes
    const eyeGeom = new THREE.SphereGeometry(0.03, 6, 6);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x444444 });
    for (const side of [-1, 1]) {
      const eye = new THREE.Mesh(eyeGeom, eyeMat);
      eye.position.set(0.58, 0.6, side * 0.08);
      group.add(eye);
    }

    // Legs
    const legGeom = new THREE.CylinderGeometry(0.04, 0.035, 0.35, 6);
    const legMat = new THREE.MeshStandardMaterial({ color: legColor, roughness: 0.9 });
    const legPositions = [
      [0.2, 0.17, 0.15], [0.2, 0.17, -0.15],
      [-0.2, 0.17, 0.15], [-0.2, 0.17, -0.15]
    ];
    for (const [lx, ly, lz] of legPositions) {
      const leg = new THREE.Mesh(legGeom, legMat);
      leg.position.set(lx, ly, lz);
      leg.castShadow = true;
      group.add(leg);
    }

    // Color marker band around body
    if (colorName !== 'WHITE') {
      const bandGeom = new THREE.TorusGeometry(0.36, 0.04, 8, 16);
      const bandMat = new THREE.MeshStandardMaterial({
        color: markerColor,
        emissive: markerColor,
        emissiveIntensity: 0.4,
        roughness: 0.3
      });
      const band = new THREE.Mesh(bandGeom, bandMat);
      band.position.set(0, 0.5, 0);
      band.rotation.x = Math.PI / 2;
      group.add(band);
    }

    // ID number sprite
    if (id > 0) {
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      const hexColor = '#' + markerColor.toString(16).padStart(6, '0');

      // Draw circle background with dark outline
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 4;
      ctx.fillStyle = hexColor;
      ctx.beginPath();
      ctx.arc(64, 64, 52, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Choose text color based on marker brightness
      const r = (markerColor >> 16) & 0xff;
      const g = (markerColor >> 8) & 0xff;
      const b = markerColor & 0xff;
      const brightness = (r * 299 + g * 587 + b * 114) / 1000;
      const textColor = brightness > 160 ? '#000000' : '#ffffff';
      const strokeColor = brightness > 160 ? '#ffffff' : '#000000';

      ctx.font = 'bold 64px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      // Draw text outline for contrast
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 5;
      ctx.strokeText(String(id), 64, 66);
      // Draw text fill
      ctx.fillStyle = textColor;
      ctx.fillText(String(id), 64, 66);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.set(0, 1.3, 0);
      sprite.scale.set(0.9, 0.9, 1);
      group.add(sprite);
    }

    // Random Y rotation for variety
    group.rotation.y = Math.random() * Math.PI * 2;

    return group;
  }

  /**
   * Place sheep across multiple partition cells
   */
  static placeSheepInPartitions(options = {}) {
    const {
      cells = [],
      numSheep = 8,
      sceneRadius = 12,
      minDistance = 2.0,
      outsideCount = 2,
      colors = ['RED', 'BLUE'],
      minFenceDist = 2.0
    } = options;

    const sheep = [];
    const insideSheep = Math.max(0, numSheep - outsideCount);

    // Build a distribution with a single strict winner so Q3 always has one
    // correct answer. Unlike the old tie-breaking pass, this preserves the
    // total inside-sheep count instead of silently dropping sheep.
    const cellCounts = this._buildUniqueMaxCellCounts(cells.length, insideSheep);

    // Place sheep in each cell
    let id = 1;
    for (let c = 0; c < cells.length; c++) {
      const cell = cells[c];
      for (let s = 0; s < cellCounts[c]; s++) {
        let placed = false;
        for (let attempt = 0; attempt < 500 && !placed; attempt++) {
          const pos = this._randomPointInPolygon(cell, sceneRadius);
          if (pos &&
              RegionAnalyzer.pointInPolygon(pos, cell) &&
              this._minDistanceOk(pos, sheep, minDistance) &&
              this._minDistToPolygonEdge(pos, cell) > minFenceDist) {
            sheep.push({
              x: pos.x, z: pos.z,
              isInside: true,
              cellIndex: c,
              cellLabel: String.fromCharCode(65 + c),
              nestingDepth: 1,
              regionLabel: String(c),
              color: colors[(id - 1) % colors.length],
              id
            });
            placed = true;
            id++;
          }
        }
        if (!placed) {
          // Fallback: search near centroid with fence distance check
          const centroid = this._centroid(cell);
          let fallbackPos = null;
          for (let attempt = 0; attempt < 300; attempt++) {
            const candidate = {
              x: centroid.x + (Math.random() - 0.5) * 2.0,
              z: centroid.z + (Math.random() - 0.5) * 2.0
            };
            if (RegionAnalyzer.pointInPolygon(candidate, cell) &&
                this._minDistanceOk(candidate, sheep, minDistance * 0.7) &&
                this._minDistToPolygonEdge(candidate, cell) > minFenceDist * 0.95) {
              fallbackPos = candidate;
              break;
            }
          }
          if (!fallbackPos) {
            // Last resort: push centroid away from cell edges, but never force
            // an overlapping sheep into the cell.
            const adjustedPos = this._pushAwayFromPolygonEdges(
              { x: centroid.x, z: centroid.z }, cell, minFenceDist
            );
            if (
              RegionAnalyzer.pointInPolygon(adjustedPos, cell) &&
              this._minDistanceOk(adjustedPos, sheep, minDistance * 0.85) &&
              this._minDistToPolygonEdge(adjustedPos, cell) > minFenceDist * 0.9
            ) {
              fallbackPos = adjustedPos;
            }
          }
          if (!fallbackPos) {
            continue;
          }
          sheep.push({
            x: fallbackPos.x, z: fallbackPos.z,
            isInside: true,
            cellIndex: c,
            cellLabel: String.fromCharCode(65 + c),
            nestingDepth: 1,
            regionLabel: String(c),
            color: colors[(id - 1) % colors.length],
            id
          });
          id++;
        }
      }
    }

    // Place sheep outside all cells — keep close to partitions
    let cMinX = Infinity, cMaxX = -Infinity, cMinZ = Infinity, cMaxZ = -Infinity;
    for (const cell of cells) {
      for (const v of cell) {
        cMinX = Math.min(cMinX, v.x); cMaxX = Math.max(cMaxX, v.x);
        cMinZ = Math.min(cMinZ, v.z); cMaxZ = Math.max(cMaxZ, v.z);
      }
    }
    if (!isFinite(cMinX)) { cMinX = -5; cMaxX = 5; cMinZ = -5; cMaxZ = 5; }
    const cellMargin = 3.5;

    for (let i = 0; i < outsideCount; i++) {
      let placed = false;
      for (let attempt = 0; attempt < 500 && !placed; attempt++) {
        const pos = {
          x: cMinX - cellMargin + Math.random() * (cMaxX - cMinX + cellMargin * 2),
          z: cMinZ - cellMargin + Math.random() * (cMaxZ - cMinZ + cellMargin * 2)
        };
        let insideAny = false;
        for (const cell of cells) {
          if (RegionAnalyzer.pointInPolygon(pos, cell)) { insideAny = true; break; }
        }
        if (!insideAny &&
            this._minDistanceOk(pos, sheep, minDistance) &&
            this._allCellEdgesOk(pos, cells, minFenceDist)) {
          sheep.push({
            x: pos.x, z: pos.z,
            isInside: false,
            cellIndex: -1,
            cellLabel: 'outside',
            nestingDepth: 0,
            regionLabel: 'outside',
            color: colors[(id - 1) % colors.length],
            id
          });
          placed = true;
          id++;
        }
      }
      if (!placed) {
        // Fallback: place at safe angle outside all cells
        let fallbackPos = null;
        for (let attempt = 0; attempt < 300; attempt++) {
          const angle = Math.random() * Math.PI * 2;
          const cellExtent = Math.max(cMaxX - cMinX, cMaxZ - cMinZ) / 2;
          const dist = cellExtent + minFenceDist + Math.random() * 2.0;
          const candidate = {
            x: Math.cos(angle) * dist,
            z: Math.sin(angle) * dist
          };
          let insideAny = false;
          for (const cell of cells) {
            if (RegionAnalyzer.pointInPolygon(candidate, cell)) { insideAny = true; break; }
          }
          if (!insideAny &&
              this._minDistanceOk(candidate, sheep, minDistance * 0.7) &&
              this._allCellEdgesOk(candidate, cells, minFenceDist * 0.95)) {
            fallbackPos = candidate;
            break;
          }
        }
        if (!fallbackPos) {
          const cellExtent = Math.max(cMaxX - cMinX, cMaxZ - cMinZ) / 2;
          const angle = i * 2.4 + 0.7;
          fallbackPos = {
            x: (cellExtent + minFenceDist + 1.5) * Math.cos(angle),
            z: (cellExtent + minFenceDist + 1.5) * Math.sin(angle)
          };
        }
        let insideAny = false;
        for (const cell of cells) {
          if (RegionAnalyzer.pointInPolygon(fallbackPos, cell)) {
            insideAny = true;
            break;
          }
        }
        if (
          fallbackPos &&
          !insideAny &&
          this._minDistanceOk(fallbackPos, sheep, minDistance * 0.85) &&
          this._allCellEdgesOk(fallbackPos, cells, minFenceDist * 0.9)
        ) {
          sheep.push({
            x: fallbackPos.x, z: fallbackPos.z,
            isInside: false,
            cellIndex: -1,
            cellLabel: 'outside',
            nestingDepth: 0,
            regionLabel: 'outside',
            color: colors[(id - 1) % colors.length],
            id
          });
          id++;
        }
      }
    }

    return sheep;
  }

  /**
   * After placement, resolve any label overlaps by offsetting sprite positions.
   * Call this in the renderer after placing all sheep meshes.
   * @param {Array<THREE.Group>} sheepMeshes - array of sheep group meshes
   * @param {THREE.Camera} camera - the scene camera
   */
  static resolveLabelsOverlap(sheepMeshes, camera) {
    if (sheepMeshes.length < 2) return;

    // Screen-space label radius (empirically tuned for 128px sprite at scale 0.9)
    const labelScreenRadius = 0.035;
    const minSep = labelScreenRadius * 2.2; // minimum separation between label centers
    const minSepSq = minSep * minSep;

    // Build entries with screen-space positions
    const entries = sheepMeshes.map(mesh => {
      const worldPos = new THREE.Vector3();
      mesh.getWorldPosition(worldPos);
      const labelWorld = worldPos.clone();
      labelWorld.y += 1.3;
      const screen = labelWorld.clone().project(camera);
      return { mesh, sx: screen.x, sy: screen.y, offsetY: 0 };
    });

    // Iterative repulsion: push overlapping labels apart over several passes
    const maxPasses = 12;
    for (let pass = 0; pass < maxPasses; pass++) {
      let anyOverlap = false;
      for (let i = 0; i < entries.length; i++) {
        for (let j = i + 1; j < entries.length; j++) {
          const dx = (entries[i].sx) - (entries[j].sx);
          const dy = (entries[i].sy + entries[i].offsetY) - (entries[j].sy + entries[j].offsetY);
          const distSq = dx * dx + dy * dy;
          if (distSq < minSepSq) {
            anyOverlap = true;
            const dist = Math.sqrt(distSq) || 0.001;
            const overlap = minSep - dist;
            // Push apart vertically in screen space (map to world Y offset)
            const push = overlap * 0.55;
            // Push the one with the higher index upward
            if (entries[i].sy + entries[i].offsetY >= entries[j].sy + entries[j].offsetY) {
              entries[i].offsetY += push;
              entries[j].offsetY -= push * 0.3;
            } else {
              entries[j].offsetY += push;
              entries[i].offsetY -= push * 0.3;
            }
          }
        }
      }
      if (!anyOverlap) break;
    }

    // Convert screen-space Y offsets to world-space Y offsets
    // Approximate: screen Y range [-1,1] maps to roughly 2*camDist*tan(fov/2) in world
    const camPos = camera.position;
    const camDist = camPos.length();
    const fovRad = (camera.fov || 50) * Math.PI / 180;
    const worldPerScreen = camDist * Math.tan(fovRad / 2);

    // Apply offsets to sprite positions
    for (const entry of entries) {
      if (Math.abs(entry.offsetY) > 0.001) {
        const worldOffset = entry.offsetY * worldPerScreen;
        entry.mesh.traverse(child => {
          if (child.isSprite) {
            child.position.y += worldOffset;
            // Scale up offset labels slightly so they're still readable
            child.scale.set(0.95, 0.95, 1);
          }
        });
      }
    }
  }

  // ── Private helpers ──

  /**
   * Minimum distance from point to any edge of a polygon
   */
  static _minDistToPolygonEdge(pos, polygon) {
    let minDist = Infinity;
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const dist = this._pointToSegmentDist(pos, polygon[i], polygon[(i + 1) % n]);
      if (dist < minDist) minDist = dist;
    }
    return minDist;
  }

  /**
   * Check min distance from point to all edges of all cells
   */
  static _allCellEdgesOk(pos, cells, minDist) {
    for (const cell of cells) {
      if (this._minDistToPolygonEdge(pos, cell) < minDist) return false;
    }
    return true;
  }

  /**
   * Push a position away from the nearest fence segments until it meets minDist.
   * Iteratively nudges the point away from the closest fence edge.
   */
  static _pushAwayFromFences(pos, fenceLayers, minDist) {
    let p = { x: pos.x, z: pos.z };
    for (let iter = 0; iter < 10; iter++) {
      let closestDist = Infinity;
      let closestNormal = null;
      for (const polygon of fenceLayers) {
        const n = polygon.length;
        for (let i = 0; i < n; i++) {
          const a = polygon[i];
          const b = polygon[(i + 1) % n];
          const dist = this._pointToSegmentDist(p, a, b);
          if (dist < closestDist) {
            closestDist = dist;
            // Compute push direction (away from closest point on segment)
            const proj = this._closestPointOnSegment(p, a, b);
            const dx = p.x - proj.x;
            const dz = p.z - proj.z;
            const len = Math.sqrt(dx * dx + dz * dz) || 1;
            closestNormal = { x: dx / len, z: dz / len };
          }
        }
      }
      if (closestDist >= minDist) break;
      // Push away
      const push = minDist - closestDist + 0.1;
      p = {
        x: p.x + closestNormal.x * push,
        z: p.z + closestNormal.z * push
      };
    }
    return p;
  }

  /**
   * Push a position away from the nearest edges of a single polygon
   */
  static _pushAwayFromPolygonEdges(pos, polygon, minDist) {
    let p = { x: pos.x, z: pos.z };
    for (let iter = 0; iter < 10; iter++) {
      let closestDist = Infinity;
      let closestNormal = null;
      const n = polygon.length;
      for (let i = 0; i < n; i++) {
        const a = polygon[i];
        const b = polygon[(i + 1) % n];
        const dist = this._pointToSegmentDist(p, a, b);
        if (dist < closestDist) {
          closestDist = dist;
          const proj = this._closestPointOnSegment(p, a, b);
          const dx = p.x - proj.x;
          const dz = p.z - proj.z;
          const len = Math.sqrt(dx * dx + dz * dz) || 1;
          closestNormal = { x: dx / len, z: dz / len };
        }
      }
      if (closestDist >= minDist) break;
      const push = minDist - closestDist + 0.1;
      p = {
        x: p.x + closestNormal.x * push,
        z: p.z + closestNormal.z * push
      };
    }
    return p;
  }

  /**
   * Closest point on segment AB to point P
   */
  static _closestPointOnSegment(p, a, b) {
    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const lenSq = dx * dx + dz * dz;
    if (lenSq < 1e-8) return { x: a.x, z: a.z };
    let t = ((p.x - a.x) * dx + (p.z - a.z) * dz) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return { x: a.x + t * dx, z: a.z + t * dz };
  }

  static _randomPointInPolygon(polygon, fallbackRadius) {
    if (polygon.length < 3) {
      return {
        x: (Math.random() - 0.5) * fallbackRadius,
        z: (Math.random() - 0.5) * fallbackRadius
      };
    }
    // Bounding box sampling with a shape-relative margin. Fixed large margins
    // starve small inner enclosures and collapse Q1 into mostly zero trapped
    // sheep, so narrow cells keep a smaller but nonzero inset.
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const v of polygon) {
      minX = Math.min(minX, v.x);
      maxX = Math.max(maxX, v.x);
      minZ = Math.min(minZ, v.z);
      maxZ = Math.max(maxZ, v.z);
    }
    const span = Math.min(maxX - minX, maxZ - minZ);
    const margin = Math.min(1.0, Math.max(0.2, span * 0.12));
    const usableW = maxX - minX - 2 * margin;
    const usableH = maxZ - minZ - 2 * margin;
    const pos = {
      x: usableW > 0 ? minX + margin + Math.random() * usableW : minX + Math.random() * (maxX - minX),
      z: usableH > 0 ? minZ + margin + Math.random() * usableH : minZ + Math.random() * (maxZ - minZ)
    };
    if (RegionAnalyzer.pointInPolygon(pos, polygon)) return pos;
    return null;
  }

  static _minDistanceOk(pos, existingSheep, minDist) {
    for (const s of existingSheep) {
      const dx = pos.x - s.x;
      const dz = pos.z - s.z;
      if (dx * dx + dz * dz < minDist * minDist) return false;
    }
    return true;
  }

  /**
   * Check minimum distance from a point to all fence segments across all layers.
   */
  static _minDistanceToFenceOk(pos, fenceLayers, minDist) {
    for (const polygon of fenceLayers) {
      const n = polygon.length;
      for (let i = 0; i < n; i++) {
        const a = polygon[i];
        const b = polygon[(i + 1) % n];
        const dist = this._pointToSegmentDist(pos, a, b);
        if (dist < minDist) return false;
      }
    }
    return true;
  }

  /**
   * Distance from point P to line segment AB (in xz plane)
   */
  static _pointToSegmentDist(p, a, b) {
    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const lenSq = dx * dx + dz * dz;
    if (lenSq < 1e-8) {
      const ex = p.x - a.x, ez = p.z - a.z;
      return Math.sqrt(ex * ex + ez * ez);
    }
    let t = ((p.x - a.x) * dx + (p.z - a.z) * dz) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const projX = a.x + t * dx;
    const projZ = a.z + t * dz;
    const ex = p.x - projX, ez = p.z - projZ;
    return Math.sqrt(ex * ex + ez * ez);
  }

  static _centroid(polygon) {
    let cx = 0, cz = 0;
    for (const v of polygon) {
      cx += v.x;
      cz += v.z;
    }
    return { x: cx / polygon.length, z: cz / polygon.length };
  }

  static _buildUniqueMaxCellCounts(numCells, sheepCount) {
    const cellCounts = new Array(numCells).fill(0);
    if (numCells <= 0 || sheepCount <= 0) return cellCounts;
    if (numCells === 1) {
      cellCounts[0] = sheepCount;
      return cellCounts;
    }

    const winnerIdx = Math.floor(Math.random() * numCells);
    const winnerCount = Math.max(1, Math.ceil((sheepCount + numCells - 1) / numCells));
    const loserCap = Math.max(0, winnerCount - 1);
    const otherIndices = [];

    for (let i = 0; i < numCells; i++) {
      if (i !== winnerIdx) otherIndices.push(i);
    }

    cellCounts[winnerIdx] = winnerCount;
    let remaining = sheepCount - winnerCount;

    while (remaining > 0) {
      const eligible = otherIndices.filter(index => cellCounts[index] < loserCap);
      if (!eligible.length) {
        cellCounts[winnerIdx] += remaining;
        break;
      }
      const targetIdx = eligible[Math.floor(Math.random() * eligible.length)];
      cellCounts[targetIdx]++;
      remaining--;
    }

    return cellCounts;
  }

  static _hasUniquePositiveMax(values) {
    if (!values.length) return false;
    const maxCount = Math.max(...values);
    if (maxCount <= 0) return false;
    let winners = 0;
    for (const value of values) {
      if (value === maxCount) winners++;
      if (winners > 1) return false;
    }
    return true;
  }

  static _isValidMaxCellDataset(values) {
    if (!this._hasUniquePositiveMax(values)) return false;
    return Math.max(...values) >= 2;
  }
}
