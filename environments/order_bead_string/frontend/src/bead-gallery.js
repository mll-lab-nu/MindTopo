/**
 * Bead String Gallery & Dataset Generator
 *
 * Main application: Three.js scene, UI controls, camera presets, export.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createBeadString, BEAD_PALETTE, PALETTE_NAMES } from './bead-renderer.js';
import { makeRng } from './bead-curve-library.js';
import { computeBeadStringDifficulty } from './bead-difficulty-controller.js';

// ============= Constants =============

const CAMERA_PRESETS = {
  iso_fr:  { pos: [4, 4, 6],    name: 'Isometric Front' },
  front:   { pos: [0, 1, 8],    name: 'Front' },
  oblique: { pos: [3, 7, 4],    name: 'Oblique' },
  top:     { pos: [0, 10, 0.1], name: 'Top' },
  back:    { pos: [0, 1, -8],   name: 'Back' },
  left:    { pos: [-8, 1, 0],   name: 'Left' },
  right:   { pos: [8, 1, 0],    name: 'Right' },
  iso_bk:  { pos: [-4, 4, -6],  name: 'Isometric Back' },
};

const CURVE_TYPES = [
  { value: 'straight', label: '— Straight (直线)' },
  { value: 'arc', label: '⌒ Arc (弧线)' },
  { value: 's_curve', label: '∿ S-Curve (S 形)' },
  { value: 'helix', label: '🧬 Helix (螺旋)' },
  { value: 'random_spline', label: '🎲 Random Spline (随机曲线)' },
];

const RING_CURVE_TYPES = [
  { value: 'ring', label: '○ Ring (圆环)' },
  { value: 'wavy_ring', label: '〰 Wavy Ring (波浪环)' },
];

function datasetBeadSize(rng) {
  return 0.24 + rng() * 0.07;
}

function datasetRopeThickness(rng) {
  return 0.055 + rng() * 0.025;
}

// ============= State =============

let scene, camera, renderer, controls, shadowPlane, grid;
let root = new THREE.Group();
let currentObjects = { meshes: [], groups: [] };
let currentMetadata = null;

const viewEl = document.getElementById('view');

// ============= Scene Init =============

function initThree() {
  // Check WebGL availability
  const testCanvas = document.createElement('canvas');
  const gl = testCanvas.getContext('webgl2') || testCanvas.getContext('webgl');
  if (!gl) {
    throw new Error('WebGL 不可用，请关闭一些浏览器标签页后重试，或检查浏览器是否支持 WebGL');
  }

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a2236);

  const w = viewEl.clientWidth;
  const h = viewEl.clientHeight;
  camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 500);
  camera.position.set(4, 4, 6);

  // Try creating renderer, fallback to no-antialias if it fails
  let canvas;
  try {
    canvas = document.createElement('canvas');
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
  } catch (e) {
    console.warn('WebGLRenderer with antialias failed, retrying without:', e);
    canvas = document.createElement('canvas');
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      preserveDrawingBuffer: true,
    });
  }

  // Handle WebGL context loss & restore
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    console.warn('WebGL context lost');
    const statusEl = document.getElementById('status');
    if (statusEl) statusEl.innerHTML = '<div style="color:#ff6b6b"><b>WebGL 上下文丢失</b>，正在等待恢复...</div>';
  });
  canvas.addEventListener('webglcontextrestored', () => {
    console.log('WebGL context restored');
    // Re-render current scene
    try {
      renderer.render(scene, camera);
      const statusEl = document.getElementById('status');
      if (statusEl) statusEl.innerHTML = '<div><b>WebGL 上下文已恢复</b></div>';
    } catch (err) {
      console.error('Context restore render failed:', err);
    }
  });

  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewEl.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.target.set(0, 0, 0);

  // --- Lighting (same as knot gallery) ---
  scene.add(new THREE.AmbientLight(0xffffff, 0.46));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a335a, 0.18));

  const dir = new THREE.DirectionalLight(0xffffff, 1.95);
  dir.position.set(10, 18, 10);
  dir.castShadow = true;
  dir.shadow.mapSize.width = 2048;
  dir.shadow.mapSize.height = 2048;
  dir.shadow.camera.near = 0.1;
  dir.shadow.camera.far = 80;
  dir.shadow.camera.left = -18;
  dir.shadow.camera.right = 18;
  dir.shadow.camera.top = 18;
  dir.shadow.camera.bottom = -18;
  dir.shadow.bias = -0.0006;
  scene.add(dir);

  const fill = new THREE.DirectionalLight(0xffffff, 0.55);
  fill.position.set(-10, 5, -10);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0xffffff, 0.42);
  rim.position.set(0, 6, -14);
  scene.add(rim);

  // Shadow plane
  shadowPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(220, 220),
    new THREE.ShadowMaterial({ opacity: 0.18 }),
  );
  shadowPlane.rotation.x = -Math.PI * 0.5;
  shadowPlane.position.y = -2.5;
  shadowPlane.receiveShadow = true;
  scene.add(shadowPlane);

  // Grid
  grid = new THREE.GridHelper(60, 30, 0x2a335a, 0x1a2040);
  grid.position.y = -2.5;
  if (Array.isArray(grid.material)) {
    grid.material.forEach(m => { m.transparent = true; m.opacity = 0.3; });
  } else if (grid.material) {
    grid.material.transparent = true;
    grid.material.opacity = 0.3;
  }
  scene.add(grid);

  scene.add(root);

  window.addEventListener('resize', () => {
    const ww = viewEl.clientWidth;
    const hh = viewEl.clientHeight;
    camera.aspect = ww / hh;
    camera.updateProjectionMatrix();
    renderer.setSize(ww, hh);
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  // Guard against rendering with a lost context
  if (renderer.getContext && !renderer.getContext().isContextLost()) {
    renderer.render(scene, camera);
  }
}

// ============= Dispose =============

function disposeCurrent() {
  for (const g of currentObjects.groups) {
    root.remove(g);
    g.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
        else child.material.dispose();
      }
    });
  }
  currentObjects = { meshes: [], groups: [] };
  currentMetadata = null;
}

// ============= Read UI =============

function readUI() {
  const val = (id, fallback) => {
    const el = document.getElementById(id);
    return el ? el.value : fallback;
  };
  const num = (id, fallback, min, max) => {
    const v = Number(val(id, fallback));
    if (!Number.isFinite(v)) return fallback;
    return Math.max(min, Math.min(max, v));
  };

  const isRing = document.getElementById('isRing')?.checked || false;

  return {
    seed: val('seed', 'bead-gallery-v1'),
    numBeads: num('numBeads', 5, 3, 10),
    curveType: val('curveType', 'arc'),
    curveComplexity: num('curveComplexity', 0.3, 0, 1),
    beadSize: num('beadSize', 0.35, 0.15, 0.8),
    ropeThickness: num('ropeThickness', 0.08, 0.03, 0.2),
    isRing,
    occlusionLevel: val('occlusionLevel', 'none'),
    cameraAngle: val('cameraAngle', 'iso_fr'),
    colorMode: val('colorMode', 'distinct'),
    count: num('count', 1, 1, 200),
    cols: num('cols', 4, 1, 20),
    showStartMarker: document.getElementById('showStartMarker')?.checked !== false,
  };
}

// ============= Build Scene =============

function buildScene(params) {
  disposeCurrent();

  const { count, cols, cameraAngle } = params;
  const allMeta = [];
  const spacing = 6;

  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const itemSeed = count === 1 ? params.seed : `${params.seed}-${i}`;

    const { group, metadata } = createBeadString({
      curveType: params.curveType,
      curveComplexity: params.curveComplexity,
      numBeads: params.numBeads,
      colorMode: params.colorMode,
      beadSize: params.beadSize,
      ropeThickness: params.ropeThickness,
      seed: itemSeed,
      isRing: params.isRing,
      showStartMarker: params.showStartMarker,
    });

    // Compute difficulty
    const diff = computeBeadStringDifficulty({
      numBeads: params.numBeads,
      curveComplexity: params.curveComplexity,
      occlusionLevel: params.occlusionLevel,
      cameraAngle,
      beadColors: metadata.bead_sequence,
    });

    metadata.difficulty_score = diff.score;
    metadata.difficulty = diff.level;
    metadata.difficulty_factors = diff.factors;
    metadata.camera_angle = cameraAngle;
    metadata.occlusion_level = params.occlusionLevel;

    // Position in grid
    if (count > 1) {
      const offsetX = (col - (cols - 1) / 2) * spacing;
      const offsetZ = (row - Math.floor((count - 1) / cols) / 2) * spacing;
      group.position.set(offsetX, 0, offsetZ);
    }

    root.add(group);
    currentObjects.groups.push(group);
    allMeta.push(metadata);
  }

  currentMetadata = count === 1 ? allMeta[0] : allMeta;

  // Set camera
  setCameraPreset(cameraAngle);

  // Update status
  updateStatus(params, allMeta);
}

// ============= Camera =============

function setCameraPreset(name, group = root) {
  const preset = CAMERA_PRESETS[name] || CAMERA_PRESETS.iso_fr;
  const bounds = new THREE.Box3().setFromObject(group);
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());

  // Keep the transparent shadow surface below the complete model. A fixed
  // ground height can depth-occlude low transparent beads, especially bead 0
  // at the lower endpoint of a helix.
  const groundY = bounds.min.y - Math.max(0.5, sphere.radius * 0.08);
  if (shadowPlane) shadowPlane.position.y = groundY;
  if (grid) grid.position.y = groundY;

  const dirVec = new THREE.Vector3(...preset.pos).normalize();
  const fovRad = camera.fov * Math.PI / 180;
  const dist = Math.max(
    (sphere.radius * 1.15) / Math.sin(fovRad / 2),
    new THREE.Vector3(...preset.pos).length(),
  );
  camera.position.copy(dirVec).multiplyScalar(dist);
  camera.lookAt(sphere.center);
  controls.target.copy(sphere.center);
  controls.update();
}

// ============= Status =============

function updateStatus(params, allMeta) {
  const statusEl = document.getElementById('status');
  if (!statusEl) return;

  const meta = Array.isArray(allMeta) ? allMeta[0] : allMeta;
  const diff = meta?.difficulty || '-';
  const score = meta?.difficulty_score?.toFixed(3) || '-';
  const seq = meta?.bead_sequence?.join(', ') || '-';

  // Difficulty pill
  const pillClass = diff === 'easy' ? 'pill-easy' : diff === 'medium' ? 'pill-medium' : diff === 'hard' ? 'pill-hard' : '';

  statusEl.innerHTML = `
    <div><b>Three.js</b>: r${THREE.REVISION}</div>
    <div><b>曲线</b>: ${meta?.curve_type || '-'} | <b>珠子数</b>: ${meta?.num_beads || '-'}</div>
    <div><b>序列</b>: ${seq}</div>
    <div><b>难度</b>: <span class="pill ${pillClass}">${diff}</span> (${score})</div>
    <div><b>实例数</b>: ${Array.isArray(allMeta) ? allMeta.length : 1}</div>
  `;
}

// ============= Export JSON =============

function exportJSON() {
  if (!currentMetadata) {
    alert('请先生成 bead string');
    return;
  }
  const data = {
    version: 1,
    task: 'bead_string',
    generated_at: new Date().toISOString(),
    image_size: [1024, 1024],
    ...(Array.isArray(currentMetadata)
      ? { samples: currentMetadata }
      : currentMetadata),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bead_string_metadata_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ============= Capture PNG =============

function capturePNG(width = 1024, height = 1024) {
  const gl = renderer.getContext();
  if (gl.isContextLost()) {
    throw new Error('WebGL 上下文已丢失，无法截图。请刷新页面重试。');
  }

  const prevW = renderer.domElement.width;
  const prevH = renderer.domElement.height;
  const prevAspect = camera.aspect;

  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);

  const dataUrl = renderer.domElement.toDataURL('image/png');

  // Restore
  renderer.setSize(prevW, prevH);
  camera.aspect = prevAspect;
  camera.updateProjectionMatrix();

  return dataUrl;
}

function downloadPNG() {
  const dataUrl = capturePNG();
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `bead_string_${Date.now()}.png`;
  a.click();
}

// ============= Multi-Angle Batch Capture =============

async function batchCapture8Angles() {
  if (!currentMetadata) {
    alert('请先生成 bead string');
    return;
  }

  const angles = ['front', 'back', 'left', 'right', 'top', 'iso_fr', 'iso_bk', 'oblique'];
  const results = [];

  for (const angle of angles) {
    setCameraPreset(angle);
    await new Promise(r => requestAnimationFrame(r));
    results.push({ angle, dataUrl: capturePNG() });
  }

  // Reset camera
  const params = readUI();
  setCameraPreset(params.cameraAngle);

  // Download each
  for (const { angle, dataUrl } of results) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `bead_string_${angle}_${Date.now()}.png`;
    a.click();
    await new Promise(r => setTimeout(r, 200));
  }
}

// ============= Full Dataset Generation =============

export function normalizeDifficultyLevels(values = ['easy', 'medium', 'hard']) {
  const allowed = new Set(['easy', 'medium', 'hard']);
  const out = [];
  for (const value of values) {
    const normalized = String(value || '').trim().toLowerCase();
    if (!normalized || !allowed.has(normalized) || out.includes(normalized)) continue;
    out.push(normalized);
  }
  return out.length ? out : ['easy', 'medium', 'hard'];
}

// Shared difficulty recipes.
// The tier ladder was shifted: the old trivial "easy" was retired, so easy now
// uses the old medium recipe, medium uses the old hard recipe, and the new hard
// swaps in the self-crossing TangledLoopCurve — much harder for VLMs to trace,
// still human-readable thanks to clear over/under cues from the z displacement.
const BEAD_DIFFICULTY_CONFIGS = {
  easy: {
    numBeads: [2, 3, 4],
    curveTypes: ['arc', 's_curve', 'helix'],
    complexity: [0.3, 0.6],
    colorMode: 'mixed',
    occlusion: 'partial',
    cameras: ['iso_fr', 'oblique'],
  },
  medium: {
    numBeads: [5, 6, 7],
    curveTypes: ['helix', 'random_spline'],
    complexity: [0.6, 1.0],
    colorMode: 'similar',
    occlusion: 'heavy',
    cameras: ['oblique', 'top'],
  },
  hard: {
    numBeads: [3, 4, 5],
    curveTypes: ['tangled_loop'],
    complexity: [0.2, 0.85],
    colorMode: 'distinct',
    occlusion: 'none',
    // 'top' removed: top-down kills the z over/under signal, so xy crossings
    // read as a single bead threaded by two separate ropes.
    cameras: ['iso_fr', 'oblique'],
    forceRing: false,
  },
};

/**
 * Generate a complete dataset with various configurations.
 */
export async function generateFullDataset(config = {}, progressCb = () => {}, sampleCb = null) {
  const {
    seed = 'bead-dataset-v1',
    samplesPerDifficulty = 50,
    renderWidth = 1024,
    renderHeight = 1024,
    difficulties = ['easy', 'medium', 'hard'],
    retainSamples = true,
  } = config;

  const rng = makeRng(seed);
  const allSamples = [];
  const selectedDifficulties = normalizeDifficultyLevels(difficulties);

  const difficultyConfigs = BEAD_DIFFICULTY_CONFIGS;

  const totalBaseSamples = samplesPerDifficulty * selectedDifficulties.length;
  let current = 0;
  const difficultyCounts = { easy: 0, medium: 0, hard: 0 };

  async function recordSample(sample) {
    if (sampleCb) {
      await sampleCb(sample);
    }
    if (retainSamples) {
      allSamples.push(sample);
      return;
    }
    const { image_data_url, ...metadataOnlySample } = sample;
    allSamples.push(metadataOnlySample);
  }

  for (const level of selectedDifficulties) {
    const cfg = difficultyConfigs[level];
    for (let i = 0; i < samplesPerDifficulty; i++) {
      const numBeads = cfg.numBeads[Math.floor(rng() * cfg.numBeads.length)];
      const curveType = cfg.curveTypes[Math.floor(rng() * cfg.curveTypes.length)];
      const complexity = cfg.complexity[0] + rng() * (cfg.complexity[1] - cfg.complexity[0]);
      const cam = cfg.cameras[Math.floor(rng() * cfg.cameras.length)];
      const itemSeed = `${seed}-${level}-${i}`;
      const isRing = cfg.forceRing !== undefined ? cfg.forceRing : (rng() < 0.2);

      // Color mode distribution: 15% palindrome, 15% periodic, 70% default
      const roll = rng();
      const effectiveColorMode = roll < 0.15 ? 'palindrome'
                               : roll < 0.30 ? 'periodic'
                               : cfg.colorMode;

      // Build the bead string
      disposeCurrent();
      const { group, metadata } = createBeadString({
        curveType,
        curveComplexity: complexity,
        numBeads,
        colorMode: effectiveColorMode,
        beadSize: datasetBeadSize(rng),
        ropeThickness: datasetRopeThickness(rng),
        seed: itemSeed,
        isRing,
        showStartMarker: true,
      });
      root.add(group);
      currentObjects.groups.push(group);

      setCameraPreset(cam, group);

      // Render
      await new Promise(r => requestAnimationFrame(r));
      const dataUrl = capturePNG(renderWidth, renderHeight);

      // Compute difficulty
      const diff = computeBeadStringDifficulty({
        numBeads,
        curveComplexity: complexity,
        occlusionLevel: cfg.occlusion,
        cameraAngle: cam,
        beadColors: metadata.bead_sequence,
      });

      const sample = {
        ...metadata,
        camera_angle: cam,
        camera_position: [...CAMERA_PRESETS[cam].pos],
        occlusion_level: cfg.occlusion,
        difficulty_score: diff.score,
        difficulty: level,
        difficulty_bucket: diff.level,
        difficulty_factors: diff.factors,
        image_size: [renderWidth, renderHeight],
        image_data_url: dataUrl,
        filename: `bead_${level}_${String(i).padStart(3, '0')}_${cam}.png`,
      };
      await recordSample(sample);
      difficultyCounts[level] += 1;

      current++;
      await progressCb(current, totalBaseSamples, `Generating [${level}]`);
    }
  }

  // ── Generate cyclic rotation pairs ──
  // Pick some existing samples and create rotated versions with different curves/cameras.
  // This fills the CYCLIC_ROTATION gap for T_BS02_pair_relationship.
  const cyclicPairCount = Math.max(0, Number(config.cyclicPairs ?? 15) || 0);
  const cyclicSource = allSamples.filter(s => s.num_beads >= 4 && !s.is_ring);
  const cyclicSamples = [];

  for (let ci = 0; ci < Math.min(cyclicPairCount, cyclicSource.length); ci++) {
    const source = cyclicSource[ci];
    const seq = source.bead_sequence;
    const n = seq.length;
    // Rotate by 1..n-1 positions (pick a random shift that gives a different sequence)
    const shift = 1 + Math.floor(rng() * (n - 1));
    const rotatedSeq = [...seq.slice(shift), ...seq.slice(0, shift)];

    // Skip if rotation equals original or its reverse (would be IDENTICAL/REVERSED, not CYCLIC)
    if (JSON.stringify(rotatedSeq) === JSON.stringify(seq)) continue;
    if (JSON.stringify(rotatedSeq) === JSON.stringify([...seq].reverse())) continue;

    // Render with a different curve and camera
    const curveTypes = ['arc', 's_curve', 'helix', 'random_spline'];
    const cameras = ['front', 'iso_fr', 'oblique'];
    const cType = curveTypes[Math.floor(rng() * curveTypes.length)];
    const cam = cameras[Math.floor(rng() * cameras.length)];
    const complexity = 0.2 + rng() * 0.5;
    const itemSeed = `${seed}-cyclic-${ci}`;

    disposeCurrent();
    const { group, metadata } = createBeadString({
      curveType: cType,
      curveComplexity: complexity,
      numBeads: n,
      beadColors: rotatedSeq,
      beadSize: datasetBeadSize(rng),
      ropeThickness: datasetRopeThickness(rng),
      seed: itemSeed,
      isRing: false,
      showStartMarker: true,
    });
    root.add(group);
    currentObjects.groups.push(group);

    setCameraPreset(cam, group);

    await new Promise(r => requestAnimationFrame(r));
    const dataUrl = capturePNG(renderWidth, renderHeight);

    const diff = computeBeadStringDifficulty({
      numBeads: n,
      curveComplexity: complexity,
      occlusionLevel: 'none',
      cameraAngle: cam,
      beadColors: rotatedSeq,
    });

    const inheritedLevel = source.difficulty || 'easy';
    const sample = {
      ...metadata,
      camera_angle: cam,
      camera_position: [...CAMERA_PRESETS[cam].pos],
      occlusion_level: 'none',
      difficulty_score: diff.score,
      difficulty: inheritedLevel,
      difficulty_bucket: diff.level,
      difficulty_factors: diff.factors,
      image_size: [renderWidth, renderHeight],
      image_data_url: dataUrl,
      filename: `bead_cyclic_${String(ci).padStart(3, '0')}_${cam}.png`,
      _cyclic_source: source.filename,
      _cyclic_shift: shift,
    };
    await recordSample(sample);
    cyclicSamples.push(sample);

    current++;
    await progressCb(
      totalBaseSamples + cyclicSamples.length,
      totalBaseSamples + cyclicPairCount,
      'Generating [cyclic-pairs]',
    );
  }

  // Clean up
  disposeCurrent();

  return {
    version: 2,
    task: 'bead_string',
    generated_at: new Date().toISOString(),
    samples: allSamples,
    stats: {
      total: allSamples.length,
      easy: difficultyCounts.easy,
      medium: difficultyCounts.medium,
      hard: difficultyCounts.hard,
      cyclic_pairs: cyclicSamples.length,
    },
  };
}

/**
 * Generate base samples for a single shard of the dataset.
 *
 * Each shard processes the per-difficulty indices where `i % numWorkers === workerIdx`.
 * Combined with workerIdx-seeded RNG, the shards make independent random choices,
 * so the union of all shards covers `samplesPerDifficulty` items per difficulty
 * without overlap and without coordinating between workers.
 *
 * Returns lightweight per-sample summaries (no image data) suitable for handing
 * to `generateCyclicPairs` after all shards finish.
 */
export async function generateBaseSamplesShard(config = {}, progressCb = () => {}, sampleCb = null) {
  const {
    seed = 'bead-dataset-v1',
    samplesPerDifficulty = 50,
    renderWidth = 1024,
    renderHeight = 1024,
    difficulties = ['easy', 'medium', 'hard'],
    workerIdx = 0,
    numWorkers = 1,
  } = config;

  const totalWorkers = Math.max(1, Number(numWorkers) || 1);
  const idx = Math.max(0, Math.min(totalWorkers - 1, Number(workerIdx) || 0));
  // For workers=1 we keep the original RNG seed so single-worker output stays
  // bit-compatible with the previous integrated path; with multiple workers,
  // each worker gets its own seed suffix to avoid identical sample streams.
  const rngSeed = totalWorkers === 1 ? seed : `${seed}-w${idx}-of${totalWorkers}`;
  const rng = makeRng(rngSeed);

  const selectedDifficulties = normalizeDifficultyLevels(difficulties);
  const difficultyConfigs = BEAD_DIFFICULTY_CONFIGS;

  const sampleSummaries = [];
  const difficultyCounts = { easy: 0, medium: 0, hard: 0 };

  let shardTotal = 0;
  for (const level of selectedDifficulties) {
    for (let i = idx; i < samplesPerDifficulty; i += totalWorkers) shardTotal++;
  }

  let current = 0;
  for (const level of selectedDifficulties) {
    const cfg = difficultyConfigs[level];
    for (let i = idx; i < samplesPerDifficulty; i += totalWorkers) {
      const numBeads = cfg.numBeads[Math.floor(rng() * cfg.numBeads.length)];
      const curveType = cfg.curveTypes[Math.floor(rng() * cfg.curveTypes.length)];
      const complexity = cfg.complexity[0] + rng() * (cfg.complexity[1] - cfg.complexity[0]);
      const cam = cfg.cameras[Math.floor(rng() * cfg.cameras.length)];
      const itemSeed = `${seed}-${level}-${i}`;
      const isRing = cfg.forceRing !== undefined ? cfg.forceRing : (rng() < 0.2);

      const roll = rng();
      const effectiveColorMode = roll < 0.15 ? 'palindrome'
                               : roll < 0.30 ? 'periodic'
                               : cfg.colorMode;

      disposeCurrent();
      const { group, metadata } = createBeadString({
        curveType,
        curveComplexity: complexity,
        numBeads,
        colorMode: effectiveColorMode,
        beadSize: datasetBeadSize(rng),
        ropeThickness: datasetRopeThickness(rng),
        seed: itemSeed,
        isRing,
        showStartMarker: true,
      });
      root.add(group);
      currentObjects.groups.push(group);

      setCameraPreset(cam, group);
      await new Promise(r => requestAnimationFrame(r));
      const dataUrl = capturePNG(renderWidth, renderHeight);

      const diff = computeBeadStringDifficulty({
        numBeads,
        curveComplexity: complexity,
        occlusionLevel: cfg.occlusion,
        cameraAngle: cam,
        beadColors: metadata.bead_sequence,
      });

      const sample = {
        ...metadata,
        camera_angle: cam,
        camera_position: [...CAMERA_PRESETS[cam].pos],
        occlusion_level: cfg.occlusion,
        difficulty_score: diff.score,
        difficulty: level,
        difficulty_bucket: diff.level,
        difficulty_factors: diff.factors,
        image_size: [renderWidth, renderHeight],
        image_data_url: dataUrl,
        filename: `bead_${level}_${String(i).padStart(3, '0')}_${cam}.png`,
      };

      if (sampleCb) await sampleCb(sample);
      sampleSummaries.push({
        filename: sample.filename,
        bead_sequence: metadata.bead_sequence,
        num_beads: numBeads,
        is_ring: !!metadata.is_ring,
        difficulty: level,
        seed: itemSeed,
        camera_angle: cam,
        curve_type: metadata.curve_type,
        occlusion_level: cfg.occlusion,
      });
      difficultyCounts[level] += 1;

      current++;
      await progressCb(current, shardTotal, `Generating [${level}] worker=${idx}/${totalWorkers}`);
    }
  }

  disposeCurrent();
  return {
    workerIdx: idx,
    numWorkers: totalWorkers,
    sampleSummaries,
    stats: { ...difficultyCounts },
  };
}

/**
 * Generate cyclic-rotation pair samples from a pool of base-sample summaries.
 *
 * Uses its own RNG seeded from `${seed}-cyclic` so this phase is independent
 * of how the base samples were sharded across workers.
 */
export async function generateCyclicPairs(config = {}, sources = [], progressCb = () => {}, sampleCb = null) {
  const {
    seed = 'bead-dataset-v1',
    cyclicPairs = 15,
    renderWidth = 1024,
    renderHeight = 1024,
  } = config;

  const cyclicPairCount = Math.max(0, Number(cyclicPairs) || 0);
  if (cyclicPairCount === 0) {
    return { cyclic_pairs: 0 };
  }

  const rng = makeRng(`${seed}-cyclic`);
  const cyclicSource = (sources || []).filter(s => s && s.num_beads >= 4 && !s.is_ring);
  let emitted = 0;

  for (let ci = 0; ci < Math.min(cyclicPairCount, cyclicSource.length); ci++) {
    const source = cyclicSource[ci];
    const seq = source.bead_sequence || [];
    const n = seq.length;
    if (n < 2) continue;
    const shift = 1 + Math.floor(rng() * (n - 1));
    const rotatedSeq = [...seq.slice(shift), ...seq.slice(0, shift)];

    if (JSON.stringify(rotatedSeq) === JSON.stringify(seq)) continue;
    if (JSON.stringify(rotatedSeq) === JSON.stringify([...seq].reverse())) continue;

    const curveTypes = ['arc', 's_curve', 'helix', 'random_spline'];
    const cameras = ['front', 'iso_fr', 'oblique'];
    const cType = curveTypes[Math.floor(rng() * curveTypes.length)];
    const cam = cameras[Math.floor(rng() * cameras.length)];
    const complexity = 0.2 + rng() * 0.5;
    const itemSeed = `${seed}-cyclic-${ci}`;

    disposeCurrent();
    const { group, metadata } = createBeadString({
      curveType: cType,
      curveComplexity: complexity,
      numBeads: n,
      beadColors: rotatedSeq,
      beadSize: datasetBeadSize(rng),
      ropeThickness: datasetRopeThickness(rng),
      seed: itemSeed,
      isRing: false,
      showStartMarker: true,
    });
    root.add(group);
    currentObjects.groups.push(group);

    setCameraPreset(cam, group);
    await new Promise(r => requestAnimationFrame(r));
    const dataUrl = capturePNG(renderWidth, renderHeight);

    const diff = computeBeadStringDifficulty({
      numBeads: n,
      curveComplexity: complexity,
      occlusionLevel: 'none',
      cameraAngle: cam,
      beadColors: rotatedSeq,
    });

    const inheritedLevel = source.difficulty || 'easy';
    const sample = {
      ...metadata,
      camera_angle: cam,
      camera_position: [...CAMERA_PRESETS[cam].pos],
      occlusion_level: 'none',
      difficulty_score: diff.score,
      difficulty: inheritedLevel,
      difficulty_bucket: diff.level,
      difficulty_factors: diff.factors,
      image_size: [renderWidth, renderHeight],
      image_data_url: dataUrl,
      filename: `bead_cyclic_${String(ci).padStart(3, '0')}_${cam}.png`,
      _cyclic_source: source.filename,
      _cyclic_shift: shift,
    };

    if (sampleCb) await sampleCb(sample);
    emitted++;
    await progressCb(emitted, cyclicPairCount, 'Generating [cyclic-pairs]');
  }

  disposeCurrent();
  return { cyclic_pairs: emitted };
}

const PAIR_RELATIONS = ['IDENTICAL', 'REVERSED', 'CYCLIC_ROTATION', 'DIFFERENT'];

function sequenceRelation(seqA, seqB) {
  if (JSON.stringify(seqA) === JSON.stringify(seqB)) return 'IDENTICAL';
  if (JSON.stringify(seqA) === JSON.stringify([...seqB].reverse())) return 'REVERSED';
  if (seqA.length === seqB.length) {
    const doubled = [...seqA, ...seqA];
    for (let k = 1; k < seqA.length; k++) {
      if (JSON.stringify(doubled.slice(k, k + seqA.length)) === JSON.stringify(seqB)) {
        return 'CYCLIC_ROTATION';
      }
    }
  }
  return 'DIFFERENT';
}

function chooseDifferent(options, current, rng) {
  const candidates = options.filter(value => value !== current);
  return candidates[Math.floor(rng() * candidates.length)] || options[0];
}

function makeDifferentSequence(seq, rng) {
  const base = [...seq];
  for (let attempt = 0; attempt < 64; attempt++) {
    const candidate = [...base];
    const changes = 1 + Math.floor(rng() * Math.min(3, candidate.length));
    for (let i = 0; i < changes; i++) {
      const idx = Math.floor(rng() * candidate.length);
      candidate[idx] = chooseDifferent(PALETTE_NAMES, candidate[idx], rng);
    }
    if (sequenceRelation(base, candidate) === 'DIFFERENT') return candidate;
  }
  const candidate = [...base].reverse();
  candidate[0] = chooseDifferent(PALETTE_NAMES, candidate[0], rng);
  return candidate;
}

function makeRelationSequence(seq, relation, rng) {
  if (relation === 'IDENTICAL') return [...seq];
  if (relation === 'REVERSED') {
    const reversed = [...seq].reverse();
    return sequenceRelation(seq, reversed) === 'REVERSED' ? reversed : null;
  }
  if (relation === 'CYCLIC_ROTATION') {
    for (let attempt = 0; attempt < 32; attempt++) {
      const shift = 1 + Math.floor(rng() * (seq.length - 1));
      const rotated = [...seq.slice(shift), ...seq.slice(0, shift)];
      if (sequenceRelation(seq, rotated) === 'CYCLIC_ROTATION') return rotated;
    }
    return null;
  }
  return makeDifferentSequence(seq, rng);
}

/**
 * Generate source-linked pair variants for balanced pair-relation questions.
 */
export async function generatePairVariants(config = {}, sources = [], progressCb = () => {}, sampleCb = null) {
  const {
    seed = 'bead-dataset-v1',
    variantsPerRelation = 1,
    renderWidth = 1024,
    renderHeight = 1024,
    difficulties = ['easy', 'medium', 'hard'],
    relations = PAIR_RELATIONS,
  } = config;

  const perRelation = Math.max(0, Number(variantsPerRelation) || 0);
  if (perRelation === 0) return { pair_variants: 0, by_relation: {} };

  const selectedDifficulties = normalizeDifficultyLevels(difficulties);
  const selectedRelations = relations.filter(relation => PAIR_RELATIONS.includes(relation));
  const rng = makeRng(`${seed}-pair-variants`);
  const curveTypes = ['arc', 's_curve', 'helix', 'random_spline'];
  const cameras = ['front', 'iso_fr', 'oblique', 'top'];
  const occlusions = ['none', 'partial', 'heavy'];
  const byDifficulty = new Map();
  for (const source of sources || []) {
    const difficulty = source?.difficulty || 'easy';
    if (!selectedDifficulties.includes(difficulty)) continue;
    if (!source?.filename || !Array.isArray(source?.bead_sequence) || source.bead_sequence.length < 2) continue;
    if (!byDifficulty.has(difficulty)) byDifficulty.set(difficulty, []);
    byDifficulty.get(difficulty).push(source);
  }

  let emitted = 0;
  const byRelation = {};
  const total = selectedDifficulties.length * selectedRelations.length * perRelation;

  for (const difficulty of selectedDifficulties) {
    const pool = byDifficulty.get(difficulty) || [];
    if (!pool.length) continue;
    byRelation[difficulty] = {};
    for (const relation of selectedRelations) {
      let relationCount = 0;
      let attempts = 0;
      while (relationCount < perRelation && attempts < pool.length * 8) {
        const source = pool[(relationCount + attempts) % pool.length];
        attempts++;
        const seq = source.bead_sequence || [];
        if (relation === 'CYCLIC_ROTATION' && seq.length < 4) continue;
        const variantSeq = makeRelationSequence(seq, relation, rng);
        if (!variantSeq || sequenceRelation(seq, variantSeq) !== relation) continue;

        const curveType = chooseDifferent(curveTypes, source.curve_type, rng);
        const cam = chooseDifferent(cameras, source.camera_angle, rng);
        const occlusion = chooseDifferent(occlusions, source.occlusion_level, rng);
        const complexity = 0.35 + rng() * 0.45;
        const itemSeed = `${seed}-pair-${difficulty}-${relation}-${relationCount}`;

        disposeCurrent();
        const { group, metadata } = createBeadString({
          curveType,
          curveComplexity: complexity,
          numBeads: variantSeq.length,
          beadColors: variantSeq,
          beadSize: datasetBeadSize(rng),
          ropeThickness: datasetRopeThickness(rng),
          seed: itemSeed,
          isRing: false,
          showStartMarker: true,
        });
        root.add(group);
        currentObjects.groups.push(group);
        setCameraPreset(cam, group);

        await new Promise(r => requestAnimationFrame(r));
        const dataUrl = capturePNG(renderWidth, renderHeight);
        const diff = computeBeadStringDifficulty({
          numBeads: variantSeq.length,
          curveComplexity: complexity,
          occlusionLevel: occlusion,
          cameraAngle: cam,
          beadColors: metadata.bead_sequence,
        });
        const sample = {
          ...metadata,
          camera_angle: cam,
          camera_position: [...CAMERA_PRESETS[cam].pos],
          occlusion_level: occlusion,
          difficulty_score: diff.score,
          difficulty,
          difficulty_bucket: diff.level,
          difficulty_factors: diff.factors,
          image_size: [renderWidth, renderHeight],
          image_data_url: dataUrl,
          filename: `bead_pair_${difficulty}_${relation.toLowerCase()}_${String(relationCount).padStart(3, '0')}_${cam}.png`,
          _pair_variant: true,
          _pair_source: source.filename,
          _pair_relation_target: relation,
          _pair_variant_index: relationCount,
        };
        if (sampleCb) await sampleCb(sample);
        relationCount++;
        emitted++;
        byRelation[difficulty][relation] = relationCount;
        await progressCb(emitted, total, `Generating [pair-${relation}]`);
      }
    }
  }

  disposeCurrent();
  return { pair_variants: emitted, by_relation: byRelation };
}

/**
 * Download dataset as ZIP (requires JSZip loaded externally).
 */
export async function downloadDatasetAsZip(dataset, progressCb = () => {}) {
  if (typeof JSZip === 'undefined') {
    // Dynamically load JSZip
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  const zip = new JSZip();
  const imgFolder = zip.folder('images');
  const metaList = [];

  for (let i = 0; i < dataset.samples.length; i++) {
    const sample = dataset.samples[i];
    const { image_data_url, filename, ...meta } = sample;

    // Convert data URL to binary
    const base64 = image_data_url.split(',')[1];
    imgFolder.file(filename, base64, { base64: true });

    meta.filename = filename;
    metaList.push(meta);

    progressCb(i + 1, dataset.samples.length, 'Packing ZIP');
  }

  // Add metadata JSON
  zip.file('dataset_metadata.json', JSON.stringify({
    version: dataset.version,
    task: dataset.task,
    generated_at: dataset.generated_at,
    stats: dataset.stats,
    samples: metaList,
  }, null, 2));

  const blob = await zip.generateAsync({ type: 'blob' }, (meta) => {
    progressCb(Math.round(meta.percent), 100, 'Compressing ZIP');
  });

  // Download
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bead_string_dataset_${Date.now()}.zip`;
  a.click();
  URL.revokeObjectURL(url);

  return { zipSize: blob.size };
}

// ============= Init & Wire UI =============

function init() {
  try {
    initThree();
    animate();
  } catch (e) {
    console.error('Three.js init failed:', e);
    alert('Three.js 初始化失败: ' + e.message);
    return;
  }

  // Generate button
  document.getElementById('btnGenerate')?.addEventListener('click', () => {
    try {
      buildScene(readUI());
    } catch (e) {
      console.error('Build scene failed:', e);
      alert('生成失败: ' + e.message);
    }
  });

  // Export JSON
  document.getElementById('btnExport')?.addEventListener('click', exportJSON);

  // Capture PNG
  document.getElementById('btnCapture')?.addEventListener('click', downloadPNG);

  // 8-angle batch
  document.getElementById('btnBatchCapture')?.addEventListener('click', batchCapture8Angles);

  // Camera angle change → live preview
  document.getElementById('cameraAngle')?.addEventListener('change', (e) => {
    setCameraPreset(e.target.value);
  });

  // Auto-generate on load
  try {
    buildScene(readUI());
  } catch (e) {
    console.error('Initial build failed:', e);
    alert('生成失败: ' + e.message);
  }
}

// Start
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
