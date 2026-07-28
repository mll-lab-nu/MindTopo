import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "https://unpkg.com/three@0.161.0/examples/jsm/loaders/OBJLoader.js";

const CATALOG_URL = "./catalog.json";
const TASK_CATEGORY = "separation";
const QUESTION_PROMPT =
  "These images show multiple views of separated subassemblies from the same object. If the subassemblies are combined, which object category does the complete object belong to?";
const VIEW_SPECS = [
  { key: "front", label: "Front view", direction: [0, 0, 1], up: [0, 1, 0] },
  { key: "top", label: "Top view", direction: [0, 1, 0], up: [0, 0, -1] },
  { key: "side", label: "Side view", direction: [1, 0, 0], up: [0, 1, 0] },
  { key: "oblique", label: "45° oblique view", direction: [1, 1, 1], up: [0, 1, 0] }
];
const FRONT_UPPER_RIGHT_OBLIQUE_VIEW = {
  key: "oblique_front_upper_right",
  label: "Front upper right 45° oblique",
  direction: [1, 1, 1],
  up: [0, 1, 0]
};
const BACK_UPPER_LEFT_OBLIQUE_VIEW = {
  key: "oblique_back_upper_left",
  label: "Back upper left 45° oblique",
  direction: [-1, 1, -1],
  up: [0, 1, 0]
};
const COMPLETE_OBJECT_VIEW_SPECS = [
  FRONT_UPPER_RIGHT_OBLIQUE_VIEW,
  BACK_UPPER_LEFT_OBLIQUE_VIEW
];
const OPTION_TASK_VIEW_SPECS = [FRONT_UPPER_RIGHT_OBLIQUE_VIEW];
const COMPLETE_OBJECT_PANEL_SHIFT_Y = 50;
const RENDER_WIDTH = 420;
const RENDER_HEIGHT = 320;
const OPTION_VISIBILITY_MIN_PIXELS = 13;
const OPTION_VISIBILITY_MIN_SOLO_RATIO = 0.13;
const OPTION_VISIBILITY_WHITE_THRESHOLD = 200;
const OPTION_MODES = new Set(["revised"]);
const COMPLETE_VISIBILITY_EXCLUDED_OBJECT_KEYS = new Set([
  "Bench/tjusig",
  "Chair/applaro_2",
  "Chair/dalfred",
  "Chair/klappsta",
  "Chair/lisabo",
  "Chair/skogsta",
  "Chair/stig",
  "Chair/teodores",
  "Desk/alex",
  "Desk/fredrik",
  "Misc/vaniljstang",
  "Table/gladom",
  "Table/tornviken",
  "Table/vadholma",
  "Table/voxlov"
]);
const PART_COLOR = "#8a8f93";

const objLoader = new OBJLoader();
const assetCache = new Map();

const categorySelect = document.getElementById("category-select");
const objectSelect = document.getElementById("object-select");
const seedInput = document.getElementById("seed-input");
const optionModeSelect = document.getElementById("option-mode-select");
const generateButton = document.getElementById("generate-btn");
const statusRoot = document.getElementById("status");
const metadataRoot = document.getElementById("metadata");
const viewGridRoot = document.getElementById("view-grid");
const badgeObject = document.getElementById("badge-object");
const badgeParts = document.getElementById("badge-parts");
const questionText = document.getElementById("question-text");
const optionsList = document.getElementById("options-list");
const interactiveViewerRoot = document.getElementById("interactive-viewer");
const completeViewerRoot = document.getElementById("complete-viewer");
const optionViewerRoot = document.getElementById("option-viewer");
const mainPreviewTitle = document.getElementById("main-preview-title");
const mainPreviewCopy = document.getElementById("main-preview-copy");
const partSelectorRoot = document.getElementById("part-selector");
const selectorCountRoot = document.getElementById("selector-count");
const selectAllPartsButton = document.getElementById("select-all-parts");
const clearAllPartsButton = document.getElementById("clear-all-parts");

let catalog = null;
let eligibleEntries = [];
let binaryEligibleEntries = [];
let binaryTargetEntries = [];
let revisedTargetEntries = [];
let revisedThreePartTargetEntries = [];
let currentScene = null;
let interactiveSelectedIndices = [];

function setStatus(message) {
  if (statusRoot) {
    statusRoot.textContent = message;
  }
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(seedLike) {
  const seedText = String(seedLike ?? "12345");
  let hash = 2166136261;
  for (let index = 0; index < seedText.length; index += 1) {
    hash ^= seedText.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function shuffleWithRng(items, rng) {
  const next = items.slice();
  for (let index = next.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [next[index], next[swapIndex]] = [next[swapIndex], next[index]];
  }
  return next;
}

function normalizeOptionMode(value) {
  const mode = String(value || "revised").trim().toLowerCase();
  return OPTION_MODES.has(mode) ? mode : "revised";
}

function getSubassemblyCount(entry) {
  return entry?.category === "Chair" || entry?.category === "Table" ? 3 : 2;
}

function objectKey(entry) {
  return `${entry?.category || ""}/${entry?.name || ""}`;
}

function passesCompleteVisibilityFilter(entry) {
  return !COMPLETE_VISIBILITY_EXCLUDED_OBJECT_KEYS.has(objectKey(entry));
}

function listBinaryEligibleEntries() {
  if (!catalog) {
    return [];
  }
  return Object.values(catalog.categories)
    .flat()
    .filter((entry) => entry && Number(entry.part_count) >= 2 && passesCompleteVisibilityFilter(entry));
}

function listBinaryTargetEntries(entries = binaryEligibleEntries) {
  if (!entries.length) {
    return [];
  }
  const categoryCounts = new Map();
  for (const entry of entries) {
    categoryCounts.set(entry.category, (categoryCounts.get(entry.category) || 0) + 1);
  }
  const distinctCategories = new Set(entries.map((entry) => entry.category));
  return entries.filter(
    (entry) =>
      Number(entry.part_count) >= 3 &&
      (categoryCounts.get(entry.category) || 0) >= 2 &&
      distinctCategories.size >= 2
  );
}

function listRevisedTargetEntries(entries = binaryEligibleEntries) {
  return entries.filter((entry) => {
    if (Number(entry.part_count) < 3) {
      return false;
    }
    return entries.some(
      (other) =>
        other.category === entry.category &&
        other.name !== entry.name
    );
  });
}

function listRevisedThreePartTargetEntries(entries = binaryEligibleEntries) {
  return listRevisedTargetEntries(entries).filter((entry) => Number(entry.part_count) >= 4);
}

function targetEntriesForOptionMode(optionMode) {
  const normalizedMode = normalizeOptionMode(optionMode);
  if (normalizedMode === "revise_3parts") {
    return revisedThreePartTargetEntries;
  }
  if (normalizedMode === "revised") {
    return revisedTargetEntries;
  }
  return binaryTargetEntries;
}

function subassemblyCountForOptionMode(optionMode) {
  return 2;
}

function difficultyLabelForPartCount(partCount) {
  const value = Number(partCount);
  if (value <= 5) {
    return "easy";
  }
  if (value <= 10) {
    return "medium";
  }
  return "hard";
}

function chooseBalancedTwoPartition(entry, rng) {
  if (!entry || entry.part_count < 2) {
    throw new Error(`Entry ${entry?.category}/${entry?.name} does not have enough parts for a two-way split.`);
  }
  const shuffled = shuffleWithRng(
    Array.from({ length: entry.part_count }, (_value, index) => index),
    rng
  );
  const leftSize = Math.ceil(entry.part_count / 2);
  const groups = [
    shuffled.slice(0, leftSize).sort((left, right) => left - right),
    shuffled.slice(leftSize).sort((left, right) => left - right)
  ];
  return {
    groups,
    splitBalanceGap: Math.abs(groups[0].length - groups[1].length)
  };
}

function chooseBalancedNPartition(entry, rng, subassemblyCount) {
  if (!entry || entry.part_count < subassemblyCount) {
    throw new Error(
      `Entry ${entry?.category}/${entry?.name} does not have enough parts for a ${subassemblyCount}-way split.`
    );
  }
  const shuffled = shuffleWithRng(
    Array.from({ length: entry.part_count }, (_value, index) => index),
    rng
  );
  const baseSize = Math.floor(entry.part_count / subassemblyCount);
  const remainder = entry.part_count % subassemblyCount;
  const groups = [];
  let cursor = 0;
  for (let groupIndex = 0; groupIndex < subassemblyCount; groupIndex += 1) {
    const groupSize = baseSize + (groupIndex < remainder ? 1 : 0);
    groups.push(
      shuffled
        .slice(cursor, cursor + groupSize)
        .sort((left, right) => left - right)
    );
    cursor += groupSize;
  }
  return {
    groups,
    splitBalanceGap: Math.max(...groups.map((group) => group.length)) - Math.min(...groups.map((group) => group.length))
  };
}

function partitionSignature(groups) {
  return groups
    .map((group) => group.slice().sort((left, right) => left - right).join(","))
    .sort()
    .join("|");
}

function chooseDistinctBalancedPartitions(entry, rng, count, subassemblyCount) {
  const partitions = [];
  const seen = new Set();
  const maxAttempts = Math.max(80, count * 80);
  for (let attempt = 0; attempt < maxAttempts && partitions.length < count; attempt += 1) {
    const partition = chooseBalancedNPartition(entry, rng, subassemblyCount);
    const signature = partitionSignature(partition.groups);
    if (seen.has(signature)) {
      continue;
    }
    seen.add(signature);
    partitions.push(partition);
  }
  if (partitions.length < count) {
    throw new Error(
      `Entry ${entry.category}/${entry.name} cannot produce ${count} distinct ${subassemblyCount}-part partitions.`
    );
  }
  return partitions;
}

function chooseDistinctBalancedTwoPartitions(entry, rng, count) {
  return chooseDistinctBalancedPartitions(entry, rng, count, 2);
}

function clonePartRef(partRef) {
  return {
    ...partRef,
    transform: partRef.transform ? { ...partRef.transform } : undefined,
    optionDetail: partRef.optionDetail ? { ...partRef.optionDetail } : undefined,
  };
}

function clonePartRefGroups(groups) {
  return groups.map((group) => group.map((partRef) => clonePartRef(partRef)));
}

function buildPartSpecs(count) {
  return Array.from({ length: count }, (_value, index) => {
    const labelIndex = String.fromCharCode(65 + index);
    return {
      key: String.fromCharCode(97 + index),
      label: `Subassembly ${labelIndex}`,
      shortLabel: labelIndex
    };
  });
}

function setSelectOptions(select, options, selectedValue) {
  if (!select) {
    return;
  }
  select.innerHTML = "";
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    if (option.value === selectedValue) {
      element.selected = true;
    }
    select.appendChild(element);
  }
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }
  if (material && typeof material.dispose === "function") {
    material.dispose();
  }
}

function disposeObject(root) {
  if (!root) {
    return;
  }
  root.traverse((child) => {
    if (child.geometry && typeof child.geometry.dispose === "function") {
      child.geometry.dispose();
    }
    if (child.material) {
      disposeMaterial(child.material);
    }
  });
}

function colorizeObject(root, colorHex) {
  const color = new THREE.Color(colorHex);
  root.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    child.geometry = child.geometry.clone();
    child.material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.58,
      metalness: 0.06,
      side: THREE.DoubleSide
    });
  });
}

function applyMaskMaterial(root, colorHex) {
  const color = new THREE.Color(colorHex);
  root.traverse((child) => {
    if (!child.isMesh) {
      return;
    }
    child.geometry = child.geometry.clone();
    child.material = new THREE.MeshBasicMaterial({
      color,
      side: THREE.DoubleSide
    });
  });
}

function setMaskColor(root, colorHex) {
  const color = new THREE.Color(colorHex);
  root.traverse((child) => {
    if (!child.isMesh || !child.material) {
      return;
    }
    child.material.color.copy(color);
    child.material.needsUpdate = true;
  });
}

function cloneSceneDeep(root) {
  return root.clone(true);
}

function createDisplayObject(root) {
  root.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) {
    return root;
  }
  const center = box.getCenter(new THREE.Vector3());
  const wrapper = new THREE.Group();
  root.position.sub(center);
  wrapper.add(root);
  return wrapper;
}

async function loadObject(url) {
  if (!assetCache.has(url)) {
    assetCache.set(
      url,
      new Promise((resolve, reject) => {
        objLoader.load(
          url,
          (object) => resolve(object),
          undefined,
          (error) => reject(error)
        );
      })
    );
  }
  const original = await assetCache.get(url);
  return cloneSceneDeep(original);
}

function fitCameraToObject(camera, controls, object) {
  object.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) {
    return;
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.5);
  const fov = camera.fov * (Math.PI / 180);
  let distance = maxDim / (2 * Math.tan(fov / 2));
  distance *= 1.9;

  camera.position.set(center.x + distance, center.y + distance * 0.58, center.z + distance);
  camera.near = Math.max(distance / 100, 0.01);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}

class ModelViewer {
  constructor(root) {
    this.root = root;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#f8f3ea");
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.01, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.renderer.domElement.style.display = "block";

    this.root.innerHTML = "";
    this.root.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.minDistance = 0.2;
    this.controls.maxDistance = 160;
    this.controls.addEventListener("change", () => this.renderNow());

    this.stage = new THREE.Group();
    this.scene.add(this.stage);

    const hemi = new THREE.HemisphereLight("#ffffff", "#d9cfbf", 1.25);
    const key = new THREE.DirectionalLight("#ffffff", 1.15);
    key.position.set(4, 6, 5);
    const fill = new THREE.DirectionalLight("#f1e1c7", 0.7);
    fill.position.set(-5, 3, -4);
    this.scene.add(hemi, key, fill);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.root);
    this.resize();
  }

  setObject(object) {
    if (this.currentObject) {
      this.stage.remove(this.currentObject);
      disposeObject(this.currentObject);
    }
    this.currentObject = object;
    this.stage.add(object);
    fitCameraToObject(this.camera, this.controls, object);
    this.renderNow();
  }

  resize() {
    const width = Math.max(this.root.clientWidth, 1);
    const height = Math.max(this.root.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.renderNow();
  }

  renderNow() {
    this.renderer.render(this.scene, this.camera);
  }
}

function chooseBalancedPartition(entry, rng) {
  const subassemblyCount = getSubassemblyCount(entry);
  if (entry.part_count < subassemblyCount) {
    throw new Error(
      `Entry ${entry.category}/${entry.name} does not have enough parts for ${subassemblyCount} subassemblies.`
    );
  }

  const shuffled = shuffleWithRng(
    Array.from({ length: entry.part_count }, (_value, index) => index),
    rng
  );
  const baseSize = Math.floor(entry.part_count / subassemblyCount);
  const remainder = entry.part_count % subassemblyCount;
  const targetSizes = Array.from(
    { length: subassemblyCount },
    (_value, index) => baseSize + (index < remainder ? 1 : 0)
  );

  const groups = [];
  let cursor = 0;
  for (const targetSize of targetSizes) {
    groups.push(
      shuffled
        .slice(cursor, cursor + targetSize)
        .sort((left, right) => left - right)
    );
    cursor += targetSize;
  }

  return {
    groups,
    splitBalanceGap: Math.max(...targetSizes) - Math.min(...targetSizes)
  };
}

function chooseEntry(config, rng) {
  if (config.category && config.objectName) {
    const exact = eligibleEntries.find(
      (entry) => entry.category === config.category && entry.name === config.objectName
    );
    if (exact) {
      return exact;
    }
    throw new Error(`No eligible object found for ${config.category}/${config.objectName}.`);
  }

  if (config.category) {
    const categoryEntries = eligibleEntries.filter((entry) => entry.category === config.category);
    if (categoryEntries.length > 0) {
      return categoryEntries[Math.floor(rng() * categoryEntries.length)];
    }
  }

  if (config.objectName) {
    const byName = eligibleEntries.filter((entry) => entry.name === config.objectName);
    if (byName.length > 0) {
      return byName[Math.floor(rng() * byName.length)];
    }
  }

  return eligibleEntries[Math.floor(rng() * eligibleEntries.length)];
}

async function buildSubsetGroup(entry, indices) {
  const group = new THREE.Group();
  for (const index of indices) {
    const object = await loadObject(entry.parts[index]);
    colorizeObject(object, PART_COLOR);
    group.add(object);
  }
  return createDisplayObject(group);
}

function buildPartRefs(entry, indices) {
  return indices.map((index) => ({
    entry,
    partIndex: index,
    colorHex: PART_COLOR,
    partFile: entry.part_files[index],
  }));
}

function applyLocalPartTransform(object, transform = null) {
  if (!transform) {
    return object;
  }

  object.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) {
    return object;
  }

  const center = box.getCenter(new THREE.Vector3());
  const pivot = new THREE.Group();
  object.position.sub(center);
  pivot.position.copy(center);
  pivot.add(object);

  if (transform.mirrorX) {
    pivot.scale.x *= -1;
  }
  if (transform.mirrorZ) {
    pivot.scale.z *= -1;
  }
  if (transform.rotateX) {
    pivot.rotation.x += Number(transform.rotateX);
  }
  if (transform.rotateY) {
    pivot.rotation.y += Number(transform.rotateY);
  }
  if (transform.rotateZ) {
    pivot.rotation.z += Number(transform.rotateZ);
  }
  return pivot;
}

async function buildCompositeGroup(partRefs) {
  const group = new THREE.Group();
  for (const partRef of partRefs) {
    let object = await loadObject(partRef.entry.parts[partRef.partIndex]);
    colorizeObject(object, partRef.colorHex);
    object = applyLocalPartTransform(object, partRef.transform);
    group.add(object);
  }
  return createDisplayObject(group);
}

async function buildVisibilityCompositeGroup(partRefs) {
  const group = new THREE.Group();
  const partObjects = [];
  for (const partRef of partRefs) {
    let object = await loadObject(partRef.entry.parts[partRef.partIndex]);
    applyMaskMaterial(object, "#000000");
    object = applyLocalPartTransform(object, partRef.transform);
    group.add(object);
    partObjects.push(object);
  }
  return {
    object: createDisplayObject(group),
    partObjects
  };
}

class StaticViewRenderer {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#f8f3ea");
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1000);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.stage = new THREE.Group();
    this.scene.add(this.stage);

    const hemi = new THREE.HemisphereLight("#ffffff", "#d9cfbf", 1.28);
    const key = new THREE.DirectionalLight("#ffffff", 1.2);
    key.position.set(4, 6, 5);
    const fill = new THREE.DirectionalLight("#f1e1c7", 0.7);
    fill.position.set(-5, 3, -4);
    this.scene.add(hemi, key, fill);
  }

  renderToDataUrl(object, viewSpec, width, height) {
    const safeWidth = Math.max(width, 1);
    const safeHeight = Math.max(height, 1);
    const aspect = safeWidth / safeHeight;
    this.stage.add(object);

    object.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(object);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 0.6);
    const frustum = radius * 1.4;
    this.camera.left = -frustum * aspect;
    this.camera.right = frustum * aspect;
    this.camera.top = frustum;
    this.camera.bottom = -frustum;
    this.camera.near = 0.01;
    this.camera.far = radius * 20;

    const direction = new THREE.Vector3(...viewSpec.direction).normalize();
    const up = new THREE.Vector3(...viewSpec.up).normalize();
    this.camera.position.copy(sphere.center.clone().add(direction.multiplyScalar(radius * 3.2)));
    this.camera.up.copy(up);
    this.camera.lookAt(sphere.center);
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(safeWidth, safeHeight, false);
    this.renderer.render(this.scene, this.camera);
    const dataUrl = this.renderer.domElement.toDataURL("image/png");
    this.stage.remove(object);
    return dataUrl;
  }
}

class VisibilityRenderer {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#000000");
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1000);
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: true
    });
    this.renderer.setPixelRatio(1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.stage = new THREE.Group();
    this.scene.add(this.stage);
  }

  attach(object, viewSpec, width, height) {
    const safeWidth = Math.max(width, 1);
    const safeHeight = Math.max(height, 1);
    const aspect = safeWidth / safeHeight;
    this.stage.add(object);

    object.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(object);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 0.6);
    const frustum = radius * 1.4;
    this.camera.left = -frustum * aspect;
    this.camera.right = frustum * aspect;
    this.camera.top = frustum;
    this.camera.bottom = -frustum;
    this.camera.near = 0.01;
    this.camera.far = radius * 20;

    const direction = new THREE.Vector3(...viewSpec.direction).normalize();
    const up = new THREE.Vector3(...viewSpec.up).normalize();
    this.camera.position.copy(sphere.center.clone().add(direction.multiplyScalar(radius * 3.2)));
    this.camera.up.copy(up);
    this.camera.lookAt(sphere.center);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(safeWidth, safeHeight, false);
  }

  detach(object) {
    this.stage.remove(object);
  }

  countWhitePixels(width, height) {
    const safeWidth = Math.max(width, 1);
    const safeHeight = Math.max(height, 1);
    this.renderer.render(this.scene, this.camera);
    const pixels = new Uint8Array(safeWidth * safeHeight * 4);
    const gl = this.renderer.getContext();
    gl.readPixels(0, 0, safeWidth, safeHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    let count = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      if (
        pixels[index] >= OPTION_VISIBILITY_WHITE_THRESHOLD &&
        pixels[index + 1] >= OPTION_VISIBILITY_WHITE_THRESHOLD &&
        pixels[index + 2] >= OPTION_VISIBILITY_WHITE_THRESHOLD
      ) {
        count += 1;
      }
    }
    return count;
  }
}

const viewRenderer = new StaticViewRenderer();
const visibilityRenderer = new VisibilityRenderer();
const interactiveViewer = null;

async function renderNamedViews(partRefs, viewSpecs, prefix) {
  const group = await buildCompositeGroup(partRefs);
  const rendered = [];
  for (const viewSpec of viewSpecs) {
    const dataUrl = viewRenderer.renderToDataUrl(group, viewSpec, RENDER_WIDTH, RENDER_HEIGHT);
    rendered.push({
      imageKey: `${prefix}_${viewSpec.key}`,
      viewKey: viewSpec.key,
      viewLabel: viewSpec.label,
      dataUrl,
    });
  }
  disposeObject(group);
  return rendered;
}

async function evaluateSubassemblyOptionVisibility(partRefs, viewSpec) {
  if (!partRefs.length) {
    return { pass: true, parts: [], failures: [] };
  }

  const { object, partObjects } = await buildVisibilityCompositeGroup(partRefs);
  const parts = [];
  const failures = [];
  visibilityRenderer.attach(object, viewSpec, RENDER_WIDTH, RENDER_HEIGHT);

  try {
    for (let targetIndex = 0; targetIndex < partObjects.length; targetIndex += 1) {
      partObjects.forEach((partObject, index) => {
        partObject.visible = true;
        setMaskColor(partObject, index === targetIndex ? "#ffffff" : "#000000");
      });
      const visiblePixels = visibilityRenderer.countWhitePixels(RENDER_WIDTH, RENDER_HEIGHT);

      partObjects.forEach((partObject, index) => {
        partObject.visible = index === targetIndex;
        setMaskColor(partObject, "#ffffff");
      });
      const soloPixels = visibilityRenderer.countWhitePixels(RENDER_WIDTH, RENDER_HEIGHT);

      partObjects.forEach((partObject) => {
        partObject.visible = true;
        setMaskColor(partObject, "#000000");
      });

      const requiredPixels =
        soloPixels > 0
          ? Math.max(1, Math.min(OPTION_VISIBILITY_MIN_PIXELS, Math.ceil(soloPixels * OPTION_VISIBILITY_MIN_SOLO_RATIO)))
          : 1;
      const partRef = partRefs[targetIndex];
      const detail = {
        objectCategory: partRef.entry.category,
        objectName: partRef.entry.name,
        partIndex: partRef.partIndex,
        partFile: partRef.partFile,
        visiblePixels,
        soloPixels,
        requiredPixels,
        pass: visiblePixels >= requiredPixels
      };
      parts.push(detail);
      if (!detail.pass) {
        failures.push(detail);
      }
    }
  } finally {
    visibilityRenderer.detach(object);
    disposeObject(object);
  }

  return {
    pass: failures.length === 0,
    parts,
    failures
  };
}

function chooseEntryFromList(entries, rng, predicate) {
  const filtered = entries.filter(predicate);
  if (!filtered.length) {
    throw new Error("No matching object entry found for option task.");
  }
  return filtered[Math.floor(rng() * filtered.length)];
}

function decompositionPartLabel(subassemblyCount) {
  return Number(subassemblyCount) === 3 ? "three-part" : "two-part";
}

function candidateSubassemblyLabel(subassemblyCount) {
  return Number(subassemblyCount) === 3 ? "three candidate subassemblies" : "two candidate subassemblies";
}

function optionTaskQuestion(subassemblyCount = 2) {
  return (
    `Options A-E each show ${candidateSubassemblyLabel(subassemblyCount)}, with one view per subassembly. ` +
    `Which option shows the correct ${decompositionPartLabel(subassemblyCount)} decomposition of the complete object?`
  );
}

function chooseSimilarSameCategoryEntry(targetEntry, rng) {
  const sourceEntries = revisedTargetEntries.length ? revisedTargetEntries : binaryEligibleEntries;
  return chooseEntryFromList(
    sourceEntries,
    rng,
    (entry) => entry.category === targetEntry.category && entry.name !== targetEntry.name
  );
}

function replacementPartIndexFor(targetPartIndex, sourceEntry, rng) {
  if (Number(sourceEntry.part_count) > targetPartIndex) {
    return targetPartIndex;
  }
  return Math.floor(rng() * Number(sourceEntry.part_count));
}

function replaceOneTargetPart(groups, targetEntry, rng, optionType, usedTargetPartIndices = new Set()) {
  const clonedGroups = clonePartRefGroups(groups);
  const replacementCandidates = [];
  clonedGroups.forEach((group, groupIndex) => {
    group.forEach((partRef, partRefIndex) => {
      const alreadyUsed = usedTargetPartIndices.has(partRef.partIndex);
      replacementCandidates.push({ groupIndex, partRefIndex, partRef, alreadyUsed });
    });
  });

  const preferred = replacementCandidates.filter((candidate) => !candidate.alreadyUsed);
  const candidatePool = preferred.length ? preferred : replacementCandidates;
  const selected = candidatePool[Math.floor(rng() * candidatePool.length)];
  usedTargetPartIndices.add(selected.partRef.partIndex);

  const sourceEntry = chooseSimilarSameCategoryEntry(targetEntry, rng);
  const sourcePartIndex = replacementPartIndexFor(selected.partRef.partIndex, sourceEntry, rng);
  clonedGroups[selected.groupIndex][selected.partRefIndex] = {
    entry: sourceEntry,
    partIndex: sourcePartIndex,
    colorHex: selected.partRef.colorHex,
    partFile: sourceEntry.part_files[sourcePartIndex],
    optionDetail: {
      optionType,
      replacedTargetPartIndex: selected.partRef.partIndex,
      replacementCategory: sourceEntry.category,
      replacementObjectName: sourceEntry.name,
      replacementPartIndex: sourcePartIndex,
    },
  };
  return clonedGroups;
}

function removeOneTargetPart(groups, rng, usedTargetPartIndices = new Set()) {
  const clonedGroups = clonePartRefGroups(groups);
  const candidates = [];
  clonedGroups.forEach((group, groupIndex) => {
    group.forEach((partRef, partRefIndex) => {
      if (group.length <= 1) {
        return;
      }
      const alreadyUsed = usedTargetPartIndices.has(partRef.partIndex);
      candidates.push({ groupIndex, partRefIndex, partRef, alreadyUsed });
    });
  });
  if (!candidates.length) {
    throw new Error("Cannot create a missing-component option without an oversized subassembly.");
  }
  const preferred = candidates.filter((candidate) => !candidate.alreadyUsed);
  const candidatePool = preferred.length ? preferred : candidates;
  const selected = candidatePool[Math.floor(rng() * candidatePool.length)];
  usedTargetPartIndices.add(selected.partRef.partIndex);

  clonedGroups[selected.groupIndex] = clonedGroups[selected.groupIndex]
    .filter((_partRef, partRefIndex) => partRefIndex !== selected.partRefIndex);
  return clonedGroups;
}

function addOneSimilarSameCategoryPart(groups, targetEntry, rng) {
  const clonedGroups = clonePartRefGroups(groups);
  if (!clonedGroups.length) {
    throw new Error("Cannot create an extra-component option without subassemblies.");
  }

  const sourceEntry = chooseSimilarSameCategoryEntry(targetEntry, rng);
  const sourcePartIndex = Math.floor(rng() * Number(sourceEntry.part_count));
  const sourcePartRef = buildPartRefs(sourceEntry, [sourcePartIndex])[0];
  const targetGroupIndex = Math.floor(rng() * clonedGroups.length);
  clonedGroups[targetGroupIndex] = clonedGroups[targetGroupIndex].concat({
    ...sourcePartRef,
    optionDetail: {
      optionType: "extra_same_category_component_from_different_partition",
      extraCategory: sourceEntry.category,
      extraObjectName: sourceEntry.name,
      extraPartIndex: sourcePartIndex,
    },
  });
  return clonedGroups;
}

function buildLegacyOptionSpecs(targetEntry, seedValue, partitionRng) {
  const correctPartition = chooseBalancedTwoPartition(targetEntry, partitionRng);
  const correctPartRefs = correctPartition.groups.map((group) => buildPartRefs(targetEntry, group));

  const removableGroupIndices = correctPartRefs
    .map((group, index) => ({ group, index }))
    .filter(({ group }) => group.length > 1);
  if (!removableGroupIndices.length) {
    throw new Error(`Target entry ${targetEntry.category}/${targetEntry.name} cannot produce a missing-part distractor.`);
  }
  const removalChoice = removableGroupIndices[Math.floor(partitionRng() * removableGroupIndices.length)];
  const removePartAt = Math.floor(partitionRng() * removalChoice.group.length);
  const missingPartRefs = correctPartRefs.map((group) => group.map((partRef) => ({ ...partRef })));
  missingPartRefs[removalChoice.index] = missingPartRefs[removalChoice.index]
    .filter((_partRef, index) => index !== removePartAt);

  const foreignPartSource = chooseEntryFromList(
    binaryEligibleEntries,
    partitionRng,
    (entry) => !(entry.category === targetEntry.category && entry.name === targetEntry.name)
  );
  const foreignPartIndex = Math.floor(partitionRng() * foreignPartSource.part_count);
  const foreignPartRef = buildPartRefs(foreignPartSource, [foreignPartIndex])[0];
  const extraPartRefs = correctPartRefs.map((group) => group.map((partRef) => ({ ...partRef })));
  const extraTargetGroup = Math.floor(partitionRng() * extraPartRefs.length);
  extraPartRefs[extraTargetGroup] = extraPartRefs[extraTargetGroup]
    .concat({ ...foreignPartRef });

  const sameCategoryEntry = chooseEntryFromList(
    binaryEligibleEntries,
    partitionRng,
    (entry) => entry.category === targetEntry.category && entry.name !== targetEntry.name
  );
  const sameCategoryPartition = chooseBalancedTwoPartition(
    sameCategoryEntry,
    mulberry32(hashSeed(`option-same-${seedValue}-${sameCategoryEntry.category}-${sameCategoryEntry.name}`))
  );
  const sameCategoryRefs = sameCategoryPartition.groups.map((group) => buildPartRefs(sameCategoryEntry, group));

  const differentCategoryEntry = chooseEntryFromList(
    binaryEligibleEntries,
    partitionRng,
    (entry) => entry.category !== targetEntry.category
  );
  const differentCategoryPartition = chooseBalancedTwoPartition(
    differentCategoryEntry,
    mulberry32(hashSeed(`option-diff-${seedValue}-${differentCategoryEntry.category}-${differentCategoryEntry.name}`))
  );
  const differentCategoryRefs = differentCategoryPartition.groups.map((group) => buildPartRefs(differentCategoryEntry, group));

  return {
    correctPartition,
    optionSpecs: [
      {
        optionType: "correct_two_parts",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        groups: correctPartRefs,
      },
      {
        optionType: "missing_one_part",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        groups: missingPartRefs,
      },
      {
        optionType: "extra_one_part",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        groups: extraPartRefs,
      },
      {
        optionType: "same_category_other_object",
        sourceCategory: sameCategoryEntry.category,
        sourceObjectName: sameCategoryEntry.name,
        groups: sameCategoryRefs,
      },
      {
        optionType: "different_category_other_object",
        sourceCategory: differentCategoryEntry.category,
        sourceObjectName: differentCategoryEntry.name,
        groups: differentCategoryRefs,
      }
    ],
  };
}

function buildRevisedOptionSpecs(targetEntry, seedValue, partitionRng, subassemblyCount = 2) {
  const partitions = chooseDistinctBalancedPartitions(targetEntry, partitionRng, 3, subassemblyCount);
  const correctPartition = partitions[0];
  const correctPartRefs = correctPartition.groups.map((group) => buildPartRefs(targetEntry, group));
  const usedTargetPartIndices = new Set();

  const replacementBGroups = replaceOneTargetPart(
    partitions[1].groups.map((group) => buildPartRefs(targetEntry, group)),
    targetEntry,
    partitionRng,
    "one_target_part_replaced_by_similar_same_category_part",
    usedTargetPartIndices
  );
  const replacementCGroups = replaceOneTargetPart(
    partitions[2].groups.map((group) => buildPartRefs(targetEntry, group)),
    targetEntry,
    partitionRng,
    "another_target_part_replaced_by_similar_same_category_part",
    usedTargetPartIndices
  );

  const missingGroups = removeOneTargetPart(
    partitions[1].groups.map((group) => buildPartRefs(targetEntry, group)),
    partitionRng,
    usedTargetPartIndices
  );

  const extraGroups = addOneSimilarSameCategoryPart(
    partitions[2].groups.map((group) => buildPartRefs(targetEntry, group)),
    targetEntry,
    partitionRng
  );

  return {
    correctPartition,
    optionSpecs: [
      {
        optionType: "correct_two_parts",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        groups: correctPartRefs,
      },
      {
        optionType: "one_target_part_replaced_by_similar_same_category_part",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        optionDetail: "different_target_partition_with_one_same_category_component_replacement",
        groups: replacementBGroups,
      },
      {
        optionType: "another_target_part_replaced_by_similar_same_category_part",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        optionDetail: "another_different_target_partition_with_one_same_category_component_replacement",
        groups: replacementCGroups,
      },
      {
        optionType: "missing_one_target_component_from_different_partition",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        optionDetail: "different_target_partition_with_one_target_component_removed",
        groups: missingGroups,
      },
      {
        optionType: "extra_same_category_component_from_different_partition",
        sourceCategory: targetEntry.category,
        sourceObjectName: targetEntry.name,
        optionDetail: "different_target_partition_with_one_same_category_component_added",
        groups: extraGroups,
      }
    ],
  };
}

async function generateTwoPartOptionTask(config = {}) {
  if (!catalog) {
    throw new Error("Catalog not loaded yet.");
  }
  const optionMode = normalizeOptionMode(config.optionMode ?? optionModeSelect?.value ?? "revised");
  const subassemblyCount = subassemblyCountForOptionMode(optionMode);
  const targetEntries = targetEntriesForOptionMode(optionMode);
  if (!targetEntries.length) {
    throw new Error(`No eligible objects found for the ${optionMode} option task.`);
  }

  const seedValue = clampInt(config.seed ?? seedInput?.value ?? 12345, 0, 2147483647, 12345);
  const targetSeedKey = optionMode === "legacy" ? `option-target-${seedValue}` : `option-target-${optionMode}-${seedValue}`;
  const targetRng = mulberry32(hashSeed(targetSeedKey));

  const targetEntry = (() => {
    if (config.category && config.objectName) {
      const exact = targetEntries.find(
        (entry) => entry.category === config.category && entry.name === config.objectName
      );
      if (!exact) {
        throw new Error(
          `No eligible ${optionMode} option-task object found for ${config.category}/${config.objectName}.`
        );
      }
      return exact;
    }
    if (config.category) {
      return chooseEntryFromList(
        targetEntries,
        targetRng,
        (entry) => entry.category === config.category
      );
    }
    return targetEntries[Math.floor(targetRng() * targetEntries.length)];
  })();

  const partitionSeedKey =
    optionMode === "legacy"
      ? `option-partition-${seedValue}-${targetEntry.category}-${targetEntry.name}`
      : `option-partition-${optionMode}-${seedValue}-${targetEntry.category}-${targetEntry.name}`;
  const partitionRng = mulberry32(hashSeed(partitionSeedKey));
  const { correctPartition, optionSpecs } =
    optionMode === "revised" || optionMode === "revise_3parts"
      ? buildRevisedOptionSpecs(targetEntry, seedValue, partitionRng, subassemblyCount)
      : buildLegacyOptionSpecs(targetEntry, seedValue, partitionRng);

  const optionOrder = shuffleWithRng(
    optionSpecs.map((optionSpec, index) => ({ optionSpec, originalIndex: index })),
    mulberry32(hashSeed(
      optionMode === "legacy"
        ? `option-order-${seedValue}-${targetEntry.category}-${targetEntry.name}`
        : `option-order-${optionMode}-${seedValue}-${targetEntry.category}-${targetEntry.name}`
    ))
  );

  const wholeViews = await renderNamedViews(
    buildPartRefs(targetEntry, Array.from({ length: targetEntry.part_count }, (_value, index) => index)),
    COMPLETE_OBJECT_VIEW_SPECS,
    "whole"
  );

  const options = [];
  const imageOrder = [];
  wholeViews.forEach((view) => imageOrder.push({ ...view, optionLetter: null, subassemblyIndex: null }));

  for (let optionIndex = 0; optionIndex < optionOrder.length; optionIndex += 1) {
    const optionLetter = String.fromCharCode(65 + optionIndex);
    const { optionSpec } = optionOrder[optionIndex];
    const subassemblies = [];
    for (let subassemblyIndex = 0; subassemblyIndex < optionSpec.groups.length; subassemblyIndex += 1) {
      const subassemblyPartRefs = optionSpec.groups[subassemblyIndex];
      const subassemblyViews = await renderNamedViews(
        subassemblyPartRefs,
        OPTION_TASK_VIEW_SPECS,
        `option_${optionLetter}_subassembly_${subassemblyIndex + 1}`
      );
      const visibility = await evaluateSubassemblyOptionVisibility(
        subassemblyPartRefs,
        OPTION_TASK_VIEW_SPECS[0]
      );
      subassemblies.push({
        subassemblyIndex,
        partCount: subassemblyPartRefs.length,
        partFiles: subassemblyPartRefs.map((partRef) => partRef.partFile),
        visibility,
        views: subassemblyViews,
      });
      subassemblyViews.forEach((view) =>
        imageOrder.push({
          ...view,
          optionLetter,
          subassemblyIndex,
        })
      );
    }
    options.push({
      letter: optionLetter,
      optionType: optionSpec.optionType,
      sourceCategory: optionSpec.sourceCategory,
      sourceObjectName: optionSpec.sourceObjectName,
      optionDetail: optionSpec.optionDetail || null,
      subassemblies,
    });
  }

  const optionVisibilityFailures = [];
  options.forEach((option) => {
    option.subassemblies.forEach((subassembly) => {
      (subassembly.visibility?.failures || []).forEach((failure) => {
        optionVisibilityFailures.push({
          optionLetter: option.letter,
          subassemblyIndex: subassembly.subassemblyIndex,
          ...failure
        });
      });
    });
  });

  const correctOption = options.find((option) => option.optionType === "correct_two_parts");
  if (!correctOption) {
    throw new Error("Failed to build a correct option for the decomposition option task.");
  }

  return {
    taskCategory: "separation",
    questionType: "subassembly_option_mcq",
    optionMode,
    question: optionTaskQuestion(subassemblyCount),
    objectCategory: targetEntry.category,
    objectName: targetEntry.name,
    partCount: targetEntry.part_count,
    subassemblyCount,
    imageCount: 1 + options.length,
    splitBalanceGap: correctPartition.splitBalanceGap,
    seed: seedValue,
    correctOptionLetter: correctOption.letter,
    correctOptionType: correctOption.optionType,
    optionVisibilityPass: optionVisibilityFailures.length === 0,
    optionVisibilityFailures,
    wholeViews,
    options,
    imageOrder,
  };
}

function viewSelector(partKey, viewKey) {
  return `#view-${partKey}-${viewKey}`;
}

function buildViewGrid(partSpecs) {
  if (!viewGridRoot) {
    return;
  }
  viewGridRoot.innerHTML = "";
  for (const partSpec of partSpecs) {
    for (const viewSpec of VIEW_SPECS) {
      const article = document.createElement("article");
      article.id = `view-${partSpec.key}-${viewSpec.key}`;
      article.className = "view-card";

      const heading = document.createElement("div");
      const title = document.createElement("h2");
      title.textContent = partSpec.label;
      const subtitle = document.createElement("div");
      subtitle.className = "view-subtitle";
      subtitle.textContent = viewSpec.label;
      heading.append(title, subtitle);

      const frame = document.createElement("div");
      frame.className = "view-frame";
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "Loading...";
      frame.appendChild(empty);

      article.append(heading, frame);
      viewGridRoot.appendChild(article);
    }
  }
}

function setViewImage(partKey, viewKey, dataUrl, altText) {
  const root = document.querySelector(viewSelector(partKey, viewKey));
  const frame = root?.querySelector(".view-frame");
  if (!frame) {
    return;
  }
  frame.innerHTML = "";
  const image = document.createElement("img");
  image.className = "view-image";
  image.alt = altText;
  image.src = dataUrl;
  frame.appendChild(image);
}

function updateBadges(scene) {
  if (badgeObject) {
    badgeObject.textContent = `Object: ${scene.objectCategory}/${scene.objectName}`;
  }
  if (badgeParts) {
    badgeParts.textContent = `Difficulty: ${difficultyLabelForPartCount(scene.partCount)} · Parts: ${scene.partCount}`;
  }
}

function syncInteractiveSelector(entry) {
  if (!partSelectorRoot || !entry) {
    return;
  }

  if (selectorCountRoot) {
    selectorCountRoot.textContent = `Selected ${interactiveSelectedIndices.length} / ${entry.part_count}`;
  }

  for (const option of partSelectorRoot.querySelectorAll(".part-option")) {
    const index = Number(option.dataset.index);
    const selected = interactiveSelectedIndices.includes(index);
    option.classList.toggle("is-off", !selected);
    const input = option.querySelector("input");
    if (input) {
      input.checked = selected;
    }
  }
}

function buildInteractiveSelector(entry) {
  if (!partSelectorRoot) {
    return;
  }
  partSelectorRoot.innerHTML = "";
  for (let index = 0; index < entry.part_count; index += 1) {
    const option = document.createElement("label");
    option.className = "part-option";
    option.dataset.index = String(index);

    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.index = String(index);
    input.checked = interactiveSelectedIndices.includes(index);

    const swatch = document.createElement("span");
    swatch.className = "part-swatch";
    swatch.style.background = PART_COLOR;

    const meta = document.createElement("span");
    meta.className = "part-meta";

    const name = document.createElement("span");
    name.className = "part-name";
    name.textContent = `Part ${String(index).padStart(2, "0")}`;

    const file = document.createElement("span");
    file.className = "part-file-small";
    file.textContent = entry.part_files[index];

    meta.append(name, file);
    option.append(input, swatch, meta);
    partSelectorRoot.appendChild(option);
  }
  syncInteractiveSelector(entry);
}

async function renderInteractiveViewer(entry) {
  if (!interactiveViewer || !entry) {
    return;
  }
  const group = await buildSubsetGroup(entry, interactiveSelectedIndices);
  interactiveViewer.setObject(group);
}

async function renderCompletePreview(scene) {
  const item = await buildCompletePreviewItem(scene);
  setCompletePreview(item);
}

async function buildCompletePreviewItem(scene) {
  const wholeViewItems = scene.wholeViews.filter((view) => view.dataUrl);
  const dataUrl = await composeCaptionedImage(
    wholeViewItems.map((view) => view.dataUrl),
    wholeViewItems.map((view) => view.viewLabel || "Oblique view"),
    "",
    "Complete Object"
  );
  return {
    key: "complete_object",
    title: "Complete Object",
    copy: "Whole-object image used as the target.",
    alt: `${scene.objectName} complete object view`,
    dataUrl,
  };
}

async function buildOptionPreviewItem(option) {
  const dataUrl = await composeCaptionedImage(
    option.subassemblies.map((subassembly) => subassembly.views[0]?.dataUrl || ""),
    option.subassemblies.map((_subassembly, index) => `Subassembly ${index + 1}`),
    `Option ${option.letter}`
  );
  return {
    key: `option_${option.letter}`,
    title: `Option ${option.letter}`,
    copy: `${option.subassemblies.length} subassemblies`,
    alt: `Option ${option.letter} candidate decomposition`,
    dataUrl,
  };
}

function syncThumbnailSelection(selectedKey) {
  if (!viewGridRoot) {
    return;
  }
  for (const button of viewGridRoot.querySelectorAll(".thumbnail-card")) {
    button.classList.toggle("is-selected", button.dataset.previewKey === selectedKey);
  }
}

function setMainPreview(item) {
  if (!optionViewerRoot || !item) {
    return;
  }
  if (mainPreviewTitle) {
    mainPreviewTitle.textContent = item.title;
  }
  if (mainPreviewCopy) {
    mainPreviewCopy.textContent = item.copy || "";
  }
  optionViewerRoot.innerHTML = "";
  const image = document.createElement("img");
  image.className = "view-image";
  image.alt = item.alt;
  image.src = item.dataUrl;
  optionViewerRoot.appendChild(image);
  syncThumbnailSelection(item.key);
}

function setCompletePreview(item) {
  if (!completeViewerRoot || !item) {
    return;
  }
  completeViewerRoot.innerHTML = "";
  const image = document.createElement("img");
  image.className = "view-image";
  image.alt = item.alt;
  image.src = item.dataUrl;
  completeViewerRoot.appendChild(image);
}

async function renderOptionTaskGrid(scene) {
  if (!viewGridRoot) {
    return;
  }
  viewGridRoot.innerHTML = "";
  const completeItem = await buildCompletePreviewItem(scene);
  setCompletePreview(completeItem);
  const previewItems = [];
  for (const option of scene.options) {
    previewItems.push(await buildOptionPreviewItem(option));
  }

  for (const item of previewItems) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thumbnail-card";
    button.dataset.previewKey = item.key;

    const title = document.createElement("span");
    title.className = "thumbnail-title";
    title.textContent = item.title;

    const frame = document.createElement("span");
    frame.className = "thumbnail-frame";
    const image = document.createElement("img");
    image.alt = item.alt;
    image.src = item.dataUrl;

    frame.appendChild(image);
    button.append(title, frame);
    button.addEventListener("click", () => setMainPreview(item));
    viewGridRoot.appendChild(button);
  }
  setMainPreview(previewItems[0]);
}

function updateMetadata(scene) {
  if (!metadataRoot) {
    return;
  }
  const rows = [
    ["Task Category", TASK_CATEGORY],
    ["Object Category", scene.objectCategory],
    ["Object", scene.objectName],
    ["Primitive Parts", String(scene.partCount)],
    ["Subassemblies", String(scene.subassemblyCount)],
    ["Question Images", String(scene.imageCount)],
    ["Split Balance Gap", String(scene.splitBalanceGap)],
    ["Seed", String(scene.seed)],
    ["Option Mode", scene.optionMode || "revised"],
    ["Option Visibility", scene.optionVisibilityPass ? "pass" : "fail"],
    ["Question Type", scene.questionType],
    ["Correct Option", scene.correctOptionLetter],
    ["Correct Option Type", scene.correctOptionType]
  ];
  scene.options.forEach((option) => {
    rows.push([
      `Option ${option.letter}`,
      `${option.optionType} · ${option.sourceCategory}/${option.sourceObjectName}`
    ]);
  });
  metadataRoot.innerHTML = rows
    .map(
      ([key, value]) => `
        <div class="meta-row">
          <span class="meta-key">${key}</span>
          <span>${value}</span>
        </div>
      `
    )
    .join("");
}

function updateQuestionPreview(scene) {
  if (questionText) {
    questionText.textContent = scene.question;
  }
  if (optionsList) {
    optionsList.innerHTML = "";
  }
}

function updateDebugState(scene) {
  window.__lastScene = scene;
}

function updateCategoryOptions(optionMode, selectedCategory = "") {
  if (!categorySelect) {
    return selectedCategory;
  }
  const targetEntries = targetEntriesForOptionMode(optionMode);
  const categoryNames = Array.from(new Set(targetEntries.map((entry) => entry.category))).sort();
  const nextSelectedCategory = selectedCategory && categoryNames.includes(selectedCategory) ? selectedCategory : "";
  setSelectOptions(
    categorySelect,
    [{ value: "", label: "Random eligible object" }].concat(
      categoryNames.map((categoryName) => ({
        value: categoryName,
        label: `${categoryName} (${targetEntries.filter((entry) => entry.category === categoryName).length})`
      }))
    ),
    nextSelectedCategory
  );
  return nextSelectedCategory;
}

function updateSelects(scene) {
  const optionMode = scene.optionMode || "revised";
  const targetEntries = targetEntriesForOptionMode(optionMode);
  updateCategoryOptions(optionMode, scene.objectCategory);
  const objects = scene.objectCategory
    ? targetEntries.filter((entry) => entry.category === scene.objectCategory)
    : targetEntries;
  setSelectOptions(
    objectSelect,
    [{ value: "", label: "Random in category" }].concat(
      objects.map((entry) => ({ value: entry.name, label: entry.name }))
    ),
    scene.objectName
  );
  if (optionModeSelect) {
    optionModeSelect.value = normalizeOptionMode(optionMode);
  }
}

function updateUrl(scene) {
  const url = new URL(window.location.href);
  url.searchParams.set("category", scene.objectCategory);
  url.searchParams.set("object", scene.objectName);
  url.searchParams.set("seed", String(scene.seed));
  url.searchParams.set("optionMode", normalizeOptionMode(scene.optionMode || "revised"));
  history.replaceState({}, "", url);
}

async function dataUrlToImage(dataUrl) {
  return await new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });
}

function drawCenteredTopLabel(context, text, font, left, right, top = 14) {
  context.save();
  context.font = font;
  context.textAlign = "center";
  context.textBaseline = "top";
  context.fillStyle = "black";
  const centerX = left + (right - left) / 2;
  context.fillText(text, centerX, top);
  context.restore();
}

function drawBottomRightLabel(context, text, font, canvasWidth, canvasHeight) {
  context.save();
  context.font = font;
  context.textAlign = "right";
  context.textBaseline = "bottom";
  context.fillStyle = "black";
  const right = Math.max(0, canvasWidth - 16);
  const bottom = Math.max(0, canvasHeight - 14);
  context.fillText(text, right, bottom);
  context.restore();
}

async function composeCaptionedImage(imageDataUrls, imageCaptions, optionLabel = "", topLabel = "") {
  if (!Array.isArray(imageDataUrls) || !imageDataUrls.length) {
    return "";
  }
  if (imageDataUrls.length !== imageCaptions.length) {
    throw new Error("composeCaptionedImage requires one caption per image.");
  }

  const images = await Promise.all(imageDataUrls.map((dataUrl) => dataUrlToImage(dataUrl)));
  const panelShiftY = topLabel ? COMPLETE_OBJECT_PANEL_SHIFT_Y : 0;
  const canvas = document.createElement("canvas");
  canvas.width = images.reduce((sum, image) => sum + image.width, 0);
  canvas.height = Math.max(...images.map((image) => image.height));
  const context = canvas.getContext("2d");

  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);

  let left = 0;
  images.forEach((image, index) => {
    if (panelShiftY) {
      context.drawImage(image, 0, 0, image.width, 1, left, 0, image.width, panelShiftY);
    }
    context.drawImage(image, left, panelShiftY);
    drawCenteredTopLabel(
      context,
      imageCaptions[index],
      "700 24px Arial",
      left,
      left + image.width,
      topLabel ? 52 : 14
    );
    left += image.width;
  });

  if (topLabel) {
    drawCenteredTopLabel(context, topLabel, "700 30px Arial", 0, canvas.width);
  }

  if (optionLabel) {
    drawBottomRightLabel(context, optionLabel, "700 30px Arial", canvas.width, canvas.height);
  }

  return canvas.toDataURL("image/png");
}

async function buildContactSheet(viewItems) {
  const tileWidth = 420;
  const tileHeight = 320;
  const columns = Math.min(4, Math.max(1, viewItems.length));
  const rows = Math.max(1, Math.ceil(viewItems.length / columns));
  const padding = 18;
  const headerHeight = 44;
  const canvas = document.createElement("canvas");
  canvas.width = columns * tileWidth + padding * (columns + 1);
  canvas.height = rows * (tileHeight + headerHeight) + padding * (rows + 1);
  const context = canvas.getContext("2d");

  context.fillStyle = "#f7f3ea";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.font = "700 18px Avenir Next";
  context.fillStyle = "#27333f";

  const images = await Promise.all(viewItems.map((item) => dataUrlToImage(item.dataUrl)));
  images.forEach((image, index) => {
    const item = viewItems[index];
    const column = index % columns;
    const row = Math.floor(index / columns);
    const left = padding + column * (tileWidth + padding);
    const top = padding + row * (tileHeight + headerHeight + padding);

    context.fillText(item.title, left, top + 20);
    context.strokeStyle = "rgba(70, 81, 92, 0.14)";
    context.strokeRect(left, top + 30, tileWidth, tileHeight);
    context.drawImage(image, left, top + 30, tileWidth, tileHeight);
  });

  return canvas.toDataURL("image/png");
}

function buildQuestion(scene, rng) {
  const categories = scene.availableObjectCategories.slice();
  const distractors = shuffleWithRng(
    categories.filter((categoryName) => categoryName !== scene.objectCategory),
    rng
  ).slice(0, 3);
  const options = shuffleWithRng([scene.objectCategory].concat(distractors), rng);
  return {
    question: QUESTION_PROMPT,
    options,
    answer: scene.objectCategory
  };
}

async function renderScene(entry, partitions, seedValue) {
  const partSpecs = buildPartSpecs(partitions.length);
  buildViewGrid(partSpecs);
  const viewItems = [];
  const partGroups = [];
  for (let partIndex = 0; partIndex < partSpecs.length; partIndex += 1) {
    const partSpec = partSpecs[partIndex];
    const indices = partitions[partIndex];
    const group = await buildSubsetGroup(entry, indices);
    partGroups.push(group);
    for (const viewSpec of VIEW_SPECS) {
      const dataUrl = viewRenderer.renderToDataUrl(group, viewSpec, 420, 320);
      setViewImage(
        partSpec.key,
        viewSpec.key,
        dataUrl,
        `${entry.name} ${partSpec.label} ${viewSpec.label}`
      );
      viewItems.push({
        partKey: partSpec.key,
        partLabel: partSpec.label,
        viewKey: viewSpec.key,
        viewLabel: viewSpec.label,
        selector: `${viewSelector(partSpec.key, viewSpec.key)} .view-frame`,
        imageKey: `${partSpec.key}_${viewSpec.key}`,
        title: `${partSpec.label} ${viewSpec.label}`,
        dataUrl
      });
    }
  }

  const rng = mulberry32(hashSeed(`question-${seedValue}-${entry.category}-${entry.name}`));
  const baseScene = {
    taskCategory: TASK_CATEGORY,
    objectCategory: entry.category,
    objectName: entry.name,
    partCount: entry.part_count,
    subassemblyCount: partitions.length,
    imageCount: viewItems.length,
    connectionRelationCount: entry.connection_relation_count,
    splitBalanceGap: Math.max(...partitions.map((group) => group.length)) - Math.min(...partitions.map((group) => group.length)),
    partitions: partitions.map((group) => group.slice()),
    seed: seedValue,
    questionType: "category_mcq",
    availableObjectCategories: Object.keys(catalog.categories),
    viewOrder: viewItems.map(({ selector, imageKey, partKey, viewKey }) => ({
      selector,
      imageKey,
      partKey,
      viewKey
    }))
  };
  const qa = buildQuestion(baseScene, rng);
  const contactSheetDataUrl = await buildContactSheet(viewItems);

  partGroups.forEach(disposeObject);

  return {
    ...baseScene,
    question: qa.question,
    options: qa.options,
    answer: qa.answer,
    contactSheetDataUrl
  };
}

async function loadCatalog() {
  const response = await fetch(CATALOG_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch catalog: ${response.status}`);
  }
  return response.json();
}

function updateObjectOptions(selectedCategory, selectedObjectName = "", optionMode = optionModeSelect?.value || "revised") {
  if (!objectSelect) {
    return;
  }
  const targetEntries = targetEntriesForOptionMode(optionMode);
  const entries = selectedCategory
    ? targetEntries.filter((entry) => entry.category === selectedCategory)
    : targetEntries;
  setSelectOptions(
    objectSelect,
    [{ value: "", label: "Random in category" }].concat(
      entries.map((entry) => ({ value: entry.name, label: entry.name }))
    ),
    selectedObjectName
  );
}

function readQueryConfig() {
  const params = new URLSearchParams(window.location.search);
  return {
    category: params.get("category") || "",
    objectName: params.get("object") || "",
    seed: params.get("seed") || "12345",
    optionMode: normalizeOptionMode(params.get("optionMode") || params.get("option_mode") || "revised")
  };
}

async function generateScene(config = {}) {
  const scene = await generateTwoPartOptionTask(config);
  currentScene = scene;
  updateBadges(scene);
  updateMetadata(scene);
  updateQuestionPreview(scene);
  updateDebugState(scene);
  updateSelects(scene);
  updateUrl(scene);
  await renderOptionTaskGrid(scene);
  setStatus(
    `Generated ${scene.objectCategory}/${scene.objectName} (${scene.optionMode || "revised"}). Correct option: ${scene.correctOptionLetter}.`
  );

  window.__sceneReady = {
    objectCategory: scene.objectCategory,
    objectName: scene.objectName,
    partCount: scene.partCount,
    subassemblyCount: scene.subassemblyCount,
    splitBalanceGap: scene.splitBalanceGap,
    optionMode: scene.optionMode || "revised",
    optionVisibilityPass: Boolean(scene.optionVisibilityPass),
    optionCount: scene.options.length,
    correctOptionLetter: scene.correctOptionLetter
  };
  return scene;
}

function screenshot() {
  if (!currentScene) {
    return "";
  }
  return optionViewerRoot?.querySelector("img")?.src || completeViewerRoot?.querySelector("img")?.src || "";
}

function bindEvents() {
  if (categorySelect) {
    categorySelect.addEventListener("change", () => {
      updateObjectOptions(categorySelect.value, "", optionModeSelect?.value || "revised");
    });
  }

  if (optionModeSelect) {
    optionModeSelect.addEventListener("change", () => {
      const selectedCategory = updateCategoryOptions(optionModeSelect.value, categorySelect?.value || "");
      updateObjectOptions(selectedCategory, "", optionModeSelect.value);
    });
  }

  if (generateButton) {
    generateButton.addEventListener("click", async () => {
      try {
        await generateScene({
          category: categorySelect?.value || "",
          objectName: objectSelect?.value || "",
          seed: seedInput?.value || 12345,
          optionMode: optionModeSelect?.value || "revised"
        });
      } catch (error) {
        console.error(error);
        setStatus(`Failed to generate scene: ${String(error)}`);
      }
    });
  }

}

async function bootstrap() {
  catalog = await loadCatalog();
  eligibleEntries = Object.values(catalog.categories)
    .flat()
    .filter((entry) => entry.part_count >= getSubassemblyCount(entry) && passesCompleteVisibilityFilter(entry));
  binaryEligibleEntries = listBinaryEligibleEntries();
  binaryTargetEntries = listBinaryTargetEntries(binaryEligibleEntries);
  revisedTargetEntries = listRevisedTargetEntries(binaryEligibleEntries);
  revisedThreePartTargetEntries = listRevisedThreePartTargetEntries(binaryEligibleEntries);

  const query = readQueryConfig();
  const optionMode = normalizeOptionMode(query.optionMode);
  if (optionModeSelect) {
    optionModeSelect.value = optionMode;
  }
  updateCategoryOptions(optionMode, "");
  updateObjectOptions("", "", optionMode);
  bindEvents();
  if (seedInput) {
    seedInput.value = String(query.seed || 12345);
  }
  if (categorySelect) {
    categorySelect.value = query.category || "";
  }
  updateObjectOptions(query.category || "", query.objectName || "", optionMode);
  if (objectSelect) {
    objectSelect.value = query.objectName || "";
  }

  const api = {
    async generate(config = {}) {
      return await generateScene(config);
    },
    async generateTwoPartOptionTask(config = {}) {
      return await generateTwoPartOptionTask(config);
    },
    screenshot
  };

  window.topoBench = api;
  await api.generate({
    category: query.category || "",
    objectName: query.objectName || "",
    seed: query.seed || 12345,
    optionMode
  });
}

bootstrap().catch((error) => {
  console.error(error);
  setStatus("Failed to initialize renderer.");
  window.__lastScene = { error: String(error) };
});
