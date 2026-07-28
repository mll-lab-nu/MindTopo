#!/usr/bin/env node
/**
 * regenerate_dataset.js
 * Headless dataset regeneration using Puppeteer.
 * Serves sheep_gallery.html locally, injects batch generation code,
 * saves images + metadata one by one.
 *
 * Usage:
 *   node script/regenerate_dataset.js [--count 115] [--difficulties easy,medium,hard]
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const http = require('http');

const SHEEP_TASK_DIR = path.resolve(__dirname, '..', 'frontend');
const DATASET_DIR = path.resolve(__dirname, '..', 'dataset');
const IMAGES_DIR = path.join(DATASET_DIR, 'images');
const SERVER_PORT = 8377;

// Parse args
const args = process.argv.slice(2);
let count = 115;
let difficulties = ['easy', 'medium', 'hard'];

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--count' && args[i + 1]) count = parseInt(args[i + 1]);
  if (args[i] === '--difficulties' && args[i + 1]) difficulties = args[i + 1].split(',');
}

async function main() {
  const total = count * difficulties.length * 2; // fence + partitioned
  console.log(`Regenerating dataset: ${count} × ${difficulties.length} difficulties × 2 types = ${total} total`);
  console.log(`Output: ${DATASET_DIR}`);

  fs.mkdirSync(IMAGES_DIR, { recursive: true });

  // Clear old images
  const oldImages = fs.readdirSync(IMAGES_DIR).filter(f => f.endsWith('.png'));
  for (const f of oldImages) fs.unlinkSync(path.join(IMAGES_DIR, f));
  console.log(`Cleared ${oldImages.length} old images`);

  // HTTP server for ES modules
  const mimeTypes = {
    '.html': 'text/html', '.js': 'application/javascript',
    '.css': 'text/css', '.json': 'application/json', '.png': 'image/png'
  };
  const server = http.createServer((req, res) => {
    const filePath = path.join(SHEEP_TASK_DIR, req.url === '/' ? 'sheep_gallery.html' : req.url);
    try {
      const data = fs.readFileSync(filePath);
      res.writeHead(200, { 'Content-Type': mimeTypes[path.extname(filePath)] || 'application/octet-stream' });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  await new Promise(resolve => server.listen(SERVER_PORT, resolve));

  const browser = await puppeteer.launch({
    headless: false,
    args: ['--no-sandbox', '--use-gl=angle', '--use-angle=metal',
           '--enable-webgl', '--ignore-gpu-blocklist'],
    defaultViewport: { width: 1200, height: 1200 }
  });

  const page = await browser.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error') console.error(`[BROWSER] ${msg.text()}`);
  });

  console.log('Loading gallery...');
  await page.goto(`http://localhost:${SERVER_PORT}/`, { waitUntil: 'networkidle0', timeout: 60000 });

  // Wait for the module to finish loading — the gallery script sets window.startBatchGeneration
  await page.waitForFunction(() => typeof window.startBatchGeneration === 'function', { timeout: 30000 });
  console.log('Gallery loaded.');

  // Step 1: Generate all configs and expose renderer in browser
  const configCount = await page.evaluate(async (count, difficulties) => {
    const { DifficultyController } = await import('./src/difficulty-controller.js');
    const { DatasetGenerator } = await import('./src/dataset-generator.js');
    const { FenceSceneRenderer } = await import('./src/fence-scene-renderer.js');

    // Create renderer (reuse existing or create new)
    if (!window._renderer) {
      const container = document.getElementById('viewportCanvas');
      window._renderer = new FenceSceneRenderer(container);
    }
    // Generate configs for both fence and partitioned scene types
    const fenceConfigs = DatasetGenerator.generateBatch({ sceneType: 'fence', count, difficulties });
    const partConfigs = DatasetGenerator.generateBatch({ sceneType: 'partitioned', count, difficulties });
    window._configs = [...fenceConfigs, ...partConfigs];
    window._DatasetGenerator = DatasetGenerator;
    return window._configs.length;
  }, count, difficulties);

  console.log(`Generated ${configCount} scene configs. Rendering...`);

  // Step 2: Render one scene at a time and save
  const allMetadata = [];
  const startTime = Date.now();

  for (let i = 0; i < configCount; i++) {
    // Render scene i and return metadata + base64 image
    const result = await page.evaluate((idx) => {
      const config = window._configs[idx];
      const r = window._renderer;
      const renderMeta = r.generateScene(config.params);
      r.render();
      const dataUrl = r.renderer.domElement.toDataURL('image/png');
      const metadata = window._DatasetGenerator.createMetadataEntry(config.id, renderMeta, config);
      return {
        id: config.id,
        difficulty: config.difficulty,
        base64: dataUrl.split(',')[1],
        metadata
      };
    }, i);

    // Save image
    fs.writeFileSync(path.join(IMAGES_DIR, `${result.id}.png`), Buffer.from(result.base64, 'base64'));
    allMetadata.push(result.metadata);

    // Progress
    if ((i + 1) % 10 === 0 || i === configCount - 1) {
      const elapsed = (Date.now() - startTime) / 1000;
      const eta = (elapsed / (i + 1)) * (configCount - i - 1);
      console.log(`  [${i + 1}/${configCount}] ${result.id} (${result.difficulty}) — ETA ${Math.round(eta)}s`);
    }
  }

  // Save metadata
  fs.writeFileSync(path.join(DATASET_DIR, 'dataset_metadata.json'), JSON.stringify(allMetadata, null, 2));

  // Save summary
  const summary = {
    totalSamples: allMetadata.length,
    sceneTypes: ['fence', 'partitioned'],
    difficulties,
    samplesPerDifficulty: count,
    generatedAt: new Date().toISOString(),
    breakdown: Object.fromEntries(difficulties.map(d => [d, allMetadata.filter(m => m.difficulty === d).length])),
    byType: {
      fence: allMetadata.filter(m => m.sceneType === 'fence').length,
      partitioned: allMetadata.filter(m => m.sceneType === 'partitioned').length
    }
  };
  fs.writeFileSync(path.join(DATASET_DIR, 'summary.json'), JSON.stringify(summary, null, 2));

  await browser.close();
  server.close();

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\nDone! ${allMetadata.length} samples in ${elapsed}s`);
  console.log(`  Breakdown: ${JSON.stringify(summary.breakdown)}`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
