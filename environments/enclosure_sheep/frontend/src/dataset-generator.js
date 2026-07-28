/**
 * dataset-generator.js
 * Batch dataset generation with metadata export and ZIP download
 */

import { DifficultyController } from './difficulty-controller.js';

export class DatasetGenerator {

  /**
   * Generate a batch of scene configurations
   * @param {Object} options
   * @param {string} options.sceneType - 'fence' | 'hole' | 'hexgrid' | 'laser'
   * @param {number} options.count - number of samples per difficulty
   * @param {Array} options.difficulties - ['easy', 'medium', 'hard']
   * @returns {Array} Array of parameter configs with IDs
   */
  static generateBatch(options = {}) {
    const {
      sceneType = 'fence',
      count = 10,
      difficulties = ['easy', 'medium', 'hard']
    } = options;

    const configs = [];
    let id = 1;

    for (const diff of difficulties) {
      for (let i = 0; i < count; i++) {
        const params = DifficultyController.randomize(sceneType, diff);
        const score = DifficultyController.computeScore(sceneType, params);
        configs.push({
          id: `${sceneType}_${String(id).padStart(4, '0')}`,
          sceneType,
          difficulty: diff,
          difficultyScore: score,
          params,
          timestamp: new Date().toISOString()
        });
        id++;
      }
    }

    return configs;
  }

  /**
   * Generate metadata JSON for a rendered scene
   * @param {string} id - sample ID
   * @param {Object} renderMetadata - metadata from renderer
   * @param {Object} config - generation config
   * @returns {Object} complete metadata entry
   */
  static createMetadataEntry(id, renderMetadata, config) {
    return {
      id,
      sceneType: config.sceneType,
      difficulty: config.difficulty,
      difficultyScore: config.difficultyScore,
      params: config.params,
      groundTruth: renderMetadata.groundTruth,
      imageFile: `${id}.png`,
      timestamp: config.timestamp
    };
  }

  /**
   * Download metadata as JSON
   * @param {Array} entries - metadata entries
   * @param {string} filename
   */
  static downloadMetadata(entries, filename = 'dataset_metadata.json') {
    const blob = new Blob([JSON.stringify(entries, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Download a single image
   * @param {string} dataUrl - PNG data URL
   * @param {string} filename
   */
  static downloadImage(dataUrl, filename) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    a.click();
  }

  /**
   * Create ZIP file of images + metadata (requires JSZip)
   * @param {Array} items - [{id, imageBlob, metadata}]
   * @param {string} zipName
   */
  static async createZip(items, zipName = 'sheep_dataset.zip') {
    if (typeof JSZip === 'undefined') {
      console.error('JSZip not loaded. Include JSZip library for ZIP export.');
      // Fallback: download metadata only
      const allMeta = items.map(i => i.metadata);
      this.downloadMetadata(allMeta);
      return;
    }

    const zip = new JSZip();
    const metadataArr = [];

    for (const item of items) {
      zip.file(`images/${item.id}.png`, item.imageBlob);
      metadataArr.push(item.metadata);
    }

    zip.file('dataset_metadata.json', JSON.stringify(metadataArr, null, 2));

    const content = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.download = zipName;
    a.click();
    URL.revokeObjectURL(url);
  }
}
