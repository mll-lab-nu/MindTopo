/**
 * difficulty-controller.js
 * Controls difficulty parameters for fence & sheep scenes
 */

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function randInt(min, max) { return min + Math.floor(Math.random() * (max - min + 1)); }
function randFloat(min, max) { return min + Math.random() * (max - min); }

export class DifficultyController {

  static PRESETS = {
    fence: {
      easy: {
        fenceShape: 'concave', fenceSegments: 12, fenceRadius: 9,
        nestingDepth: 2, numGaps: 1, gapSize: 0.45, numSheep: 10,
        insideRatio: 0.58, cameraAngle: 'tilt_15', numGates: 2,
        sheepColors: ['RED', 'BLUE', 'GREEN', 'WHITE']
      },
      medium: {
        fenceShape: 'irregular', fenceSegments: 17, fenceRadius: 12,
        nestingDepth: 3, numGaps: 2, gapSize: 0.3, numSheep: 16,
        insideRatio: 0.64, cameraAngle: 'tilt_30', numGates: 3,
        sheepColors: ['RED', 'BLUE', 'GREEN', 'YELLOW', 'WHITE']
      },
      hard: {
        fenceShape: 'irregular', fenceSegments: 24, fenceRadius: 16,
        nestingDepth: 4, numGaps: 4, gapSize: 0.22, numSheep: 20,
        insideRatio: 0.7, cameraAngle: 'tilt_45', numGates: 5,
        sheepColors: ['RED', 'BLUE', 'GREEN', 'YELLOW', 'WHITE', 'PURPLE']
      }
    },
    partitioned: {
      easy: {
        fenceShape: 'partitioned', fenceRadius: 8, numSheep: 9,
        insideRatio: 0.62, cameraAngle: 'tilt_15',
        numPartitionCols: 3, numPartitionRows: 2, partitionSeparated: false,
        sheepColors: ['RED', 'BLUE', 'GREEN', 'WHITE']
      },
      medium: {
        fenceShape: 'partitioned', fenceRadius: 10, numSheep: 12,
        insideRatio: 0.58, cameraAngle: 'tilt_30',
        numPartitionCols: 3, numPartitionRows: 3, partitionSeparated: false,
        sheepColors: ['RED', 'BLUE', 'GREEN', 'YELLOW', 'WHITE']
      },
      hard: {
        fenceShape: 'partitioned', fenceRadius: 12, numSheep: 15,
        insideRatio: 0.55, cameraAngle: 'tilt_45',
        numPartitionCols: 4, numPartitionRows: 3, partitionSeparated: false,
        sheepColors: ['RED', 'BLUE', 'GREEN', 'YELLOW', 'WHITE', 'PURPLE']
      }
    }
  };

  static getPreset(sceneType, difficulty) {
    return { ...this.PRESETS[sceneType]?.[difficulty] } || {};
  }

  /**
   * Generate highly varied random parameters within difficulty bounds.
   */
  static randomize(sceneType, difficulty) {
    if (sceneType === 'partitioned') return this.randomizePartitioned(difficulty);
    const allColors = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'WHITE', 'PURPLE', 'ORANGE'];

    if (difficulty === 'easy') {
      return {
        fenceShape: pick(['concave', 'irregular', 'convex']),
        fenceSegments: randInt(9, 14),
        fenceRadius: randFloat(8, 10.5),
        nestingDepth: pick([2, 2, 3]),
        numGaps: pick([0, 1, 1, 2]),
        gapSize: randFloat(0.28, 0.65),
        numSheep: randInt(7, 10),
        insideRatio: randFloat(0.45, 0.72),
        cameraAngle: pick(['tilt_15', 'tilt_15', 'tilt_30']),
        numGates: pick([1, 2, 3]),
        sheepColors: allColors.slice(0, randInt(3, 5)),
        groundTexture: pick(['grass', 'grass', 'dirt', 'snow'])
      };
    }
    if (difficulty === 'medium') {
      return {
        fenceShape: pick(['irregular', 'concave', 'irregular', 'convex']),
        fenceSegments: randInt(13, 19),
        fenceRadius: randFloat(10.5, 14),
        nestingDepth: pick([2, 3, 3, 4]),
        numGaps: pick([0, 1, 2, 2, 3]),
        gapSize: randFloat(0.18, 0.5),
        numSheep: randInt(10, 14),
        insideRatio: randFloat(0.5, 0.76),
        cameraAngle: pick(['tilt_15', 'tilt_30', 'tilt_30', 'tilt_45']),
        numGates: pick([2, 3, 4]),
        sheepColors: allColors.slice(0, randInt(4, 6)),
        groundTexture: pick(['grass', 'dirt', 'snow'])
      };
    }
    // hard
    return {
      fenceShape: pick(['irregular', 'irregular', 'concave']),
      fenceSegments: randInt(18, 26),
      fenceRadius: randFloat(13.5, 17.5),
      nestingDepth: pick([3, 3, 4, 4]),
      numGaps: pick([1, 2, 3, 4, 5]),
      gapSize: randFloat(0.12, 0.35),
      numSheep: randInt(13, 18),
      insideRatio: randFloat(0.55, 0.8),
      cameraAngle: pick(['tilt_30', 'tilt_30', 'tilt_45', 'tilt_45']),
      numGates: pick([4, 5, 6]),
      sheepColors: allColors.slice(0, randInt(5, 7)),
      groundTexture: pick(['grass', 'dirt', 'snow'])
    };
  }

  /**
   * Generate random parameters for partitioned scenes
   */
  static randomizePartitioned(difficulty) {
    const allColors = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'WHITE', 'PURPLE', 'ORANGE'];

    if (difficulty === 'easy') {
      const partitionLayout = pick(['polygon_star', 'hex_cross', 'nested_polygon', 'grid']);
      return {
        fenceShape: 'partitioned',
        partitionLayout,
        fenceRadius: randFloat(7.5, 9.5),
        numSheep: randInt(8, 11),
        insideRatio: randFloat(0.55, 0.72),
        cameraAngle: pick(['tilt_15', 'tilt_30', 'tilt_30']),
        numPartitionCols: pick([2, 3, 3]),
        numPartitionRows: pick([2, 2, 3]),
        numSectors: pick([5, 6]),
        numPolygonSides: pick([5, 6, 7]),
        partitionSeparated: false,
        numGates: pick([1, 2, 3]),
        sheepColors: allColors.slice(0, randInt(3, 5)),
        groundTexture: pick(['grass', 'grass', 'dirt', 'snow'])
      };
    }
    if (difficulty === 'medium') {
      const partitionLayout = pick(['polygon_star', 'nested_polygon', 'radial', 'hex_cross', 'grid']);
      return {
        fenceShape: 'partitioned',
        partitionLayout,
        fenceRadius: randFloat(8.5, 11),
        numSheep: randInt(10, 14),
        insideRatio: randFloat(0.52, 0.68),
        cameraAngle: pick(['tilt_30', 'tilt_30', 'tilt_45']),
        numPartitionCols: pick([3, 3, 4]),
        numPartitionRows: pick([2, 3, 3]),
        numSectors: pick([6, 7]),
        numPolygonSides: pick([6, 7, 8]),
        partitionSeparated: false,
        numGates: pick([2, 3, 4]),
        sheepColors: allColors.slice(0, randInt(4, 6)),
        groundTexture: pick(['grass', 'dirt', 'snow'])
      };
    }
    const partitionLayout = pick(['polygon_star', 'nested_polygon', 'radial', 'hex_cross', 'polygon_star', 'grid']);
    return {
      fenceShape: 'partitioned',
      partitionLayout,
      fenceRadius: randFloat(10, 13),
      numSheep: randInt(12, 16),
      insideRatio: randFloat(0.48, 0.64),
      cameraAngle: pick(['tilt_30', 'tilt_45', 'tilt_45']),
      numPartitionCols: pick([3, 4, 4]),
      numPartitionRows: pick([3, 3, 4]),
      numSectors: pick([7, 8]),
      numPolygonSides: pick([7, 8, 8]),
      partitionSeparated: false,
      numGates: pick([3, 4, 5]),
      sheepColors: allColors.slice(0, randInt(5, 7)),
      groundTexture: pick(['grass', 'dirt', 'snow'])
    };
  }

  static computeScore(sceneType, params) {
    const factors = {};
    if (params.fenceShape === 'partitioned') {
      const totalCells = (params.numPartitionCols || 2) * (params.numPartitionRows || 2);
      factors.cellComplexity = Math.min(4, Math.ceil(totalCells / 4));
      factors.sheepCount = Math.min(4, Math.ceil((params.numSheep || 5) / 7));
      factors.separated = params.partitionSeparated ? 2 : 1;
      factors.gateDifficulty = (params.numGates || 0) === 0 ? 1 : Math.min(4, Math.ceil(params.numGates / 2));
      factors.cameraAngle = params.cameraAngle === 'top_down' ? 1 :
                            params.cameraAngle === 'tilt_15' ? 2 :
                            params.cameraAngle === 'tilt_30' ? 3 : 4;
    } else {
      factors.fenceComplexity = params.fenceShape === 'convex' ? 1 :
                                params.fenceShape === 'concave' ? 2 : 3;
      factors.sheepCount = Math.min(4, Math.ceil((params.numSheep || 5) / 7));
      factors.nesting = Math.min(4, params.nestingDepth || 1);
      factors.gapDifficulty = (params.numGaps || 0) === 0 ? 1 :
                              Math.min(4, Math.ceil((params.numGaps || 0) / 2));
      factors.cameraAngle = params.cameraAngle === 'top_down' ? 1 :
                            params.cameraAngle === 'tilt_15' ? 2 :
                            params.cameraAngle === 'tilt_30' ? 3 : 4;
    }
    const score = Object.values(factors).reduce((a, b) => a + b, 0) / (Object.keys(factors).length * 4);
    const level = score < 0.45 ? 'easy' : score < 0.7 ? 'medium' : 'hard';
    return { score: Math.round(score * 100) / 100, level, factors };
  }
}
