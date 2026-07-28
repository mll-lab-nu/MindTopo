// Canvas export utilities

export const DEFAULT_JPEG_QUALITY = 0.85;

/**
 * Export the canvas content as a PNG image
 */
export function exportCanvasAsPNG(
  canvas: HTMLCanvasElement,
  filename: string = 'scene.png'
): void {
  const link = document.createElement('a');
  link.download = filename;
  link.href = getSquareCanvasDataURL(canvas, 'png');
  link.click();
}

/**
 * Export the canvas content as a JPG image
 */
export function exportCanvasAsJPG(
  canvas: HTMLCanvasElement,
  filename: string = 'scene.jpg',
  quality: number = DEFAULT_JPEG_QUALITY
): void {
  const link = document.createElement('a');
  link.download = filename;
  link.href = getSquareCanvasDataURL(canvas, 'jpg', quality);
  link.click();
}

/**
 * Get the canvas as a data URL
 */
export function getCanvasDataURL(
  canvas: HTMLCanvasElement,
  format: 'png' | 'jpg' = 'png',
  quality: number = 0.9
): string {
  if (format === 'jpg') {
    return canvas.toDataURL('image/jpeg', quality);
  }
  return canvas.toDataURL('image/png');
}

/**
 * Get the canvas as a square data URL (center-cropped)
 */
export function getSquareCanvasDataURL(
  canvas: HTMLCanvasElement,
  format: 'png' | 'jpg' = 'png',
  quality: number = 0.9
): string {
  const squareCanvas = getSquareCanvas(canvas);
  return getCanvasDataURL(squareCanvas, format, quality);
}

/**
 * Get the canvas as a Blob
 */
export function getCanvasBlob(
  canvas: HTMLCanvasElement,
  format: 'png' | 'jpg' = 'png',
  quality: number = 0.9
): Promise<Blob | null> {
  return new Promise((resolve) => {
    const mimeType = format === 'jpg' ? 'image/jpeg' : 'image/png';
    canvas.toBlob(resolve, mimeType, format === 'jpg' ? quality : undefined);
  });
}

/**
 * Batch export interface for saving multiple images
 */
export interface BatchExportItem {
  dataUrl: string;
  filename: string;
  seed: number;
  index: number;
}

/**
 * Create a batch export handler
 */
export class BatchExporter {
  private items: BatchExportItem[] = [];

  /**
   * Add an item to the batch
   */
  add(canvas: HTMLCanvasElement, seed: number, index: number): void {
    const filename = `scene_${seed}_${index.toString().padStart(4, '0')}.jpg`;
    this.items.push({
      dataUrl: getSquareCanvasDataURL(canvas, 'jpg', DEFAULT_JPEG_QUALITY),
      filename,
      seed,
      index
    });
  }

  /**
   * Get all items
   */
  getItems(): BatchExportItem[] {
    return [...this.items];
  }

  /**
   * Clear all items
   */
  clear(): void {
    this.items = [];
  }

  /**
   * Download all items as individual files
   * Note: This may be blocked by browser popup blockers for large batches
   */
  async downloadAll(delayMs: number = 100): Promise<void> {
    for (const item of this.items) {
      const link = document.createElement('a');
      link.download = item.filename;
      link.href = item.dataUrl;
      link.click();
      
      // Small delay to prevent browser issues
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }

  /**
   * Create a zip file containing all images (requires JSZip library)
   * Returns a Blob that can be downloaded
   */
  async createZipBlob(): Promise<Blob> {
    // Dynamic import JSZip if available
    // @ts-ignore
    if (typeof JSZip === 'undefined') {
      throw new Error('JSZip library is required for zip export');
    }
    
    // @ts-ignore
    const zip = new JSZip();
    
    for (const item of this.items) {
      // Convert data URL to blob
      const response = await fetch(item.dataUrl);
      const blob = await response.blob();
      zip.file(item.filename, blob);
    }
    
    return zip.generateAsync({ type: 'blob' });
  }
}

function getSquareCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const size = Math.min(canvas.width, canvas.height);
  const sx = Math.floor((canvas.width - size) / 2);
  const sy = Math.floor((canvas.height - size) / 2);
  
  const squareCanvas = document.createElement('canvas');
  squareCanvas.width = size;
  squareCanvas.height = size;
  
  const ctx = squareCanvas.getContext('2d');
  if (ctx) {
    ctx.drawImage(canvas, sx, sy, size, size, 0, 0, size, size);
  }
  
  return squareCanvas;
}

/**
 * Simple image download helper
 */
export function downloadImage(dataUrl: string, filename: string): void {
  const link = document.createElement('a');
  link.download = filename;
  link.href = dataUrl;
  link.click();
}
