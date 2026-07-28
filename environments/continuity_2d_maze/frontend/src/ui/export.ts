import { MazeConfig } from '../maze/types';

export function downloadCanvasPng(
  canvas: HTMLCanvasElement,
  config: Required<
    Omit<MazeConfig, 'question_type' | 'bar_count' | 'min_pairwise_distance' | 'min_correct_removals'>
  >,
): void {
  const wd = (config.wall_density ?? 0).toFixed(2);
  const name =
    `maze_seed${config.seed}_grid${config.grid_size}` +
    `_wd${wd}_pts${config.point_count}.png`;
  const url = canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
