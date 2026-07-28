export interface DifficultyPreset {
  label: string;
  grid_size: number;
  wall_density: number;
}

export type DifficultyKey = 'easy' | 'medium' | 'hard';

export const DIFFICULTY_PRESETS: Record<DifficultyKey, DifficultyPreset> = {
  easy: { label: 'Easy 4x4', grid_size: 4, wall_density: 0.4 },
  medium: { label: 'Medium 5x5', grid_size: 5, wall_density: 0.425 },
  hard: { label: 'Hard 6x6', grid_size: 6, wall_density: 0.47 },
};
