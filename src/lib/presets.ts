// Preset management for the calculator.
//
// Two sources of presets are merged into the dropdown:
//   1. Built-in (default) presets  — JSON files in src/configs/models/, discovered
//      at build time. These are the shared defaults, grouped by `family`.
//   2. User presets                — saved at runtime to localStorage, so users can
//      dynamically add/delete their own presets without rebuilding.
//
// Pure construction/grouping helpers live in ./presetUtils (importable under
// plain Node for tests). This module holds the React hook + the browser-only
// export helper.
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ModelPreset } from '../configs/schema';
import { normalizePreset } from '../configs/schema';
import { BUILTIN_PRESETS, CUSTOM_PRESET } from '../configs/builtins';
import { presetFromState, groupPresets } from './presetUtils';

export { presetFromState } from './presetUtils';

const STORAGE_KEY = 'kvcache.userPresets.v1';

function readUserPresets(): ModelPreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr
      .map((r) => {
        try {
          return normalizePreset(r);
        } catch {
          return null;
        }
      })
      .filter((p): p is ModelPreset => p !== null);
  } catch {
    return [];
  }
}

function writeUserPresets(presets: ModelPreset[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
  } catch {
    /* ignore quota / private-mode errors */
  }
}

export interface UsePresets {
  /** Family-grouped, ordered presets for the dropdown optgroups. */
  groupedPresets: Record<string, ModelPreset[]>;
  /** All presets flat (custom + built-ins + user), for id lookups. */
  flatPresets: ModelPreset[];
  userPresets: ModelPreset[];
  findPreset: (id: string) => ModelPreset | undefined;
  isUserPreset: (id: string) => boolean;
  addUserPreset: (preset: ModelPreset) => void;
  removeUserPreset: (id: string) => void;
}

export function usePresets(): UsePresets {
  const [userPresets, setUserPresets] = useState<ModelPreset[]>(() => readUserPresets());

  // Stay in sync if presets change in another tab/window.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setUserPresets(readUserPresets());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const addUserPreset = useCallback((preset: ModelPreset) => {
    setUserPresets((prev) => {
      const next = prev.some((p) => p.id === preset.id)
        ? prev.map((p) => (p.id === preset.id ? preset : p))
        : [...prev, preset];
      writeUserPresets(next);
      return next;
    });
  }, []);

  const removeUserPreset = useCallback((id: string) => {
    setUserPresets((prev) => {
      const next = prev.filter((p) => p.id !== id);
      writeUserPresets(next);
      return next;
    });
  }, []);

  const flatPresets = useMemo<ModelPreset[]>(
    () => [CUSTOM_PRESET, ...BUILTIN_PRESETS, ...userPresets],
    [userPresets],
  );

  const findPreset = useCallback(
    (id: string) => flatPresets.find((p) => p.id === id),
    [flatPresets],
  );

  const isUserPreset = useCallback(
    (id: string) => userPresets.some((p) => p.id === id),
    [userPresets],
  );

  const groupedPresets = useMemo(
    () => groupPresets(userPresets, BUILTIN_PRESETS, CUSTOM_PRESET),
    [userPresets],
  );

  return {
    groupedPresets,
    flatPresets,
    userPresets,
    findPreset,
    isUserPreset,
    addUserPreset,
    removeUserPreset,
  };
}

/**
 * Download a preset as a JSON file in the same format used by
 * src/configs/models/, so a user-saved preset can be dropped into that folder
 * and rebuilt into a shared built-in.
 */
export function downloadPresetJson(preset: ModelPreset) {
  const blob = new Blob([JSON.stringify(preset, null, 2) + '\n'], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${preset.id}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
