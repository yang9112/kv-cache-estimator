// Preset management for the calculator.
//
// Built-in (default) presets — JSON files in src/configs/models/, discovered at
// build time — are grouped by `family` for the dropdown. The "Custom"
// pseudo-preset is prepended for ad-hoc configurations.
//
// Pure construction/grouping helpers live in ./presetUtils (importable under
// plain Node for tests). This module holds the React hook + the browser-only
// export helper.
import { useCallback, useMemo } from 'react';
import type { ModelPreset } from '../configs/schema';
import { BUILTIN_PRESETS, CUSTOM_PRESET } from '../configs/builtins';
import { groupPresets } from './presetUtils';

export { presetFromState } from './presetUtils';

export interface UsePresets {
  /** Family-grouped, ordered presets for the dropdown optgroups. */
  groupedPresets: Record<string, ModelPreset[]>;
  /** All presets flat (custom + built-ins), for id lookups. */
  flatPresets: ModelPreset[];
  findPreset: (id: string) => ModelPreset | undefined;
}

export function usePresets(): UsePresets {
  const flatPresets = useMemo<ModelPreset[]>(
    () => [CUSTOM_PRESET, ...BUILTIN_PRESETS],
    [],
  );

  const findPreset = useCallback(
    (id: string) => flatPresets.find((p) => p.id === id),
    [flatPresets],
  );

  const groupedPresets = useMemo(
    () => groupPresets([], BUILTIN_PRESETS, CUSTOM_PRESET),
    [],
  );

  return {
    groupedPresets,
    flatPresets,
    findPreset,
  };
}

/**
 * Download a preset as a JSON file in the same format used by
 * src/configs/models/, so an exported preset can be dropped into that folder
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
