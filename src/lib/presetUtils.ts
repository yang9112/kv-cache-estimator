// Pure preset construction & grouping helpers.
//
// Deliberately free of React and of import.meta.glob so they can be exercised
// directly under plain Node by scripts/test-presets.ts. The usePresets hook
// (./presets.ts) consumes these; the UI imports presetFromState from here too.
import type { ModelPreset } from '../configs/schema';
import { FAMILY_ORDER, USER_PRESETS_FAMILY } from '../configs/schema';
import type { CalculatorState } from '../types';

/** Build a preset object from the current calculator architecture state. */
export function presetFromState(name: string, state: CalculatorState): ModelPreset {
  const slug =
    name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') ||
    'preset';
  const isMla = state.attentionType === 'mla';
  return {
    id: `user-${slug}-${Date.now().toString(36)}`,
    name: name.trim(),
    family: USER_PRESETS_FAMILY,
    attentionType: state.attentionType,
    parameters: state.parameters,
    layers: state.layers,
    hiddenSize: state.hiddenSize,
    qHeads: state.qHeads,
    kvHeads: state.kvHeads,
    // headDim is unused by the MLA formula; drop it so an MLA preset never
    // carries a stale head_dim left over from a previously-selected standard model.
    headDim: isMla ? undefined : state.headDim || undefined,
    mlaDc: isMla ? state.mlaDc : undefined,
    mlaDr: isMla ? state.mlaDr : undefined,
    fullAttnLayers: state.attentionType === 'hybrid' ? state.fullAttnLayers : undefined,
    isMoe: state.isMoe || undefined,
    numExperts: state.numExperts || undefined,
    moeInterSize: state.moeInterSize || undefined,
    moeLayers: state.moeLayers || undefined,
  };
}

/**
 * Group presets by family in a stable, human-friendly order:
 * the custom pseudo-preset first, then user presets, then built-in families in
 * FAMILY_ORDER, then any remaining families alphabetically. Pure (no React).
 */
export function groupPresets(
  userPresets: ModelPreset[],
  builtinPresets: ModelPreset[],
  customPreset: ModelPreset,
): Record<string, ModelPreset[]> {
  const groups: Record<string, ModelPreset[]> = {};
  const place = (p: ModelPreset) => {
    (groups[p.family] ??= []).push(p);
  };

  groups[customPreset.family] = [customPreset];
  [...userPresets]
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach(place);
  [...builtinPresets]
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach(place);

  const ordered: Record<string, ModelPreset[]> = {};
  const seen = new Set<string>();
  const emit = (f: string) => {
    if (groups[f] && !seen.has(f)) {
      ordered[f] = groups[f];
      seen.add(f);
    }
  };
  emit(customPreset.family);
  emit(USER_PRESETS_FAMILY);
  FAMILY_ORDER.forEach(emit);
  Object.keys(groups)
    .sort((a, b) => a.localeCompare(b))
    .forEach(emit);
  return ordered;
}
