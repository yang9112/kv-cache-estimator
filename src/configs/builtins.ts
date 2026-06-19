// Built-in (default) presets are individual JSON files under ./models/.
//
// They are discovered at build time via Vite's import.meta.glob, so adding or
// removing a file in that folder automatically adds or removes a preset — there
// is no central registry to edit. This is the "dynamic add/delete" mechanism for
// the default configuration set.
import type { ModelPreset } from './schema';
import { normalizePreset, dedupeById, DEFAULT_PRESET_TEMPLATE } from './schema';

// Import each preset file as RAW TEXT (not as parsed JSON) so we can parse and
// validate it ourselves. This keeps a single malformed file from failing the
// whole build: both JSON syntax errors and schema-validation errors are caught
// here and the offending file is skipped with a console.error, rather than
// aborting the Vite/Rollup build at the json-plugin transform step.
const files = import.meta.glob('./models/*.json', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>;

export const BUILTIN_PRESETS: ModelPreset[] = dedupeById(
  Object.entries(files)
    .map(([path, text]) => {
      try {
        const data = JSON.parse(text);
        return normalizePreset(data);
      } catch (e) {
        console.error(`[presets] Skipping invalid built-in preset file ${path}:`, e);
        return null;
      }
    })
    .filter((p): p is ModelPreset => p !== null),
  (kept, dropped) => {
    console.warn(
      `[presets] Duplicate preset id "${dropped.id}" (${dropped.name}); ` +
        `keeping first occurrence "${kept.name}".`,
    );
  },
);

// The "Custom" pseudo-preset lets users edit architecture freely without tying
// to a specific model. It is intentionally not a file — it is the blank slate.
export const CUSTOM_PRESET: ModelPreset = { ...DEFAULT_PRESET_TEMPLATE };
