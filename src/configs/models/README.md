# Model preset files

Each `.json` file in this folder is a **built-in model preset**, auto-discovered at
build time by Vite (`import.meta.glob` in [`../builtins.ts`](../builtins.ts)).

**To add a preset:** drop a new `.json` file here → rebuild → it appears in the
dropdown under its `family`. **To remove one:** delete the file. No registry or
source code needs to change — that is the whole point of this folder.

## File format

A preset is a JSON object matching `ModelPreset` from [`../schema.ts`](../schema.ts).
Required fields: `id`, `name`, `family`, `attentionType`, `parameters`, `layers`,
`hiddenSize`, `qHeads`, `kvHeads`. Everything else is optional and omitted when
the model does not use it (the calculator falls back to live state).

`attentionType` must be one of `standard` | `mla` | `hybrid`.

A `description` field may be included for provenance/source notes; it is ignored
at runtime (only known fields are loaded).

### Minimal example (standard GQA)

```json
{
  "id": "my-model-8b",
  "name": "My Model (8B)",
  "family": "Mine",
  "attentionType": "standard",
  "parameters": 8,
  "layers": 32,
  "hiddenSize": 4096,
  "qHeads": 32,
  "kvHeads": 8
}
```

### MLA + MoE example (DeepSeek-style)

```json
{
  "id": "my-moe-mla",
  "name": "My MoE MLA (600B)",
  "family": "Mine",
  "attentionType": "mla",
  "parameters": 600,
  "layers": 61,
  "hiddenSize": 7168,
  "qHeads": 128,
  "kvHeads": 128,
  "mlaDc": 512,
  "mlaDr": 64,
  "isMoe": true,
  "numExperts": 256,
  "moeInterSize": 2048,
  "moeLayers": 58
}
```

## Promoting a configuration to a built-in

Use the UI's **Export** button to download the current configuration as a
`.json` file in this same format, then drop it into this folder and rebuild to
promote it to a shared built-in preset.

## Verifying

`npm run verify` reads every file here and checks MoE/EP math and head_dim
resolution against the calculator.
