#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HOME/Desktop/NucleotideTransformer.zip}"
cd "$(dirname "$ROOT")"
zip -r "$OUT" "$(basename "$ROOT")" \
  -x "$(basename "$ROOT")/.venv/*" \
  -x "$(basename "$ROOT")/outputs/*" \
  -x "$(basename "$ROOT")/.git/*"
echo "Upload this zip in the Colab notebook: $OUT"
