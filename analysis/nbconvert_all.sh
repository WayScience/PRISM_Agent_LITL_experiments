#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
if git rev-parse --show-toplevel &>/dev/null; then
  REPO_ROOT="$(git rev-parse --show-toplevel)"
else
  # Fallback: repo root is parent of this script's directory twice
  REPO_ROOT="$(realpath "$(dirname "$0")"/..)"
fi

NB_ROOT="$REPO_ROOT/analysis/notebooks"
OUT_ROOT="$REPO_ROOT/analysis/scripts"

# Optional --force flag to rebuild everything
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

echo "Repo root: $REPO_ROOT"
echo "Notebook root: $NB_ROOT"
echo "Output root: $OUT_ROOT"
echo

mkdir -p "$OUT_ROOT"

# Find all notebooks (skip checkpoints)
while IFS= read -r -d '' nb; do
  rel="${nb#$NB_ROOT/}"                         # path relative to NB_ROOT
  dest_dir="$OUT_ROOT/$(dirname "$rel")"        # mirror folder structure
  base="$(basename "$rel" .ipynb)"              # filename without .ipynb
  out_path="$dest_dir/$base.py"

  mkdir -p "$dest_dir"

  if [[ $FORCE -eq 0 && -f "$out_path" && "$nb" -ot "$out_path" ]]; then
    echo "Up to date: $rel"
    continue
  fi

  echo "Converting: $rel  ->  ${out_path#$REPO_ROOT/}"
  jupyter nbconvert \
    --to script \
    --output-dir "$dest_dir" \
    "$nb"

done < <(find "$NB_ROOT" -type f -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0)

echo
echo "Done."
