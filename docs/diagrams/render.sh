#!/usr/bin/env bash
set -euo pipefail

diagram_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 0 ]]; then
  set -- run-flow configurable-system configurable-system-agents
fi

for diagram in "$@"; do
  d2 "$diagram_dir/$diagram.d2" "$diagram_dir/$diagram.svg"
  d2 "$diagram_dir/$diagram.d2" "$diagram_dir/$diagram.png"
done
