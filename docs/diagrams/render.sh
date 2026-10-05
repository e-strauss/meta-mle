#!/usr/bin/env bash
set -euo pipefail

diagram_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

d2 "$diagram_dir/run-flow.d2" "$diagram_dir/run-flow.svg"
d2 "$diagram_dir/run-flow.d2" "$diagram_dir/run-flow.png"
