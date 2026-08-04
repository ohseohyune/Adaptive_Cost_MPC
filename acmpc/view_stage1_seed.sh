#!/usr/bin/env bash
set -euo pipefail

seed="${1:-1001}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$(command -v python3)"
python_prefix="$($python_bin -c 'import sys; print(sys.prefix)')"
loader=/lib64/ld-linux-x86-64.so.2
system_lib=/lib/x86_64-linux-gnu

# ponytail: this bypasses Conda's incompatible X11 RPATH on glibc/x86_64;
# other platforms fall back to their normal Python/GLFW graphics stack.
if [[ -x "$loader" && -e "$system_lib/libglfw.so.3" ]]; then
  exec env PYGLFW_LIBRARY="$system_lib/libglfw.so.3" \
    "$loader" --inhibit-rpath '' \
    --library-path "$system_lib:/usr/lib/x86_64-linux-gnu:$python_prefix/lib" \
    "$python_bin" "$root/acmpc/stage1_evaluation.py" --viewer-seed "$seed"
fi

exec "$python_bin" "$root/acmpc/stage1_evaluation.py" --viewer-seed "$seed"
