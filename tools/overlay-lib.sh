# Shared helper: emit `-v` args that overlay ONLY the .py files this branch
# changed, one bind mount per file.
#
# Bind-mounting the whole worktree `vllm/` over `/vllm/vllm` shadows the 1564
# non-.py files the image ships -- including `_C*.so`, `_moe_C*.so`,
# `cumem_allocator*.so` and every `__pycache__`. `import vllm._C` then fails and
# vLLM silently falls back to non-compiled paths: measured 60.4 -> 14 tok/s on
# TP4 + MTP k=3, which looked like a model regression and was not one.
#
# Usage:  mapfile -t OVERLAY < <(py_overlay_args "$WORKTREE" "$BASE_REF")
py_overlay_args() {
  local wt="$1" base="${2:-glm53-sm80}"
  git -C "$wt" diff --name-only "$base"...HEAD -- 'vllm/**/*.py' 'vllm/*.py' \
    | while read -r f; do
        [ -f "$wt/$f" ] || continue          # deleted files cannot be overlaid
        printf -- '-v\n%s/%s:/vllm/%s:ro\n' "$wt" "$f" "$f"
      done
}

# Fail loudly if the overlay would be empty or absurdly large -- either means
# the base ref is wrong and the run would silently test the wrong code.
py_overlay_check() {
  local n="$1"
  if [ "$n" -eq 0 ]; then
    echo "REFUSING: overlay is empty; wrong base ref?" >&2; return 1
  fi
  if [ "$n" -gt 200 ]; then
    echo "REFUSING: $n files in overlay; that is a whole-tree mount, not a patch" >&2; return 1
  fi
  echo "[overlay] $n changed .py file(s)" >&2
}
