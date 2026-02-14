#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config
# ----------------------------
CWD="$(pwd)"

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autocommit"
LLMA_PATH_FILE="$CONFIG_DIR/llama_path"

# Max diff size sent to the model (bytes). Diffs beyond this are summarized via --stat.
MAX_DIFF_BYTES="${MAX_DIFF_BYTES:-100000}"

# Inference timeout in seconds (0 = no timeout)
TIMEOUT="${TIMEOUT:-120}"

# ----------------------------
# System prompt (embedded)
# ----------------------------
SYSTEM_PROMPT='You are a git commit message generator. Given a diff, output a descriptive commit message.

Format:
- First line: type(scope): short description
- Types: feat, fix, refactor, docs, test, chore, perf, style
- Scope: primary module/file affected (optional)
- Blank line, then bullet points for details if non-trivial

Guidelines:
- Describe WHAT changed and WHY, not HOW (the diff itself shows how)
- Infer intent from: log messages, variable names, comments, control flow patterns
- For bug fixes: "fix: prevent X when Y" not "fix: add null check"
- For features: "feat: add X for Y" describing capability, not implementation
- For refactors: "refactor: extract/simplify/reorganize X" with motivation if clear
- For diagnostics/debugging: "fix/chore: add diagnostic for X" or "debug: track X"
- Multiple related changes → summarize the theme, bullet the parts, mention any key files/modules changed
- Unrelated changes → note this is a mixed commit, list separately

Output ONLY the commit message, no preamble or explanation.'

# ----------------------------
# Colors & formatting (check stderr since all UI goes there)
# ----------------------------
if [[ -t 2 ]]; then
  BOLD='\033[1m'
  DIM='\033[2m'
  CYAN='\033[36m'
  GREEN='\033[32m'
  YELLOW='\033[33m'
  RED='\033[31m'
  RESET='\033[0m'
  SEP="${DIM}$(printf '─%.0s' {1..60})${RESET}"
else
  BOLD='' DIM='' CYAN='' GREEN='' YELLOW='' RED='' RESET=''
  SEP="$(printf -- '-%.0s' {1..60})"
fi

# ----------------------------
# Logging (verbose gated)
# ----------------------------
VERBOSE=false

log() {
  if [[ "$VERBOSE" == true ]]; then
    printf '%s\n' "$*" >&2
  fi
}

# ----------------------------
# Temp file management (single trap cleans everything)
# ----------------------------
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

# ----------------------------
# Ensure we're in a git repo
# ----------------------------
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '%s\n' "Error: not inside a git repository." >&2
  exit 1
fi

# ----------------------------
# Resolve llama.cpp location
# ----------------------------
resolve_llama() {
  if [[ -n "${LLMA:-}" && -d "$LLMA" ]]; then
    printf '%s\n' "$LLMA"; return
  fi

  if [[ -f "$LLMA_PATH_FILE" ]]; then
    local saved
    saved="$(cat "$LLMA_PATH_FILE")"
    if [[ -d "$saved" ]]; then
      printf '%s\n' "$saved"; return
    fi
  fi

  for candidate in /opt/llama.cpp "$HOME/llama.cpp" /usr/local/llama.cpp; do
    if [[ -d "$candidate/build/bin" ]]; then
      printf '%s\n' "$candidate"; return
    fi
  done

  printf '%s\n' "llama.cpp not found. Cloning and building in /opt/llama.cpp..." >&2
  sudo mkdir -p /opt/llama.cpp
  sudo chown "$(whoami)" /opt/llama.cpp
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp >&2

  printf '%s\n' "Building llama.cpp (this may take a few minutes)..." >&2
  cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF >&2
  cmake --build /opt/llama.cpp/build --config Release \
    -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)" >&2

  mkdir -p "$CONFIG_DIR"
  printf '%s\n' "/opt/llama.cpp" > "$LLMA_PATH_FILE"
  printf '%s\n' "/opt/llama.cpp"
}

LLMA="$(resolve_llama)"
LLAMA_BIN="${LLAMA_BIN:-$LLMA/build/bin/llama-cli}"

# Allow external system prompt file to override the embedded one
if [[ -n "${SYSF:-}" && -f "$SYSF" ]]; then
  SYSTEM_PROMPT="$(cat "$SYSF")"
fi

# HuggingFace model repo + file
HF_REPO="${HF_REPO:-Qwen/Qwen3-1.7B-GGUF:Q8_0}"

# Inference params
TEMP="${TEMP:-0.8}"
CTXK="${CTXK:-q8_0}"
CTXV="${CTXV:-q8_0}"

# ----------------------------
# Args
# ----------------------------
MODE="unstaged"
PUSH=false

for arg in "$@"; do
  case "$arg" in
    --staged|-s)  MODE="staged" ;;
    --push|-p)    PUSH=true ;;
    --verbose|-v) VERBOSE=true ;;
    --help|-h)
      cat >&2 <<'USAGE'
Usage: autocommit [OPTIONS]

Options:
  -s, --staged    Commit only staged changes (default: all changes)
  -p, --push      Push after committing
  -v, --verbose   Show debug/progress output
  -h, --help      Show this help message

Environment variables:
  LLMA            Path to llama.cpp directory
  LLAMA_BIN       Path to llama-cli binary
  SYSF            Path to custom system prompt file
  HF_REPO         HuggingFace model repo (default: Qwen/Qwen3-1.7B-GGUF:Q8_0)
  TEMP            Sampling temperature (default: 0.8)
  TIMEOUT         Inference timeout in seconds (default: 120, 0 = none)
  MAX_DIFF_BYTES  Max diff size before falling back to --stat (default: 100000)
USAGE
      exit 0
      ;;
    *) printf '%s\n' "Unknown argument: $arg (see --help)" >&2; exit 1 ;;
  esac
done

log "Generating commit message for $MODE changes..."

# ----------------------------
# Handle untracked files
# ----------------------------
UNTRACKED_LIST=()
while IFS= read -r -d '' f; do
  UNTRACKED_LIST+=("$f")
done < <(git ls-files --others --exclude-standard -z)

if [[ ${#UNTRACKED_LIST[@]} -gt 0 ]]; then
  echo "" >&2
  echo -e "${YELLOW}Untracked (new) files detected:${RESET}" >&2
  for f in "${UNTRACKED_LIST[@]}"; do
    printf '  %s\n' "$f" >&2
  done
  echo "" >&2

  if [[ "$MODE" == "staged" ]]; then
    echo -e "${DIM}Tip: use 'git add <file>' to stage new files before using --staged.${RESET}" >&2
  fi

  read -n 1 -s -r -p "$(echo -e "Include all new files in this commit? ${GREEN}y${RESET}/${RED}n${RESET} ")" ADD_RESP
  echo >&2

  if [[ "$ADD_RESP" == "y" ]]; then
    git ls-files --others --exclude-standard -z | xargs -0 git add --intent-to-add
    echo -e "${GREEN}New files will be included.${RESET}" >&2
  fi
fi

# ----------------------------
# Build DIFF payload
# ----------------------------
if [[ "$MODE" == "staged" ]]; then
  DIFF_CONTENT="$(git diff --staged -p)"
else
  DIFF_CONTENT="$(git diff -p)"
fi

if [[ -z "$DIFF_CONTENT" ]]; then
  printf '%s\n' "No changes detected. Nothing to commit." >&2
  exit 0
fi

log "Raw diff length: ${#DIFF_CONTENT} chars"

# If diff is too large (e.g. binary blobs from new files), fall back to --stat + truncated diff
if [[ ${#DIFF_CONTENT} -gt $MAX_DIFF_BYTES ]]; then
  echo -e "${YELLOW}Diff is very large (${#DIFF_CONTENT} chars). Truncating and including --stat summary.${RESET}" >&2

  if [[ "$MODE" == "staged" ]]; then
    STAT_CONTENT="$(git diff --staged --stat)"
  else
    STAT_CONTENT="$(git diff --stat)"
  fi

  TRUNCATED="${DIFF_CONTENT:0:$MAX_DIFF_BYTES}"

  DIFF_PAYLOAD="$(cat <<EOF
File change summary:
\`\`\`
$STAT_CONTENT
\`\`\`

Diff (truncated to first ${MAX_DIFF_BYTES} bytes):
\`\`\`\`diff
$TRUNCATED
\`\`\`\`
EOF
)"
else
  DIFF_PAYLOAD="$(cat <<EOF
\`\`\`\`diff
$DIFF_CONTENT
\`\`\`\`
EOF
)"
fi

log "Diff payload prepared (length: ${#DIFF_PAYLOAD} chars)"

SCHEMA='{
  "type":"object",
  "properties":{"message":{"type":"string"}},
  "required":["message"],
  "additionalProperties":false
}'

log "Running model inference..."

# ----------------------------
# Write system prompt to temp file
# ----------------------------
SYSF_TMP="$TMPDIR_WORK/sysprompt.txt"
printf '%s\n' "$SYSTEM_PROMPT" > "$SYSF_TMP"

# ----------------------------
# Run model (stderr to log file, only capture stdout)
# ----------------------------
INFERENCE_LOG="$TMPDIR_WORK/inference.log"

TIMEOUT_CMD=()
if [[ "$TIMEOUT" -gt 0 ]] 2>/dev/null; then
  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout "$TIMEOUT")
  elif command -v gtimeout >/dev/null 2>&1; then
    # macOS with coreutils installed via brew
    TIMEOUT_CMD=(gtimeout "$TIMEOUT")
  else
    log "Warning: no timeout command found, running without timeout"
  fi
fi

if ! RAW="$(
  "${TIMEOUT_CMD[@]}" \
  "$LLAMA_BIN" \
    -hf "$HF_REPO" \
    --single-turn \
    --reasoning-budget 0 \
    --simple-io \
    -fa on \
    --temp "$TEMP" \
    --no-display-prompt \
    --log-disable \
    --no-perf \
    --no-warmup \
    --json-schema "$SCHEMA" \
    -ctk "$CTXK" -ctv "$CTXV" \
    -sysf "$SYSF_TMP" \
    -p "$DIFF_PAYLOAD" \
    2>"$INFERENCE_LOG"
)"; then
  echo -e "${RED}Model inference failed.${RESET}" >&2
  if [[ "$VERBOSE" == true && -s "$INFERENCE_LOG" ]]; then
    echo -e "${DIM}Inference log:${RESET}" >&2
    cat "$INFERENCE_LOG" >&2
  else
    echo -e "${DIM}Run with --verbose for details.${RESET}" >&2
  fi
  exit 1
fi

log "Model inference completed."

# ----------------------------
# Extract commit message from JSON
# ----------------------------
COMMIT_MSG="$(
  python3 - "$RAW" <<'PY'
import json, sys

s = sys.argv[1]

decoder = json.JSONDecoder()
last_obj = None
i = 0
while True:
    j = s.find('{', i)
    if j == -1:
        break
    try:
        obj, end = decoder.raw_decode(s[j:])
        if isinstance(obj, dict) and "message" in obj:
            last_obj = obj
        i = j + max(end, 1)
    except Exception:
        i = j + 1

if not last_obj:
    print(s.strip(), file=sys.stderr)
    print("Failed to extract commit message from model output.", file=sys.stderr)
    sys.exit(1)

msg = str(last_obj["message"]).strip()
print(msg)
PY
)"

# ----------------------------
# Present commit message & prompt
# ----------------------------
echo "" >&2
echo -e "$SEP" >&2
echo "" >&2
echo -e "${BOLD}Generated commit message:${RESET}" >&2
echo "" >&2

while IFS= read -r line; do
  echo -e "  ${CYAN}${line}${RESET}" >&2
done <<< "$COMMIT_MSG"

echo "" >&2
echo -e "$SEP" >&2
echo "" >&2

read -n 1 -s -r -p "$(echo -e "${BOLD}Apply?${RESET}  ${GREEN}y${RESET} = commit  ${RED}n${RESET} = abort  ${YELLOW}e${RESET} = edit  ")" RESP
echo >&2

if [[ "$RESP" == "n" ]]; then
  echo "" >&2
  echo -e "${RED}Aborted.${RESET} Run again to regenerate." >&2
  exit 1
elif [[ "$RESP" == "e" ]]; then
  EDITMSG="$TMPDIR_WORK/editmsg.txt"
  printf '%s\n' "$COMMIT_MSG" > "$EDITMSG"
  ${EDITOR:-nano} "$EDITMSG"
  COMMIT_MSG="$(cat "$EDITMSG")"
fi

cd "$CWD"

echo "" >&2
if [[ "$MODE" == "staged" ]]; then
  printf '%s\n' "$COMMIT_MSG" | git commit -F -
else
  printf '%s\n' "$COMMIT_MSG" | git commit -aF -
fi

if [[ "$PUSH" == true ]]; then
  echo -e "${DIM}Pushing...${RESET}" >&2
  git push
  echo -e "${GREEN}✓ Committed and pushed.${RESET}" >&2
else
  echo -e "${GREEN}✓ Committed.${RESET} Push when ready." >&2
fi