#!/usr/bin/env sh
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
VERSION=${VERSION:-0.3.0}
OUTPUT=${1:-"$PROJECT/dist"}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

PAYLOAD="$STAGE/payload.tar.gz"
ARCHIVE_ROOT="$STAGE/Distributed-LLM-Universal"
mkdir -p "$ARCHIVE_ROOT" "$OUTPUT"

cp "$PROJECT/run.py" "$PROJECT/README.md" "$PROJECT/LICENSE" "$PROJECT/requirements.txt" "$ARCHIVE_ROOT/"
cp -R "$PROJECT/dllm" "$PROJECT/static" "$PROJECT/packaging" "$PROJECT/portable" "$ARCHIVE_ROOT/"
find "$ARCHIVE_ROOT" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$ARCHIVE_ROOT" -type f -name '*.pyc' -delete
tar -czf "$PAYLOAD" -C "$STAGE" Distributed-LLM-Universal

INSTALLER="$OUTPUT/Distributed-LLM-Universal-${VERSION}-Linux.run"
cat >"$INSTALLER" <<'EOF'
#!/usr/bin/env sh
set -eu

if [ "${1:-}" = "--help" ]; then
  echo "Distributed LLM Universal self-extracting Linux installer"
  echo "Usage: sudo ./Distributed-LLM-Universal-Linux.run"
  echo "       ./Distributed-LLM-Universal-Linux.run --extract DIRECTORY"
  exit 0
fi

PAYLOAD_LINE=$(awk '/^__DLLM_ARCHIVE_BELOW__$/ { print NR + 1; exit }' "$0")

if [ "${1:-}" = "--extract" ]; then
  if [ -z "${2:-}" ]; then
    echo "Choose a destination directory."
    exit 2
  fi
  mkdir -p "$2"
  tail -n +"$PAYLOAD_LINE" "$0" | tar -xz -C "$2"
  echo "Extracted to $2"
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this installer with sudo."
  exit 1
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
tail -n +"$PAYLOAD_LINE" "$0" | tar -xz -C "$STAGE"
"$STAGE/Distributed-LLM-Universal/packaging/linux/install.sh"
exit 0
__DLLM_ARCHIVE_BELOW__
EOF
cat "$PAYLOAD" >>"$INSTALLER"
chmod 0755 "$INSTALLER"
echo "$INSTALLER"
