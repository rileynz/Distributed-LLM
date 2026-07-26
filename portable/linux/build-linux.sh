#!/usr/bin/env sh
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
VERSION=${VERSION:-0.3.0}
ARCH=${ARCH:-$(uname -m)}
SKIP_BACKEND=${SKIP_BACKEND:-0}
DIST="$PROJECT/dist/portable"
WORK="$PROJECT/build/portable-linux"
RUNTIME="$PROJECT/portable/runtime/backend"
VENV="$WORK/venv"

case "$ARCH" in
  x86_64|amd64) PACKAGE_ARCH=x64 ;;
  aarch64|arm64) PACKAGE_ARCH=arm64 ;;
  *) echo "Unsupported portable architecture: $ARCH"; exit 2 ;;
esac

mkdir -p "$DIST" "$RUNTIME"
cd "$PROJECT"
python3 -m venv "$VENV"
PYTHON="$VENV/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check \
  -r requirements.txt "pyinstaller==6.14.2"
"$PYTHON" -m unittest discover -v
if [ "$SKIP_BACKEND" != "1" ]; then
  "$PYTHON" portable/prepare_backend.py "$RUNTIME" --variant cpu
fi
"$PYTHON" -m PyInstaller --noconfirm --clean \
  --distpath "$DIST" \
  --workpath "$WORK" \
  portable/linux/PortableWorker.spec

PACKAGE="$DIST/DistributedLLM-Portable-Worker"
test -x "$PACKAGE/DistributedLLM-Worker"
"$PACKAGE/DistributedLLM-Worker" --version
cp "$PROJECT/portable/Start-Portable-Worker.sh" "$PACKAGE/Start-Portable-Worker.sh"
chmod 0755 "$PACKAGE/DistributedLLM-Worker" "$PACKAGE/Start-Portable-Worker.sh"

ARCHIVE="$DIST/Distributed-LLM-Portable-Worker-Linux-${PACKAGE_ARCH}-${VERSION}.tar.gz"
tar -czf "$ARCHIVE" -C "$DIST" DistributedLLM-Portable-Worker
sha256sum "$ARCHIVE" >"$DIST/SHA256SUMS-linux-${PACKAGE_ARCH}-portable.txt"
echo "$ARCHIVE"
