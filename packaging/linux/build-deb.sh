#!/usr/bin/env sh
set -eu

PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
VERSION=${VERSION:-0.3.0}
OUTPUT=${1:-"$PROJECT/dist"}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

ROOT="$STAGE/distributed-llm"
mkdir -p "$ROOT/DEBIAN" "$ROOT/opt/distributed-llm/dllm" "$ROOT/opt/distributed-llm/static"
mkdir -p "$ROOT/opt/distributed-llm/packaging/linux"
mkdir -p "$ROOT/usr/bin" "$ROOT/lib/systemd/system" "$ROOT/var/lib/distributed-llm"
mkdir -p "$ROOT/usr/share/applications" "$ROOT/usr/share/icons/hicolor/scalable/apps"
mkdir -p "$ROOT/usr/share/doc/distributed-llm"

cp "$PROJECT/run.py" "$ROOT/opt/distributed-llm/"
cp "$PROJECT"/dllm/*.py "$ROOT/opt/distributed-llm/dllm/"
cp "$PROJECT/static/index.html" "$ROOT/opt/distributed-llm/static/"
cp "$PROJECT/packaging/linux/uninstall.sh" "$ROOT/opt/distributed-llm/packaging/linux/"
cp "$PROJECT/packaging/linux/distributed-llm.service" "$ROOT/lib/systemd/system/"
cp "$PROJECT/packaging/linux/distributed-llm.desktop" "$ROOT/usr/share/applications/"
cp "$PROJECT/packaging/linux/distributed-llm.svg" "$ROOT/usr/share/icons/hicolor/scalable/apps/"
cp "$PROJECT/README.md" "$PROJECT/LICENSE" "$ROOT/usr/share/doc/distributed-llm/"
chmod 0755 "$ROOT/opt/distributed-llm/packaging/linux/uninstall.sh"

cat >"$ROOT/usr/bin/distributed-llm" <<'EOF'
#!/usr/bin/env sh
export DLLM_HOME=/var/lib/distributed-llm
export DLLM_NO_BROWSER=1
exec /usr/bin/python3 /opt/distributed-llm/run.py "$@"
EOF
chmod 0755 "$ROOT/usr/bin/distributed-llm"

cat >"$ROOT/DEBIAN/control" <<EOF
Package: distributed-llm
Version: $VERSION
Section: utils
Priority: optional
Architecture: all
Depends: python3 (>= 3.10), python3-cryptography, ca-certificates
Maintainer: Distributed LLM Project
Description: Lightweight cross-platform llama.cpp cluster manager
 Discovers, approves, benchmarks, and manages low-power inference workers.
EOF

cat >"$ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if ! getent group distributed-llm >/dev/null 2>&1; then
  addgroup --system distributed-llm
fi
if ! id distributed-llm >/dev/null 2>&1; then
  adduser --system --ingroup distributed-llm --home /var/lib/distributed-llm --no-create-home --disabled-login distributed-llm
fi
chown -R distributed-llm:distributed-llm /var/lib/distributed-llm
chmod 0750 /var/lib/distributed-llm
systemctl daemon-reload >/dev/null 2>&1 || true
exit 0
EOF
chmod 0755 "$ROOT/DEBIAN/postinst"

cat >"$ROOT/DEBIAN/prerm" <<'EOF'
#!/bin/sh
set -e
if [ "$1" = remove ]; then
  systemctl stop distributed-llm.service >/dev/null 2>&1 || true
  systemctl disable distributed-llm.service >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "$ROOT/DEBIAN/prerm"

cat >"$ROOT/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
systemctl daemon-reload >/dev/null 2>&1 || true
exit 0
EOF
chmod 0755 "$ROOT/DEBIAN/postrm"

mkdir -p "$OUTPUT"
dpkg-deb --root-owner-group --build "$ROOT" "$OUTPUT/distributed-llm_${VERSION}_all.deb"
