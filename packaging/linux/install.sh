#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this installer with sudo."
  exit 1
fi

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
INSTALL_DIR=/opt/distributed-llm
DATA_DIR=/var/lib/distributed-llm

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required."
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(sys.version_info.major * 100 + sys.version_info.minor)')
if [ "$PY_VERSION" -lt 310 ]; then
  echo "Python 3.10 or newer is required."
  exit 1
fi

if ! python3 -c 'import cryptography' >/dev/null 2>&1; then
  echo "The python3-cryptography package is required for portable-worker TLS."
  echo "On Debian or Ubuntu, install it with: sudo apt install python3-cryptography"
  exit 1
fi

if ! getent group distributed-llm >/dev/null 2>&1; then
  groupadd --system distributed-llm
fi
if ! id distributed-llm >/dev/null 2>&1; then
  useradd --system --gid distributed-llm --home-dir "$DATA_DIR" --shell /usr/sbin/nologin distributed-llm
fi

install -d -m 0755 "$INSTALL_DIR" "$INSTALL_DIR/dllm" "$INSTALL_DIR/static" "$INSTALL_DIR/packaging/linux"
install -m 0755 "$SOURCE_DIR/run.py" "$INSTALL_DIR/run.py"
install -m 0644 "$SOURCE_DIR"/dllm/*.py "$INSTALL_DIR/dllm/"
install -m 0644 "$SOURCE_DIR/static/index.html" "$INSTALL_DIR/static/index.html"
install -m 0644 "$SOURCE_DIR/README.md" "$INSTALL_DIR/README.md"
install -m 0644 "$SOURCE_DIR/LICENSE" "$INSTALL_DIR/LICENSE"
install -m 0755 "$SOURCE_DIR/packaging/linux/uninstall.sh" "$INSTALL_DIR/packaging/linux/uninstall.sh"
install -d -o distributed-llm -g distributed-llm -m 0750 "$DATA_DIR"
install -m 0644 "$SOURCE_DIR/packaging/linux/distributed-llm.service" /etc/systemd/system/distributed-llm.service
install -d -m 0755 /usr/share/applications /usr/share/icons/hicolor/scalable/apps
install -m 0644 "$SOURCE_DIR/packaging/linux/distributed-llm.desktop" /usr/share/applications/distributed-llm.desktop
install -m 0644 "$SOURCE_DIR/packaging/linux/distributed-llm.svg" /usr/share/icons/hicolor/scalable/apps/distributed-llm.svg

cat >/usr/local/bin/distributed-llm <<'EOF'
#!/usr/bin/env sh
export DLLM_HOME=/var/lib/distributed-llm
export DLLM_NO_BROWSER=1
exec /usr/bin/python3 /opt/distributed-llm/run.py "$@"
EOF
chmod 0755 /usr/local/bin/distributed-llm
systemctl daemon-reload

echo
echo "Distributed LLM installed."
echo "Desktop Linux: open Distributed LLM from the applications menu."
echo
echo "Configure this machine:"
echo "  sudo -u distributed-llm distributed-llm setup --configure-only"
echo
echo "Or configure a headless worker:"
echo "  sudo -u distributed-llm distributed-llm join --coordinator http://IP:7000 --code 123456 --configure-only"
echo
echo "Then start it:"
echo "  sudo systemctl enable --now distributed-llm"
echo
echo "Dashboard after coordinator setup:"
echo "  http://THIS-COMPUTER-IP:7000"
