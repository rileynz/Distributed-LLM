#!/usr/bin/env sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this uninstaller with sudo."
  exit 1
fi

systemctl stop distributed-llm.service >/dev/null 2>&1 || true
systemctl disable distributed-llm.service >/dev/null 2>&1 || true

rm -f /etc/systemd/system/distributed-llm.service
rm -f /usr/local/bin/distributed-llm
rm -f /usr/share/applications/distributed-llm.desktop
rm -f /usr/share/icons/hicolor/scalable/apps/distributed-llm.svg
rm -rf /opt/distributed-llm
systemctl daemon-reload >/dev/null 2>&1 || true

echo "Distributed LLM was removed."
echo "Configuration, models, and logs remain in /var/lib/distributed-llm."
echo "Remove that directory manually only if you no longer need its data."
