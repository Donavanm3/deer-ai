#!/usr/bin/env bash
# Deer AI installer for an Ubuntu/Debian VPS. Safe to run again (it also repairs a broken setup).
# From a phone: upload deer-ai.zip with your SSH app, then in the SSH terminal:
#   cd /root && unzip -o deer-ai.zip && bash deer-ai/tools/install_vps.sh ai.yourdomain.com
# The domain is optional; without it Deer AI only listens on the server itself.
set -euo pipefail

DOMAIN="${1:-}"
APP=/opt/deer-ai
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY="${DRY_RUN:-0}"

run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else bash -c "$*"; fi; }
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
write() {  # write <path> <content>
  if [ "$DRY" = 1 ]; then echo "+ write $1"; else printf '%s\n' "$2" > "$1"; fi
}

if [ "$DRY" != 1 ] && [ "$(id -u)" != 0 ]; then echo "Please run this as root (or put sudo in front)."; exit 1; fi
if [ -n "$DOMAIN" ] && ! [[ "$DOMAIN" =~ ^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
  echo "\"$DOMAIN\" doesn't look like a domain name (for example ai.dashlith.com)."; exit 1
fi
[ -f "$SRC/run.py" ] && [ -f "$SRC/deer/server/app.py" ] || { echo "Run this from inside the unzipped deer-ai folder."; exit 1; }

say "Installing system packages"
run "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q python3 python3-venv python3-pip git openssh-client unzip sqlite3 ufw curl rsync gnupg debian-keyring debian-archive-keyring apt-transport-https"

say "Setting up the deer user and $APP"
run "id deer >/dev/null 2>&1 || adduser --system --group --home $APP deer"
if [ "$SRC" != "$APP" ]; then
  run "mkdir -p $APP && rsync -a --exclude deer_data --exclude deer.env --exclude venv --exclude .git '$SRC/' '$APP/'"
fi

say "Python environment"
run "[ -x $APP/venv/bin/python ] || python3 -m venv $APP/venv"
run "$APP/venv/bin/pip install -q --upgrade pip && $APP/venv/bin/pip install -q -r $APP/requirements.txt"
run "[ -f $APP/deer.env ] || cp $APP/deer.env.example $APP/deer.env"
run "chmod 600 $APP/deer.env && chown -R deer:deer $APP"

say "Swap space (helps training on small servers)"
run "swapon --show | grep -q . || { fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile; grep -q swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab; }"

say "Running Deer AI as a service"
write /etc/systemd/system/deer-ai.service "[Unit]
Description=Deer AI
After=network.target

[Service]
User=deer
WorkingDirectory=$APP
ExecStart=$APP/venv/bin/python run.py
Restart=always
RestartSec=5
TimeoutStopSec=60
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target"
run "systemctl daemon-reload && systemctl enable -q deer-ai && systemctl restart deer-ai"

say "Firewall"
run "ufw allow OpenSSH >/dev/null && ufw allow 80 >/dev/null && ufw allow 443 >/dev/null && ufw --force enable >/dev/null"

if [ -n "$DOMAIN" ]; then
  say "HTTPS for $DOMAIN"
  run "command -v caddy >/dev/null || { curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg && curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list && apt-get update -q && apt-get install -y -q caddy; }"
  BLOCK="$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:8000
}"
  if [ "$DRY" = 1 ]; then
    echo "+ add $DOMAIN to /etc/caddy/Caddyfile"
  elif grep -q "^$DOMAIN " /etc/caddy/Caddyfile 2>/dev/null; then
    echo "$DOMAIN is already in the Caddyfile; leaving it as it is."
  elif grep -qE '^:80|^\s*$' /etc/caddy/Caddyfile 2>/dev/null && ! grep -qE '^[A-Za-z0-9.-]+\.[A-Za-z]{2,} ' /etc/caddy/Caddyfile; then
    printf '%s\n' "$BLOCK" > /etc/caddy/Caddyfile   # only Caddy's placeholder page was there
  else
    printf '\n%s\n' "$BLOCK" >> /etc/caddy/Caddyfile  # keep your other sites
  fi
  run "systemctl reload caddy 2>/dev/null || systemctl restart caddy"
fi

say "Checking Deer AI"
if [ "$DRY" = 1 ]; then echo "+ wait for http://127.0.0.1:8000"; else
  for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/api/auth/status >/dev/null; then break; fi
    sleep 2
    if [ "$i" = 30 ]; then echo "Deer AI didn't start. Last log lines:"; journalctl -u deer-ai -n 40 --no-pager; exit 1; fi
  done
fi

echo
echo "Deer AI is running."
if [ -n "$DOMAIN" ]; then
  echo "Open https://$DOMAIN and create your account right away: the first account becomes the admin."
  echo "(If the page doesn't load yet, check that $DOMAIN points to this server; HTTPS starts by itself once it does.)"
fi
echo "Future updates: Admin > Updates (upload a zip or update from GitHub). No PC needed."
