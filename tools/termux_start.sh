#!/data/data/com.termux/files/usr/bin/bash
# Starts Deer AI on the phone and keeps it running with the screen off.
# Usage: bash tools/termux_start.sh          (only this phone)
#        bash tools/termux_start.sh public   (also gives a temporary public https link)
cd "$(dirname "$0")/.."
termux-wake-lock 2>/dev/null || true
PORT="${DEER_PORT:-8000}"
if [ "$1" = "public" ]; then
  python run.py --port "$PORT" &
  SERVER=$!
  sleep 4
  echo "Starting a Cloudflare tunnel. Look for the https://....trycloudflare.com link below."
  cloudflared tunnel --url "http://127.0.0.1:$PORT"
  kill $SERVER
else
  python run.py --port "$PORT"
fi
