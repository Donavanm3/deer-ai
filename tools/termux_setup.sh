#!/data/data/com.termux/files/usr/bin/bash
# Deer AI setup for Android (Termux). Run from the deer-ai folder:  bash tools/termux_setup.sh
set -e
echo "==> Updating Termux packages"
pkg update -y && pkg upgrade -y
echo "==> Installing Python, NumPy, Pillow (prebuilt for Android) and build tools"
pkg install -y python python-numpy python-pillow rust binutils clang make cloudflared
# pydantic-core (used by FastAPI) is compiled from source on Android; this tells its build which API level to target
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
echo "==> Installing web server packages (the first time can take 10-20 minutes)"
pip install --upgrade pip wheel
pip install fastapi uvicorn python-multipart
echo "==> Creating starter data"
python tools/make_demo_data.py
[ -f deer.env ] || cp deer.env.example deer.env
echo
echo "Done. Start Deer AI with:   bash tools/termux_start.sh"
