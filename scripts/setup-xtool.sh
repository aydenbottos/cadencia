#!/usr/bin/env bash
# Setup xtool & USB device dependencies on Arch Linux for free Apple ID signing
# and sideloading Cadencia onto your iPhone 18 Pro Max.
set -euo pipefail

echo "=== Cadencia Linux -> iPhone (xtool) Setup ==="

# 1. Install Arch Linux USB / iOS pairing tools if missing
MISSING_PKGS=()
for pkg in usbmuxd libimobiledevice libplist fuse2; do
  if ! pacman -Qi "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [ "${#MISSING_PKGS[@]}" -gt 0 ]; then
  echo "Installing Arch packages: ${MISSING_PKGS[*]}"
  sudo pacman -S --needed --noconfirm "${MISSING_PKGS[@]}"
fi

if ! systemctl is-active --quiet usbmuxd; then
  echo "Enabling and starting usbmuxd service..."
  sudo systemctl enable --now usbmuxd
fi

# 2. Install xtool AppImage to ~/.local/bin/xtool
mkdir -p "$HOME/.local/bin"
XTOOL_BIN="$HOME/.local/bin/xtool"

if ! command -v xtool >/dev/null 2>&1 && [ ! -x "$XTOOL_BIN" ]; then
  echo "Downloading latest xtool Linux x86_64 AppImage..."
  DOWNLOAD_URL=$(curl -s https://api.github.com/repos/xtool-org/xtool/releases/latest \
    | grep -Eo 'https://[^"]+xtool-x86_64\.AppImage' | head -n 1 || true)
  if [ -z "$DOWNLOAD_URL" ]; then
    DOWNLOAD_URL="https://github.com/xtool-org/xtool/releases/latest/download/xtool-x86_64.AppImage"
  fi
  curl -fL "$DOWNLOAD_URL" -o "$XTOOL_BIN"
  chmod +x "$XTOOL_BIN"
  echo "Installed xtool to $XTOOL_BIN"
else
  echo "xtool already installed: $(command -v xtool || echo "$XTOOL_BIN")"
fi

export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "=== Checking USB-connected iPhone ==="
if command -v idevice_id >/dev/null 2>&1; then
  idevice_id -l || true
fi

echo ""
echo "Next step — authenticate once with your free Apple Account (stored locally by xtool):"
echo "  $XTOOL_BIN auth"
echo ""
echo "Then connect your iPhone 18 Pro Max via USB, trust this computer, and run:"
echo "  ./scripts/deploy-iphone.sh"
