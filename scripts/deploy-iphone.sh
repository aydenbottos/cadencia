#!/usr/bin/env bash
# Sign & sideload Cadencia onto your USB-connected iPhone 18 Pro Max using xtool
# and a free Apple account!
#
# Modes:
#   ./scripts/deploy-iphone.sh                    # Downloads latest CI Cadencia-unsigned.ipa via `gh` and installs via xtool
#   ./scripts/deploy-iphone.sh path/to/file.ipa   # Signs & installs a specific local .ipa
#   ./scripts/deploy-iphone.sh --local-xtool      # Builds locally with `xtool dev` (requires local Swift + Darwin SDK)
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v xtool >/dev/null 2>&1; then
  echo "ERROR: xtool not found. Run ./scripts/setup-xtool.sh first."
  exit 1
fi

if [ "${1:-}" = "--local-xtool" ]; then
  echo "Building & installing Cadencia directly via xtool dev..."
  xtool dev
  exit 0
fi

IPA_PATH="${1:-}"

if [ -z "$IPA_PATH" ]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: Provide an .ipa path OR install/login GitHub CLI (`gh auth login`)."
    exit 1
  fi
  echo "Fetching latest successful Cadencia-unsigned-ipa from GitHub Actions..."
  RUN_ID=$(gh run list --workflow "build-ios.yml" --status success --limit 1 --json databaseId --jq '.[0].databaseId')
  if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
    echo "ERROR: No successful GitHub Actions build found yet. Push to main or trigger workflow first."
    exit 1
  fi
  TMP_DL="$ROOT_DIR/.build/ci-ipa"
  rm -rf "$TMP_DL"
  mkdir -p "$TMP_DL"
  gh run download "$RUN_ID" -n Cadencia-unsigned-ipa -D "$TMP_DL"
  IPA_PATH="$TMP_DL/Cadencia-unsigned.ipa"
fi

if [ ! -f "$IPA_PATH" ]; then
  echo "ERROR: IPA not found at $IPA_PATH"
  exit 1
fi

echo "=== Signing & installing $IPA_PATH onto iPhone 18 Pro Max via xtool ==="
xtool install "$IPA_PATH"

echo ""
echo "Installed Cadencia! On first launch with a free Apple account:"
echo "  1. iPhone Settings -> General -> VPN & Device Management -> Trust your Apple ID."
echo "  2. Open Cadencia -> enter your OpenAI API key and Hugging Face Read token into the iOS Keychain vaults."
echo "  3. ZIPA CR Large 300M (FP16) is already built into the app and ready on Metal GPU!"
echo "     Tap Install in Settings to download Gradium Phonon-40m-es (Spanish), Kokoro-82M (English Tutor) & Parakeet Ultra INT8 directly on iPhone."
