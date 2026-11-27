#!/usr/bin/env bash
# Build PhononCore.xcframework from gradium-ai/xn-ptts v0.7.1 with on-device pure-Rust
# `ptts_export_coreml` support (aarch64-apple-ios + aarch64-apple-darwin).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PTTS_TAG="${PTTS_TAG:-v0.7.1}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/.build/xn-ptts-build}"
OUT_XCFRAMEWORK="$REPO_ROOT/Frameworks/PhononCore.xcframework"

echo "==> Preparing gradium-ai/xn-ptts @ ${PTTS_TAG} in ${WORK_DIR}"
rm -rf "$WORK_DIR"
mkdir -p "$(dirname "$WORK_DIR")" "$(dirname "$OUT_XCFRAMEWORK")"
git clone --depth 1 --branch "$PTTS_TAG" https://github.com/gradium-ai/xn-ptts.git "$WORK_DIR"

echo "==> Patching ptts-ffi with on-device ptts_export_coreml FFI"
cp "$REPO_ROOT/tools/phonon_export_ffi.rs" "$WORK_DIR/ptts-ffi/src/export_coreml_ffi.rs"

python3 - "$WORK_DIR" <<'PY'
import pathlib, sys

root = pathlib.Path(sys.argv[1])

# 1. Patch ptts-ffi/Cargo.toml to enable gguf + safetensors + sha2 + xn on Apple
cargo_toml = root / "ptts-ffi" / "Cargo.toml"
text = cargo_toml.read_text()
old_dep = 'ptts-coreml = { path = "../ptts-coreml" }'
new_deps = '''ptts-coreml = { path = "../ptts-coreml", features = ["gguf"] }
safetensors = { workspace = true }
sha2 = { workspace = true }
xn = { workspace = true }'''
assert old_dep in text, "Expected ptts-coreml dependency in ptts-ffi/Cargo.toml"
cargo_toml.write_text(text.replace(old_dep, new_deps, 1))

# 2. Patch ptts-ffi/include/ptts.h to declare ptts_export_coreml
header = root / "ptts-ffi" / "include" / "ptts.h"
htext = header.read_text()
decl = '''/// Export a local Phonon checkpoint + voices into a Core ML bundle directory on-device.
/// Returns false on failure; call ptts_last_error(NULL) on the same thread for details.
bool ptts_export_coreml(const char *checkpoint_dir, const char *voices_dir, const char *out_dir);

#endif'''
assert "#endif" in htext, "Expected #endif in ptts.h"
idx = htext.rfind("#endif")
header.write_text(htext[:idx] + decl + "\n")

# 3. Patch ptts-ffi/src/lib.rs to register export_coreml_ffi module & C ABI symbol
lib_rs = root / "ptts-ffi" / "src" / "lib.rs"
ltext = lib_rs.read_text()
anchor = 'mod coreml;'
assert anchor in ltext, "Expected mod coreml; in ptts-ffi/src/lib.rs"
ltext = ltext.replace(anchor, 'mod coreml;\n#[cfg(target_vendor = "apple")]\nmod export_coreml_ffi;', 1)

ffi_fn = '''
/// Export a local checkpoint directory + optional voices directory into an Apple Core ML bundle.
///
/// # Safety
/// `checkpoint_dir` and `out_dir` must be NUL-terminated UTF-8. `voices_dir` may be null.
#[cfg(target_vendor = "apple")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_export_coreml(
    checkpoint_dir: *const c_char,
    voices_dir: *const c_char,
    out_dir: *const c_char,
) -> bool {
    let ck = unsafe { CStr::from_ptr(checkpoint_dir) }.to_string_lossy().to_string();
    let vd = if voices_dir.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(voices_dir) }.to_string_lossy().to_string();
        if s.is_empty() { None } else { Some(s) }
    };
    let out = unsafe { CStr::from_ptr(out_dir) }.to_string_lossy().to_string();
    match guarded(|| {
        export_coreml_ffi::export_bundle(
            Path::new(&ck),
            vd.as_deref().map(Path::new),
            Path::new(&out),
        )
    }) {
        Ok(()) => true,
        Err(e) => {
            LAST_ERROR.with(|l| *l.borrow_mut() = Some(c_string(&e)));
            false
        }
    }
}
'''
ltext = ltext + "\n" + ffi_fn
lib_rs.write_text(ltext)
print("Patched ptts-ffi successfully.")
PY

cd "$WORK_DIR"
export RUSTFLAGS="-C embed-bitcode=no"
unset CARGO_ENCODED_RUSTFLAGS || true

TARGETS=(aarch64-apple-ios aarch64-apple-darwin)
rustup target add "${TARGETS[@]}"

EXPORT_LIST="$(mktemp)"
cat > "$EXPORT_LIST" <<'SYM'
_ptts_new
_ptts_speak
_ptts_voices
_ptts_languages
_ptts_set_voice
_ptts_set_language
_ptts_set_temperature
_ptts_set_seed
_ptts_last_error
_ptts_free
_ptts_export_coreml
SYM

for t in "${TARGETS[@]}"; do
    echo "==> Building libptts_ffi.a for ${t}"
    cargo build --release -p ptts-ffi --target "$t"

    # Merge into a single relocatable object and localize all non-API Rust stdlib symbols
    # (including _rust_eh_personality) so PhononCore never collides with other Rust xcframeworks
    # such as FluidAudio's NemoTextProcessing (libtext_processing_rs.a).
    RAW_LIB="target/$t/release/libptts_ffi.a"
    MERGED_OBJ="target/$t/release/ptts_ffi_merged.o"
    CLEAN_LIB="target/$t/release/libptts_ffi_clean.a"
    rm -f "$MERGED_OBJ" "$CLEAN_LIB"
    if [ "$t" = "aarch64-apple-ios" ]; then
        SDK_VER="$(xcrun --sdk iphoneos --show-sdk-version)"
        ld -r -arch arm64 -platform_version ios 18.0 "$SDK_VER" -all_load "$RAW_LIB" -o "$MERGED_OBJ"
    else
        SDK_VER="$(xcrun --sdk macosx --show-sdk-version)"
        ld -r -arch arm64 -platform_version macos 15.0 "$SDK_VER" -all_load "$RAW_LIB" -o "$MERGED_OBJ"
    fi
    nmedit -s "$EXPORT_LIST" "$MERGED_OBJ"
    libtool -static -o "$CLEAN_LIB" "$MERGED_OBJ"

    echo "==> Exported global symbols in ${t}:"
    nm -gU "$CLEAN_LIB"
    if nm -gU "$CLEAN_LIB" | grep -q "_rust_eh_personality"; then
        echo "ERROR: _rust_eh_personality was not localized in ${t}" >&2
        exit 1
    fi
done
rm -f "$EXPORT_LIST"

HEADERS="$(mktemp -d)"
cp ptts-ffi/include/ptts.h "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'MAP'
module PhononCore {
    header "ptts.h"
    export *
}
MAP

rm -rf "$OUT_XCFRAMEWORK"
xcodebuild -create-xcframework \
    -library "target/aarch64-apple-ios/release/libptts_ffi_clean.a" -headers "$HEADERS" \
    -library "target/aarch64-apple-darwin/release/libptts_ffi_clean.a" -headers "$HEADERS" \
    -output "$OUT_XCFRAMEWORK"
rm -rf "$HEADERS"

echo "==> Built ${OUT_XCFRAMEWORK}:"
du -sh "$OUT_XCFRAMEWORK"/*/libptts_ffi_clean.a
