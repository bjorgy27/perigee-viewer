#!/usr/bin/env bash
# Build the native Perigee viewer for the Fire TV (arm64) and package it as an APK, no Gradle.
# Bakes the cast token and certificate fingerprint from tv/secrets into the binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export JAVA_HOME="${JAVA_HOME:-$HOME/.local/opt/jdk17}"
export PATH="$JAVA_HOME/bin:$PATH"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$(ls -d "$ANDROID_HOME"/ndk/* | sort -V | tail -1)}"
BT="$ANDROID_HOME/build-tools/34.0.0"
PLATFORM="$ANDROID_HOME/platforms/android-34/android.jar"
export PERIGEE_CAST_TOKEN="$(cat "$HERE/secrets/token.txt")"
export PERIGEE_CAST_CERT_SHA256="$(cat "$HERE/secrets/cert_sha256.txt")"
OUT="$HERE/build-android"; rm -rf "$OUT"; mkdir -p "$OUT/jni" "$OUT/res"

# The Fire TV (ginza) reports armeabi-v7a, a 32-bit userland; build that. Add "-t arm64-v8a" for 64-bit boxes.
ABIS="${PERIGEE_ABIS:-armeabi-v7a}"
echo "== cargo ndk (release, $ABIS)"
TFLAGS=""; for a in $ABIS; do TFLAGS="$TFLAGS -t $a"; done
(cd "$ROOT" && cargo ndk $TFLAGS --platform 28 -o "$OUT/jni" build --release --lib)
STRIP="$(ls "$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)"
for a in $ABIS; do "$STRIP" --strip-unneeded "$OUT/jni/$a/libperigee_viewer.so"; ls -la "$OUT/jni/$a/libperigee_viewer.so" | awk '{print $5" bytes  "$9}'; done

echo "== package"
"$BT/aapt2" compile --dir "$HERE/android/res" -o "$OUT/res/compiled.zip"
"$BT/aapt2" link -o "$OUT/base.apk" -I "$PLATFORM" --manifest "$HERE/android/AndroidManifest.xml" \
    --min-sdk-version 28 --target-sdk-version 34 "$OUT/res/compiled.zip"
python3 - "$OUT/base.apk" "$OUT/jni" $ABIS <<'PY'
import sys, zipfile, os
apk, jni, abis = sys.argv[1], sys.argv[2], sys.argv[3:]
with zipfile.ZipFile(apk, 'a') as z:
    for a in abis:
        z.write(os.path.join(jni, a, 'libperigee_viewer.so'), f'lib/{a}/libperigee_viewer.so', compress_type=zipfile.ZIP_STORED)  # stored: page-alignable
PY
"$BT/zipalign" -f -p 4 "$OUT/base.apk" "$OUT/aligned.apk"
KS="$HERE/secrets/signing.keystore"
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:perigee-tv --ks-key-alias perigee --out "$HERE/perigee-viewer-tv.apk" "$OUT/aligned.apk"
"$BT/apksigner" verify "$HERE/perigee-viewer-tv.apk" && ls -la "$HERE/perigee-viewer-tv.apk" | awk '{print $5" bytes  "$9}'
