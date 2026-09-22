#!/usr/bin/env bash
# Build the Perigee TV app without Gradle: aapt2 + javac + d8 + zipalign + apksigner.
# Bakes in the cast host, token and certificate fingerprint from tv/secrets at build time.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export JAVA_HOME="${JAVA_HOME:-$HOME/.local/opt/jdk17}"
export PATH="$JAVA_HOME/bin:$PATH"
SDK="${ANDROID_HOME:-$HOME/Android/Sdk}"
BT="$SDK/build-tools/34.0.0"
PLATFORM="$SDK/platforms/android-34/android.jar"
HOST="${PERIGEE_CAST_BIND:-192.168.1.75}"
PORT="${PERIGEE_CAST_PORT:-8443}"
TOKEN="$(cat "$HERE/secrets/token.txt")"
FP="$(cat "$HERE/secrets/cert_sha256.txt")"
OUT="$HERE/build"; rm -rf "$OUT"; mkdir -p "$OUT/gen/org/perigee/cast" "$OUT/classes" "$OUT/res" "$OUT/res_src/raw" "$OUT/res_src/xml"

# Generated constants (never committed: they hold the token)
cat > "$OUT/gen/org/perigee/cast/CastConfig.java" <<JAVA
package org.perigee.cast;
final class CastConfig {
    static final String HOST = "$HOST";
    static final String URL = "https://$HOST:$PORT/$TOKEN/";
    static final String CERT_SHA256 = "$FP";
    private CastConfig() {}
}
JAVA

# Resources: copy, fill in the host, add the certificate as a raw resource for the network security config
cp -r "$HERE/app/res/." "$OUT/res_src/"
sed -i "s/__HOST__/$HOST/" "$OUT/res_src/xml/network_security.xml"
cp "$HERE/secrets/server.crt" "$OUT/res_src/raw/cast_cert"

"$BT/aapt2" compile --dir "$OUT/res_src" -o "$OUT/res/compiled.zip"
"$BT/aapt2" link -o "$OUT/base.apk" -I "$PLATFORM" --manifest "$HERE/app/AndroidManifest.xml" \
    --java "$OUT/gen" --min-sdk-version 22 --target-sdk-version 34 "$OUT/res/compiled.zip"

javac -source 8 -target 8 -encoding UTF-8 -bootclasspath "$PLATFORM" -d "$OUT/classes" \
    "$HERE"/app/src/org/perigee/cast/*.java "$OUT"/gen/org/perigee/cast/*.java 2>&1 | grep -v 'bootstrap class path' || true
"$BT/d8" --min-api 22 --output "$OUT" $(find "$OUT/classes" -name '*.class') --lib "$PLATFORM"

# classes.dex into the APK (python: no zip tool on this box), then align and sign
python3 - "$OUT/base.apk" "$OUT/classes.dex" <<'PY'
import sys, zipfile
apk, dex = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(apk, 'a', zipfile.ZIP_DEFLATED) as z:
    z.write(dex, 'classes.dex')
PY
"$BT/zipalign" -f -p 4 "$OUT/base.apk" "$OUT/aligned.apk"

KS="$HERE/secrets/signing.keystore"
if [ ! -f "$KS" ]; then
    keytool -genkeypair -keystore "$KS" -storepass perigee-tv -keypass perigee-tv -alias perigee \
        -keyalg EC -groupname secp256r1 -validity 3650 -dname "CN=perigee-tv" >/dev/null 2>&1
    chmod 600 "$KS"
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:perigee-tv --ks-key-alias perigee --out "$HERE/perigee-tv.apk" "$OUT/aligned.apk"
"$BT/apksigner" verify --print-certs "$HERE/perigee-tv.apk" | head -2
ls -la "$HERE/perigee-tv.apk" | awk '{print $5" bytes  "$9}'
