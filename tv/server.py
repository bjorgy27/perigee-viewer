#!/usr/bin/env python3
"""Perigee TV cast server.

Captures the running perigee-viewer window (it must be visible on screen) with grim and serves it as an
MJPEG stream over HTTPS to the Fire TV app.  Security: binds only to the LAN address below, TLS with the
self-signed certificate in secrets/ (the TV app pins its SHA-256), every request must carry the random
token from secrets/token.txt in its path, anything else gets 404 and is logged.
"""
import http.server, ssl, subprocess, threading, time, json, os, sys, socketserver, datetime, struct, array, math

HERE = os.path.dirname(os.path.abspath(__file__))
BIND = os.environ.get("PERIGEE_CAST_BIND", "192.168.1.75")
PORT = int(os.environ.get("PERIGEE_CAST_PORT", "8443"))
FPS = float(os.environ.get("PERIGEE_CAST_FPS", "8"))
QUALITY = os.environ.get("PERIGEE_CAST_JPEG_QUALITY", "72")
WINDOW_TITLE = os.environ.get("PERIGEE_CAST_TITLE", "PERIGEE")
DATA_DIR = os.environ.get("PERIGEE_DATA_DIR", os.path.join(HERE, "..", "..", "Perigee", "src"))
STEP = 60.0            # engine propagate step, seconds
KEEP_BEFORE_MIN = 45   # how much history before "now" the TV gets (for trails)

with open(os.path.join(HERE, "secrets", "token.txt")) as f:
    TOKEN = f.read().strip()
if len(TOKEN) < 32:
    sys.exit("secrets/token.txt is too short")

_frame = b""
_frame_lock = threading.Condition()

def window_geometry():
    """x,y wxh of the viewer window via Hyprland, or None if it is not open."""
    try:
        out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=2).stdout
        for c in json.loads(out):
            if WINDOW_TITLE in c.get("title", "") and c.get("mapped", True):
                x, y = c["at"]; w, h = c["size"]
                return f"{x},{y} {w}x{h}"
    except Exception:
        pass
    return None

def capture_loop():
    global _frame
    period = 1.0 / FPS
    geom = None; last_geom_check = 0.0
    while True:
        t0 = time.time()
        if t0 - last_geom_check > 2.0:
            geom = window_geometry(); last_geom_check = t0
        if geom:
            r = subprocess.run(["grim", "-t", "jpeg", "-q", QUALITY, "-g", geom, "-"], capture_output=True, timeout=5)
            if r.returncode == 0 and r.stdout:
                with _frame_lock:
                    _frame = r.stdout
                    _frame_lock.notify_all()
        time.sleep(max(0.0, period - (time.time() - t0)))

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>PERIGEE</title>
<style>html,body{margin:0;background:#000;height:100%%;overflow:hidden}
img{width:100vw;height:100vh;object-fit:contain;display:block}</style></head>
<body><img src="/%s/stream" alt=""></body></html>"""

# ---------------------------------------------------------------- data for the native TV app
# The TV runs the viewer itself and only needs Perigee's outputs.  ORBIT_DATA.json is 150+ MB of text from
# each epoch onwards; the TV gets a compact binary of just the window from (now - 45 min) to the end, f32,
# with each satellite's epoch rebased to its first kept column so the viewer's time maths is unchanged.
_data_lock = threading.Lock()
_data_cache = {"mtime": None, "built": 0.0, "orbits": None, "sorted": None, "raw_epochs": None}

def now_jd():
    return time.time() / 86400.0 + 2440587.5

def build_data():
    op = os.path.join(DATA_DIR, "ORBIT_DATA.json"); sp = os.path.join(DATA_DIR, "SORTED_SATS.json")
    mtime = os.path.getmtime(op)
    with _data_lock:
        fresh = _data_cache["mtime"] == mtime and time.time() - _data_cache["built"] < 600
        if fresh:
            return _data_cache["orbits"], _data_cache["sorted"]
    t0 = time.time()
    with open(op) as f: orbits = json.load(f)
    with open(sp) as f: sorted_sats = json.load(f)
    sdata, _, ncols = sorted_sats; nr = 9
    jd0 = now_jd() - KEEP_BEFORE_MIN / 1440.0
    out = bytearray(); out += b"PGO1"; out += struct.pack("<If", len(orbits), 0) [:4]; out += struct.pack("<d", STEP)
    new_sdata = list(sdata)
    for col, m in enumerate(orbits):
        data, _, n = m
        epoch = sdata[col * nr + 1]
        first = int(max(0, math.floor((jd0 - epoch) * 86400.0 / STEP)))
        first = min(first, max(0, n - 2))
        keep = n - first
        out += struct.pack("<IdI", int(sdata[col * nr]), epoch + first * STEP / 86400.0, keep)
        out += array.array("f", data[first * 6:(first + keep) * 6]).tobytes()
        new_sdata[col * nr + 1] = epoch + first * STEP / 86400.0
    built = bytes(out)
    sorted_json = json.dumps([new_sdata, None, ncols]).encode()
    with _data_lock:
        _data_cache.update(mtime=mtime, built=time.time(), orbits=built, sorted=sorted_json)
    sys.stderr.write("data rebuilt: %d sats, %.1f MB, %.1f s\n" % (len(orbits), len(built) / 1e6, time.time() - t0))
    return built, sorted_json

def elset_min():
    keep = ("NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_ID", "OBJECT_TYPE", "COUNTRY_CODE", "LAUNCH_DATE", "SITE", "ELEMENT_SET_NO", "REV_AT_EPOCH")
    with open(os.path.join(DATA_DIR, "ELSET.json")) as f: recs = json.load(f)
    return json.dumps([{k: r.get(k) for k in keep} for r in recs]).encode()

def rerank_now():
    """Run `perigee rank` in the data folder right away (non-blocking); the app picks the new file up."""
    exe = os.path.join(DATA_DIR, "..", "target", "release", "perigee")
    try:
        subprocess.Popen([exe, "rank"], cwd=DATA_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        sys.stderr.write("rerank failed to start: %s\n" % e)

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "perigee-cast"
    sys_version = ""

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != TOKEN or parts[1] != "data":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 4096:
            self.send_error(413); return
        body = self.rfile.read(length)
        if parts[2] == "view_region":
            #Only the five expected fields, all validated, then written for Perigee and re-ranked
            try:
                v = json.loads(body)
                region = {"name": str(v["name"])[:40],
                          "az_from": float(v["az_from"]) % 360.0, "az_to": float(v["az_to"]),
                          "el_min": max(-90.0, min(90.0, float(v["el_min"]))),
                          "el_max": max(-90.0, min(90.0, float(v["el_max"])))}
                if region["az_to"] < 0 or region["az_to"] > 360: raise ValueError("az_to")
            except Exception as e:
                self.send_error(400, "bad region: %s" % e); return
            with open(os.path.join(DATA_DIR, "VIEW_REGION.json"), "w") as f:
                json.dump(region, f)
            sys.stderr.write("view region set to %s; re-ranking\n" % region["name"])
            rerank_now()
            self.send_response(204); self.end_headers(); return
        self.send_error(404)
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s %s\n" % (datetime.datetime.now().strftime("%H:%M:%S"), self.client_address[0], fmt % args))
    def do_GET(self):
        parts = self.path.strip("/").split("/")
        if not parts or parts[0] != TOKEN:
            self.send_error(404); return
        tail = parts[1] if len(parts) > 1 else ""
        if tail == "":
            body = PAGE % TOKEN.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body); return
        if tail == "stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = None
            try:
                while True:
                    with _frame_lock:
                        _frame_lock.wait(timeout=2.0)
                        frame = _frame
                    if not frame or frame is last:
                        continue
                    last = frame
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame))
                    self.wfile.write(frame); self.wfile.write(b"\r\n"); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        if tail == "data" and len(parts) > 2:
            name = parts[2]
            try:
                if name == "orbits.bin":
                    body, _ = build_data(); ctype = "application/octet-stream"
                elif name == "sorted_sats.json":
                    _, body = build_data(); ctype = "application/json"
                elif name == "elset_min.json":
                    body = elset_min(); ctype = "application/json"
                elif name == "elsets.json":
                    #Raw SORTED_SATS.json (true epochs, 88 KB): the TV integrates these itself instead of pulling orbits.bin
                    body = open(os.path.join(DATA_DIR, "SORTED_SATS.json"), "rb").read(); ctype = "application/json"
                elif name == "categories.json":
                    body = open(os.path.join(DATA_DIR, "CATEGORIES.json"), "rb").read(); ctype = "application/json"
                elif name == "transmitters.json":
                    body = open(os.path.join(DATA_DIR, "NORADs.json"), "rb").read(); ctype = "application/json"
                elif name == "viewer.toml":
                    body = open(os.path.join(HERE, "viewer-tv.toml"), "rb").read(); ctype = "application/toml"
                elif name == "ranks.json":
                    body = open(os.path.join(DATA_DIR, "SATELLITE_RANKS.json"), "rb").read(); ctype = "application/json"
                else:
                    self.send_error(404); return
            except FileNotFoundError as e:
                self.send_error(503, "data not ready: %s" % e); return
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body); return
        if tail == "still":
            with _frame_lock:
                frame = _frame
            if not frame: self.send_error(503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(frame); return
        self.send_error(404)

class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(os.path.join(HERE, "secrets", "server.crt"), os.path.join(HERE, "secrets", "server.key"))
    threading.Thread(target=capture_loop, daemon=True).start()
    srv = Server((BIND, PORT), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"perigee cast: https://{BIND}:{PORT}/<token>/   capturing window '{WINDOW_TITLE}' at {FPS} fps")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
