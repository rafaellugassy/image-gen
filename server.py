"""Tiny phone-friendly web page for generating images through ComfyUI.

    python server.py                  # http://127.0.0.1:8080  (ComfyUI must be running)

Type a prompt. If you add photos they are used as reference images (edit/restyle),
otherwise it is a normal text-to-image generation. Everything else uses the
profile defaults.

It listens on localhost only. To reach it from your phone privately, run
    tailscale serve --bg 8080
"""
import argparse
import base64
import json
import random
import re
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from generate import COMFY_DIR, OUTPUT_DIR, Comfy, build_flux3_workflow, build_workflow
from profiles import DEFAULT_PROFILE, PROFILES

MAX_BODY = 100 * 1024 * 1024
MAX_IMAGES = 3  # keep VRAM sane on 16 GB
SIZES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}

GEN_LOCK = threading.Lock()  # one generation at a time
COMFY_HOST = "127.0.0.1:8188"


def installed_profiles() -> list:
    return [name for name, p in PROFILES.items()
            if all((COMFY_DIR / "models" / f["folder"] / f["name"]).exists() for f in p["files"])]


def generate(prompt: str, profile_name: str, size: str, data_urls: list) -> list:
    """Returns the saved output filenames. Raises RuntimeError with a user-facing message."""
    profile = PROFILES.get(profile_name)
    if profile is None or profile_name not in installed_profiles():
        raise RuntimeError(f"Profile '{profile_name}' is not installed.")
    comfy = Comfy(COMFY_HOST)
    if not comfy.is_up():
        raise RuntimeError("ComfyUI is not running. Start it with run_comfyui.ps1.")

    seed = random.randint(0, 2**48)
    steps, cfg = profile["steps"], profile["cfg"]
    is_flux3 = profile.get("family") == "flux3"

    with tempfile.TemporaryDirectory() as tmp, GEN_LOCK:
        uploaded = []
        for url in data_urls:
            m = re.match(r"data:image/(png|jpeg|webp);base64,(.+)", url, re.S)
            if not m:
                raise RuntimeError("Unsupported image.")
            path = Path(tmp) / f"web_{uuid.uuid4().hex}.{m.group(1).replace('jpeg', 'jpg')}"
            path.write_bytes(base64.b64decode(m.group(2)))
            uploaded.append(comfy.upload(path))

        # With reference images the output follows the first image's shape (flux2);
        # otherwise use the chosen size preset.
        width, height = SIZES.get(size, SIZES["square"])
        if uploaded and not is_flux3:
            width = height = None
        if is_flux3:
            wf = build_flux3_workflow(profile, prompt, seed, steps, width, height,
                                      f"web_{profile_name}", images=uploaded)
        else:
            wf = build_workflow(profile, prompt, "", width, height, seed, steps, cfg,
                                f"web_{profile_name}", images=uploaded)
        try:
            pngs = comfy.run(wf)
        except SystemExit as e:  # Comfy.run() reports errors via sys.exit(message)
            raise RuntimeError(str(e.code))

    OUTPUT_DIR.mkdir(exist_ok=True)
    names = []
    for n, data in enumerate(pngs):
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{profile_name}_{seed}_{n}.png"
        (OUTPUT_DIR / name).write_bytes(data)
        names.append(name)
    return names


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>Image Gen</title>
<style>
  :root { --bg:#111; --fg:#f2f2f2; --card:#1c1c1e; --muted:#8e8e93; --accent:#3b82f6; --danger:#ef4444; }
  @media (prefers-color-scheme: light) { :root { --bg:#f5f5f7; --fg:#111; --card:#fff; --muted:#6b7280; } }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; background:var(--bg); color:var(--fg); font:17px/1.4 -apple-system,system-ui,sans-serif;
         padding:16px 16px calc(24px + env(safe-area-inset-bottom)); max-width:560px; margin-inline:auto; }
  textarea, select { width:100%; background:var(--card); color:var(--fg); border:1px solid #8884; border-radius:14px;
                     padding:14px; font:inherit; }
  textarea { min-height:120px; resize:vertical; }
  button { font:inherit; border:0; border-radius:14px; padding:14px 16px; cursor:pointer; color:var(--fg); background:var(--card); }
  button.primary { width:100%; background:var(--accent); color:#fff; font-weight:600; margin-top:14px; padding:16px; }
  button:disabled { opacity:.5; }
  .row { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  .row > * { flex:1; }
  .chip { text-align:center; border:1px solid #8884; }
  .chip.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  .thumbs { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  .thumb { position:relative; width:88px; height:88px; }
  .thumb img { width:100%; height:100%; object-fit:cover; border-radius:12px; }
  .thumb button { position:absolute; top:-6px; right:-6px; width:28px; height:28px; padding:0; border-radius:50%;
                  background:var(--danger); color:#fff; line-height:28px; }
  .hint { color:var(--muted); font-size:14px; margin-top:8px; }
  #result img { width:100%; border-radius:14px; margin-top:16px; display:block; }
  #result a { display:block; text-align:center; margin-top:10px; color:var(--accent); }
  #status { margin-top:14px; text-align:center; color:var(--muted); }
  #error { margin-top:14px; color:var(--danger); text-align:center; }
  details { margin-top:14px; color:var(--muted); }
  summary { cursor:pointer; }
  details select { margin-top:8px; }
  input[type=file] { display:none; }
</style></head>
<body>
  <textarea id="prompt" placeholder="Describe the image (or what to change in your photos)..." autofocus></textarea>

  <div class="row">
    <button id="addBtn" type="button">+ Add photos</button>
    <input id="file" type="file" accept="image/*" multiple>
  </div>
  <div class="thumbs" id="thumbs"></div>
  <div class="hint" id="hint">No photos: a new image is generated from your text.</div>

  <div class="row" id="sizes">
    <button class="chip on" data-size="square" type="button">Square</button>
    <button class="chip" data-size="portrait" type="button">Portrait</button>
    <button class="chip" data-size="landscape" type="button">Landscape</button>
  </div>

  <details><summary>Model</summary><select id="profile"></select></details>

  <button class="primary" id="go" type="button">Generate</button>
  <div id="status"></div>
  <div id="error"></div>
  <div id="result"></div>

<script>
const MAX_IMAGES = __MAX_IMAGES__, MAX_SIDE = 1536;
const $ = id => document.getElementById(id);
let photos = [];            // data URLs
let size = "square";

fetch("/api/info").then(r => r.json()).then(info => {
  const sel = $("profile");
  info.profiles.forEach(p => { const o = new Option(p, p); sel.add(o); });
  sel.value = localStorage.profile && info.profiles.includes(localStorage.profile) ? localStorage.profile : info.default;
  if (!info.profiles.length) $("error").textContent = "No models installed. Run download_models.py.";
});
$("profile").onchange = e => localStorage.profile = e.target.value;
$("prompt").value = localStorage.prompt || "";
$("prompt").oninput = e => localStorage.prompt = e.target.value;

document.querySelectorAll("#sizes .chip").forEach(b => b.onclick = () => {
  size = b.dataset.size;
  document.querySelectorAll("#sizes .chip").forEach(c => c.classList.toggle("on", c === b));
});

$("addBtn").onclick = () => $("file").click();
$("file").onchange = async e => {
  for (const f of e.target.files) {
    if (photos.length >= MAX_IMAGES) break;
    photos.push(await shrink(f));
  }
  e.target.value = "";
  renderPhotos();
};

async function shrink(file) {   // phone photos are huge; downscale + fix rotation
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.92);
}

function renderPhotos() {
  const t = $("thumbs"); t.innerHTML = "";
  photos.forEach((src, i) => {
    const d = document.createElement("div"); d.className = "thumb";
    d.innerHTML = '<img><button type="button">&times;</button>';
    d.querySelector("img").src = src;
    d.querySelector("button").onclick = () => { photos.splice(i, 1); renderPhotos(); };
    t.appendChild(d);
  });
  $("addBtn").textContent = photos.length ? "+ Add more photos" : "+ Add photos";
  $("addBtn").disabled = photos.length >= MAX_IMAGES;
  $("sizes").style.display = photos.length ? "none" : "";
  $("hint").textContent = photos.length
    ? "Using your photo" + (photos.length > 1 ? "s (call them image 1, image 2... in the prompt)." : " as a reference.")
    : "No photos: a new image is generated from your text.";
}

$("go").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { $("error").textContent = "Write a prompt first."; return; }
  $("error").textContent = ""; $("result").innerHTML = "";
  $("go").disabled = true;
  const t0 = Date.now();
  const timer = setInterval(() => $("status").textContent =
    "Generating... " + Math.round((Date.now() - t0) / 1000) + "s", 500);
  try {
    const r = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, profile: $("profile").value, size, images: photos }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Failed");
    $("result").innerHTML = j.images.map(n =>
      `<img src="/outputs/${n}"><a href="/outputs/${n}" download>Save image</a>`).join("");
  } catch (e) {
    $("error").textContent = e.message;
  } finally {
    clearInterval(timer); $("status").textContent = ""; $("go").disabled = false;
  }
};
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__MAX_IMAGES__", str(MAX_IMAGES))
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._json(200, {"profiles": installed_profiles(), "default": DEFAULT_PROFILE})
        elif self.path.startswith("/outputs/"):
            f = OUTPUT_DIR / Path(self.path[len("/outputs/"):]).name  # .name blocks path traversal
            if f.is_file() and f.suffix == ".png":
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self._json(413, {"error": "Upload too large."})
        try:
            req = json.loads(self.rfile.read(length))
            prompt = str(req.get("prompt", "")).strip()
            images = list(req.get("images", []))[:MAX_IMAGES]
            if not prompt:
                return self._json(400, {"error": "Write a prompt first."})
            names = generate(prompt, req.get("profile") or DEFAULT_PROFILE, req.get("size", "square"), images)
            self._json(200, {"images": names})
        except RuntimeError as e:
            self._json(500, {"error": str(e)})
        except Exception as e:  # keep the server alive on unexpected failures
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")


def main() -> None:
    global COMFY_HOST
    ap = argparse.ArgumentParser(description="Phone-friendly web page for ComfyUI image generation")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1", help="keep 127.0.0.1 and use 'tailscale serve' for remote access")
    ap.add_argument("--comfy", default=COMFY_HOST, help="ComfyUI host:port")
    args = ap.parse_args()
    COMFY_HOST = args.comfy
    print(f"Open http://{args.bind}:{args.port}  (ComfyUI at {COMFY_HOST} must be running)")
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
