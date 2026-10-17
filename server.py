"""Tiny phone-friendly web page for generating images through ComfyUI.

    python server.py                  # http://127.0.0.1:8080

If ComfyUI is already running it is used as is. Otherwise it is started when a
request arrives and stopped again once every pending request has finished or failed.

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
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from generate import COMFY_DIR, OUTPUT_DIR, Comfy, build_flux3_workflow, build_workflow
from profiles import DEFAULT_PROFILE, PROFILES

MAX_BODY = 100 * 1024 * 1024
MAX_IMAGES = 3  # keep VRAM sane on 16 GB
SIZES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}

GEN_LOCK = threading.Lock()  # one generation at a time
COMFY_HOST = "127.0.0.1:8188"

_lease_lock = threading.Lock()
_active = 0       # requests currently holding ComfyUI open (running or queued)
_comfy = None     # shared Comfy; only has a process if we started it


@contextmanager
def comfy_lease():
    """Keep ComfyUI up for the duration of a request.

    Already running -> reuse it and never stop it. Not running -> start it for the first
    request; it is stopped when the last overlapping request finishes or fails.
    """
    global _active, _comfy
    with _lease_lock:
        if _active == 0:
            c = Comfy(COMFY_HOST)
            try:
                c.ensure_running(COMFY_HOST)  # no-op if already up
            except SystemExit as e:  # ensure_running reports failures via sys.exit(message)
                c.stop()
                raise RuntimeError(str(e.code))
            _comfy = c
        _active += 1
        comfy = _comfy
    try:
        yield comfy
    finally:
        with _lease_lock:
            _active -= 1
            if _active == 0:
                _comfy.stop()  # does nothing if ComfyUI was already running before us
                _comfy = None


def installed_profiles() -> list:
    return [name for name, p in PROFILES.items()
            if all((COMFY_DIR / "models" / f["folder"] / f["name"]).exists() for f in p["files"])]


def generate(prompt: str, profile_name: str, size: str, data_urls: list) -> list:
    """Returns the saved output filenames. Raises RuntimeError with a user-facing message."""
    profile = PROFILES.get(profile_name)
    if profile is None or profile_name not in installed_profiles():
        raise RuntimeError(f"Profile '{profile_name}' is not installed.")
    with comfy_lease() as comfy:
        return _generate(comfy, prompt, profile_name, profile, size, data_urls)


def _generate(comfy, prompt: str, profile_name: str, profile: dict, size: str, data_urls: list) -> list:
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

        # With reference images the output follows the first image's shape;
        # otherwise use the chosen size preset.
        width, height = SIZES.get(size, SIZES["square"])
        if uploaded:
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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<meta name="theme-color" content="#0d0f14">
<title>FLUX Studio</title>
<style>
  :root {
    --bg: #090a0f;
    --card: #13151f;
    --card-hover: #181b28;
    --card-subtle: #1c202f;
    --fg: #f1f5f9;
    --muted: #94a3b8;
    --border: #252b3d;
    --border-hover: #38425d;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.35);
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
    --danger: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.12);
    --success: #10b981;
    --radius: 16px;
    --radius-sm: 10px;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --card-hover: #f1f5f9;
      --card-subtle: #f1f5f9;
      --fg: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --border-hover: #cbd5e1;
      --accent: #4f46e5;
      --accent-glow: rgba(79, 70, 229, 0.2);
      --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
      --danger: #dc2626;
      --danger-bg: rgba(220, 38, 38, 0.08);
      --success: #059669;
    }
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, system-ui, sans-serif;
    line-height: 1.5;
    padding: 20px 16px calc(32px + env(safe-area-inset-bottom));
    max-width: 600px;
    margin-inline: auto;
  }
  
  /* App Header */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0 18px 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .brand-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    background: var(--accent-gradient);
    border-radius: 8px;
    color: #fff;
    -webkit-text-fill-color: #fff;
    font-size: 15px;
    box-shadow: 0 4px 14px var(--accent-glow);
  }
  .status-pill {
    font-size: 12px;
    font-weight: 500;
    color: var(--muted);
    background: var(--card-subtle);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s;
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 6px var(--success);
    transition: all 0.2s;
  }
  .status-dot.busy {
    background: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Main Form Panel */
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  textarea {
    width: 100%;
    background: var(--card-subtle);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px;
    font: inherit;
    font-size: 15px;
    min-height: 105px;
    resize: vertical;
    transition: border-color 0.15s, box-shadow 0.15s;
    outline: none;
  }
  textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  textarea::placeholder {
    color: var(--muted);
    opacity: 0.8;
  }

  /* Reference Photos */
  .photo-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .photo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .section-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }
  .add-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    background: var(--card-subtle);
    color: var(--fg);
    border: 1px dashed var(--border-hover);
    border-radius: 8px;
    padding: 7px 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .add-btn:hover {
    border-color: var(--accent);
    background: var(--card-hover);
  }
  .thumbs {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .thumb {
    position: relative;
    width: 76px;
    height: 76px;
  }
  .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
  }
  .thumb button {
    position: absolute;
    top: -6px;
    right: -6px;
    width: 22px;
    height: 22px;
    padding: 0;
    border-radius: 50%;
    background: var(--danger);
    color: #fff;
    border: 2px solid var(--card);
    font-size: 12px;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  }
  .hint {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.4;
  }

  /* Aspect Ratio Selector */
  .aspect-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }
  .aspect-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 8px;
    background: var(--card-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  .aspect-btn:hover {
    border-color: var(--border-hover);
    color: var(--fg);
  }
  .aspect-btn.on {
    background: var(--card-hover);
    border-color: var(--accent);
    color: var(--fg);
    box-shadow: 0 0 0 2px var(--accent-glow);
  }
  .aspect-icon {
    border: 1.5px solid currentColor;
    border-radius: 3px;
  }
  .aspect-sq { width: 16px; height: 16px; }
  .aspect-pt { width: 13px; height: 18px; }
  .aspect-ls { width: 20px; height: 14px; }

  /* Model & Engine Select */
  details {
    border-top: 1px solid var(--border);
    padding-top: 12px;
  }
  summary {
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    user-select: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  summary::-webkit-details-marker { display: none; }
  summary::after {
    content: "▾";
    font-size: 12px;
    transition: transform 0.2s;
  }
  details[open] summary::after {
    transform: rotate(180deg);
  }
  select {
    width: 100%;
    margin-top: 10px;
    background: var(--card-subtle);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font: inherit;
    font-size: 14px;
    outline: none;
    cursor: pointer;
  }

  /* Primary Action Button */
  .primary-btn {
    width: 100%;
    background: var(--accent-gradient);
    color: #fff;
    font-weight: 600;
    font-size: 16px;
    letter-spacing: 0.01em;
    padding: 15px;
    border: 0;
    border-radius: var(--radius-sm);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 6px 20px var(--accent-glow);
    transition: transform 0.1s, opacity 0.15s, box-shadow 0.15s;
  }
  .primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px var(--accent-glow);
  }
  .primary-btn:active {
    transform: translateY(1px);
  }

  /* Form Top Error Message */
  #formError:not(:empty) {
    margin-top: 10px;
    background: var(--danger-bg);
    color: var(--danger);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-size: 13px;
    text-align: center;
  }

  /* Feed / Queue Area */
  .feed-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 28px;
    margin-bottom: 14px;
  }
  .feed-title {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }
  .clear-btn {
    background: transparent;
    border: 0;
    color: var(--muted);
    font-size: 12px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 6px;
  }
  .clear-btn:hover {
    color: var(--fg);
    background: var(--card-subtle);
  }

  .feed {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Job Card */
  .job-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,0.14);
    animation: slideIn 0.25s ease-out;
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .job-meta {
    padding: 12px 16px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
  }
  .job-prompt {
    font-size: 14px;
    font-weight: 600;
    color: var(--fg);
    line-height: 1.4;
    word-break: break-word;
  }
  .job-tags {
    display: flex;
    gap: 6px;
    margin-top: 6px;
    flex-wrap: wrap;
  }
  .tag {
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 12px;
    background: var(--card-subtle);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .card-close {
    background: transparent;
    border: 0;
    color: var(--muted);
    font-size: 16px;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }
  .card-close:hover { color: var(--danger); }

  /* Processing State */
  .job-processing {
    padding: 28px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: var(--card-subtle);
  }
  .spinner {
    width: 24px;
    height: 24px;
    border: 2.5px solid var(--border-hover);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .job-status-text {
    font-size: 14px;
    color: var(--fg);
    font-weight: 500;
  }
  .job-timer {
    font-size: 12px;
    color: var(--muted);
  }

  /* Completed State */
  .job-img-wrap {
    position: relative;
    background: #000;
  }
  .job-img-wrap img {
    width: 100%;
    height: auto;
    display: block;
  }
  .job-actions {
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: var(--card);
  }
  .job-info-sub {
    font-size: 12px;
    color: var(--muted);
  }
  .job-btn-group {
    display: flex;
    gap: 8px;
  }
  .btn-sm {
    font-size: 13px;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--card-subtle);
    color: var(--fg);
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s;
  }
  .btn-sm:hover {
    border-color: var(--border-hover);
    background: var(--card-hover);
  }
  .btn-primary-sm {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .btn-primary-sm:hover { opacity: 0.9; }

  /* Error State */
  .job-error {
    padding: 20px 16px;
    background: var(--danger-bg);
    color: var(--danger);
    font-size: 13px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    text-align: center;
  }

  input[type=file] { display: none; }
</style>
</head>
<body>
  <div class="header">
    <div class="brand">
      <div class="brand-icon">✦</div>
      FLUX Studio
    </div>
    <div class="status-pill" id="globalStatus">
      <div class="status-dot" id="globalDot"></div>
      <span id="globalText">Ready</span>
    </div>
  </div>

  <div class="panel">
    <textarea id="prompt" placeholder="Describe the image you want to create (e.g. 'a futuristic city in rain at twilight')..." autofocus></textarea>

    <div class="photo-section">
      <div class="photo-header">
        <span class="section-label">Reference Images</span>
        <button class="add-btn" id="addBtn" type="button">+ Add Photos</button>
        <input id="file" type="file" accept="image/*" multiple>
      </div>
      <div class="thumbs" id="thumbs"></div>
      <div class="hint" id="hint">No photos: generate fresh artwork directly from prompt.</div>
    </div>

    <div id="sizesContainer">
      <div class="section-label" style="margin-bottom:8px">Aspect Ratio</div>
      <div class="aspect-grid" id="sizes">
        <button class="aspect-btn on" data-size="square" type="button">
          <div class="aspect-icon aspect-sq"></div>
          Square (1:1)
        </button>
        <button class="aspect-btn" data-size="portrait" type="button">
          <div class="aspect-icon aspect-pt"></div>
          Portrait (9:16)
        </button>
        <button class="aspect-btn" data-size="landscape" type="button">
          <div class="aspect-icon aspect-ls"></div>
          Landscape (16:9)
        </button>
      </div>
    </div>

    <details>
      <summary>Model & Engine Profile</summary>
      <select id="profile"></select>
    </details>

    <button class="primary-btn" id="go" type="button">
      <span>Generate Image</span>
      <span>✦</span>
    </button>
    <div id="formError"></div>
  </div>

  <div class="feed-header">
    <span class="feed-title">Gallery & History</span>
    <button class="clear-btn" id="clearBtn" type="button">Clear History</button>
  </div>

  <div class="feed" id="feed"></div>

<script>
const MAX_IMAGES = __MAX_IMAGES__, MAX_SIDE = 1536;
const STORAGE_KEY = "flux_gallery_history";
const $ = id => document.getElementById(id);
let photos = [];            // data URLs for reference
let size = "square";
let activeJobCount = 0;
let jobCounter = 0;

// Persistent History Storage Helpers
function getStoredHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch (e) {
    return [];
  }
}

function saveToHistory(item) {
  try {
    const list = getStoredHistory().filter(x => x.id !== item.id);
    list.unshift(item);
    if (list.length > 80) list.pop(); // keep last 80 items
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch (e) {}
}

function removeFromHistory(id) {
  try {
    const list = getStoredHistory().filter(x => x.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch (e) {}
}

function clearAllHistory() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (e) {}
}

// Initial Model Loading
fetch("/api/info").then(r => r.json()).then(info => {
  const sel = $("profile");
  info.profiles.forEach(p => { const o = new Option(p, p); sel.add(o); });
  sel.value = localStorage.profile && info.profiles.includes(localStorage.profile) ? localStorage.profile : info.default;
  if (!info.profiles.length) $("formError").textContent = "No models installed. Run download_models.py.";
});
$("profile").onchange = e => localStorage.profile = e.target.value;

document.querySelectorAll("#sizes .aspect-btn").forEach(b => b.onclick = () => {
  size = b.dataset.size;
  document.querySelectorAll("#sizes .aspect-btn").forEach(c => c.classList.toggle("on", c === b));
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
  $("addBtn").textContent = photos.length ? "+ Add more photos" : "+ Add Photos";
  $("addBtn").disabled = photos.length >= MAX_IMAGES;
  $("sizesContainer").style.display = photos.length ? "none" : "";
  $("hint").textContent = photos.length
    ? "Using your photo" + (photos.length > 1 ? "s (call them image 1, image 2... in the prompt)." : " as a reference.")
    : "No photos: generate fresh artwork directly from prompt.";
}

function updateGlobalStatus() {
  const dot = $("globalDot");
  const txt = $("globalText");
  if (activeJobCount > 0) {
    dot.className = "status-dot busy";
    txt.textContent = activeJobCount === 1 ? "1 task generating/queued" : `${activeJobCount} tasks in queue`;
  } else {
    dot.className = "status-dot";
    txt.textContent = "Ready";
  }
}

function renderCompletedCard(item, prepend = true) {
  const card = document.createElement("div");
  card.className = "job-card";
  card.id = item.id;
  const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  
  card.innerHTML = `
    <div class="job-meta">
      <div>
        <div class="job-prompt">${escapeHtml(item.prompt || item.name || 'Generated image')}</div>
        <div class="job-tags">
          ${item.size ? `<span class="tag">${item.size}</span>` : ''}
          ${item.profile ? `<span class="tag">${item.profile}</span>` : ''}
          ${timeStr ? `<span class="tag">${timeStr}</span>` : ''}
        </div>
      </div>
      <button class="card-close" title="Remove from history" onclick="deleteCard('${item.id}')">&times;</button>
    </div>
    ${(item.images || [item.name]).map(imgName => `
      <div class="job-img-wrap">
        <img src="/outputs/${imgName}" alt="Generated image" loading="lazy">
      </div>
      <div class="job-actions">
        <span class="job-info-sub">${item.elapsed ? `✓ ${item.elapsed}s` : imgName}</span>
        <div class="job-btn-group">
          ${item.prompt ? `<button class="btn-sm" type="button" onclick="reusePrompt('${escapeJs(item.prompt)}')">↺ Reuse</button>` : ''}
          <a class="btn-sm btn-primary-sm" href="/outputs/${imgName}" download>⬇ Save</a>
        </div>
      </div>
    `).join("")}
  `;

  if (prepend) {
    $("feed").prepend(card);
  } else {
    $("feed").appendChild(card);
  }
}

function deleteCard(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
  removeFromHistory(id);
}

function submitRequest() {
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("formError").textContent = "Write a prompt first.";
    $("prompt").focus();
    return;
  }
  $("formError").textContent = "";

  const currentProfile = $("profile").value;
  const currentSize = size;
  const currentPhotos = [...photos];
  const jobId = "job_" + (++jobCounter) + "_" + Date.now();
  const startTime = Date.now();

  // Create job card in feed immediately
  const card = document.createElement("div");
  card.className = "job-card";
  card.id = jobId;
  card.innerHTML = `
    <div class="job-meta">
      <div>
        <div class="job-prompt">${escapeHtml(prompt)}</div>
        <div class="job-tags">
          <span class="tag">${currentSize}</span>
          <span class="tag">${currentProfile}</span>
          ${currentPhotos.length ? `<span class="tag">${currentPhotos.length} ref photo(s)</span>` : ''}
        </div>
      </div>
      <button class="card-close" title="Remove" onclick="document.getElementById('${jobId}').remove()">&times;</button>
    </div>
    <div class="job-processing" id="proc_${jobId}">
      <div class="spinner"></div>
      <div class="job-status-text">Processing image...</div>
      <div class="job-timer" id="timer_${jobId}">Elapsed: 0s</div>
    </div>
    <div id="content_${jobId}"></div>
  `;
  $("feed").prepend(card);

  activeJobCount++;
  updateGlobalStatus();

  // Keep live timer running for this specific job
  const timer = setInterval(() => {
    const el = $(`timer_${jobId}`);
    if (el) {
      el.textContent = `Elapsed: ${Math.round((Date.now() - startTime) / 1000)}s`;
    }
  }, 500);

  // Send request in background without blocking the UI
  fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, profile: currentProfile, size: currentSize, images: currentPhotos })
  })
  .then(async r => {
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Generation failed.");
    return j;
  })
  .then(res => {
    clearInterval(timer);
    const elapsed = Math.round((Date.now() - startTime) / 1000);

    // Save item to persistent localStorage history
    const historyItem = {
      id: jobId,
      prompt,
      profile: currentProfile,
      size: currentSize,
      images: res.images,
      timestamp: Date.now(),
      elapsed
    };
    saveToHistory(historyItem);

    const cardEl = document.getElementById(jobId);
    if (cardEl) {
      cardEl.remove();
      renderCompletedCard(historyItem, true);
    }
  })
  .catch(err => {
    clearInterval(timer);
    const procEl = $(`proc_${jobId}`);
    if (procEl) procEl.remove();

    const contentEl = $(`content_${jobId}`);
    if (contentEl) {
      contentEl.innerHTML = `
        <div class="job-error">
          <div>⚠️ ${escapeHtml(err.message)}</div>
          <button class="btn-sm" type="button" onclick="document.getElementById('${jobId}').remove()">Dismiss</button>
        </div>
      `;
    }
  })
  .finally(() => {
    activeJobCount = Math.max(0, activeJobCount - 1);
    updateGlobalStatus();
  });
}

function reusePrompt(text) {
  $("prompt").value = text;
  $("prompt").focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escapeJs(s) {
  return s.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n");
}

$("go").onclick = submitRequest;
$("prompt").addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    submitRequest();
  }
});

$("clearBtn").onclick = () => {
  clearAllHistory();
  document.querySelectorAll(".job-card").forEach(c => {
    if (!c.querySelector(".job-processing")) c.remove();
  });
};

// Load saved history on startup
function initHistory() {
  const localItems = getStoredHistory();
  if (localItems.length > 0) {
    localItems.forEach(item => renderCompletedCard(item, false));
  } else {
    // If no localStorage, fallback to server outputs
    fetch("/api/history")
      .then(r => r.json())
      .then(data => {
        if (data.images && data.images.length) {
          data.images.forEach(img => {
            renderCompletedCard({
              id: "hist_" + img.name,
              name: img.name,
              images: [img.name],
              timestamp: img.time * 1000
            }, false);
          });
        }
      })
      .catch(() => {});
  }
}

initHistory();
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
        elif self.path == "/api/history":
            if not OUTPUT_DIR.exists():
                self._json(200, {"images": []})
            else:
                files = sorted(OUTPUT_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
                history = [{"name": f.name, "time": int(f.stat().st_mtime)} for f in files[:80]]
                self._json(200, {"images": history})
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
    print(f"Open http://{args.bind}:{args.port}  (ComfyUI is started on demand)")
    try:
        ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _comfy is not None:  # Ctrl+C mid-generation: don't leave our ComfyUI process behind
            _comfy.stop()


if __name__ == "__main__":
    main()
