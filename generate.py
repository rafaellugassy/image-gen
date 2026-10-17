"""Generate images with FLUX.2 / FLUX.3 through a local ComfyUI server.

Examples:
    python generate.py "a lighthouse at dusk, cinematic"
    python generate.py "a red fox in snow" --width 1024 --height 768 --count 4
    python generate.py "a castle" --profile klein-4b-base --negative "blurry, low quality"
    python generate.py "make the bag blue" --image bag.png            # edit / restyle a reference image
    python generate.py "dress the cat in image 1 like image 2" --image cat.png --image style.png
    python generate.py "a lighthouse at dusk" --profile flux3-klein   # FLUX.3 with built-in prompt upsampler

If ComfyUI is not already running it is started automatically (and stopped
again when done). Start it yourself with run_comfyui.ps1 to keep models loaded
between runs.
"""
import argparse
import json
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from profiles import DEFAULT_PROFILE, PROFILES

ROOT = Path(__file__).resolve().parent
FLUX3_WORKFLOWS = ROOT / "flux3_workflows"

COMFY_DIR = ROOT / "ComfyUI"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"


def build_workflow(profile: dict, prompt: str, negative: str, width, height,
                   seed: int, steps: int, cfg: float, prefix: str, images=None) -> dict:
    """Build a ComfyUI API-format graph equivalent to the official FLUX.2 templates.

    images: optional list of filenames already in ComfyUI's input folder (see Comfy.upload).
    They are used as reference images for editing/restyling. If width/height are None
    and images are given, the output size follows the first image (scaled to ~1 megapixel).
    """
    images = images or []
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": profile["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": profile["clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": profile["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }

    mode = profile["mode"]
    # pos / neg are the conditioning outputs fed to the guider (reference images get chained on).
    if mode == "dev":
        w["9"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": cfg}}
        pos, neg = ["9", 0], None
    else:
        if mode == "distilled":
            w["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
        else:  # base
            w["9"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}}
        pos, neg = ["4", 0], ["9", 0]

    size_src = None
    for i, name in enumerate(images):
        load, scale, enc = f"2{i}0", f"2{i}1", f"2{i}2"
        w[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        w[scale] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [load, 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1}}
        w[enc] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        w[f"2{i}3"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [enc, 0]}}
        pos = [f"2{i}3", 0]
        if neg is not None:
            w[f"2{i}4"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [enc, 0]}}
            neg = [f"2{i}4", 0]
        if size_src is None:
            size_src = scale

    if size_src is None:
        width, height = width or 1024, height or 1024
    elif width is None or height is None:
        w["30"] = {"class_type": "GetImageSize", "inputs": {"image": [size_src, 0]}}
        width, height = ["30", 0], ["30", 1]

    w["5"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    w["6"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}}

    if mode == "dev":
        w["10"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": pos}}
    else:
        w["10"] = {"class_type": "CFGGuider", "inputs": {
            "model": ["1", 0], "positive": pos, "negative": neg, "cfg": cfg}}

    w["11"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["8", 0], "guider": ["10", 0], "sampler": ["7", 0],
        "sigmas": ["6", 0], "latent_image": ["5", 0]}}
    w["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    w["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}}
    return w


def _get_flux3_system_prompt(tpl_path: Path) -> str:
    """Extract the t2i system prompt string from the workflow JSON."""
    wf = json.loads(tpl_path.read_text(encoding="utf-8"))
    for sg in wf.get("definitions", {}).get("subgraphs", []):
        for node in sg.get("nodes", []):
            if node.get("title") == "System prompt: plain (t2i)":
                wvn = node.get("widgets_values_named") or {}
                return wvn.get("value", node.get("widgets_values", [""])[0])
    return ""


def build_flux3_workflow(profile: dict, prompt: str, seed: int, steps: int,
                         width: Optional[int], height: Optional[int],
                         prefix: str, images: Optional[list] = None) -> dict:
    """Build a flat ComfyUI API-format prompt for FLUX.3 klein.

    Expands the subgraph definitions from the official workflow JSONs into
    primitive nodes that the /prompt API endpoint understands directly.

    images: list of filenames already uploaded to ComfyUI's input folder.
    - 0 images → text-to-image
    - 1 image   → image-to-image  (ReferenceMultiLatent with 1 image)
    - 2-4 images → multi-image-to-image
    """
    images = images or []
    n_images = len(images)
    w = width or 1024
    h = height or 1024
    alpha = 16.0   # timestep shift; model reference is 6.93
    t2i_tpl = FLUX3_WORKFLOWS / "flux3_klein_int8_t2i.json"
    system_prompt = _get_flux3_system_prompt(t2i_tpl)

    # ── Prompt upsampler ──────────────────────────────────────────────────────
    # 1  CLIPLoader for the upsampler LLM
    # 2  TextGenerate: produces an expanded caption from the short user prompt
    # ── Diffusion ─────────────────────────────────────────────────────────────
    # 3  UNETLoader
    # 4  CLIPLoader (text encoder for the diffusion model)
    # 5  VAELoader
    # 6  CLIPTextEncode (positive, fed the upsampled prompt)
    # 7  CLIPTextEncode (negative, empty — cfg=1 distilled)
    # 8  EmptyFlux2LatentImage  (t2i) / image path (i2i)
    # 9  PrimitiveFloat (alpha)
    # 10 ComfyMathExpression (mu = ln(alpha))
    # 11 ModelSamplingFlux (timestep shift)
    # 12 KSampler
    # 13 VAEDecode
    # 14 SaveImage
    # For i2i: add LoadImage + ImageScaleToTotalPixels + VAEEncode + ReferenceMultiLatent

    p: dict = {}

    # Upsampler CLIP loader
    p["1"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": profile["upsampler"], "type": "flux3", "device": "default"}}

    # TextGenerate: upsampler LLM
    p["2"] = {"class_type": "TextGenerate", "inputs": {
        "clip": ["1", 0],
        "system_prompt": system_prompt,
        "prompt": prompt,
        "max_length": 3584,
        "sampling_mode": "on",
        "sampling_mode.temperature": 1,
        "sampling_mode.top_k": 1,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0,
        "sampling_mode.repetition_penalty": 1,
        "sampling_mode.seed": seed,
        "sampling_mode.presence_penalty": 1.5,
        "thinking": False,
        "use_default_template": True,
        "mtp": "off",
    }}

    # Diffusion model loaders
    p["3"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": profile["unet"], "weight_dtype": "default"}}
    p["4"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": profile["clip"], "type": "flux3", "device": "default"}}
    p["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": profile["vae"]}}

    # Text conditioning — positive uses the upsampled prompt output
    p["6"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["4", 0], "text": ["2", 0]}}
    p["7"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["4", 0], "text": ""}}

    # Timestep shift
    p["9"]  = {"class_type": "PrimitiveFloat",        "inputs": {"value": alpha}}
    p["10"] = {"class_type": "ComfyMathExpression",   "inputs": {
        "values.a": ["9", 0], "expression": "log(a)"}}
    p["11"] = {"class_type": "ModelSamplingFlux",     "inputs": {
        "model": ["3", 0], "max_shift": ["10", 0], "base_shift": ["10", 0],
        "width": w, "height": h}}

    pos_src, neg_src = ["6", 0], ["7", 0]
    if n_images == 0:
        # t2i: plain empty latent
        p["8"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": w, "height": h, "batch_size": 1}}
        latent_src = ["8", 0]
    else:
        # i2i / mi2i: load + scale each image, feed into ReferenceMultiLatent
        ref_inputs: dict = {
            "positive": ["6", 0],
            "negative": ["7", 0],
            "vae":      ["5", 0],
        }
        # ReferenceMultiLatent takes IMAGE inputs and encodes them itself with the VAE.
        for idx, img_name in enumerate(images[:4]):
            load_id  = str(20 + idx * 2)
            scale_id = str(21 + idx * 2)
            p[load_id]  = {"class_type": "LoadImage", "inputs": {
                "image": img_name, "upload": "image"}}
            p[scale_id] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": [load_id, 0], "upscale_method": "lanczos",
                "megapixels": 1.0, "resolution_steps": 1}}
            ref_inputs[f"images.image{idx}"] = [scale_id, 0]

        # Output latent — use explicit size if given, otherwise 1024×1024 (images
        # are already scaled to ~1 MP by ImageScaleToTotalPixels above)
        p["51"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": w, "height": h, "batch_size": 1}}

        # ReferenceMultiLatent outputs (positive, negative, empty_latent): the sampler
        # takes its conditioning from here; the latent comes from the sized empty latent.
        p["52"] = {"class_type": "ReferenceMultiLatent", "inputs": ref_inputs}
        p["53"] = {"class_type": "RepeatLatentBatch", "inputs": {
            "samples": ["51", 0], "amount": 1}}
        latent_src = ["53", 0]
        pos_src, neg_src = ["52", 0], ["52", 1]

    # KSampler
    p["12"] = {"class_type": "KSampler", "inputs": {
        "model":        ["11", 0],
        "positive":     pos_src,
        "negative":     neg_src,
        "latent_image": latent_src,
        "seed":         seed,
        "steps":        steps,
        "cfg":          1.0,
        "sampler_name": "euler",
        "scheduler":    "simple",
        "denoise":      1.0,
    }}

    # Decode + save
    p["13"] = {"class_type": "VAEDecode",  "inputs": {"samples": ["12", 0], "vae": ["5", 0]}}
    p["14"] = {"class_type": "SaveImage",  "inputs": {"images": ["13", 0], "filename_prefix": prefix}}

    return p


class Comfy:
    def __init__(self, host: str):
        self.base = f"http://{host}"
        self.client_id = str(uuid.uuid4())
        self.proc = None
        self.log = None

    def is_up(self) -> bool:
        try:
            return requests.get(f"{self.base}/system_stats", timeout=2).ok
        except requests.RequestException:
            return False

    def ensure_running(self, host: str, timeout: int = 300) -> None:
        if self.is_up():
            print(f"Using ComfyUI already running at {self.base}")
            return
        main_py = COMFY_DIR / "main.py"
        if not main_py.exists():
            sys.exit("ComfyUI not found. Run setup.ps1 first.")
        LOG_DIR.mkdir(exist_ok=True)
        self.log = open(LOG_DIR / "comfyui.log", "w", encoding="utf-8")
        hostname, port = host.split(":")
        print(f"Starting ComfyUI (log: logs/comfyui.log) ...")
        self.proc = subprocess.Popen(
            [sys.executable, str(main_py), "--listen", hostname, "--port", port, "--disable-auto-launch"],
            cwd=COMFY_DIR, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                sys.exit("ComfyUI exited during startup. See logs/comfyui.log")
            if self.is_up():
                return
            time.sleep(1)
        sys.exit("Timed out waiting for ComfyUI to start. See logs/comfyui.log")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log:
            self.log.close()

    def upload(self, path) -> str:
        """Upload a local image to ComfyUI's input folder and return the name to reference."""
        path = Path(path)
        with open(path, "rb") as f:
            r = requests.post(f"{self.base}/upload/image", files={"image": (path.name, f)},
                              data={"type": "input"}, timeout=120)
        if r.status_code != 200:
            sys.exit(f"Upload of {path} failed ({r.status_code}): {r.text}")
        info = r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def run(self, workflow: dict, timeout: int = 1800) -> list:
        r = requests.post(f"{self.base}/prompt",
                          json={"prompt": workflow, "client_id": self.client_id}, timeout=30)
        if r.status_code != 200:
            sys.exit(f"ComfyUI rejected the workflow ({r.status_code}): {r.text}")
        prompt_id = r.json()["prompt_id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = requests.get(f"{self.base}/history/{prompt_id}", timeout=30).json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    sys.exit(f"Generation failed: {msgs[-1][1] if msgs else status}")
                images = []
                for out in entry["outputs"].values():
                    for img in out.get("images", []):
                        data = requests.get(f"{self.base}/view", params=img, timeout=60)
                        data.raise_for_status()
                        images.append(data.content)
                return images
            time.sleep(0.5)
        sys.exit("Timed out waiting for the image.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate images with FLUX.2 / FLUX.3 via ComfyUI")
    ap.add_argument("prompt")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=PROFILES)
    ap.add_argument("--negative", default="", help="negative prompt (klein-4b-base only)")
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="reference image to edit/restyle; repeat for several (refer to them as 'image 1', 'image 2' in the prompt)")
    ap.add_argument("--width", type=int, help="default 1024, or the first --image's size if given")
    ap.add_argument("--height", type=int, help="default 1024, or the first --image's size if given")
    ap.add_argument("--steps", type=int, help="override profile default")
    ap.add_argument("--cfg", type=float, help="override profile default (guidance for dev)")
    ap.add_argument("--seed", type=int, help="random if omitted; incremented per image with --count")
    ap.add_argument("--count", type=int, default=1, help="number of images")
    ap.add_argument("--host", default="127.0.0.1:8188")
    args = ap.parse_args()

    for dim in (args.width, args.height):
        if dim is not None and dim % 16:
            sys.exit("Width and height must be multiples of 16.")
    for img in args.image:
        if not Path(img).is_file():
            sys.exit(f"Reference image not found: {img}")

    profile = PROFILES[args.profile]
    for f in profile["files"]:
        if not (COMFY_DIR / "models" / f["folder"] / f["name"]).exists():
            sys.exit(f"Missing model {f['folder']}/{f['name']}. Run: python download_models.py --profile {args.profile}")

    base_seed = args.seed if args.seed is not None else random.randint(0, 2**48)
    steps = args.steps or profile["steps"]
    cfg = args.cfg if args.cfg is not None else profile["cfg"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    comfy = Comfy(args.host)
    try:
        comfy.ensure_running(args.host)
        uploaded = [comfy.upload(p) for p in args.image]
        is_flux3 = profile.get("family") == "flux3"
        if is_flux3:
            print("Note: FLUX.3 uses a built-in prompt upsampler — your prompt will be "
                  "expanded automatically before generation.")
        for i in range(args.count):
            seed = base_seed + i
            if is_flux3:
                wf = build_flux3_workflow(profile, args.prompt, seed, steps,
                                          args.width, args.height,
                                          f"flux3_{args.profile}", images=uploaded)
            else:
                wf = build_workflow(profile, args.prompt, args.negative, args.width, args.height,
                                    seed, steps, cfg, f"flux2_{args.profile}", images=uploaded)
            print(f"Generating {i + 1}/{args.count} (seed {seed}) ...")
            t0 = time.time()
            for n, data in enumerate(comfy.run(wf)):
                path = OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.profile}_{seed}_{n}.png"
                path.write_bytes(data)
                print(f"  saved {path.relative_to(ROOT)} ({time.time() - t0:.1f}s)")
    finally:
        comfy.stop()


if __name__ == "__main__":
    main()
