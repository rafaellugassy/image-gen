"""Generate images with FLUX.2 through a local ComfyUI server.

Examples:
    python generate.py "a lighthouse at dusk, cinematic"
    python generate.py "a red fox in snow" --width 1024 --height 768 --count 4
    python generate.py "a castle" --profile klein-4b-base --negative "blurry, low quality"

If ComfyUI is not already running it is started automatically (and stopped
again when done). Start it yourself with run_comfyui.ps1 to keep models loaded
between runs.
"""
import argparse
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

from profiles import DEFAULT_PROFILE, PROFILES

ROOT = Path(__file__).resolve().parent
COMFY_DIR = ROOT / "ComfyUI"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"


def build_workflow(profile: dict, prompt: str, negative: str, width: int, height: int,
                   seed: int, steps: int, cfg: float, prefix: str) -> dict:
    """Build a ComfyUI API-format graph equivalent to the official FLUX.2 templates."""
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": profile["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": profile["clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": profile["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
    }

    mode = profile["mode"]
    if mode == "dev":
        w["9"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": cfg}}
        w["10"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["9", 0]}}
    else:
        if mode == "distilled":
            w["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
        else:  # base
            w["9"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}}
        w["10"] = {"class_type": "CFGGuider", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["9", 0], "cfg": cfg}}

    w["11"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["8", 0], "guider": ["10", 0], "sampler": ["7", 0],
        "sigmas": ["6", 0], "latent_image": ["5", 0]}}
    w["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    w["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}}
    return w


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

    def run(self, workflow: dict, timeout: int = 1800) -> list:
        r = requests.post(f"{self.base}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=30)
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
    ap = argparse.ArgumentParser(description="Generate images with FLUX.2 via ComfyUI")
    ap.add_argument("prompt")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=PROFILES)
    ap.add_argument("--negative", default="", help="negative prompt (klein-4b-base only)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, help="override profile default")
    ap.add_argument("--cfg", type=float, help="override profile default (guidance for dev)")
    ap.add_argument("--seed", type=int, help="random if omitted; incremented per image with --count")
    ap.add_argument("--count", type=int, default=1, help="number of images")
    ap.add_argument("--host", default="127.0.0.1:8188")
    args = ap.parse_args()

    for dim in (args.width, args.height):
        if dim % 16:
            sys.exit("Width and height must be multiples of 16.")

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
        for i in range(args.count):
            seed = base_seed + i
            wf = build_workflow(profile, args.prompt, args.negative, args.width, args.height,
                                seed, steps, cfg, f"flux2_{args.profile}")
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
