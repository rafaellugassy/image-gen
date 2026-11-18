"""Large-image pipeline: generate at ~1 MP, upscale to an exact size, optionally add real detail.

Stage 1  FLUX generates (or edits) at ~1 MP, the upscale model enlarges it 4x and it is
         resized to exactly the requested width x height.
Stage 2  (optional "detail") the large image is cut into overlapping ~1 MP tiles. Every tile
         is re-rendered by FLUX.3 at low denoise (the upscaled pixels guide it, so the
         composition is kept and only fine detail is redrawn), then the tiles are blended
         back together with feathered overlaps so no seams show.
"""
import io
import math
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from generate import (COMFY_DIR, FLUX3_ALPHA, MAX_TARGET_SIDE, base_size_for,
                      build_flux3_workflow, build_workflow)

TILE = 1024              # tile edge in pixels (~1 MP per tile)
OVERLAP = 128            # minimum overlap between neighbouring tiles
REFINE_STEPS = 8         # sampler steps per tile
REFINE_DENOISE = 0.3     # 0 = keep the upscale as is, 1 = redraw completely
DIRECT_PIXELS = 1_500_000  # targets up to this size are generated directly, no upscale
DETAIL_PROMPT = "ultra detailed, sharp focus, natural fine texture, high quality"


class Tile(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    l: int   # overlap with the tile on the left (0 at the image border)
    t: int   # ... above
    r: int   # ... on the right
    b: int   # ... below


def parse_target(text: str) -> tuple:
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m:
        raise RuntimeError("Target must look like 4000x4000.")
    w, h = int(m.group(1)), int(m.group(2))
    if not (256 <= w <= MAX_TARGET_SIDE and 256 <= h <= MAX_TARGET_SIDE):
        raise RuntimeError(f"Target sides must be between 256 and {MAX_TARGET_SIDE} pixels.")
    return w, h


def target_from_long_side(long_side: int, aspect: float) -> tuple:
    """Exact target size whose longer side is `long_side`, with the given width/height aspect."""
    if not 256 <= long_side <= MAX_TARGET_SIDE:
        raise RuntimeError(f"Resolution must be between 256 and {MAX_TARGET_SIDE} pixels.")
    if aspect >= 1:
        return long_side, max(256, round(long_side / aspect))
    return max(256, round(long_side * aspect)), long_side


def _axis(length: int, tile: int, overlap: int) -> list:
    """Evenly spaced (start, size, overlap_before, overlap_after) along one axis."""
    if length <= tile:
        return [(0, length, 0, 0)]
    n = math.ceil((length - overlap) / (tile - overlap))
    starts = [round(i * (length - tile) / (n - 1)) for i in range(n)]
    return [(s, tile,
             starts[i - 1] + tile - s if i else 0,
             s + tile - starts[i + 1] if i < n - 1 else 0)
            for i, s in enumerate(starts)]


def plan_tiles(width: int, height: int, tile: int = TILE, overlap: int = OVERLAP) -> list:
    """Overlapping tiles that cover the whole image."""
    return [Tile(x, y, w, h, l, t, r, b)
            for (y, h, t, b) in _axis(height, tile, overlap)
            for (x, w, l, r) in _axis(width, tile, overlap)]


def _ramp(size: int, before: int, after: int) -> np.ndarray:
    """1-D blend weights: fade in over `before` px, fade out over `after` px.

    With equal overlaps the fade-out of one tile and the fade-in of its neighbour sum to 1.
    """
    r = np.ones(size, np.float32)
    if before:
        r[:before] = (np.arange(before, dtype=np.float32) + 1) / (before + 1)
    if after:
        r[size - after:] = np.minimum(
            r[size - after:], np.arange(after, 0, -1, dtype=np.float32) / (after + 1))
    return r


def tile_weights(t: Tile) -> np.ndarray:
    return np.outer(_ramp(t.h, t.t, t.b), _ramp(t.w, t.l, t.r))


def build_refine_workflow(profile: dict, prompt: str, seed: int, steps: int, image: str,
                          width: int, height: int, denoise: float) -> dict:
    """FLUX.3 low-denoise img2img of one tile.

    Skips the prompt upsampler (it would write a different caption for every tile) and
    starts from the encoded tile instead of an empty latent, so the composition is kept
    and only fine detail is redrawn. The result goes to ComfyUI's temp folder.
    """
    return {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": profile["unet"], "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": profile["clip"], "type": "flux3", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": profile["vae"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": ""}},
        "20": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        "21": {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["5", 0]}},
        "9": {"class_type": "PrimitiveFloat", "inputs": {"value": FLUX3_ALPHA}},
        "10": {"class_type": "ComfyMathExpression", "inputs": {"values.a": ["9", 0], "expression": "log(a)"}},
        "11": {"class_type": "ModelSamplingFlux", "inputs": {
            "model": ["3", 0], "max_shift": ["10", 0], "base_shift": ["10", 0],
            "width": width, "height": height}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["11", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["21", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "14": {"class_type": "PreviewImage", "inputs": {"images": ["13", 0]}},
    }


def refine_tiles(comfy, profile: dict, prompt: str, seed: int, img: Image.Image,
                 denoise: float = REFINE_DENOISE, steps: int = REFINE_STEPS,
                 log=print) -> Image.Image:
    """Re-render every tile of `img` with FLUX.3 and blend the results back together."""
    width, height = img.size
    tiles = plan_tiles(width, height)
    original = np.asarray(img)
    acc = np.zeros((height, width, 3), np.float32)
    wsum = np.zeros((height, width), np.float32)
    sub = f"hires_{uuid.uuid4().hex[:8]}"   # one input subfolder per run, removed afterwards
    log(f"Adding detail: {len(tiles)} tiles of up to {TILE}x{TILE}, overlap >= {OVERLAP}px")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for i, t in enumerate(tiles):
                log(f"  tile {i + 1}/{len(tiles)}")
                path = Path(tmp) / f"tile_{i}.png"
                img.crop((t.x, t.y, t.x + t.w, t.y + t.h)).save(path)
                name = comfy.upload(path, subfolder=sub)
                wf = build_refine_workflow(profile, prompt, seed, steps, name, t.w, t.h, denoise)
                out = np.asarray(Image.open(io.BytesIO(comfy.run(wf)[0])).convert("RGB"), np.float32)
                h, w = min(out.shape[0], t.h), min(out.shape[1], t.w)   # VAE may trim to a multiple of 8/16
                wt = tile_weights(t)[:h, :w]
                acc[t.y:t.y + h, t.x:t.x + w] += out[:h, :w] * wt[..., None]
                wsum[t.y:t.y + h, t.x:t.x + w] += wt
    finally:
        shutil.rmtree(COMFY_DIR / "input" / sub, ignore_errors=True)

    acc /= np.maximum(wsum, 1e-6)[..., None]
    result = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
    uncovered = wsum == 0                  # only if a tile came back trimmed: keep the upscaled pixels
    result[uncovered] = original[uncovered]
    return Image.fromarray(result)


def render_target(comfy, profile: dict, prompt: str, seed: int, steps: int, images: list,
                  target: tuple, detail: bool, prefix: str, negative: str = "",
                  denoise: float = REFINE_DENOISE, log=print) -> bytes:
    """Return PNG bytes at exactly `target` = (width, height).

    images: filenames already uploaded to ComfyUI (reference/edit images), may be empty.
    Raises RuntimeError for unusable settings; Comfy.run() reports failures via SystemExit.
    """
    tw, th = target
    is_flux3 = profile.get("family") == "flux3"
    if detail and not is_flux3:
        raise RuntimeError("The detail pass needs a FLUX.3 profile.")

    direct = not images and tw * th <= DIRECT_PIXELS and tw % 16 == 0 and th % 16 == 0
    if direct:
        base, upscale, tgt = (tw, th), 1, None
        if detail:
            log("Target is near native size: generating directly, detail pass not needed.")
            detail = False
    elif images:
        base, upscale, tgt = (None, None), 4, (tw, th)   # size follows the first image (~1 MP)
    else:
        base, upscale, tgt = base_size_for(tw, th), 4, (tw, th)

    if is_flux3:
        wf = build_flux3_workflow(profile, prompt, seed, steps, base[0], base[1], prefix,
                                  images=images, upscale=upscale, target=tgt)
    else:
        wf = build_workflow(profile, prompt, negative, base[0], base[1], seed, steps,
                            profile["cfg"], prefix, images=images, upscale=upscale, target=tgt)
    if not direct:
        log(f"Generating at ~1 MP and upscaling to {tw}x{th} ...")
    png = comfy.run(wf)[0]
    if not detail:
        return png

    img = Image.open(io.BytesIO(png)).convert("RGB")
    refined = refine_tiles(comfy, profile, DETAIL_PROMPT if images else prompt, seed, img,
                           denoise=denoise, log=log)
    buf = io.BytesIO()
    refined.save(buf, "PNG")
    return buf.getvalue()
