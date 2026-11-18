"""Large-image pipeline: generate at ~1 MP, upscale to an exact size, optionally add real detail.

Stage 1  FLUX generates (or edits) at ~1 MP, the upscale model enlarges it 4x and it is
         resized to exactly the requested width x height.
Stage 2  (optional "detail") the large image is cut into overlapping ~1 MP tiles. For every tile:
           - the matching region of the NEW image (upscaled) is the starting point,
           - the matching region of the ORIGINAL reference photo, zoomed to the tile size, is
             given to FLUX as a reference image so it can see how that spot originally looked,
           - a vision model looks at both and writes a caption of what the tile really contains,
           - FLUX re-renders the tile at a light denoise, then the colours/brightness are
             locked back to the upscaled image so neighbouring tiles agree.
         The tiles are blended back together with feathered overlaps.
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
from PIL import Image, ImageFilter, ImageOps

from generate import (COMFY_DIR, MAX_TARGET_SIDE, base_size_for,
                      build_flux3_workflow, build_workflow)

TILE = 1024              # tile edge in pixels (~1 MP per tile)
OVERLAP = 192            # minimum overlap between neighbouring tiles
REFINE_STEPS = 8         # sampler steps per tile
REFINE_DENOISE = 0.25    # 0 = keep the upscale as is, 1 = redraw completely
# Timestep shift for the tile pass. Generation uses 16, but at that shift a denoise of 0.3 starts
# at ~87% noise (sigma = 16t / (1 + 15t)), i.e. a near-complete redraw that produces gibberish.
# With a shift of 1 the start sigma equals the denoise value, so 0.25 is a genuinely light touch.
REFINE_SHIFT = 1.0
# After re-rendering, everything coarser than this blur radius (px) is taken from the upscaled
# image again, so only fine detail comes from FLUX. This stops colour/brightness drift between
# tiles, which is what made the blended seams visible.
LOWFREQ_RADIUS = 8
DIRECT_PIXELS = 1_500_000  # targets up to this size are generated directly, no upscale

CAPTION_SYSTEM = (
    "You write captions for an image-restoration model. You are shown one crop of a larger "
    "image. Describe exactly and only what is visible in this crop: objects, materials, "
    "surface textures, colours, lighting and any text. Do not invent anything that is not "
    "visible, do not describe the whole scene, and do not mention that it is a crop. "
    "Output a single plain caption of at most 80 words.")
CAPTION_SYSTEM_REF = (
    "You write captions for an image-restoration model. You are shown two crops of the same "
    "area: image 1 is the ORIGINAL photo, image 2 is the NEW version that is being refined. "
    "Describe exactly what is visible in image 2: objects, materials, surface textures, colours, "
    "lighting and any text or labels. Use image 1 only to read fine details, text and textures "
    "that image 2 shows too blurrily. The requested changes are already applied in image 2, so "
    "never describe anything that image 2 no longer contains. Do not invent anything. Do not "
    "mention crops or image numbers. Output a single plain caption of at most 80 words.")


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


def load_reference(path) -> Image.Image:
    """Open a reference photo the same way ComfyUI's LoadImage does (EXIF rotation applied)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def fit_aspect(img: Image.Image, aspect: float) -> Image.Image:
    """Centre-crop to the given width/height aspect (same as ImageScale crop=center)."""
    w, h = img.size
    if w / h > aspect:
        nw = max(1, round(h * aspect))
        x0 = (w - nw) // 2
        return img.crop((x0, 0, x0 + nw, h))
    nh = max(1, round(w / aspect))
    y0 = (h - nh) // 2
    return img.crop((0, y0, w, y0 + nh))


def match_low_freq(out: np.ndarray, guide: Image.Image, radius: int = LOWFREQ_RADIUS) -> np.ndarray:
    """Keep FLUX's fine detail but take colour, brightness and coarse structure from `guide`.

    out - blur(out) is the detail layer; blur(guide) is the base layer. Adding them removes the
    slight colour/exposure drift a VAE round trip and a per-tile re-render introduce.
    """
    h, w = out.shape[:2]
    out_img = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    guide = guide.crop((0, 0, w, h))
    blur = ImageFilter.GaussianBlur(radius)
    lo_out = np.asarray(out_img.filter(blur), np.float32)
    lo_guide = np.asarray(guide.filter(blur), np.float32)
    return np.clip(out - lo_out + lo_guide, 0, 255)


def build_refine_workflow(profile: dict, context: str, seed: int, steps: int, image: str,
                          ref_image, width: int, height: int, denoise: float) -> dict:
    """FLUX.3 low-denoise img2img of one tile.

    image:     the tile cut from the upscaled new image (starting point of the sampler).
    ref_image: optional crop of the ORIGINAL photo covering the same area, zoomed to the tile
               size. It is given to FLUX as a reference latent and to the captioner, so the
               tile can recover real detail from the original.
    context:   the user's request, only used as a hint for the captioner.
    The sampler starts from the encoded tile (not an empty latent) at a low shift and denoise,
    so the content is kept and only fine detail is redrawn. The result goes to ComfyUI's temp folder.
    """
    ask = ((f'The image was made from this request, which is already applied: "{context}"\n' if context else "")
           + ("Describe image 2. Output the caption." if ref_image else "Describe this crop. Output the caption."))
    w = {
        "1": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": profile["upsampler"], "type": "flux3", "device": "default"}},
        "2": {"class_type": "TextGenerate", "inputs": {
            "clip": ["1", 0], "image": ["20", 0],
            "system_prompt": CAPTION_SYSTEM_REF if ref_image else CAPTION_SYSTEM, "prompt": ask,
            "max_length": 300,
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
            "mtp": "off"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": profile["unet"], "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": profile["clip"], "type": "flux3", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": profile["vae"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": ""}},
        "20": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        "21": {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["5", 0]}},
        "9": {"class_type": "PrimitiveFloat", "inputs": {"value": REFINE_SHIFT}},
        "10": {"class_type": "ComfyMathExpression", "inputs": {"values.a": ["9", 0], "expression": "log(a)"}},
        "11": {"class_type": "ModelSamplingFlux", "inputs": {
            "model": ["3", 0], "max_shift": ["10", 0], "base_shift": ["10", 0],
            "width": width, "height": height}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "14": {"class_type": "PreviewImage", "inputs": {"images": ["13", 0]}},
    }
    pos, neg = ["6", 0], ["7", 0]
    if ref_image:
        w["22"] = {"class_type": "LoadImage", "inputs": {"image": ref_image, "upload": "image"}}
        # image 1 = original, image 2 = new version (the order the captioner prompt refers to)
        w["54"] = {"class_type": "BatchImagesNode", "inputs": {
            "images.image0": ["22", 0], "images.image1": ["20", 0]}}
        w["2"]["inputs"]["image"] = ["54", 0]
        w["52"] = {"class_type": "ReferenceMultiLatent", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["5", 0],
            "resolution": "resize",
            "resolution.megapixels": max(0.1, round(min(1.0, width * height / 1e6), 3)),
            "images.image0": ["22", 0]}}
        pos, neg = ["52", 0], ["52", 1]
    w["12"] = {"class_type": "KSampler", "inputs": {
        "model": ["11", 0], "positive": pos, "negative": neg,
        "latent_image": ["21", 0], "seed": seed, "steps": steps, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}}
    return w


def refine_tiles(comfy, profile: dict, context: str, seed: int, img: Image.Image,
                 reference: Image.Image = None, denoise: float = REFINE_DENOISE,
                 steps: int = REFINE_STEPS, log=print) -> Image.Image:
    """Re-render every tile of `img` with FLUX.3 and blend the results back together.

    reference: the original photo. Each tile gets the matching zoomed-in region of it.
    """
    width, height = img.size
    tiles = plan_tiles(width, height)
    ref = fit_aspect(reference, width / height) if reference is not None else None
    original = np.asarray(img)
    acc = np.zeros((height, width, 3), np.float32)
    wsum = np.zeros((height, width), np.float32)
    sub = f"hires_{uuid.uuid4().hex[:8]}"   # one input subfolder per run, removed afterwards
    log(f"Adding detail: {len(tiles)} tiles of up to {TILE}x{TILE}, overlap >= {OVERLAP}px"
        + (", guided by the original photo" if ref is not None else ""))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for i, t in enumerate(tiles):
                log(f"  tile {i + 1}/{len(tiles)}")
                guide = img.crop((t.x, t.y, t.x + t.w, t.y + t.h))
                gpath = Path(tmp) / f"tile_{i}.png"
                guide.save(gpath)
                gname = comfy.upload(gpath, subfolder=sub)
                rname = None
                if ref is not None:
                    rw, rh = ref.size
                    box = (t.x / width * rw, t.y / height * rh,
                           (t.x + t.w) / width * rw, (t.y + t.h) / height * rh)
                    rpath = Path(tmp) / f"ref_{i}.png"
                    ref.resize((t.w, t.h), Image.LANCZOS, box=box).save(rpath)
                    rname = comfy.upload(rpath, subfolder=sub)
                wf = build_refine_workflow(profile, context, seed, steps, gname, rname, t.w, t.h, denoise)
                out = np.asarray(Image.open(io.BytesIO(comfy.run(wf)[0])).convert("RGB"), np.float32)
                out = match_low_freq(out, guide)
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
                  reference: Image.Image = None,
                  denoise: float = REFINE_DENOISE, log=print) -> bytes:
    """Return PNG bytes at exactly `target` = (width, height).

    images:    filenames already uploaded to ComfyUI (reference/edit images), may be empty.
    reference: the first reference photo as a PIL image (see load_reference). Only used by the
               detail pass, which shows each tile the matching region of it.
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
    refined = refine_tiles(comfy, profile, prompt, seed, img, reference=reference,
                           denoise=denoise, log=log)
    buf = io.BytesIO()
    refined.save(buf, "PNG")
    return buf.getvalue()
