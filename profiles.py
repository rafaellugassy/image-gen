"""Model profiles: which files to download and how to build the FLUX.2 workflow.

All URLs were verified against Hugging Face / the official ComfyUI workflow
templates. Add your own profile by copying an entry.
"""

HF = "https://huggingface.co"

# FLUX.3 repo base (gated)
_HF_FLUX3_COMFY = f"{HF}/black-forest-labs/flux-3-image-klein-comfy/resolve/main"

_KLEIN_4B_TE = {
    "folder": "text_encoders",
    "name": "qwen_3_4b.safetensors",
    "url": f"{HF}/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
}
_FLUX2_VAE = {
    "folder": "vae",
    "name": "flux2-vae.safetensors",
    "url": f"{HF}/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
}

PROFILES = {
    # Fits 16 GB comfortably (~15 GB download). Apache 2.0, no login needed.
    "klein-4b": {
        "description": "FLUX.2 klein 4B distilled - fast (4 steps), fits 16GB VRAM easily, Apache 2.0",
        "mode": "distilled",
        "steps": 4,
        "cfg": 1.0,
        "unet": "flux-2-klein-4b.safetensors",
        "clip": "qwen_3_4b.safetensors",
        "vae": "flux2-vae.safetensors",
        "gated": False,
        "files": [
            {
                "folder": "diffusion_models",
                "name": "flux-2-klein-4b.safetensors",
                "url": f"{HF}/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors",
            },
            _KLEIN_4B_TE,
            _FLUX2_VAE,
        ],
    },
    # Non-distilled: slower (20 steps) but more varied outputs, supports a negative prompt.
    "klein-4b-base": {
        "description": "FLUX.2 klein 4B base - 20 steps, CFG + negative prompt, Apache 2.0",
        "mode": "base",
        "steps": 20,
        "cfg": 5.0,
        "unet": "flux-2-klein-base-4b.safetensors",
        "clip": "qwen_3_4b.safetensors",
        "vae": "flux2-vae.safetensors",
        "gated": False,
        "files": [
            {
                "folder": "diffusion_models",
                "name": "flux-2-klein-base-4b.safetensors",
                "url": f"{HF}/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
            },
            _KLEIN_4B_TE,
            _FLUX2_VAE,
        ],
    },
    # Better quality than 4B. The transformer repo is gated: accept the terms at
    # https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8 and set HF_TOKEN.
    "klein-9b": {
        "description": "FLUX.2 klein 9B fp8 distilled - higher quality, needs HF_TOKEN (gated, non-commercial)",
        "mode": "distilled",
        "steps": 4,
        "cfg": 1.0,
        "unet": "flux-2-klein-9b-fp8.safetensors",
        "clip": "qwen_3_8b_fp8mixed.safetensors",
        "vae": "full_encoder_small_decoder.safetensors",
        "gated": True,
        "files": [
            {
                "folder": "diffusion_models",
                "name": "flux-2-klein-9b-fp8.safetensors",
                "url": f"{HF}/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors",
            },
            {
                "folder": "text_encoders",
                "name": "qwen_3_8b_fp8mixed.safetensors",
                "url": f"{HF}/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
            },
            {
                "folder": "vae",
                "name": "full_encoder_small_decoder.safetensors",
                "url": f"{HF}/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors",
            },
        ],
    },
    # The big model. ~54 GB download; on 16 GB VRAM it relies on offloading to
    # system RAM (plan for ~64 GB RAM) and is slow. Non-commercial license.
    "dev": {
        "description": "FLUX.2 dev fp8 - best quality, ~54GB, needs ~64GB RAM on a 16GB GPU, non-commercial",
        "mode": "dev",
        "steps": 20,
        "cfg": 4.0,  # FluxGuidance value
        "unet": "flux2_dev_fp8mixed.safetensors",
        "clip": "mistral_3_small_flux2_fp8.safetensors",
        "vae": "flux2-vae.safetensors",
        "gated": False,
        "files": [
            {
                "folder": "diffusion_models",
                "name": "flux2_dev_fp8mixed.safetensors",
                "url": f"{HF}/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
            },
            {
                "folder": "text_encoders",
                "name": "mistral_3_small_flux2_fp8.safetensors",
                "url": f"{HF}/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
            },
            _FLUX2_VAE,
        ],
    },
    # FLUX.3 Image Klein (int8) — 5.6B transformer + 4B VL text encoder + 9B prompt upsampler.
    # Gated: accept terms at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
    # and set HF_TOKEN. Fits 16 GB VRAM (int8 quantised). Requires ComfyUI Nodes 2.0.
    "flux3-klein": {
        "description": "FLUX.3 Image Klein int8 - 16 steps, prompt upsampler built-in, gated, needs HF_TOKEN",
        "family": "flux3",
        "mode": "distilled",
        "steps": 16,
        "cfg": 1.0,          # no CFG guidance; kept for API compatibility
        "unet": "flux3-image-klein-int8-convrot.safetensors",
        "clip": "qwen3vl_4b_int8_convrot.safetensors",
        "upsampler": "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
        "vae": "flux2-vae.safetensors",
        "gated": True,
        "files": [
            {
                "folder": "diffusion_models",
                "name": "flux3-image-klein-int8-convrot.safetensors",
                "url": f"{_HF_FLUX3_COMFY}/diffusion_models/flux3-image-klein-int8-convrot.safetensors",
            },
            {
                "folder": "text_encoders",
                "name": "qwen3vl_4b_int8_convrot.safetensors",
                "url": f"{_HF_FLUX3_COMFY}/text_encoders/qwen3vl_4b_int8_convrot.safetensors",
            },
            {
                "folder": "text_encoders",
                "name": "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
                "url": f"{_HF_FLUX3_COMFY}/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
            },
            {
                "folder": "vae",
                "name": "flux2-vae.safetensors",
                "url": f"{HF}/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
            },
        ],
    },
}

DEFAULT_PROFILE = "klein-4b"
