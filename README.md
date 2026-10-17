# FLUX.2 image generation (ComfyUI + Python) for Windows

Local text-to-image using the open-weight FLUX.2 family. `setup.ps1` installs ComfyUI, PyTorch (CUDA) and the model files. `generate.py` builds the workflow, sends it to ComfyUI's HTTP API and saves the PNGs.

## Requirements (Windows)
- NVIDIA GPU with a current driver (tested target: 16 GB VRAM)
- [Git](https://git-scm.com/download/win)
- [Python 3.13 (64-bit)](https://www.python.org/downloads/windows/), with "Add to PATH" ticked (3.10-3.12 also work)
- Disk space: about 15 GB for `klein-4b`, about 54 GB for `dev`

## Quick start
```powershell
git pull
.\setup.ps1                 # installs everything and downloads the klein-4b model
.\generate.ps1 "a lighthouse at dusk, cinematic"
```
If PowerShell blocks scripts: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`

Images are saved to `outputs\`. The first run is slow because the models load. Later runs are faster if you keep ComfyUI running in another terminal with `.\run_comfyui.ps1`, which also gives you the web UI at http://127.0.0.1:8188. `generate.py` reuses a running server and otherwise starts and stops its own.

## Model profiles
| Profile | What | Fit for 16 GB | License | Notes |
|---|---|---|---|---|
| `klein-4b` (default) | FLUX.2 klein 4B distilled, 4 steps | Easily | Apache 2.0 | Fast, no login needed |
| `klein-4b-base` | klein 4B base, 20 steps, negative prompt | Easily | Apache 2.0 | More variety, slower |
| `klein-9b` | klein 9B fp8 distilled | Yes | Non-commercial | Gated: needs `HF_TOKEN` |
| `dev` | FLUX.2 dev fp8 (32B) | Only with CPU offload, about 64 GB system RAM | Non-commercial | About 54 GB download, slow |

Install a different profile:
```powershell
.\setup.ps1 -Profile klein-9b          # or run just the download:
$env:HF_TOKEN = "hf_..."               # needed only for klein-9b
.\.venv\Scripts\python download_models.py --profile klein-9b
```
For `klein-9b`, first accept the terms at https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8 and create a read token at https://huggingface.co/settings/tokens.

Downloads resume if interrupted. Re-run the same command.

## Generating
```powershell
.\generate.ps1 "a red fox in snow" --count 4 --width 1024 --height 768
.\generate.ps1 "a castle on a cliff" --profile klein-4b-base --negative "blurry, low quality" --steps 30
.\generate.ps1 "product photo of a watch" --seed 1234
.\generate.ps1 "..." --profile dev          # after installing the dev profile
```
Options: `--profile --width --height --steps --cfg --seed --count --negative --host`. Width and height must be multiples of 16.

## Using it from your own Python code
```python
from generate import build_workflow, Comfy
from profiles import PROFILES

p = PROFILES["klein-4b"]
wf = build_workflow(p, "a cat astronaut", "", 1024, 1024, seed=1, steps=p["steps"], cfg=p["cfg"], prefix="demo")
png_bytes_list = Comfy("127.0.0.1:8188").run(wf)   # ComfyUI must be running
```

## Troubleshooting
- **`torch.cuda.is_available()` is False.** Update your NVIDIA driver, or re-run with `.\setup.ps1 -CudaTag cu128` (or `cu126` for older drivers).
- **Out of memory.** Use a smaller profile or resolution. `run_comfyui.ps1 --lowvram` also helps.
- **"Missing model ..." error.** Run `download_models.py --profile <name>`.
- **Server errors.** See `logs\comfyui.log`.

## Layout
```
setup.ps1           one-shot installer
generate.ps1/.py    CLI image generation
run_comfyui.ps1     start ComfyUI server/web UI
download_models.py  resumable model downloader
profiles.py         model files and settings per profile
ComfyUI/ .venv/ outputs/ logs/    created locally, git-ignored
```
