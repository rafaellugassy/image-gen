# FLUX.2 image generation (ComfyUI + Python) for Windows

Local text-to-image using the open-weight FLUX.2 family. `setup.ps1` installs ComfyUI, PyTorch (CUDA for NVIDIA, ROCm for AMD) and the model files. `generate.py` builds the workflow, sends it to ComfyUI's HTTP API and saves the PNGs.

## Requirements (Windows)
- NVIDIA GPU with a current driver, or AMD GPU (RX 6000 series or newer) with a current Adrenalin driver on Windows 11. Target: 16 GB VRAM.
- [Git](https://git-scm.com/download/win)
- [Python 3.13 (64-bit)](https://www.python.org/downloads/windows/), with "Add to PATH" ticked. NVIDIA also works on 3.10-3.12, but **AMD requires 3.13**.
- Disk space: about 15 GB for `klein-4b`, about 54 GB for `dev`

## Quick start
```powershell
git pull
.\setup.ps1                 # installs everything and downloads the klein-4b model
.\generate.ps1 "a lighthouse at dusk, cinematic"
```
If PowerShell blocks scripts: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`

### AMD GPUs
`setup.ps1` detects your GPU and installs the matching PyTorch: CUDA for NVIDIA, ROCm for AMD. If PyTorch earlier reported that it can't find an NVIDIA driver, you have the CUDA build installed on an AMD machine. Re-run the setup. It removes the CUDA PyTorch and installs the ROCm build.
```powershell
.\setup.ps1 -Gpu amd                              # downloads kernels for all supported AMD GPUs
.\setup.ps1 -Gpu amd -AmdDevice device-gfx1100    # smaller download for your GPU, e.g. RX 7900 XT/XTX
```
Device targets: `device-gfx1201` (RX 9070), `device-gfx1200` (RX 9060), `device-gfx1100` (RX 7900), `device-gfx1101` (RX 7700/7800), `device-gfx1102` (RX 7600), `device-gfx1151` (Ryzen AI Max).
If the `.venv` was created with a Python other than 3.13, install 3.13, delete `.venv`, and re-run.

Images are saved to `outputs\`. The first run is slow because the models load. Later runs are faster if you keep ComfyUI running in another terminal with `.\run_comfyui.ps1`, which also gives you the web UI at http://127.0.0.1:8188. `generate.py` reuses a running server and otherwise starts and stops its own.

## Model profiles
| Profile | What | Fit for 16 GB | License | Notes |
|---|---|---|---|---|
| `klein-4b` (default) | FLUX.2 klein 4B distilled, 4 steps | Easily | Apache 2.0 | Fast, no login needed |
| `klein-4b-base` | klein 4B base, 20 steps, negative prompt | Easily | Apache 2.0 | More variety, slower |
| `klein-9b` | klein 9B fp8 distilled | Yes | Non-commercial | Gated: needs `HF_TOKEN` |
| `dev` | FLUX.2 dev fp8 (32B) | Only with CPU offload, about 64 GB system RAM | Non-commercial | About 54 GB download, slow |
| `flux3-klein` | **FLUX.3** Image Klein int8, 16 steps | Yes | FLUX Kommunity | Gated: needs `HF_TOKEN`; built-in prompt upsampler |

Install a different profile:
```powershell
.\.venv\Scripts\python download_models.py --profile klein-9b
```
For `klein-9b`, first accept the terms at https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8 and create a read token at https://huggingface.co/settings/tokens. Then store the token in a git-ignored `.env` file:
```powershell
copy .env.example .env
notepad .env          # replace hf_your_token_here with your token, save
```
`.env` is listed in `.gitignore`, so it is never committed or pushed. A real `HF_TOKEN` environment variable overrides it.

For `flux3-klein`, accept the terms at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy (same `HF_TOKEN` works), then:
```powershell
.\.venv\Scripts\python download_models.py --profile flux3-klein
```
FLUX.3 automatically expands short prompts through its built-in prompt upsampler before generating — you do not need to write long captions yourself. Requires ComfyUI **Nodes 2.0** (update ComfyUI first with `git pull` inside `ComfyUI\`).

Downloads resume if interrupted. Re-run the same command.

## Phone-friendly web page
A single simple page: type a prompt and tap Generate. If you add photos, they are used as reference images (edit/restyle); otherwise it is a normal text-to-image generation. Square/Portrait/Landscape and the model are the only options, and everything else uses the profile defaults.

On the Windows machine, one command:
```powershell
.\run_server.ps1       # web page on http://127.0.0.1:8080
```
If ComfyUI is already running (`.\run_comfyui.ps1`), it just uses it and never stops it. Otherwise it starts ComfyUI when a request comes in and stops it again once all pending requests have finished or failed. Overlapping requests share the one instance, so models load only once per burst.
For private access from your phone, install [Tailscale](https://tailscale.com/download) on the PC and the phone (same account), then on the PC:
```powershell
tailscale serve --bg 8080
```
Open the `https://<pc-name>.<tailnet>.ts.net` URL it prints. The server binds to localhost only and has no login of its own, so do not port-forward it or use `tailscale funnel`. Stop publishing with `tailscale serve reset`. Results are saved in `outputs\`.

## Generating
```powershell
.\generate.ps1 "a red fox in snow" --count 4 --width 1024 --height 768
.\generate.ps1 "a castle on a cliff" --profile klein-4b-base --negative "blurry, low quality" --steps 30
.\generate.ps1 "product photo of a watch" --seed 1234
.\generate.ps1 "..." --profile dev          # after installing the dev profile
.\generate.ps1 "a lighthouse at dusk" --profile flux3-klein   # FLUX.3 with prompt upsampler
```
Options: `--profile --image --width --height --steps --cfg --seed --count --negative --host`. Width and height must be multiples of 16.

### Using a sample (reference) image
FLUX.2 can edit or restyle an existing image. Pass `--image` and describe the change in the prompt:
```powershell
.\generate.ps1 "change the bag color to blue" --image bag.png
.\generate.ps1 "stylize the handbag in image 1 with the colours and logo from image 2" --image bag.png --image logo.png
```
- Repeat `--image` for several references and call them "image 1", "image 2" and so on in the prompt, in the order you passed them.
- Without `--width` or `--height`, the output matches the first image's shape at about 1 megapixel.
- Each reference image uses extra VRAM and time, so on 16 GB keep to 1-3 images. Results are best with the `klein-*` and `dev` profiles.
- The image is copied into `ComfyUI\input\`.

## Using it from your own Python code
```python
from generate import build_workflow, Comfy
from profiles import PROFILES

p = PROFILES["klein-4b"]
wf = build_workflow(p, "a cat astronaut", "", 1024, 1024, seed=1, steps=p["steps"], cfg=p["cfg"], prefix="demo")
png_bytes_list = Comfy("127.0.0.1:8188").run(wf)   # ComfyUI must be running

# with a reference image:
comfy = Comfy("127.0.0.1:8188")
name = comfy.upload("bag.png")
wf = build_workflow(p, "make the bag blue", "", None, None, seed=1, steps=p["steps"], cfg=p["cfg"],
                    prefix="edit", images=[name])
```

## Troubleshooting
- **`torch.cuda.is_available()` is False (NVIDIA).** Update your NVIDIA driver, or re-run with `.\setup.ps1 -CudaTag cu128` (or `cu126` for older drivers).
- **PyTorch can't find an NVIDIA driver (AMD).** You have the CUDA build. Re-run `.\setup.ps1 -Gpu amd`.
- **AMD: GPU not seen.** Update the Adrenalin driver and confirm you're on Windows 11 with an RDNA 2 or newer GPU (RX 6000 series or newer).
- **Out of memory.** Use a smaller profile or resolution. `run_comfyui.ps1 --lowvram` also helps.
- **"Missing model ..." error.** Run `download_models.py --profile <name>`.
- **Server errors.** See `logs\comfyui.log`.

## Layout
```
setup.ps1               one-shot installer
generate.ps1/.py        CLI image generation
run_comfyui.ps1         start ComfyUI server/web UI
download_models.py      resumable model downloader
profiles.py             model files and settings per profile
flux3_workflows/        official FLUX.3 ComfyUI workflow templates (JSON)
ComfyUI/ .venv/ outputs/ logs/    created locally, git-ignored
```
