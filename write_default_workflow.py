"""Write the default profile's workflow to ComfyUI so it loads automatically in the web UI.

ComfyUI looks for the default graph in two places (depending on version):
  - ComfyUI/web/assets/defaultGraph.json          (older builds)
  - ComfyUI/user/default/workflows/default.json   (newer builds)

Both are written so whichever version is installed picks it up.

Usage:
    python write_default_workflow.py                  # uses DEFAULT_PROFILE
    python write_default_workflow.py --profile dev
"""
import argparse
import json
import random
from pathlib import Path

from generate import build_workflow
from profiles import DEFAULT_PROFILE, PROFILES

ROOT = Path(__file__).resolve().parent
COMFY_DIR = ROOT / "ComfyUI"

# ComfyUI wraps the API-format prompt in a lightweight graph envelope so the
# web UI can display it.  The nodes list is empty — ComfyUI rebuilds the visual
# graph from the prompt dict on load.
GRAPH_TEMPLATE = {
    "last_node_id": 99,
    "last_link_id": 99,
    "nodes": [],
    "links": [],
    "groups": [],
    "config": {},
    "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}},
    "version": 0.4,
}


def make_graph(profile_name: str) -> dict:
    profile = PROFILES[profile_name]
    wf = build_workflow(
        profile,
        prompt="",          # user will fill this in the UI
        negative="",
        width=1024,
        height=1024,
        seed=random.randint(0, 2 ** 48),
        steps=profile["steps"],
        cfg=profile["cfg"],
        prefix=f"flux2_{profile_name}",
    )
    graph = dict(GRAPH_TEMPLATE)
    graph["workflow"] = wf
    return graph


def write(profile_name: str) -> None:
    if not COMFY_DIR.is_dir():
        print("ComfyUI directory not found — run setup.ps1 first.")
        return

    graph = make_graph(profile_name)
    data = json.dumps(graph, indent=2)

    targets = [
        COMFY_DIR / "web" / "assets" / "defaultGraph.json",
        COMFY_DIR / "user" / "default" / "workflows" / "default.json",
    ]
    for path in targets:
        if path.parent.exists():
            path.write_text(data, encoding="utf-8")
            print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=PROFILES)
    args = ap.parse_args()
    print(f"Writing default workflow for profile '{args.profile}' ...")
    write(args.profile)
    print("Done. Open http://127.0.0.1:8188 — the workflow will load automatically.")
