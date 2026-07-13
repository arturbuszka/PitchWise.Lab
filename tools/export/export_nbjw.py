"""Export the two NBJW ("No Bells, Just Whistles") HRNet calibration models to ONNX.

Self-contained: downloads the SV_kp / SV_lines checkpoints from Zenodo when missing
and imports the model code from the cloned reference repo (plugins/calibration).

    python export_nbjw.py [output_dir]

Outputs consumed by NbjwCalibKeypoints (input: RGB [N, 3, 540, 960] in [0, 1]):
  - nbjw_kp.onnx    -> heatmaps [N, 58, 270, 480] (57 keypoints + background)
  - nbjw_lines.onnx -> heatmaps [N, 24, 270, 480] (23 lines + background)
"""
import sys
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reference" / "sn-gamestate" / "plugins" / "calibration"))

from nbjw_calib.model.cls_hrnet import get_cls_net  # noqa: E402
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l  # noqa: E402

CHECKPOINTS = {
    "SV_kp": "https://zenodo.org/records/12626395/files/SV_kp?download=1",
    "SV_lines": "https://zenodo.org/records/12626395/files/SV_lines?download=1",
}

# HRNet-w48 config from sn_gamestate/configs/modules/pitch/nbjw_calib.yaml
def hrnet_cfg(num_joints):
    return {"MODEL": {
        "IMAGE_SIZE": [960, 540], "NUM_JOINTS": num_joints, "PRETRAIN": "",
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                       "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192], "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [4, 4, 4, 4], "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
        },
    }}


def fetch(name, target_dir):
    path = target_dir / name
    if not path.exists():
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(CHECKPOINTS[name], path)
    return path


def load_state(checkpoint):
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:  # older checkpoints may contain non-tensor objects
        print(f"weights_only load failed ({e}); retrying with weights_only=False", flush=True)
        return torch.load(checkpoint, map_location="cpu", weights_only=False)


def export(model, checkpoint, out_path):
    model.load_state_dict(load_state(checkpoint))
    model.eval()
    dummy = torch.randn(1, 3, 540, 960)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["images"], output_names=["heatmaps"],
        dynamic_axes={"images": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out_path}", flush=True)


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export(get_cls_net(hrnet_cfg(58)), fetch("SV_kp", out_dir), out_dir / "nbjw_kp.onnx")
    export(get_cls_net_l(hrnet_cfg(24)), fetch("SV_lines", out_dir), out_dir / "nbjw_lines.onnx")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else REPO / "pretrained_models")
