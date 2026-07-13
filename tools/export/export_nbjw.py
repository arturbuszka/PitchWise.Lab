"""Export the two NBJW ("No Bells, Just Whistles") HRNet calibration models to ONNX.

Run inside the sn-gamestate Python environment (checkpoints auto-download on first
pipeline run, or from https://zenodo.org/records/12626395):
    python export_nbjw.py path/to/SV_kp path/to/SV_lines

Outputs consumed by NbjwCalibKeypoints (input: RGB [N, 3, 540, 960] in [0, 1]):
  - nbjw_kp.onnx    -> heatmaps [N, 58, 270, 480] (57 keypoints + background)
  - nbjw_lines.onnx -> heatmaps [N, 24, 270, 480] (23 lines + background)
"""
import sys

import torch
import yaml
from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

# same configs the sn-gamestate pipeline passes to the models
CFG_KP = "config/hrnetv2_w48.yaml"
CFG_LINES = "config/hrnetv2_w48_l.yaml"


def export(model, name):
    model.eval()
    dummy = torch.randn(1, 3, 540, 960)
    torch.onnx.export(
        model, dummy, name,
        input_names=["images"], output_names=["heatmaps"],
        dynamic_axes={"images": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {name}")


def main(kp_checkpoint: str, lines_checkpoint: str):
    cfg = yaml.safe_load(open(CFG_KP))
    cfg_l = yaml.safe_load(open(CFG_LINES))

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(kp_checkpoint, map_location="cpu"))
    export(model, "nbjw_kp.onnx")

    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(lines_checkpoint, map_location="cpu"))
    export(model_l, "nbjw_lines.onnx")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
