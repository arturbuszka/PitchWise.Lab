"""Export prtreid-soccernet-baseline to ONNX for PrtReidOnnx.

Run inside the sn-gamestate Python environment (uv sync in reference/sn-gamestate),
with the baseline weights downloaded (they auto-download on first pipeline run):
    python export_prtreid.py path/to/prtreid-soccernet-baseline.pth.tar

The exported graph takes a normalized RGB batch [N, 3, 256, 128] and must expose
three named outputs consumed by the C# module:
  - embeddings        [N, D]   global ("globl") test embedding
  - visibility_scores [N, K]   part visibility scores
  - role_scores       [N, 5]   role logits (ball, goalkeeper, other, player, referee)
"""
import sys

import torch
from prtreid.scripts.main import build_config
from prtreid.tools.feature_extractor import FeatureExtractor


class PrtReidExportWrapper(torch.nn.Module):
    """Flattens the prtreid output dict to the three tensors PrtReidOnnx expects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        # prtreid returns (embeddings_dict, visibility_dict, masks, pixels, role_scores_dict)
        embeddings, visibility, _, _, role_scores = out
        return (
            embeddings["globl"],
            visibility["globl"],
            role_scores["globl"],
        )


def main(weights_path: str):
    cfg = build_config()  # default prtreid soccernet config
    cfg.model.load_weights = weights_path
    cfg.use_gpu = False
    extractor = FeatureExtractor(
        cfg, model_path=weights_path, device="cpu", image_size=(256, 128), verbose=False
    )
    wrapper = PrtReidExportWrapper(extractor.model).eval()

    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        wrapper,
        dummy,
        "prtreid-soccernet-baseline.onnx",
        input_names=["images"],
        output_names=["embeddings", "visibility_scores", "role_scores"],
        dynamic_axes={name: {0: "batch"} for name in
                      ["images", "embeddings", "visibility_scores", "role_scores"]},
        opset_version=17,
    )
    print("exported prtreid-soccernet-baseline.onnx")


if __name__ == "__main__":
    main(sys.argv[1])
