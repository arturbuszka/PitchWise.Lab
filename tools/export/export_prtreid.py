"""Export prtreid-soccernet-baseline to ONNX for PrtReidOnnx.

Run in the export venv (pip install git+https://github.com/VlSomers/prtreid.git,
matplotlib<3.9):
    python export_prtreid.py [weights.pth.tar | download_dir]

The exported graph takes a normalized RGB batch [N, 3, 256, 128] and exposes the
outputs consumed by the C# module:
  - embeddings        [N, D]   global ("globl") test embedding
  - visibility_scores [N, K]   part visibility scores
  - role_scores       [N, 5]   role scores (ball, goalkeeper, other, player, referee)
"""
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

# The baseline checkpoint stores its yacs config inside the pickle, which the
# torch>=2.6 weights_only default refuses to load. The file comes from the official
# sn-gamestate Zenodo record, so allow full unpickling here.
_orig_torch_load = torch.load
def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _load

from prtreid.scripts.default_config import get_default_config
from prtreid.scripts.main import build_config
from prtreid.tools.feature_extractor import FeatureExtractor
from prtreid.utils.tools import extract_test_embeddings

WEIGHTS_URL = "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"


def fetch_weights(target_dir):
    path = Path(target_dir) / "prtreid-soccernet-baseline.pth.tar"
    if not path.exists():
        print("downloading prtreid-soccernet-baseline.pth.tar ...", flush=True)
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    return str(path)


class PrtReidExportWrapper(torch.nn.Module):
    """Flattens the 8-tuple model output to the three tensors PrtReidOnnx expects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        embeddings, visibility, _, _, role_scores = extract_test_embeddings(out, ["globl"])
        return embeddings[:, 0, :], visibility, role_scores["globl"]


def main(weights_path: str, out_dir: Path):
    base = get_default_config()
    base.model.load_weights = weights_path
    base.model.load_config = True   # restore the model architecture from the checkpoint
    base.model.pretrained = False
    base.use_gpu = False
    base.data.save_dir = tempfile.mkdtemp()
    cfg = build_config(config=base)
    # the binary visibility path does ReduceMax on a bool tensor, which ONNX forbids;
    # keep the soft (float) visibility scores instead
    cfg.model.bpbreid.testing_binary_visibility_score = False

    extractor = FeatureExtractor(
        cfg, model_path=weights_path, device="cpu",
        image_size=(cfg.data.height, cfg.data.width), verbose=False)
    wrapper = PrtReidExportWrapper(extractor.model).eval()

    dummy = torch.randn(1, 3, cfg.data.height, cfg.data.width)
    out_path = out_dir / "prtreid-soccernet-baseline.onnx"
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["images"],
        output_names=["embeddings", "visibility_scores", "role_scores"],
        dynamic_axes={name: {0: "batch"} for name in
                      ["images", "embeddings", "visibility_scores", "role_scores"]},
        opset_version=17,
        # the dynamo exporter emits an invalid ReduceMax(bool) for the visibility head
        dynamo=False,
    )
    print(f"exported {out_path} (input {cfg.data.height}x{cfg.data.width})", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "pretrained_models"
    weights = arg if arg.endswith(".tar") else fetch_weights(arg)
    out = Path(arg).parent if arg.endswith(".tar") else Path(arg)
    main(weights, out)
