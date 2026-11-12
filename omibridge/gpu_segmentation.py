"""Optional: run DiariZen's speaker segmentation on the Mac's GPU (PyTorch MPS).

Off by default. With OMIBRIDGE_DIARIZER_DEVICE=mps the bridge loads DiariZen's
original PyTorch checkpoint (the one the ONNX file was exported from) and runs
the segmentation windows on the GPU; everything else in diarization stays as it
is. Measured on the Mac mini's M1 (2026-11-12): about 2x faster per window, the
same decisions as the ONNX model (100% of frames on two conversations,
99.999% on a third; log-prob differences below 0.004).

Fails safe: if PyTorch has no MPS device, the optional packages are missing
(requirements-gpu.txt), the checkpoint cannot be loaded, or the GPU errors
later, the bridge uses the ONNX model on the CPU as before and logs why.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("omibridge.gpu")

# The checkpoint the bridge's ONNX file was exported from, pinned to the revision tested.
DIARIZEN_REPO = "BUT-FIT/diarizen-wavlm-large-s80-md"
DIARIZEN_REVISION = "6285693ddd5b38e8229acb93f864f3d04a82bee1"


def load_diarizen_model(checkpoint: str | None = None):
    """DiariZen's WavLM-Conformer segmentation model, in eval mode, on the CPU.
    Raises when the optional packages or the checkpoint are not available."""
    import torch
    import toml
    from diarizen.models.eend.model_wavlm_conformer import Model

    if checkpoint and Path(checkpoint).is_dir():
        model_dir = Path(checkpoint)
    else:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(repo_id=checkpoint or DIARIZEN_REPO,
                                           revision=None if checkpoint else DIARIZEN_REVISION))
    args = dict(toml.load(model_dir / "config.toml").get("model", {}).get("args", {}))
    args.setdefault("max_speakers_per_chunk", 4)
    args.setdefault("max_speakers_per_frame", 2)
    accepted = inspect.signature(Model.__init__).parameters
    model = Model(**{k: v for k, v in args.items() if k in accepted})
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    return model.eval()


class MpsSegmentation:
    """Stands in for the ONNX SegmentationModel: same window, classes and binarization
    (delegated to it), but scores come from the PyTorch model on the GPU. Falls back to
    the ONNX model for good if the GPU ever fails."""

    def __init__(self, onnx_seg, model, device: str = "mps", batch: int = 4):
        import torch

        self._onnx = onnx_seg
        self._torch = torch
        self.model = model.to(device)
        self.device = device
        self.batch = max(1, batch)
        self.failed = False

    def __getattr__(self, name):            # window, num_classes, mapping, binarize, ...
        return getattr(self._onnx, name)

    @property
    def device_name(self) -> str:
        return "cpu (GPU failed)" if self.failed else self.device

    def __call__(self, chunks: np.ndarray, speech_bias: float = 0.0) -> np.ndarray:
        return self.binarize(self.scores(chunks), speech_bias)

    def scores(self, chunks: np.ndarray) -> np.ndarray:
        if self.failed:
            return self._onnx.scores(chunks)
        torch = self._torch
        try:
            out = []
            with torch.inference_mode():
                for i in range(0, len(chunks), self.batch):
                    x = torch.from_numpy(np.ascontiguousarray(chunks[i: i + self.batch], dtype=np.float32))
                    y = self.model(x[:, None, :].to(self.device))
                    out.append(y.float().cpu().numpy())
            return np.concatenate(out)
        except Exception:
            log.exception("GPU segmentation failed; using the ONNX model on the CPU from now on")
            self.failed = True
            return self._onnx.scores(chunks)

    def release(self) -> None:
        """Hand cached GPU memory back after a conversation."""
        try:
            if self.device == "mps":
                self._torch.mps.empty_cache()
        except Exception:
            pass


def maybe_gpu(onnx_seg, device: str, batch: int = 4, checkpoint: str | None = None, loader=load_diarizen_model):
    """`onnx_seg` unchanged unless `device` is "mps" and everything needed is there."""
    if device != "mps":
        return onnx_seg
    try:
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch reports no MPS device")
        seg = MpsSegmentation(onnx_seg, loader(checkpoint), "mps", batch)
        log.info("speaker segmentation runs on the GPU (MPS), batch %d", seg.batch)
        return seg
    except Exception as e:
        log.warning("OMIBRIDGE_DIARIZER_DEVICE=mps, but the GPU path is unavailable (%s: %s); using the CPU",
                    type(e).__name__, e)
        return onnx_seg
