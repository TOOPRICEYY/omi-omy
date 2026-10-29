from __future__ import annotations

import numpy as np

from .audio import SR


def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Embedder:
    """Voiceprint model (any sherpa-onnx speaker embedding ONNX file)."""

    def __init__(self, model_path: str, num_threads: int = 2):
        import sherpa_onnx as so

        cfg = so.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=num_threads)
        if not cfg.validate():
            raise ValueError(f"bad embedding model: {model_path}")
        self.model_path = model_path
        self.extractor = so.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        s = self.extractor.create_stream()
        s.accept_waveform(sr, x.astype(np.float32))
        s.input_finished()
        return l2norm(np.array(self.extractor.compute(s), np.float32))
