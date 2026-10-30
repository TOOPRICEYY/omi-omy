"""Anonymous diarization ("who spoke when") plus per-speaker voiceprints."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .audio import SR, rms_db
from .embed import Embedder, l2norm
from .types import Turn


@dataclass
class Cluster:
    label: str
    turns: list[Turn]
    centroid: np.ndarray          # voiceprint of the speaker's pooled audio
    segment_embs: list[np.ndarray]
    loudness_db: float            # median turn loudness
    speech_time: float


class Diarizer:
    def __init__(self, segmentation_model: str, embedder: Embedder, threshold: float = 0.5,
                 min_duration_on: float = 0.3, min_duration_off: float = 0.5):
        import sherpa_onnx as so

        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation_model)),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=embedder.model_path),
            clustering=so.FastClusteringConfig(threshold=threshold),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        if not cfg.validate():
            raise ValueError("bad diarization config")
        self.sd = so.OfflineSpeakerDiarization(cfg)
        self.embedder = embedder

    def turns(self, x: np.ndarray) -> list[Turn]:
        res = self.sd.process(x.astype(np.float32)).sort_by_start_time()
        return [Turn(r.start, r.end, f"S{r.speaker}") for r in res]

    def clusters(self, x: np.ndarray, turns: list[Turn], max_pool: float = 30.0,
                 min_seg: float = 0.8) -> list[Cluster]:
        by: dict[str, list[Turn]] = defaultdict(list)
        for t in turns:
            by[t.speaker].append(t)
        out = []
        for label, ts in by.items():
            clean = _non_overlapping(ts, turns)
            pieces = [x[int(a * SR): int(b * SR)] for a, b in clean if b - a >= 0.3]
            if not pieces:
                pieces = [x[int(t.start * SR): int(t.end * SR)] for t in ts]
            # Longest pieces first so the pooled voiceprint uses the best audio.
            pieces.sort(key=len, reverse=True)
            pooled, total = [], 0.0
            for p in pieces:
                pooled.append(p)
                total += len(p) / SR
                if total >= max_pool:
                    break
            centroid = self.embedder(np.concatenate(pooled))
            seg_embs = [self.embedder(p) for p in pieces[:12] if len(p) / SR >= min_seg]
            loud = float(np.median([rms_db(p) for p in pieces if len(p)]))
            out.append(Cluster(label, ts, centroid, seg_embs or [centroid], loud,
                               sum(t.duration for t in ts)))
        return out


def _non_overlapping(mine: list[Turn], all_turns: list[Turn]) -> list[tuple[float, float]]:
    """Parts of `mine` where nobody else is talking (cleaner voiceprints)."""
    others = [(t.start, t.end) for t in all_turns if t.speaker != mine[0].speaker]
    out = []
    for t in mine:
        spans = [(t.start, t.end)]
        for a, b in others:
            nxt = []
            for s, e in spans:
                if b <= s or a >= e:
                    nxt.append((s, e))
                else:
                    if a > s:
                        nxt.append((s, a))
                    if b < e:
                        nxt.append((b, e))
            spans = nxt
        out.extend(spans)
    return out


def overlap_ratio(turns: list[Turn]) -> float:
    """Share of speech time where two or more labels are active."""
    ev = sorted([(t.start, 1) for t in turns] + [(t.end, -1) for t in turns])
    active, last, speech, over = 0, 0.0, 0.0, 0.0
    for time, d in ev:
        if active >= 1:
            speech += time - last
        if active >= 2:
            over += time - last
        active += d
        last = time
    return over / speech if speech else 0.0


__all__ = ["Cluster", "Diarizer", "overlap_ratio", "l2norm"]


class PyannoteDiarizer(Diarizer):
    """A full pyannote.audio pipeline (e.g. community-1: VBx clustering + PLDA)
    for turns; our embedder still makes the voiceprints used for naming."""

    def __init__(self, pipeline_dir: str, embedder: Embedder, threads: int = 4):
        import torch
        from pyannote.audio import Pipeline

        torch.set_num_threads(threads)
        self.pipe = Pipeline.from_pretrained(pipeline_dir)
        self.embedder = embedder

    def turns(self, x: np.ndarray) -> list[Turn]:
        import torch

        out = self.pipe({"waveform": torch.from_numpy(x.astype(np.float32))[None], "sample_rate": SR})
        ann = getattr(out, "speaker_diarization", out)
        return [Turn(s.start, s.end, f"S{spk}") for s, _, spk in ann.itertracks(yield_label=True)]
