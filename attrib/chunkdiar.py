"""Diarization with chunk-level cannot-link constraints.

Same recipe as pyannote 3.x (10 s segmentation windows, one voiceprint per
local speaker per window, clustering, frame-level reconstruction), with one
change that matters for pendant audio: two local speakers found in the same
window are different people by construction, so each window's speakers are
assigned to distinct clusters (Hungarian assignment). Distant voices that
sound alike are then kept apart whenever they ever talk in the same window.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

from .audio import SR
from .diarize import Diarizer
from .embed import Embedder, l2norm
from .types import Turn


# Models exported without pyannote's metadata (window in samples).
KNOWN_META = {
    # DiariZen WavLM-large-s80-md: 16 s windows, 4 speakers, at most 2 at once.
    # The export's batch size is fixed at 1 (batch 2 fails inside the graph).
    "diarizen": {"window_size": 16 * SR, "num_speakers": 4, "num_classes": 11, "powerset_max_classes": 2,
                 "batch": 1},
}


class SegmentationModel:
    def __init__(self, path: str, threads: int = 4, meta: dict | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        meta = meta or self.sess.get_modelmeta().custom_metadata_map
        self.window = int(meta["window_size"])
        self.num_speakers = int(meta["num_speakers"])
        self.num_classes = int(meta["num_classes"])
        self.mapping = _powerset(self.num_classes, self.num_speakers, int(meta["powerset_max_classes"]))
        self.rank = len(self.sess.get_inputs()[0].shape)
        self.max_batch = int(meta.get("batch", 0))

    def __call__(self, chunks: np.ndarray, speech_bias: float = 0.0) -> np.ndarray:
        """(N, window) audio -> (N, frames, speakers) binary activity."""
        return self.binarize(self.scores(chunks), speech_bias)

    def scores(self, chunks: np.ndarray) -> np.ndarray:
        """(N, window) audio -> (N, frames, powerset classes) scores."""
        if self.max_batch:
            return np.concatenate([self._run(chunks[i: i + self.max_batch])
                                   for i in range(0, len(chunks), self.max_batch)])
        return self._run(chunks)

    def binarize(self, y: np.ndarray, speech_bias: float = 0.0) -> np.ndarray:
        """speech_bias > 0 lowers the bar for calling a frame speech (class 0 is silence)."""
        if speech_bias:
            y = y.copy()
            y[..., 0] -= speech_bias
        return self.mapping[y.argmax(-1)]

    def _run(self, chunks: np.ndarray) -> np.ndarray:
        x = chunks[:, None, :] if self.rank == 3 else chunks
        (y,) = self.sess.run(None, {self.sess.get_inputs()[0].name: x.astype(np.float32)})
        return y


class TorchSegmentationModel(SegmentationModel):
    """A pyannote.audio segmentation checkpoint directory (e.g. community-1's)."""

    def __init__(self, path: str, threads: int = 4, meta: dict | None = None):
        import torch
        from pyannote.audio import Model

        torch.set_num_threads(threads)
        self.model = Model.from_pretrained(path).eval()
        spec = self.model.specifications
        self.window = int(spec.duration * SR)
        self.num_speakers = len(spec.classes)
        self.num_classes = self.model.dimension
        self.mapping = _powerset(self.num_classes, self.num_speakers, spec.powerset_max_classes)
        self.max_batch = 0

    def _run(self, chunks: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.model(torch.from_numpy(chunks[:, None, :].astype(np.float32))).numpy()


def _powerset(num_classes, num_speakers, max_classes):
    m = np.zeros((num_classes, num_speakers))
    k = 1
    for j in range(num_speakers):
        m[k, j] = 1
        k += 1
    if max_classes >= 2:
        for j in range(num_speakers):
            for i in range(j + 1, num_speakers):
                m[k, [j, i]] = 1
                k += 1
    return m


class ChunkDiarizer(Diarizer):
    def __init__(self, segmentation_model: str, embedder: Embedder, threshold: float = 0.6,
                 hop: float = 2.0, min_active: float = 0.5, min_duration_on: float = 0.3,
                 min_duration_off: float = 0.3, batch: int = 32, seg_meta: dict | None = None,
                 speech_bias: float = 0.0):
        cls = TorchSegmentationModel if os.path.isdir(segmentation_model) else SegmentationModel
        self.seg = cls(segmentation_model, meta=seg_meta)
        self.embedder = embedder
        self.threshold = threshold        # cosine distance cut for clustering
        self.hop = int(hop * SR)
        self.min_active = min_active
        self.min_on = min_duration_on
        self.min_off = min_duration_off
        self.batch = batch
        self.speech_bias = speech_bias

    def _segment(self, x: np.ndarray):
        pad, scores = self._scores(x)
        return pad, self.seg.binarize(scores, self.speech_bias)  # acts: (n, F, S)

    def _scores(self, x: np.ndarray):
        W = self.seg.window
        n = max(1, int(np.ceil(max(len(x) - W, 0) / self.hop)) + 1)
        pad = np.zeros((n - 1) * self.hop + W, np.float32)
        pad[: len(x)] = x
        chunks = np.stack([pad[i * self.hop: i * self.hop + W] for i in range(n)])
        scores = np.concatenate([self.seg.scores(chunks[i: i + self.batch]) for i in range(0, n, self.batch)])
        return pad, scores

    def turns(self, x: np.ndarray) -> list[Turn]:
        pad, acts = self._segment(x)
        n, F, S = acts.shape
        W = self.seg.window
        fdur = W / F / SR
        # One voiceprint per (chunk, local speaker), from its non-overlapped frames.
        clean = acts * (acts.sum(-1, keepdims=True) < 2)
        keys, embs = [], []
        for i in range(n):
            for s in range(S):
                fr = clean[i, :, s].astype(bool)
                if fr.sum() * fdur < self.min_active:
                    continue
                idx = np.repeat(fr, int(np.ceil(W / F)))[:W]
                seg = pad[i * self.hop: i * self.hop + W][idx[:W]]
                keys.append((i, s))
                embs.append(self.embedder(seg))
        if not embs:
            return []
        E = np.stack(embs)
        if len(E) == 1:
            labels = np.zeros(1, int)
        else:
            Z = linkage(E, method="average", metric="cosine")
            labels = fcluster(Z, t=self.threshold, criterion="distance") - 1
        cents = l2norm(np.stack([E[labels == k].mean(0) for k in range(labels.max() + 1)]))
        # Constrained assignment: distinct clusters for one chunk's local speakers.
        assign: dict[tuple[int, int], int] = {}
        by_chunk: dict[int, list[int]] = {}
        for r, (i, s) in enumerate(keys):
            by_chunk.setdefault(i, []).append(r)
        for i, rows in by_chunk.items():
            sim = E[rows] @ cents.T
            if len(rows) <= len(cents):
                rr, cc = linear_sum_assignment(-sim)
                for a, b in zip(rr, cc):
                    assign[keys[rows[a]]] = int(b)
            else:
                for a, r in enumerate(rows):
                    assign[keys[r]] = int(sim[a].argmax())
        # Frame-level reconstruction: average votes per cluster over chunks,
        # keep the top-k clusters where k is the local speaker count.
        total = (n - 1) * self.hop // (W // F) + F + 1
        step = self.hop / (W / F)
        K = len(cents)
        votes = np.zeros((total, K))
        cover = np.zeros(total)
        count = np.zeros(total)
        for i in range(n):
            o = int(round(i * step))
            cover[o: o + F] += 1
            count[o: o + F] += acts[i].sum(-1)
            for s in range(S):
                k = assign.get((i, s))
                if k is not None:
                    votes[o: o + F, k] += acts[i, :, s]
        cover = np.maximum(cover, 1)
        votes /= cover[:, None]
        spk = np.rint(count / cover).astype(int)
        active = np.zeros_like(votes, bool)
        order = np.argsort(-votes, axis=1)
        for f in range(total):
            for j in range(min(spk[f], K)):
                if votes[f, order[f, j]] > 0:
                    active[f, order[f, j]] = True
        dur = len(x) / SR
        out = []
        for k in range(K):
            out += _runs(active[:, k], fdur * step / step, f"S{k}", self.min_on, self.min_off, dur,
                         frame=W / F / SR)
        return sorted(out, key=lambda t: t.start)


def _runs(mask, _unused, label, min_on, min_off, dur, frame):
    spans = []
    f = 0
    n = len(mask)
    while f < n:
        if mask[f]:
            g = f
            while g < n and mask[g]:
                g += 1
            spans.append([f * frame, g * frame])
            f = g
        else:
            f += 1
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < min_off:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [Turn(a, min(b, dur), label) for a, b in merged if b - a >= min_on and a < dur]
