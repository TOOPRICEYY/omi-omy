"""End-to-end run on the pyannote sample clip (two real speakers, 30 s)."""
import os

import numpy as np
import pytest

from attrib.audio import load
from attrib.chunkdiar import ChunkDiarizer
from attrib.embed import Embedder
from attrib.identify import Gallery, Identifier
from attrib.metrics import der
from attrib.pipeline import Pipeline
from attrib.types import Turn, Word
from attrib.vad import SileroVAD

SEG = "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB = "models/nemo_en_titanet_large.onnx"
needs_models = pytest.mark.skipif(not (os.path.exists(SEG) and os.path.exists(EMB)
                                       and os.path.exists("data/sample.wav")),
                                  reason="run scripts/fetch_models.sh first")


def _ref():
    out = []
    for line in open("data/sample.rttm"):
        f = line.split()
        out.append(Turn(float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return out


@needs_models
def test_sample_clip_end_to_end():
    x = load("data/sample.wav")
    ref = _ref()
    emb = Embedder(EMB)
    # Enroll speaker90 from its first long turn, then check the rest.
    t = max([r for r in ref if r.speaker == "speaker90"], key=lambda r: r.duration)
    g = Gallery()
    g.add("speaker90", emb(x[int(t.start * 16000): int(t.end * 16000)]))
    cohort = np.random.default_rng(0).standard_normal((50, len(g.matrix("speaker90")[0])))
    words = [Word("w", r.start + 0.1, r.start + 0.3) for r in ref]
    pipe = Pipeline(ChunkDiarizer(SEG, emb), Identifier(g, cohort))
    att = pipe.run(x, regions=SileroVAD()(x), words=words)
    anon = [Turn(t.start, t.end, t.cluster) for t in att.turns]
    assert der(ref, anon)["total"] < 0.35
    named = {w.speaker for w in att.words}
    assert "speaker90" in named
