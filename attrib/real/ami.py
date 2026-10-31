"""Benchmark on real meetings: AMI single distant microphone (CC-BY-4.0).

Each AMI test series (EN2002, ES2004, IS1009, TS3003) has the same four
people in four meetings, so every speaker can be enrolled from meetings a and
c and recognized in excerpts of meetings b and d. All 16 speakers are in the
gallery, and simulated calendar hints list the right four 80% of the time.
There is no wearer, so the wearer bonus is off.

    python -m attrib.real.ami --parquet-dir data/ami --diarizer chunk
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np

from ..audio import SR, resample
from ..chunkdiar import KNOWN_META, ChunkDiarizer
from ..diarize import PyannoteDiarizer
from ..embed import Embedder
from ..identify import Gallery, Identifier, IdentifyConfig, final_labels
from ..metrics import der, identification_error
from ..sim.bench import EMBEDDINGS, PIPELINES, SEGMENTATION, _cached
from ..types import Turn

TEST = ["b", "d"]
ENROLL = ["a", "c"]


@dataclass
class Meeting:
    name: str
    audio: np.ndarray
    turns: list[Turn]

    @property
    def speakers(self):
        return sorted({t.speaker for t in self.turns})


def load_ami(parquet_dir: str) -> dict[str, Meeting]:
    import pyarrow.parquet as pq
    import soundfile as sf

    out = {}
    for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            name = r["audio"]["path"].split(".")[0]
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            turns = [Turn(a, b, s) for a, b, s in zip(r["timestamps_start"], r["timestamps_end"], r["speakers"])]
            out[name] = Meeting(name, resample(x if x.ndim == 1 else x.mean(1), sr), turns)
    return out


def excerpt(m: Meeting, start: float, dur: float) -> Meeting:
    end = start + dur
    turns = [Turn(max(t.start, start) - start, min(t.end, end) - start, t.speaker)
             for t in m.turns if t.end > start and t.start < end]
    return Meeting(f"{m.name}@{int(start)}", m.audio[int(start * SR): int(end * SR)], turns)


def clean_segments(m: Meeting, min_s: float = 2.0, max_s: float = 8.0):
    """(speaker, audio) for single-speaker stretches with nobody else talking."""
    out = []
    for t in m.turns:
        if t.duration < min_s:
            continue
        if any(o is not t and o.start < t.end and o.end > t.start for o in m.turns):
            continue
        a = t.start + 0.1
        out.append((t.speaker, m.audio[int(a * SR): int(min(t.end - 0.1, a + max_s) * SR)]))
    return out


def enrollment(meetings: dict[str, Meeting], seconds: float, seed: int = 0):
    rng = random.Random(seed)
    pool: dict[str, list[np.ndarray]] = {}
    for name, m in meetings.items():
        if name[-1] in ENROLL:
            for spk, x in clean_segments(m):
                pool.setdefault(spk, []).append(x)
    out = {}
    for spk, clips in pool.items():
        rng.shuffle(clips)
        take, tot = [], 0.0
        for c in clips:
            if tot >= seconds:
                break
            take.append(c)
            tot += len(c) / SR
        out[spk] = take
    return out


def cohort(libri_parquet: str, n: int = 200, seed: int = 0):
    """Real non-target voices (LibriSpeech dev-clean) put in simulated far-field rooms."""
    import pyarrow.parquet as pq
    import soundfile as sf

    from ..sim.pendant import apply, other_acoustics

    rng = np.random.default_rng(seed)
    rows = pq.read_table(libri_parquet, columns=["audio"]).to_pylist()
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    out = []
    for k, i in enumerate(idx):
        x, sr = sf.read(io.BytesIO(rows[i]["audio"]["bytes"]), dtype="float32")
        out.append(apply(resample(x, sr), other_acoustics(rng, "hard" if k % 2 else "moderate")))
    return out


def calendar_priors(people, cast, rng, accuracy=0.8):
    pri = {}
    for p in people:
        present = p in cast
        listed = present if rng.random() < accuracy else not present
        pri[p] = 0.8 if listed else 0.3
    return pri


def run(args) -> dict:
    meetings = load_ami(args.parquet_dir)
    tests = [excerpt(m, args.start, args.minutes * 60) for n, m in sorted(meetings.items()) if n[-1] in TEST]
    emb = Embedder(EMBEDDINGS[args.embedding])
    enroll = _cached(f"ami_enroll_{args.enroll_seconds}", lambda: enrollment(meetings, args.enroll_seconds))
    gal = _cached(f"ami_gallery_{args.embedding}_{args.enroll_seconds}",
                  lambda: {p: [emb(c) for c in clips] for p, clips in enroll.items()})
    g = Gallery(max_per_person=args.gallery_size)
    for p, embs in gal.items():
        for e in embs:
            g.add(p, e)
    coh = _cached(f"ami_cohort_{args.embedding}", lambda: np.stack([emb(c) for c in cohort(args.libri)]))
    ident = Identifier(g, coh, IdentifyConfig(threshold=args.threshold, wearer=None, wearer_weight=0.0))

    if args.diarizer == "chunk":
        d = ChunkDiarizer(SEGMENTATION[args.segmentation], emb, threshold=args.cluster_threshold,
                          seg_meta=KNOWN_META.get(args.segmentation), speech_bias=args.speech_bias)
        tag = f"{args.segmentation}_t{args.cluster_threshold}_sb{args.speech_bias}"
    else:
        d = PyannoteDiarizer(PIPELINES[args.diarizer], emb)
        tag = args.diarizer
    rows = []
    t0 = time.time()
    for i, m in enumerate(tests):
        key = f"ami_diar_{m.name}_{args.minutes}_{args.embedding}_{tag}"
        raw, clusters = _cached(key, lambda m=m: (lambda t: (t, d.clusters(m.audio, t)))(d.turns(m.audio)))
        pri = calendar_priors(list(g.people), m.speakers, random.Random(i)) if args.priors else None
        matches = ident.identify(clusters, pri)
        names = final_labels(clusters, matches, 0.6)
        turns = [Turn(t.start, t.end, names[t.speaker], matches[t.speaker].confidence, t.speaker) for t in raw]
        dr = der(m.turns, raw)
        rows.append({"meeting": m.name, "der": dr["total"], "miss": dr["miss"], "fa": dr["fa"],
                     "confusion": dr["confusion"], "ider": identification_error(m.turns, turns)["total"],
                     "n_ref": len(m.speakers), "n_clusters": len(clusters)})
    s = {k: float(np.mean([r[k] for r in rows])) for k in ["der", "miss", "fa", "confusion", "ider", "n_clusters"]}
    s["audio_minutes"] = sum(len(m.audio) for m in tests) / SR / 60
    s["diarize_seconds"] = time.time() - t0
    return {"config": vars(args), "summary": s, "per_meeting": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="data/ami")
    ap.add_argument("--libri", default="data/librispeech/validation.clean.parquet")
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--embedding", choices=list(EMBEDDINGS), default="titanet")
    ap.add_argument("--segmentation", choices=list(SEGMENTATION), default="pyannote3")
    ap.add_argument("--diarizer", choices=["chunk", *PIPELINES], default="chunk")
    ap.add_argument("--cluster-threshold", type=float, default=0.5)
    ap.add_argument("--speech-bias", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=2.5)
    ap.add_argument("--gallery-size", type=int, default=20)
    ap.add_argument("--enroll-seconds", type=float, default=45)
    ap.add_argument("--priors", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = run(args)
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
