"""Benchmark the pipeline on simulated pendant conversations.

    python -m attrib.sim.bench --conversations 12 --embedding wespeaker
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time

import numpy as np

from ..align import assign_words
from ..chunkdiar import ChunkDiarizer
from ..diarize import Diarizer
from ..embed import Embedder
from ..identify import Gallery, Identifier, IdentifyConfig, final_labels
from ..metrics import der, false_accept_rate, identification_error, word_speaker_error
from ..diarize import overlap_ratio
from ..pipeline import EscalationPolicy
from ..refine import RefineConfig, WindowRefiner
from ..types import Attribution, Turn
from .pendant import World, cohort_clips, enrollment_clips, make_conversation

SEGMENTATION = {
    "pyannote3": "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    # Rev's fine-tunes: non-production license (personal and research use only).
    "reverb1": "models/sherpa-onnx-reverb-diarization-v1/model.onnx",
    "reverb2": "models/sherpa-onnx-reverb-diarization-v2/model.int8.onnx",
}
SEG = SEGMENTATION["pyannote3"]
EMBEDDINGS = {
    "wespeaker": "models/wespeaker_en_voxceleb_resnet34_LM.onnx",
    "campplus": "models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "titanet": "models/nemo_en_titanet_large.onnx",
}
CACHE = "data/sim"


def _cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    v = fn()
    with open(p, "wb") as f:
        pickle.dump(v, f)
    return v


def scenario(world: World, i: int):
    """Mix of everyday situations: 1:1 with a contact, small groups, a stranger."""
    rng = random.Random(1000 + i)
    contacts = list(world.contacts)
    kind = i % 4
    if kind == 0:
        cast = ["wearer", rng.choice(contacts)]
    elif kind == 1:
        cast = ["wearer"] + rng.sample(contacts, 2)
    elif kind == 2:
        cast = ["wearer", rng.choice(contacts), rng.choice(list(world.strangers))]
    else:
        cast = ["wearer"] + rng.sample(contacts, 3)
    return cast


def build_data(seed: int, n_conv: int, enroll_s: float, wearer_enroll_s: float, profile: str = "hard"):
    world = World.make(seed)
    tag = "" if profile == "hard" else profile
    convs = [_cached(f"conv{seed}{tag}_{i}",
                     lambda i=i: make_conversation(world, scenario(world, i), seed * 7919 + i, profile=profile))
             for i in range(n_conv)]
    enroll = _cached(f"enroll{seed}_{enroll_s}_{wearer_enroll_s}", lambda: {
        name: enrollment_clips(world, name, wearer_enroll_s if name == "wearer" else enroll_s, seed + k)
        for k, name in enumerate(["wearer"] + list(world.contacts))})
    cohort = _cached(f"cohort{seed}_mixed", lambda: cohort_clips(world))
    return world, convs, enroll, cohort


def calendar_priors(world: World, cast: list[str], rng: random.Random, accuracy: float = 0.8):
    """Simulated calendar: lists the real attendees most of the time, plus noise."""
    pri = {}
    for p in world.contacts:
        present = p in cast
        listed = present if rng.random() < accuracy else not present
        pri[p] = 0.8 if listed else 0.3
    return pri


def diarized(args, convs, emb):
    """Diarization is the slow part, so cache it per embedding model and threshold."""
    if args.diarizer == "chunk":
        d = ChunkDiarizer(SEGMENTATION[args.segmentation], emb, threshold=args.cluster_threshold)
    else:
        d = Diarizer(SEGMENTATION[args.segmentation], emb, threshold=args.cluster_threshold)
    out = []
    seg = "" if args.segmentation == "pyannote3" else "_" + args.segmentation
    seg += "" if args.diarizer == "sherpa" else "_" + args.diarizer
    for i, c in enumerate(convs):
        tag = "" if args.profile == "hard" else args.profile
        key = f"diar{args.seed}{tag}_{i}_{args.embedding}_{args.cluster_threshold}{seg}"
        out.append(_cached(key, lambda c=c: (lambda t: (t, d.clusters(c.audio, t)))(d.turns(c.audio))))
    return out


def run(args) -> dict:
    world, convs, enroll, cohort = build_data(args.seed, args.conversations, args.enroll_seconds,
                                              args.wearer_enroll_seconds, args.profile)
    emb = Embedder(EMBEDDINGS[args.embedding])
    gallery = _cached(f"gallery{args.seed}_{args.embedding}_{args.enroll_seconds}_{args.wearer_enroll_seconds}",
                      lambda: {n: [emb(c) for c in clips] for n, clips in enroll.items()})
    g = Gallery(max_per_person=args.gallery_size)
    for name, embs in gallery.items():
        for e in embs:
            g.add(name, e)
    if args.mean_gallery:
        g = Gallery(1, {p: [np.mean(v, axis=0)] for p, v in g.people.items()})
    cohort_m = _cached(f"cohortemb{args.seed}_{args.embedding}_mixed", lambda: np.stack([emb(c) for c in cohort]))
    cfg = IdentifyConfig(top_k=args.top_k, use_snorm=not args.no_snorm, threshold=args.threshold,
                         wearer=None if args.no_wearer else "wearer",
                         wearer_weight=0.0 if args.no_wearer else args.wearer_weight)
    ident = Identifier(g, cohort_m, cfg)
    refiner = WindowRefiner(emb, ident, RefineConfig(switch_penalty=args.switch_penalty)) if args.refine else None
    t0 = time.time()
    diar = diarized(args, convs, emb)
    diar_s = time.time() - t0
    enrolled = set(g.people)
    rows = []
    for i, (c, (raw_turns, clusters)) in enumerate(zip(convs, diar)):
        priors = calendar_priors(world, c.speakers, random.Random(i)) if args.priors else None
        matches = ident.identify(clusters, priors)
        names = final_labels(clusters, matches, args.unknown_merge)
        turns = [Turn(t.start, t.end, names[t.speaker], matches[t.speaker].confidence, t.speaker)
                 for t in raw_turns]
        if refiner is not None:
            bonus = None
            if priors:
                bonus = {p: args.prior_weight_refine * (np.log(v / (1 - v))) for p, v in priors.items()}
            turns = refiner.refine(c.audio, turns, priors_bonus=bonus)
        att = Attribution(turns, [], {k.label: (matches[k.label].person, matches[k.label].confidence)
                                      for k in clusters})
        att.escalate_reasons = EscalationPolicy().check(att, overlap_ratio(raw_turns))
        hyp_words = assign_words(c.words, turns)
        rows.append({
            "der_raw": der(c.turns, raw_turns)["total"],
            "der": der(c.turns, turns)["total"],
            "ider": identification_error(c.turns, turns)["total"],
            "wder": word_speaker_error(c.words, hyp_words, enrolled),
            "far": false_accept_rate(c.turns, turns, enrolled),
            "escalate": bool(att.escalate_reasons),
            "n_ref": len(set(c.speakers)),
            "n_hyp": len(set(names.values())),
            "n_clusters": len(clusters),
        })
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ["der_raw", "der", "ider", "wder", "far", "escalate"]}
    summary["speaker_count_exact"] = float(np.mean([r["n_ref"] == r["n_hyp"] for r in rows]))
    summary["clusters_per_conv"] = float(np.mean([r["n_clusters"] for r in rows]))
    summary["audio_minutes"] = sum(len(c.audio) for c in convs) / 16000 / 60
    summary["diarize_seconds"] = diar_s
    return {"config": vars(args), "summary": summary, "per_conversation": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conversations", type=int, default=12)
    ap.add_argument("--embedding", choices=list(EMBEDDINGS), default="titanet")
    ap.add_argument("--segmentation", choices=list(SEGMENTATION), default="pyannote3")
    ap.add_argument("--diarizer", choices=["sherpa", "chunk"], default="chunk")
    ap.add_argument("--profile", choices=["hard", "moderate"], default="hard")
    ap.add_argument("--cluster-threshold", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=2.5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--gallery-size", type=int, default=20)
    ap.add_argument("--mean-gallery", action="store_true")
    ap.add_argument("--no-snorm", action="store_true")
    ap.add_argument("--no-wearer", action="store_true")
    ap.add_argument("--wearer-weight", type=float, default=1.5)
    ap.add_argument("--priors", action="store_true")
    ap.add_argument("--unknown-merge", type=float, default=0.6)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--switch-penalty", type=float, default=2.0)
    ap.add_argument("--prior-weight-refine", type=float, default=1.0)
    ap.add_argument("--enroll-seconds", type=float, default=45)
    ap.add_argument("--wearer-enroll-seconds", type=float, default=150)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = run(args)
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
