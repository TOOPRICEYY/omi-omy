"""Does accuracy improve as the user confirms speakers?

Replays the simulated conversations in order. After each one the simulated
user answers up to `--taps` one-tap prompts for the least confident speakers
(the truth comes from the ground truth), and each answer adds that speaker's
in-conversation voiceprint to the gallery. Compares word speaker error on the
second half of the conversations with and without learning.

    python -m attrib.sim.learn_bench --conversations 24
"""
from __future__ import annotations

import argparse
import pickle
import random

import numpy as np

from ..align import assign_words
from ..identify import Gallery, Identifier, IdentifyConfig, final_labels
from ..metrics import word_speaker_error
from ..types import Turn
from .bench import _cached, calendar_priors
from .pendant import World


def majority(cluster, ref_turns):
    ov = {}
    for a in cluster.turns:
        for r in ref_turns:
            o = min(a.end, r.end) - max(a.start, r.start)
            if o > 0:
                ov[r.speaker] = ov.get(r.speaker, 0) + o
    if not ov:
        return None, 0.0
    top = max(ov, key=ov.get)
    return top, ov[top] / sum(ov.values())


def replay(args, learn: bool, taps: int):
    world = World.make(args.seed)
    g0 = _cached(f"gallery{args.seed}_titanet_45_150", lambda: None)
    cohort = _cached(f"cohortemb{args.seed}_titanet_mixed", lambda: None)
    g = Gallery()
    for p, es in g0.items():
        for e in es:
            g.add(p, e)
    ident = Identifier(g, cohort, IdentifyConfig(threshold=args.threshold, wearer="wearer"))
    out = []
    for i in range(args.conversations):
        c = _cached(f"conv{args.seed}_{i}", lambda: None)
        raw, clusters = _cached(f"diar{args.seed}_{i}_titanet_{args.cluster_threshold}_chunk", lambda: None)
        pri = calendar_priors(world, c.speakers, random.Random(i)) if args.priors else None
        m = ident.identify(clusters, pri)
        names = final_labels(clusters, m)
        turns = [Turn(t.start, t.end, names[t.speaker]) for t in raw]
        out.append(word_speaker_error(c.words, assign_words(c.words, turns), set(g.people)))
        if learn:
            asked = sorted([k for k in clusters if k.speech_time >= 3.0], key=lambda k: m[k.label].confidence)
            for k in asked[:taps]:
                true, purity = majority(k, c.turns)
                # The user names enrolled people; strangers they skip.
                if true in g.people and purity >= 0.7:
                    g.add(true, k.centroid)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conversations", type=int, default=24)
    ap.add_argument("--cluster-threshold", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=2.5)
    ap.add_argument("--priors", action="store_true")
    args = ap.parse_args()
    half = args.conversations // 2
    base = replay(args, False, 0)
    print(f"no learning: first half {base[:half].mean():.3f}, second half {base[half:].mean():.3f}")
    for taps in (1, 2, 4):
        r = replay(args, True, taps)
        print(f"{taps} taps/conversation: first half {r[:half].mean():.3f}, second half {r[half:].mean():.3f}")


if __name__ == "__main__":
    main()
