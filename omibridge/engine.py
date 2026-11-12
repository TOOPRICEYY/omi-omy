"""Runs one recorded conversation through the attribution pipeline and keeps
what is needed to rename speakers later without touching the audio again."""
from __future__ import annotations

import io
import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from attrib.align import assign_words
from attrib.audio import SR, resample
from attrib.chunkdiar import KNOWN_META, ChunkDiarizer
from attrib.embed import Embedder
from attrib.identify import UNKNOWN, Gallery, Identifier, IdentifyConfig, Match, final_labels, is_unknown
from attrib.maidiar import mai_clusters
from attrib.types import Region, Turn, Word
from attrib.vad import SileroVAD, pack

from .config import Config

SEGMENTATION = {
    "diarizen": "diarizen-onnx/diarizen_segmentation.onnx",
    "pyannote3": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
}
EMBEDDING = "nemo_en_titanet_large.onnx"
COHORT = Path(__file__).with_name("cohort_titanet.npy")
MAX_SAMPLES_PER_LABEL = 6  # voiceprints added to a person from one conversation
log = logging.getLogger("omibridge.engine")


class Engine:
    def __init__(self, cfg: Config, transcriber=None, summarizer=None):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.vad = SileroVAD(front=cfg.vad_front)
        # The live stream gets its own detector and lock, so the webhook never
        # waits behind a conversation that is being processed.
        self.live_vad, self.live_lock = SileroVAD(), threading.Lock()
        self.embedder = Embedder(str(cfg.models_dir / EMBEDDING))
        self.diarizer = ChunkDiarizer(str(cfg.models_dir / SEGMENTATION[cfg.segmentation]), self.embedder,
                                      threshold=cfg.cluster_threshold,
                                      seg_meta=KNOWN_META.get(cfg.segmentation))
        if cfg.segmentation == "diarizen" and cfg.diarizer_device != "cpu":
            from .gpu_segmentation import maybe_gpu

            self.diarizer.seg = maybe_gpu(self.diarizer.seg, cfg.diarizer_device, cfg.diarizer_batch,
                                          cfg.diarizen_checkpoint or None)
        self.cohort = np.load(COHORT)
        self.gallery = Gallery.load(str(cfg.gallery_path)) if cfg.gallery_path.exists() else Gallery()
        self.transcriber = transcriber
        self.summarizer = summarizer

    # -- voice detection for the live stream ---------------------------------
    def speech_seconds(self, x: np.ndarray) -> float:
        if len(x) < SR // 4:
            return 0.0
        with self.live_lock:
            return sum(r.duration for r in self.live_vad(x))

    # -- naming ----------------------------------------------------------------
    def _identifier(self) -> Identifier:
        return Identifier(self.gallery, self.cohort,
                          IdentifyConfig(threshold=self.cfg.identify_threshold, wearer=self.cfg.wearer,
                                         wearer_weight=1.5 if self.cfg.wearer else 0.0))

    def name_clusters(self, clusters, pinned: dict[str, str] | None = None) -> tuple[dict, dict]:
        """cluster label -> name, and cluster label -> confidence. `pinned`
        holds names the user set by hand; they always win."""
        pinned = pinned or {}
        free = [c for c in clusters if c.label not in pinned]
        # MAI speakers with too little clear speech for a trustworthy voiceprint stay unnamed.
        short = [c for c in free if getattr(c, "clean_time", None) is not None
                 and c.clean_time < self.cfg.min_name_seconds]
        matches = self._identifier().identify([c for c in free if c not in short])
        matches |= {c.label: Match(UNKNOWN, 0.0, 0.0, 0.0) for c in short}
        names = final_labels(free, matches)
        conf = {c.label: matches[c.label].confidence for c in free}
        for label, person in pinned.items():
            names[label], conf[label] = person, 1.0
        return names, conf

    # -- a whole conversation --------------------------------------------------
    def process(self, pcm_path: Path, meta: dict) -> dict:
        x = np.frombuffer(pcm_path.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
        x = resample(x, int(meta.get("sample_rate", SR)))
        cid = meta["id"]
        out = self.cfg.conv_dir / cid
        out.mkdir(parents=True, exist_ok=True)
        words: list[Word] = []
        error, mode = None, None
        with self.lock:
            regions = self.vad(x)
            # "all": MAI hears the whole conversation (times map 1:1); speech_seconds still from the VAD.
            sent = [Region(0.0, len(x) / SR)] if self.cfg.mai_audio == "all" and len(x) else regions
            packed, tmap = pack(x, sent)
            use_mai = self.cfg.diarizer == "mai" and self.transcriber is not None
            heard: list[Word] = []
            if self.transcriber is not None and len(packed):
                buf = io.BytesIO()
                sf.write(buf, packed, SR, format="WAV", subtype="PCM_16")
                phrases = list(self.gallery.people)
                try:
                    heard = (self.transcriber(buf.getvalue(), phrases, diarize=True) if use_mai
                             else self.transcriber(buf.getvalue(), phrases))
                except Exception as e:  # keep the diarization even if the transcript failed
                    error = f"transcription failed: {e}"
            raw, clusters = [], []
            if use_mai and any(w.speaker is not None for w in heard):
                raw, clusters = mai_clusters(packed, heard, self.embedder)
                mode = "mai"
            elif len(packed):
                if use_mai:
                    log.warning("%s: no speaker labels from MAI (%s); diarizing locally", cid,
                                error or "none returned")
                raw = self.diarizer.turns(packed)
                clusters = self.diarizer.clusters(packed, raw) if raw else []
                release = getattr(self.diarizer.seg, "release", None)
                if release:
                    release()   # GPU memory back after each conversation
                mode = "local"
        turns = [Turn(tmap.to_orig(t.start), tmap.to_orig(t.end), t.speaker) for t in raw]
        words = [Word(w.text, tmap.to_orig(w.start), tmap.to_orig(w.end)) for w in heard]
        sf.write(out / "audio.flac", x, SR, format="FLAC", subtype="PCM_16")
        with open(out / "state.pkl", "wb") as f:
            pickle.dump({"turns": turns, "words": words, "clusters": clusters, "pinned": {}}, f)
        meta = dict(meta) | {"duration": len(x) / SR, "speech_seconds": sum(r.duration for r in regions),
                             "error": error, "diarizer": mode}
        (out / "meta.json").write_text(json.dumps(meta))
        return self.render(cid)

    def render(self, cid: str) -> dict:
        """(Re)build conversation.json from the stored state and the current gallery."""
        out = self.cfg.conv_dir / cid
        with open(out / "state.pkl", "rb") as f:
            st = pickle.load(f)
        meta = json.loads((out / "meta.json").read_text())
        names, conf = self.name_clusters(st["clusters"], st["pinned"])
        turns = [Turn(t.start, t.end, names[t.speaker], conf[t.speaker], t.speaker) for t in st["turns"]]
        # Lines are cut by voice cluster, not by name: they stay the same lines (same order, same
        # ids in the app) whatever the gallery learns later. Each line carries its cluster.
        lines = segments(assign_words(st["words"], st["turns"]), st["turns"])
        for line in lines:
            cluster = line["speaker"] if line["speaker"] in names else None
            line["cluster"], line["speaker"] = cluster, names.get(cluster, "unknown")
        by_label = {c.label: c for c in st["clusters"]}
        speakers: dict[str, dict] = {}
        for label, name in names.items():
            c = by_label[label]
            sp = speakers.setdefault(name, {"name": name, "known": not is_unknown(name), "clusters": [],
                                            "speech_seconds": 0.0, "confidence": 0.0, "sample": None})
            sp["clusters"].append(label)
            sp["speech_seconds"] = round(sp["speech_seconds"] + c.speech_time, 1)
            sp["confidence"] = round(max(sp["confidence"], conf[label]), 3)
        for sp in speakers.values():
            # The longest turn, as a clip to listen to when naming someone.
            mine = [t for t in turns if t.speaker == sp["name"]]
            best = max(mine, key=lambda t: t.duration)
            sp["sample"] = [round(best.start, 2), round(min(best.end, best.start + 12), 2)]
        doc = {"id": cid, "started_at": meta.get("started_at"), "duration": round(meta.get("duration", 0), 1),
               "speech_seconds": round(meta.get("speech_seconds", 0), 1), "error": meta.get("error"),
               "speakers": sorted(speakers.values(), key=lambda s: -s["speech_seconds"]),
               "summary": None, "segments": lines}
        if (out / "summary.json").exists():
            doc["summary"] = json.loads((out / "summary.json").read_text())
        # Atomic: readers (the page, the app, a re-render in another thread) never see half a file.
        tmp = out / f"conversation.json.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, out / "conversation.json")
        return doc

    def summarize(self, cid: str) -> dict:
        """Write (or rewrite, after speakers were named) the summary of a conversation.
        Leaves the old one in place when there is no summarizer or no text."""
        doc = self.render(cid)
        if self.summarizer is None:
            return doc
        when = doc.get("started_at")
        text = time.strftime("%A %d %B %Y, %H:%M", time.localtime(when)) if when else None
        summary = self.summarizer(doc | {"started_at_text": text})
        if summary is None:
            return doc
        (self.cfg.conv_dir / cid / "summary.json").write_text(json.dumps(summary, ensure_ascii=False))
        return self.render(cid)

    def label(self, cid: str, speaker: str | None, person: str, learn: bool = True,
              clusters: list[str] | None = None) -> dict:
        """The user says a voice in this conversation is `person`. Pins it here and, if `learn`,
        adds that voice to the gallery so later conversations recognize it.

        The voice is given by `clusters` (the cluster labels of the speaker the user saw, from
        conversation.json), which stay the same whatever the gallery has learned since. Without
        them, `speaker` is a name: the one on the stored page if it is there (that is what the
        user saw; "unknown1" can mean another voice once the gallery has learned), otherwise a
        current name."""
        person = person.strip()
        if not person:
            raise ValueError("empty name")
        out = self.cfg.conv_dir / cid
        with open(out / "state.pkl", "rb") as f:
            st = pickle.load(f)
        have = {c.label for c in st["clusters"]}
        if clusters is not None:
            labels = [l for l in clusters if l in have]
        else:
            try:
                shown = json.loads((out / "conversation.json").read_text())["speakers"]
                labels = [l for sp in shown if sp["name"] == speaker for l in sp.get("clusters", []) if l in have]
            except (OSError, ValueError, KeyError):
                labels = []
            if not labels:
                names, _ = self.name_clusters(st["clusters"], st["pinned"])
                labels = [l for l, n in names.items() if n == speaker]
        if not labels:
            raise KeyError(f"no speaker {speaker!r} in {cid}")
        for c in st["clusters"]:
            if c.label in labels:
                st["pinned"][c.label] = person
                if learn:
                    for e in ([c.centroid] + list(c.segment_embs))[:MAX_SAMPLES_PER_LABEL]:
                        self.gallery.add(person, e)
        with open(out / "state.pkl", "wb") as f:
            pickle.dump(st, f)
        if learn:
            self.save_gallery()
        return self.render(cid)

    def rerender_all(self, skip: str | None = None) -> int:
        """Re-render every conversation with the current gallery (after a voice was learned), so
        pages show the new names. Returns how many were re-rendered."""
        n = 0
        for d in sorted(self.cfg.conv_dir.glob("*/state.pkl")):
            cid = d.parent.name
            if cid == skip:
                continue
            try:
                self.render(cid)
                n += 1
            except Exception:
                log.exception("re-render of %s failed", cid)
        return n

    def enroll(self, person: str, x: np.ndarray, sr: int) -> int:
        """Add a person from a clean recording of them alone. Returns the number of voiceprints added."""
        x = resample(x if x.ndim == 1 else x.mean(axis=1), sr)
        with self.lock:
            regions = self.vad(x)
        n = 0
        for r in regions:
            seg = x[int(r.start * SR): int(r.end * SR)]
            for i in range(0, len(seg), 8 * SR):
                piece = seg[i: i + 8 * SR]
                if len(piece) >= 2 * SR:
                    self.gallery.add(person.strip(), self.embedder(piece))
                    n += 1
        if n:
            self.save_gallery()
        return n

    def forget(self, person: str) -> bool:
        if self.gallery.people.pop(person, None) is None:
            return False
        self.save_gallery()
        return True

    def save_gallery(self) -> None:
        tmp = self.cfg.gallery_path.with_suffix(".tmp.npz")
        self.gallery.save(str(tmp))
        tmp.replace(self.cfg.gallery_path)

    def people(self) -> list[dict]:
        return [{"name": p, "voiceprints": len(g)} for p, g in sorted(self.gallery.people.items())]


def segments(words: list[Word], turns: list[Turn], max_gap: float = 1.5) -> list[dict]:
    """Consecutive words by one speaker become a transcript line. Without a
    transcript, the speaker turns themselves are returned with empty text."""
    if not words:
        return [{"speaker": t.speaker, "start": round(t.start, 2), "end": round(t.end, 2), "text": ""}
                for t in sorted(turns, key=lambda t: t.start)]
    out: list[dict] = []
    for w in words:
        sp = w.speaker or "unknown"
        if out and out[-1]["speaker"] == sp and w.start - out[-1]["end"] <= max_gap:
            out[-1]["text"] += " " + w.text
            out[-1]["end"] = round(w.end, 2)
        else:
            out.append({"speaker": sp, "start": round(w.start, 2), "end": round(w.end, 2), "text": w.text})
    return out

