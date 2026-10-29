# Omi Omy: speaker attribution pipeline

Turns one pendant conversation into transcript words labeled with who said
them. This is the first piece of the open-source Omi app replacement. The
design is in the project doc "Speaker attribution pipeline: max accuracy, min
cost".

```
pendant energy gate -> phone Silero VAD -> pack speech -> MAI Transcribe 2 (no diarization)
                                                   \-> diarize (pyannote seg 3.0 + TitaNet, constrained clustering)
                                                        -> identify (gallery + per-sample s-norm + priors)
                                                        -> attach words -> escalate hard ones to Precision-2
```

## Layout

| Module | What it does |
| --- | --- |
| `attrib/vad.py` | `EnergyGate` (cheap pendant gate), `SileroVAD` (phone), `pack` + `TimeMap` (speech-only upload, timestamps mapped back) |
| `attrib/transcribe.py` | MAI Transcribe 2 REST client, word timestamps, `phraseList` from enrolled names |
| `attrib/chunkdiar.py` | Diarizer: 10 s segmentation windows, one TitaNet voiceprint per window speaker, clustering with "same window = different people" constraints |
| `attrib/diarize.py` | sherpa-onnx diarizer (baseline) and per-speaker voiceprint pooling |
| `attrib/identify.py` | Gallery, per-sample adaptive s-norm, calendar priors, open-set threshold, unknown-voice tracking |
| `attrib/align.py` | Words to speakers by timestamp overlap |
| `attrib/pipeline.py` | One conversation end to end, plus the escalation policy |
| `attrib/metrics.py` | DER, identification error, word speaker error, false-accept rate |
| `attrib/sim/` | Simulated pendant conversations (904 synthetic voices, rooms, noise) and benchmarks |

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
scripts/fetch_models.sh
pytest -q
python -m attrib.sim.bench --conversations 24 --priors      # accuracy
python -m attrib.sim.gate_bench                             # pendant gate + VAD
```

The MAI client reads `AZURE_SPEECH_ENDPOINT` and `AZURE_SPEECH_KEY`. Its
response parser follows the documented fast-transcription schema and has not
been run against a live key yet.

## Results so far (simulated audio, 2026-10-29)

24 busy-room conversations (40 min) and 12 quiet-room ones (20 min). Each has
2 to 4 speakers including the wearer, and some include a stranger. Calendar
hints are simulated as right 80% of the time.

| Change | Words with the wrong speaker |
| --- | --- |
| Starting point (WeSpeaker voiceprints, default settings) | 62% |
| TitaNet voiceprints (WeSpeaker collapses on distant voices: 28% vs 4% EER) | 34% |
| Tuned threshold + gallery + s-norm | 29% |
| + calendar hints | 23% |
| + cohort from mixed rooms, threshold 2.5 | **22% busy room, 20% quiet room** |
| Perfect speaker turns (ceiling for the naming step) | 3% |

What each part is worth (busy room, priors on): per-sample s-norm 34% → 23%;
gallery of many voiceprints instead of one average 27% → 23%; calendar hints
29% → 23%; wearer loudness bonus ~0 (the wearer is already easy).

**The bottleneck is segmentation, not naming.** With perfect turns the naming
step mislabels 3% of words. The free pyannote 3.0 segmentation model gets only
89% of its window-level speaker units clean, and most of the rest fuse two
distant people. A better segmenter (pyannote community-1, DiariZen, or
pyannoteAI Precision-2) is the next lever. Hugging Face is blocked in the
build environment, so those haven't been tested. Rev's reverb-diarization
v1/v2 were worse here and are non-production licensed anyway.

**Silence gating** (31 simulated minutes, 41% speech):

| Gate | Radio on | Distant speech kept |
| --- | --- | --- |
| Pendant energy gate (3 dB) then Silero 0.15 | 49% | 96.5% |
| Silero at its default 0.4 | n/a | 73% (drops quiet distant speech) |

Silero's threshold is lowered to 0.15 for this reason.

**Learning from corrections** didn't show a clear gain in simulation. The
enrollment audio already covers the simulated rooms, so new samples add little,
and samples from impure clusters can hurt. Only confirmed, high-purity samples
are added, and this needs real recordings to judge.

Caveat: synthetic voices aren't real people. Use these numbers to compare
variants, not as a forecast.
