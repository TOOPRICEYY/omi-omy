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

| Step (measured in this order) | Test set | Words with the wrong speaker |
| --- | --- | --- |
| WeSpeaker voiceprints, sherpa-onnx diarizer, threshold 3.0 | 8 busy | 62% |
| TitaNet voiceprints (WeSpeaker collapses on distant voices: 28% vs 4% EER) | 8 busy | 57% |
| Threshold 1.5 (3.0 sent almost everyone to "unknown") | 8 busy | 29% |
| + calendar hints | 8 busy | 21% |
| Constrained chunk diarizer (3x faster, same accuracy) | 24 busy | 23% |
| + per-sample s-norm, mixed-room cohort, threshold 2.5 | 24 busy / 12 quiet | **22% / 20%** |
| Perfect speaker turns (ceiling for the naming step) | 8 busy | 3% |

Ablations on the 24 busy conversations, measured before the per-sample s-norm
change: without s-norm 34% (vs 23%), one averaged voiceprint per person
instead of a gallery 29%, no calendar hints 29%, no wearer loudness bonus 23%
(no effect, since the wearer is already easy). Strangers labeled as a known
person: 9% of their speech time busy, 3% quiet.

**The bottleneck is segmentation, not naming.** With perfect turns the naming
step mislabels 3% of words. The free pyannote 3.0 segmentation model gets only
89% of its window-level speaker units clean, and most of the rest fuse two
distant people. pyannote community-1 and DiariZen were tested on
2026-10-30 (details in `results/2026-10-30/REPORT.md`). Neither beats the
baseline in busy rooms: community-1's segmenter makes the same decisions as
3.0 and its WeSpeaker/VBx clustering merges people (29% busy, 31% quiet).
DiariZen separates speakers better but drops a quarter of busy-room speech as
silence (25% busy at best), while it is the best in quiet rooms (14% vs 19%).
Its weights are non-commercial (CC-BY-NC-4.0). Rev's reverb-diarization
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
