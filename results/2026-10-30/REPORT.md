# Segmenter benchmark: pyannote community-1 and DiariZen (2026-10-30)

Simulated pendant audio, seed 0, calendar hints on, TitaNet naming, same harness as the README.
Busy = 24 conversations (40 min), quiet = 12 (20 min). Metric: share of words given the wrong speaker.

| Diarizer | Busy | Quiet | Notes |
| --- | --- | --- | --- |
| Baseline: pyannote 3.0 segmenter + our constrained TitaNet clustering (threshold 0.5) | **22.4%** | 19.0% | |
| Baseline, threshold 0.6 | **20.5%** | 24.9% | helps busy, hurts quiet; keep 0.5 |
| community-1 full pipeline (WeSpeaker + PLDA + VBx) | 29.0% | 31.5% | merges people: 2.2 clusters per conversation |
| community-1 segmenter in our diarizer | 22.4% | n/a | identical frame decisions to 3.0, so identical result |
| DiariZen segmenter in our diarizer, 0.5 | 32.8% | 16.5% | busy: drops 23% of speech as silence |
| DiariZen, speech bias 2 | 31.2% | **14.3%** | best quiet result |
| DiariZen, speech bias 6 | 25.2% | n/a | best DiariZen busy result |

Busy-room error breakdown (share of speech time, raw diarization):

| | Missed speech | False alarm | Speaker confusion |
| --- | --- | --- | --- |
| pyannote 3.0 | 5.9% | 2.3% | 15.7% |
| DiariZen | 23.4% | 0.7% | 6.4% |
| community-1 pipeline | 6.0% | 2.0% | 18.7% |

Speed on 4 CPU cores: pyannote 3.0 chunk diarizer ~0.02x realtime, DiariZen ~0.7x (ONNX export is batch-1 only), community-1 pipeline ~1.0x.

Licence: DiariZen weights are CC-BY-NC-4.0 (non-commercial only).

New harness options: `--segmentation community1` (pyannote.audio checkpoint), `--speech-bias` (lowers the segmenter's silence log-prob before argmax).
Per-run JSON results are in this folder.
Caveat: synthetic voices, not real people. Compare variants; don't forecast.
