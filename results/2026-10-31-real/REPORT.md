# Real-speech check: AMI meetings (2026-10-31)

Why: the simulator's voices come from one TTS model, and they are much closer
together than real voices. Clean audio, 40 speakers each:

| Voiceprint model | Set | Same speaker | Different speakers | Nearest other speaker |
| --- | --- | --- | --- | --- |
| TitaNet | synthetic (piper libritts_r) | 0.84 | 0.35 | 0.70 |
| TitaNet | real (LibriSpeech dev-clean) | 0.79 | 0.07 | 0.36 |
| WeSpeaker | synthetic | 0.86 | 0.55 | 0.87 |
| WeSpeaker | real | 0.83 | 0.53 | 0.82 |

(Average cosine similarity. "Nearest other speaker" is the average similarity
of each speaker's centroid to the closest other speaker.) With TitaNet, the gap
between "same person" and "closest other person" is 0.14 for synthetic voices
and 0.43 for real ones, so in noise the simulator makes clustering look far
harder than it is. That explains why community-1 merged people in simulation.

## AMI benchmark

`python -m attrib.real.ami --priors` (see `attrib/real/ami.py`). AMI single
distant microphone (a table mic about 1 m from four people), test split from
`diarizers-community/ami` (CC-BY-4.0). 8 meetings (b and d of each series),
10 minutes each from minute 5, 80 minutes total, 16 different people. Each
person is enrolled with 45 s from their other meetings (a and c). All 16 are in
the gallery, and calendar hints list the right four 80% of the time. No wearer.
No forgiveness collar, overlap scored.

"Named wrong" is the share of speech time given the wrong name (the
identification error). There are no word timings here, so the word metric used
in simulation isn't available, but in simulation the two track each other.

| Diarizer | DER | Missed speech | Speaker confusion | Named wrong | Clusters (4 people) |
| --- | --- | --- | --- | --- | --- |
| Baseline, pyannote 3.0 + TitaNet, threshold 0.5 | 27.9% | 11.8% | 12.9% | 22.1% | 25 |
| Baseline, threshold 0.6 (best) | 22.9% | 11.9% | 8.0% | 21.1% | 16 |
| community-1 full pipeline (untuned) | 18.1% | 10.6% | 4.3% | 18.1% | 4.0 |
| DiariZen segmenter + TitaNet, threshold 0.5 | 21.0% | 8.1% | 9.6% | 16.9% | 19 |
| DiariZen segmenter + TitaNet, threshold 0.6 (best) | **18.0%** | 8.1% | 6.6% | **16.5%** | 15 |

DiariZen's speech-bias knob hurt here (bias 2: 21.9% DER, bias 4: 24.6%), since
it already hears speech fine at a table distance.

Caveats: 80 minutes and 16 people is a small sample. The baseline and DiariZen
thresholds were picked on this same test set, which flatters them slightly;
community-1 was not tuned at all. AMI is a meeting table, not a pendant: no
wearer close to the mic, no street or restaurant noise.

Speed, 4 CPU cores: baseline ~0.13x realtime, DiariZen ~0.3–0.7x, community-1 ~1.0x.
Licences: community-1 CC-BY-4.0 (commercial use OK); DiariZen weights CC-BY-NC-4.0.
