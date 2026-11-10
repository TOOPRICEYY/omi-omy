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
| `attrib/transcribe.py` | MAI Transcribe 2 client (OpenRouter), word timestamps, long audio cut at quiet points |
| `omibridge/` | Service for the Omi app: audio webhook, conversation cutting, review page, voice naming |
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

The transcription client reads `OPENROUTER_API_KEY` and calls
`microsoft/mai-transcribe-2` (word timestamps, about $0.10 per audio hour).

## Omi bridge (`omibridge/`)

Runs next to the stock Omi app, no fork needed. The app's developer setting
"Realtime audio bytes" posts the pendant's raw audio to the bridge. The bridge
cuts it into conversations (closed after 2 minutes without speech), transcribes
with MAI Transcribe 2 through OpenRouter, diarizes with DiariZen + TitaNet, and
names the speakers it knows. A local review page lists conversations, plays a
clip of each unknown voice, and lets you name it. Naming a voice adds it to the
gallery, so later conversations recognize that person.

```bash
scripts/install_mac.sh              # venv, models, .env with a random webhook token, tests
# add OPENROUTER_API_KEY to .env
.venv/bin/python -m omibridge       # webhook on :8377, review page on http://127.0.0.1:8378/
scripts/install_mac.sh --launchd    # optional: keep it running with launchd
```

Two ports on purpose. Port 8377 serves only `POST /omi/audio/<token>` and is the
one to expose (for example through a Cloudflare tunnel) so the phone can reach
it. Port 8378 has the review page and the transcripts, and listens on loopback
only. In the Omi app set Settings > Developer Mode > Realtime audio bytes to
`https://<your-host>/omi/audio/<OMIBRIDGE_TOKEN>`.

Data stays in `data/bridge/`: `conversations/<id>/` holds `audio.flac`,
`conversation.json` (speakers and transcript lines) and the voiceprints needed
to rename speakers later; `gallery.npz` holds the enrolled people.

Facts about Omi's side, from its source (commit 760499bd):

- Omi's backend sends the audio, not the phone, so the webhook URL must be
  public. Private and Tailscale addresses are rejected.
- It sends every second of a live session, silence included, as raw 16 kHz
  PCM16 with `uid` and `sample_rate` in the query and an `Idempotency-Key`
  header. Failed POSTs are retried, and after 100 failures in a row Omi turns
  the webhook off, so the endpoint answers before doing any heavy work.
- Audio synced later from the pendant's storage is not sent.

Posting conversations into Omi (`omibridge/omi_api.py`) is off by default. Omi
already keeps its own copy of each live conversation, so a posted one is a
second copy with a second round of summaries and memories. Turn it on with
`OMIBRIDGE_OMI_WRITEBACK=1` and an `OMI_API_KEY` that has `conversations:write`.
The request follows Omi's source but has not been tried against the live API. DiariZen's weights are CC-BY-NC-4.0, so
this setup is for personal use.

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
Its weights are non-commercial (CC-BY-NC-4.0).

**Real speech reverses that ranking** (`results/2026-10-31-real/REPORT.md`).
The simulator's TTS voices sit about twice as close together as real voices
(TitaNet nearest-other-speaker similarity 0.70 vs 0.36), which makes clustering
look much harder than it is. On 80 minutes of real AMI meetings (table mic,
16 people, enrolled from their other meetings), speech named wrong is: baseline
21.1%, community-1 18.1%, DiariZen 16.5%. community-1 is the best choice that can
be used commercially. Run it with:

```bash
# AMI SDM test split and LibriSpeech dev-clean (for the s-norm cohort), from Hugging Face
python -m attrib.real.ami --priors --diarizer community1
``` Rev's reverb-diarization
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
