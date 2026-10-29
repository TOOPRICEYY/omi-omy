#!/usr/bin/env bash
# Downloads the open models into ./models. All come from sherpa-onnx's GitHub
# releases, so this works where Hugging Face is blocked.
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p models data && cd models
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
curl -sSL $R/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
for m in nemo_en_titanet_large.onnx wespeaker_en_voxceleb_resnet34_LM.onnx 3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx; do
  [ -f "$m" ] || curl -sSL -o "$m" "$R/speaker-recongition-models/$m"
done
# Speech synthesizer with 904 voices, used only by the simulator.
[ -d vits-piper-en_US-libritts_r-medium ] || curl -sSL $R/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2 | tar xj
cd ../data
curl -sSL -O https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.wav \
          -O https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.rttm
