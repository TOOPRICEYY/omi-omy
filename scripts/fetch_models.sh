#!/usr/bin/env bash
# Downloads the models into ./models.
#   scripts/fetch_models.sh          bridge models: TitaNet, DiariZen (CC-BY-NC-4.0), pyannote 3.0 segmenter
#   scripts/fetch_models.sh --sim    also the simulator's speech synthesizer and extra voiceprint models
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p models data && cd models
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
[ -d sherpa-onnx-pyannote-segmentation-3-0 ] || curl -fsSL $R/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
[ -f nemo_en_titanet_large.onnx ] || curl -fsSL -o nemo_en_titanet_large.onnx $R/speaker-recongition-models/nemo_en_titanet_large.onnx
# DiariZen WavLM-large-s80-md, ONNX export. Non-commercial licence.
H=https://huggingface.co/christopherthompson81/diarizen_onnx/resolve/main
mkdir -p diarizen-onnx
for f in diarizen_segmentation.onnx diarizen_segmentation.onnx.data; do
  [ -f diarizen-onnx/$f ] || curl -fsSL -o diarizen-onnx/$f $H/$f
done
if [ "${1:-}" = "--sim" ]; then
  for m in wespeaker_en_voxceleb_resnet34_LM.onnx 3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx; do
    [ -f "$m" ] || curl -fsSL -o "$m" "$R/speaker-recongition-models/$m"
  done
  [ -d vits-piper-en_US-libritts_r-medium ] || curl -fsSL $R/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2 | tar xj
  cd ../data
  curl -fsSL -O https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.wav \
             -O https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.rttm
fi
