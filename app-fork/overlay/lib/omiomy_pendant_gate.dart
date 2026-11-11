// omi-omy: connects the silence gate (omiomy_silence_gate.dart) to the app's live
// pendant audio. CaptureController._sendPendantToSocket asks [OmiomyPendantGate.offer]
// first (app-fork/patches/0004); when it returns false the packet is sent as before.
//
// Detector: the app's own Silero VAD (services/wals/mobile_batch_vad.dart), which it
// already uses for batch uploads; it decodes a copy of each Opus packet and scores
// 512-sample windows. Only Opus pendants (10 ms and 20 ms frames) are gated; PCM and
// other codecs, a missing detector, or the switch being off mean no gate.
import 'package:omi/backend/preferences.dart';
import 'package:omi/backend/schema/bt_device/bt_device.dart';
import 'package:omi/omiomy_silence_gate.dart';
import 'package:omi/services/live_preview/local_preview_admission.dart';
import 'package:omi/services/wals/mobile_batch_vad.dart';
import 'package:omi/utils/logger.dart';

class OmiomyPendantGate {
  OmiomyPendantGate._();
  static final OmiomyPendantGate instance = OmiomyPendantGate._();

  MobileBatchVad? _vad;
  bool _loading = false;
  bool _vadUnavailable = false;
  OmiomySilenceGate<HeldPendantFrame>? _gate;
  Object? _socket;

  static bool get enabled =>
      SharedPreferencesUtil().getBool(omiomySilenceGatePref, defaultValue: omiomySilenceGateDefault);

  static set enabled(bool value) => SharedPreferencesUtil().saveBool(omiomySilenceGatePref, value);

  static int frameMsFor(BleAudioCodec? codec) => switch (codec) {
        BleAudioCodec.opus => 10,
        BleAudioCodec.opusFS320 => 20,
        _ => 0,
      };

  /// Returns true when the gate took [frame] (it will send it later, or drop it as silence).
  bool offer({
    required Object? socket,
    required BleAudioCodec? codec,
    required HeldPendantFrame frame,
    required void Function(HeldPendantFrame frame) sendNow,
    required void Function(String text) sendText,
  }) {
    try {
      final frameMs = frameMsFor(codec);
      if (socket == null || frameMs == 0 || !enabled || _vadUnavailable) {
        _gate = null;
        return false;
      }
      final vad = _vad;
      if (vad == null) {
        _load(); // until the detector is ready every packet is sent
        return false;
      }
      var gate = _gate;
      if (gate == null || !identical(_socket, socket) || gate.frameMs != frameMs) {
        _socket = socket;
        gate = _gate = OmiomySilenceGate<HeldPendantFrame>(
          detect: vad.probability,
          bytesOf: (f) => f.payload,
          send: sendNow,
          sendText: sendText,
          frameMs: frameMs,
          onDropped: (f) => OmiomyDroppedSpans.instance.add(f.capturedAt ?? DateTime.now(), frameMs),
          onFailOpen: (e) => Logger.debug('omiomy silence gate: sending everything ($e)'),
        );
      }
      gate.offer(frame);
      return true;
    } catch (e) {
      Logger.debug('omiomy silence gate: off ($e)');
      _gate = null;
      return false;
    }
  }

  void _load() {
    if (_loading) return;
    _loading = true;
    SileroMobileBatchVad.load().then((vad) {
      _vad = vad;
    }, onError: (Object e) {
      Logger.debug('omiomy silence gate: no detector ($e)');
      _vadUnavailable = true;
    });
  }
}
