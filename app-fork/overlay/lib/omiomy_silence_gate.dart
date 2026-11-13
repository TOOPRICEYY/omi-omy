// omi-omy: phone-side silence gate for live pendant audio (copied into the app by
// the omi-omy repo's scripts/build_fork.sh; hooked in by app-fork/patches/0004).
//
// While nobody speaks, Opus packets from the pendant are held instead of being sent to
// the bridge. When speech starts, the last second of held audio (pre-roll) is sent
// first; packets older than that are dropped, and the bridge is told how much audio
// was dropped with a text message {"type":"omiomy_silence_gap","ms":N} sent just before
// the first packet after the pause. The bridge puts that much silence back, so the
// conversation it builds is the same with the gate on or off; the gate only saves
// upload and battery. After the last speech the gate keeps sending for three more
// seconds (hangover).
//
// Conservative on purpose: a low speech threshold, and it fails open. Any detector
// error, a backlog the detector cannot keep up with, or a codec it does not handle
// means every packet is sent as before. Off unless the owner turns it on in
// Settings > Recording (stored in preferences), or the build sets OMIOMY_SILENCE_GATE.
import 'dart:async';
import 'dart:collection';
import 'dart:convert';

const String omiomySilenceGapType = 'omiomy_silence_gap';
const bool omiomySilenceGateDefault = bool.fromEnvironment('OMIOMY_SILENCE_GATE');
const String omiomySilenceGatePref = 'omiomySilenceGate';

/// Speech probability for one raw codec packet, 0..1. May return 0 while it is
/// still collecting samples for its next window.
typedef OmiomySpeechDetector = Future<double> Function(List<int> packet);

/// The gate's logic, independent of the app: feed packets in capture order with [offer];
/// it calls [send] for each packet that goes out, in order, and [sendText] for gap notes.
class OmiomySilenceGate<F> {
  OmiomySilenceGate({
    required this.detect,
    required this.bytesOf,
    required this.send,
    required this.sendText,
    required this.frameMs,
    this.preRollMs = 1000,
    this.hangoverMs = 3000,
    this.threshold = 0.3,
    this.maxBacklog = 300,
    this.onDropped,
    this.onFailOpen,
  }) : _lastSpeechEndMs = 0; // the first [hangoverMs] after start are always sent

  final OmiomySpeechDetector detect;
  final List<int> Function(F frame) bytesOf;
  final void Function(F frame) send;
  final void Function(String text) sendText;
  final int frameMs;
  final int preRollMs;
  final int hangoverMs;
  final double threshold;

  /// Packets waiting for the detector before the gate gives up and sends everything.
  final int maxBacklog;

  /// Called for each packet that is dropped (never sent), with the packet.
  final void Function(F frame)? onDropped;
  final void Function(Object reason)? onFailOpen;

  final Queue<F> _held = Queue<F>();
  int _audioMs = 0;
  int _lastSpeechEndMs;
  int _droppedMs = 0;
  int _backlog = 0;
  bool _failedOpen = false;
  Future<void> _chain = Future<void>.value();

  bool get failedOpen => _failedOpen;

  /// Milliseconds dropped since the last gap note (for tests and diagnostics).
  int get pendingGapMs => _droppedMs;

  /// Packets currently held as pre-roll.
  int get heldFrames => _held.length;

  void offer(F frame) {
    _backlog++;
    if (_backlog > maxBacklog && !_failedOpen) _failOpen(StateError('detector backlog'));
    _chain = _chain.then((_) => _process(frame)).catchError((Object e) {
      _failOpen(e);
      _sendNow(frame);
    });
  }

  /// Completes when every packet offered so far has been sent, held or dropped.
  Future<void> get idle => _chain;

  Future<void> _process(F frame) async {
    _backlog--;
    _audioMs += frameMs;
    if (!_failedOpen) {
      try {
        final p = await detect(bytesOf(frame));
        if (!p.isFinite) throw StateError('detector returned $p');
        if (p >= threshold) _lastSpeechEndMs = _audioMs;
      } catch (e) {
        _failOpen(e);
      }
    }
    if (_failedOpen || _audioMs - _lastSpeechEndMs <= hangoverMs) {
      _sendNow(frame);
      return;
    }
    _held.addLast(frame);
    while (_held.length * frameMs > preRollMs) {
      final old = _held.removeFirst();
      _droppedMs += frameMs;
      onDropped?.call(old);
    }
  }

  void _sendNow(F frame) {
    if (_held.isNotEmpty || _droppedMs > 0) {
      if (_droppedMs > 0) {
        sendText(jsonEncode({'type': omiomySilenceGapType, 'ms': _droppedMs}));
        _droppedMs = 0;
      }
      while (_held.isNotEmpty) {
        send(_held.removeFirst());
      }
    }
    send(frame);
  }

  void _failOpen(Object reason) {
    if (_failedOpen) return;
    _failedOpen = true;
    onFailOpen?.call(reason);
  }
}

/// Wall-clock spans (epoch seconds, phone clock) of audio the gate dropped. The app's
/// audio log counts these as handled when it decides which of its copies a saved
/// transcript covers, so dropped silence is not uploaded again through offline sync.
class OmiomyDroppedSpans {
  OmiomyDroppedSpans._();
  static final OmiomyDroppedSpans instance = OmiomyDroppedSpans._();

  final List<(int, int)> _spans = [];

  void add(DateTime at, int ms) {
    final start = at.millisecondsSinceEpoch ~/ 1000;
    final end = (at.millisecondsSinceEpoch + ms + 999) ~/ 1000;
    if (_spans.isNotEmpty && start <= _spans.last.$2 + 1) {
      final last = _spans.removeLast();
      _spans.add((last.$1, end > last.$2 ? end : last.$2));
    } else {
      _spans.add((start, end));
    }
    final cutoff = DateTime.now().millisecondsSinceEpoch ~/ 1000 - 12 * 3600;
    _spans.removeWhere((s) => s.$2 < cutoff);
  }

  List<(int, int)> get spans => List.unmodifiable(_spans);
}
