// Unit tests for the omi-omy silence gate (copied into the app's test/ by build_fork.sh).
//   flutter test test/omiomy_silence_gate_test.dart
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:omi/omiomy_silence_gate.dart';

/// Frames are ints (their index); the scripted detector says which are speech.
class Harness {
  Harness(this.speech, {int frameMs = 10, bool Function(int i)? throwsAt}) {
    gate = OmiomySilenceGate<int>(
      detect: (bytes) async {
        final i = bytes.first;
        if (throwsAt?.call(i) ?? false) throw StateError('detector broke');
        return speech(i) ? 0.9 : 0.01;
      },
      bytesOf: (i) => [i],
      send: (i) => out.add('f$i'),
      sendText: (t) => out.add('gap${jsonDecode(t)['ms']}'),
      frameMs: frameMs,
      onDropped: (i) => dropped.add(i),
    );
  }

  final bool Function(int i) speech;
  late final OmiomySilenceGate<int> gate;
  final out = <String>[];
  final dropped = <int>[];

  /// Packets arrive a few at a time (100 per second from a pendant), not all at once.
  Future<void> feed(int from, int to) async {
    for (var i = from; i < to; i++) {
      gate.offer(i);
      if (i % 10 == 9) await gate.idle;
    }
    await gate.idle;
  }
}

void main() {
  test('sends the first hangover, then holds silence as pre-roll and drops the rest', () async {
    final h = Harness((i) => false);
    await h.feed(0, 1000); // 10 s of silence at 10 ms per frame
    // 0..300 inclusive are within the 3 s start hangover (audio time 10..3010 ms).
    expect(h.out.first, 'f0');
    expect(h.out.last, 'f299');
    expect(h.out.length, 300);
    expect(h.gate.heldFrames, 100); // 1 s pre-roll
    expect(h.dropped.length, 600); // 1000 - 300 sent - 100 held
    expect(h.gate.pendingGapMs, 6000);
  });

  test('speech onset: gap note first, then the pre-roll, then live packets; gap counts only dropped audio', () async {
    final h = Harness((i) => i >= 1000 && i < 1200);
    await h.feed(0, 1200);
    final gapAt = h.out.indexWhere((s) => s.startsWith('gap'));
    expect(h.out[gapAt], 'gap6000'); // frames 300..899 dropped = 600 x 10 ms
    expect(h.out.sublist(gapAt + 1, gapAt + 3), ['f900', 'f901']); // pre-roll starts 1 s before onset
    expect(h.out.where((s) => s.startsWith('gap')).length, 1);
    expect(h.out.last, 'f1199');
    expect(h.dropped, List.generate(600, (k) => 300 + k));
  });

  test('hangover: keeps sending 3 s after the last speech, then holds again', () async {
    final h = Harness((i) => i >= 400 && i < 500);
    await h.feed(0, 1500);
    // Last speech ends at frame 499 (audio 5000 ms); sending continues to 8000 ms = frame 799.
    expect(h.out.contains('f799'), isTrue);
    expect(h.out.contains('f800'), isFalse);
    expect(h.gate.heldFrames, 100);
  });

  test('20 ms frames count 20 ms each', () async {
    final h = Harness((i) => i >= 500, frameMs: 20);
    await h.feed(0, 510);
    // Start hangover = 150 frames; 50 frames of pre-roll; 500 - 150 - 50 = 300 dropped = 6000 ms.
    expect(h.out.firstWhere((s) => s.startsWith('gap')), 'gap6000');
  });

  test('fails open: a detector error sends the held pre-roll and then everything', () async {
    final h = Harness((i) => false, throwsAt: (i) => i == 700);
    await h.feed(0, 1000);
    expect(h.gate.failedOpen, isTrue);
    final gapAt = h.out.indexWhere((s) => s.startsWith('gap'));
    expect(h.out[gapAt], 'gap3000'); // 300..599 dropped before the failure
    expect(h.out.last, 'f999');
    expect(h.out.where((s) => s.startsWith('f')).length, 300 + 100 + 300); // start, pre-roll, after failure
  });

  test('fails open on a backlog the detector cannot keep up with', () async {
    final h = Harness((i) => false);
    for (var i = 0; i < 400; i++) {
      h.gate.offer(i); // offered faster than processed
    }
    await h.gate.idle;
    expect(h.gate.failedOpen, isTrue);
    expect(h.out.length, 400);
  });

  test('dropped spans merge into wall-clock ranges', () {
    final spans = OmiomyDroppedSpans.instance;
    final t = DateTime.now();
    spans.add(t, 10);
    spans.add(t.add(const Duration(milliseconds: 10)), 10);
    expect(spans.spans.last.$1, t.millisecondsSinceEpoch ~/ 1000);
  });
}
