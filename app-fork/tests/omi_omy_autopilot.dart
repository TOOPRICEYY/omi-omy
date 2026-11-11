// Test-only entry point for the omi-omy simulator check (scripts/build_fork.sh with
// OMI_FORK_AUTOPILOT=1). Boots the real app exactly as lib/main.dart does, then taps
// through it with real pointer events: the local sign-in button, any consent or
// permission screens, and finally a conversation whose title contains
// OMIOMY_EXPECT_TITLE. It prints what is on screen at each step ("OMIOMY ..." lines).
// Not part of Omi and never built into the app you install.
import 'dart:async';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:omi/main.dart' as app;

const expectTitle = String.fromEnvironment('OMIOMY_EXPECT_TITLE');
const gates = ['Sign in (local dev)', 'I agree', 'Agree', 'Accept', 'Continue', 'Allow', 'Skip', 'Not now',
  'Got it', 'Maybe later', 'Done', 'Next'];

String? textOf(Widget w) {
  if (w is Text) return w.data ?? w.textSpan?.toPlainText();
  if (w is RichText) return w.text.toPlainText();
  return null;
}

List<(String, Element)> visibleTexts() {
  final out = <(String, Element)>[];
  void visit(Element e) {
    final s = textOf(e.widget);
    if (s != null && s.trim().isNotEmpty && e.renderObject is RenderBox && (e.renderObject as RenderBox).hasSize) {
      out.add((s, e));
    }
    e.visitChildren(visit);
  }
  WidgetsBinding.instance.rootElement?.visitChildren(visit);
  return out;
}

int _pointer = 900;
Future<void> tap(Element e) async {
  final box = e.renderObject as RenderBox;
  final c = box.localToGlobal(box.size.center(Offset.zero));
  final id = _pointer++;
  GestureBinding.instance.handlePointerEvent(PointerDownEvent(pointer: id, position: c));
  await Future<void>.delayed(const Duration(milliseconds: 90));
  GestureBinding.instance.handlePointerEvent(PointerUpEvent(pointer: id, position: c));
}

void report(String where) {
  final texts = visibleTexts().map((t) => t.$1.replaceAll('\n', ' ')).toSet().take(40).join(' | ');
  print('OMIOMY[$where] $texts');
}

Future<void> autopilot() async {
  var openedDetail = false;
  for (var i = 0; i < 90; i++) {
    await Future<void>.delayed(const Duration(seconds: 2));
    final texts = visibleTexts();
    if (expectTitle.isNotEmpty) {
      final hit = texts.where((t) => t.$1.contains(expectTitle)).toList();
      if (hit.isNotEmpty && !openedDetail) {
        report('home');
        print('OMIOMY step conversation listed');
        await tap(hit.first.$2);
        openedDetail = true;
        await Future<void>.delayed(const Duration(seconds: 4));
        report('detail');
        print('OMIOMY step detail opened');
        continue;
      }
      if (openedDetail) {
        report('detail-$i');
        if (i % 5 == 0) print('OMIOMY done');
        continue;
      }
    }
    for (final g in gates) {
      final hit = texts.where((t) => t.$1 == g).toList();
      if (hit.isNotEmpty) {
        print('OMIOMY tap "$g"');
        await tap(hit.first.$2);
        break;
      }
    }
    if (i % 3 == 0) report('t$i');
  }
}

void main() {
  app.main();
  unawaited(autopilot());
}
