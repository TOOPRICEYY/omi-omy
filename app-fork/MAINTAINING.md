# Maintaining the omi-omy build of the Omi app

This file is for whoever (person or agent) has to keep the omi-omy build of Omi's
iPhone app working against this repo's bridge, most often when moving it to a newer
Omi release. It assumes the Mac mini setup described in the README: this repo at
`~/omi-omy`, Flutter through fvm, Xcode, CocoaPods.

## Purpose and design

Omi's own app (github.com/BasedHardware/omi, directory `app/`) talks to Omi's servers
for sign-in, live transcription, offline sync, conversations, memories, chat and so
on. The owner wants a transcribe-and-summarize app that depends on no Omi server:
audio from the pendant goes to the bridge in this repo (`omibridge/`), which cuts it
into conversations, transcribes it, names the speakers it knows, and writes a
summary.

The build does not fork Omi's code into this repo. `scripts/build_fork.sh` checks out
a pinned upstream commit into a scratch tree (`~/omi-fork`), applies the
modifications kept here in `app-fork/`, and builds Omi's own **dev flavor**. The dev
flavor already avoids Omi's production services: it uses a dummy Firebase project
(`demo-omi-local`), signs in through the Firebase Auth *emulator* protocol, and
reads its API address from build-time defines. Everything Omi-specific that the
bridge does not provide is either answered with 404 (which the app reads as "none")
or hidden in the UI.

The app reaches the bridge at `https://omi.gabrielprice.com/app/<secret>/`, through
the same Cloudflare tunnel as the Omi webhook (port 8377 on the Mac). The secret is
`OMIBRIDGE_APP_SECRET` from `.env`; it is baked into the app at build time and is the
real credential. Nothing in the scratch tree or in this repo contains it.

## The backend contract

All of this is implemented in `omibridge/appapi.py` and `omibridge/appauth.py`, and
pinned down by `tests/test_appapi.py`. Paths below are relative to the API base
`/app/<secret>/` unless they start with a host name.

**Auth model.** A request is accepted only if it carries the path secret (checked in
constant time by `PublicGate`; a wrong secret falls through to the same plain 404 any
unknown path gets) and, for everything except the sign-in bootstrap and `/v1/health`,
an `Authorization: Bearer <ID token>` made by this bridge for `OMIBRIDGE_APP_UID`.
Tokens are HS256 JWTs keyed from the secret. Failed checks made without the secret
(a wrong secret, a wrong webhook token, refused sign-in tokens at the root) count per
client address (`CF-Connecting-IP`); 30 in ten minutes block that address for 15
minutes. A request under the right secret is never counted and never blocked, so a
phone with a stale token cannot lock itself out while it is recording. Bodies are capped (128 MB for sync
uploads, 1 MB for the webhook, 256 KB otherwise).

**Sign-in.** The app's local-dev sign-in button calls
`POST v1/auth/local-dev/custom-token` (form field `uid`, ignored) and gets
`{"custom_token": ...}` (valid 60 s, single use). It hands that to the Firebase
SDK, which talks to the "emulator" at the root of the public host:
`POST /www.googleapis.com/identitytoolkit/v3/relyingparty/verifyCustomToken`,
`.../getAccountInfo`, `.../setAccountInfo` (the iOS SDK 11.x names; the v1
`/identitytoolkit.googleapis.com/v1/accounts:*` names are served too) and
`POST /securetoken.googleapis.com/v1/token` (refresh, JSON or form). Before signing
in the app probes `GET /` on the emulator host. Refresh tokens never expire; rotating
the secret signs every device out. If refresh failed, the app would show its sign-in
screen again, so never return 401 from the API for anything but a bad token.

**Routes the app needs** (everything else is 404):

- `GET`/`HEAD v1/health` → `{"status":"ok"}` (dev connectivity probe, no token). The probe uses HEAD; a 405 here makes the app think it is offline and it sits on a spinner.
- `GET v1/users/onboarding` → `{"completed":true,"acquisition_source":"","device_onboarding_completed":true}`; without it the app runs Omi's onboarding. `PATCH` → 200.
- `GET v1/account/cutover/control` → the legacy-allow object in `CUTOVER_CONTROL`. A 503 or malformed body here puts up a maintenance screen; a 404 also works.
- `GET v1/conversations?limit&offset&statuses&include_discarded&start_date&end_date&folder_id&starred` → a bare JSON array. `statuses=in_progress` is polled while recording and must return `[]`.
- `GET v1/conversations/{id}` → one conversation; queued-but-unprocessed ones come back with `status:"processing"`.
- `DELETE v1/conversations/{id}?cascade=true` → **204** (the folder moves to `data/bridge/trash`).
- `POST v1/conversations` with `{}` (the stop button) → 200 `{"conversation": ..., "messages": []}`, or 404 when nothing is being recorded (benign for the app).
- `GET v1/users/people` → enrolled people `{id, name, created_at, updated_at, speech_samples}`; transcript lines link to them by `person_id` (`p-<slug>`).
- `WS v4/listen?codec&sample_rate&uid&language&...` (Authorization header). Binary messages are single raw codec packets (opus 10 ms, opus_fs320 20 ms, pcm8/pcm16), no header. Text messages (`client_state`, `capture_evidence_frame`, `suggested_transcript`, `"ping"`) are ignored. The bridge sends `{"type":"service_status","status":"ready"}` on connect, `{"type":"memory_processing_started","memory":<stub>}` when a conversation is cut, and `{"type":"memory_created","memory":<conversation>,"messages":[]}` when it is processed; `{"type":"last_memory","memory_id":...}` on the next connect if the phone was away. No live transcript segments are sent; the recording screen shows an empty live transcript. If `memory_created` never arrives the phone keeps its audio log as unsynced and uploads it later through offline sync, which the bridge then de-duplicates.
  - Silence gate (the one message the bridge reads besides `"ping"`): a phone that leaves quiet audio out sends `{"type":"omiomy_silence_gap","ms":N}` just before the first packet after the pause, where N is the duration of the packets it did not send. The bridge puts that much silence back into the open conversation (at most the idle timeout; nothing if no conversation is open) and marks the time as heard for the offline-sync de-duplication, so audio, timestamps and conversation cuts are the same with the gate on or off. Independently, when no audio arrives for a second the bridge hands on the partial chunk it holds. A bridge without this ignores the note and only loses the pause lengths.
- `POST v2/sync-local-files` (multipart field `files`, names ending `_<unix s or ms>.bin`; each file is repeated `[uint32 LE length][packet]`) → 202 `{job_id,status,total_files,total_segments,poll_after_ms,lane}`; `GET v2/sync-local-files/{job_id}` → `{status:"completed", result:{new_memories,updated_memories,failed_segments:0,total_segments,errors}}`. 400 for bad names (terminal), 404 for an unknown job (the app re-uploads). Spans the bridge already heard live (or in an earlier upload) are skipped.

**Conversation shape** (`app/lib/backend/schema/gen/conversation_wire.g.dart`):
required `id`, `created_at` (ISO), `structured` (object); `started_at` and
`finished_at` must be present (may be null). Segments need `text`, `start`, `end`,
`is_user`. `structured` = `title`, `overview`, `emoji`, `category` (one of
`omibridge/summarize.py` `CATEGORIES`), `action_items` (`{"description","completed"}`),
`events`. The mapping lives in `appapi.to_omi`.

## The modifications

Applied by `scripts/build_fork.sh` in this order. Each one says how to redo it by
hand if it no longer applies.

1. **`app-fork/patches/0001-allow-public-dev-backend.patch`** (`app/lib/env/env.dart`, +5 lines).
   Intent: the dev profile only accepts loopback, private-network and Tailscale API
   addresses (`Env._isLocalDevelopmentApi`, called from `validateStartupRouting`).
   The patch adds the define `OMI_DEV_PUBLIC_API_HOST` and lets exactly that host
   through over https. By hand: in `_isLocalDevelopmentApi`, after the localhost
   check, return true when `uri.scheme == 'https'` and the host equals the define.
   Verify: the app starts instead of failing with "Profile local_dev requires a
   loopback or private-network API endpoint".

2. **`app-fork/patches/0002-https-emulator-probe.patch`** (`app/lib/services/auth_service.dart`, 1 line).
   Intent: `signInWithLocalDevToken` probes the emulator with a plain-http GET before
   signing in; on port 443 it must use https. By hand: in that probe's `Uri(...)`, use
   scheme `https` when `Env.firebaseAuthEmulatorPort == 443`. Verify: sign-in does not
   fail with "Cannot reach the local Firebase Auth emulator".

3. **`app-fork/patches/0003-transcribe-summarize-mode.patch`** (new `app/lib/omiomy.dart`; `backend/preferences.dart`, `pages/home/page.dart`, `pages/settings/settings_drawer.dart`, `widgets/conversation_bottom_bar.dart`, `services/app_review_service.dart`; about 27 lines).
   Intent: `omiomyBuild` (define `OMIOMY=true`) skips Omi's AI-vendor consent screen
   (`aiConsentGiven` getter; its text names Deepgram and OpenAI, which does not apply),
   hides the Home tab switcher (Tasks) and the chat bar (the record button stays), hides
   the settings rows for upgrade, plan and usage, referral, assistant voice,
   integrations, connect an agent, memories and goals (in `_row`), hides the "Ask Omi"
   bar and button on the conversation page (`_buildAskBar`, `_buildAskButton`), and
   turns off the App Store rating prompt (`AppReviewService.requestReview`). By hand:
   find those widgets and methods by these names; the intent matters more than the
   exact lines. Verify: after sign-in the app lands on the conversation list with no
   tabs and no chat bar, and a conversation opens with Summary and Transcript tabs and
   no Ask Omi bar.

4. **`app-fork/pod_patches.py`** (FirebaseAuth pod, run after `pod install`).
   Intent: the Firebase iOS SDK builds emulator URLs with `http:`. When the emulator
   port is 443 they must use https (two lines in
   `Backend/IdentityToolkitRequest.swift`, one in `Backend/RPC/SecureTokenRequest.swift`).
   The script requires each expected line to match exactly and stops the build
   otherwise. By hand: in the new SDK, find where `emulatorHostAndPort` picks the
   protocol and make it `https` for `:443`. Verify: the bridge log shows the
   `verifyCustomToken` and `token` requests arriving (over the tunnel they arrive as
   HTTPS); nothing is requested from the public host over plain http.

5. **Build-time settings** in `build_fork.sh`: `.dev.env` without `POSTHOG_*` or
   `INTERCOM_*` keys (analytics and support chat stay off; the keys must be absent,
   not empty), and the defines `OMIOMY=true`, `OMI_API_BASE_URL=<base>/app/<secret>/`,
   `OMI_FIREBASE_AUTH_EMULATOR_HOST=<host>`, `OMI_FIREBASE_AUTH_EMULATOR_PORT=443`
   (or the loopback port), `OMI_DEV_PUBLIC_API_HOST=<host>`.

6. **Phone-side silence gate**: `app-fork/overlay/lib/omiomy_silence_gate.dart` (the logic),
   `app-fork/overlay/lib/omiomy_pendant_gate.dart` (the glue), copied into `app/lib/` by the
   build script, and `app-fork/patches/0004-silence-gate-hook.patch` (`capture_controller.dart`
   and `settings_groups.dart`, +34 lines).
   Intent: while nobody speaks, hold pendant Opus packets instead of sending them; at speech
   onset send `{"type":"omiomy_silence_gap","ms":N}` (N = audio dropped, not the pre-roll),
   then the last second of held packets, then live packets; keep sending 3 s after the last
   speech. The bridge refills the gap with silence, so the output does not change. Detector:
   the app's own `SileroMobileBatchVad` (`services/wals/mobile_batch_vad.dart`), threshold 0.3,
   on Opus 10 ms and 20 ms frames only. Fails open on any detector error, a backlog over 300
   packets, a missing model or a non-Opus codec. Off by default; Settings > Recording &
   transcription > "Skip silence on the phone" (preference `omiomySilenceGate`; the define
   `OMIOMY_SILENCE_GATE=true` changes the default).
   Hooks: `CaptureController._sendPendantToSocket` asks `OmiomyPendantGate.instance.offer`
   first and otherwise sends as before (the original body is now `_sendPendantToSocketNow`).
   In `_confirmSessionTranscript`, the wall-clock spans the gate dropped
   (`OmiomyDroppedSpans`) are added to the transcript spans, so the app's audio log counts
   that silence as covered and does not upload it again through offline sync. Silence with
   no conversation at all was never covered by a transcript and is still uploaded later,
   gate or not (Omi's own behaviour); the bridge skips it as already heard.
   Checked in the source: the socket stays open with nothing sent (WebSocket pings every
   20 s from both ends, no idle timeout in `pure_socket.dart`); the capture watchdog's
   "bytes sent, no transcript" rule is telemetry only, and its "no bytes" rule watches
   Bluetooth input, which the gate does not touch. Turning the switch off mid-session drops
   up to one second of held silence without a gap note (a sub-second timing shift).
   By hand: wrap the send function so packets pass through the gate first; keep the gap note
   in the same socket and in order before the first packet after a pause.
   Verify: `flutter test test/omiomy_silence_gate_test.dart` in the build tree (onset with
   pre-roll, hangover, exact gap ms, 20 ms frames, fail open, backlog). End to end it needs a
   pendant; with the switch on, the bridge log shows fewer packets and the same conversation.


## Building

    zsh scripts/build_fork.sh                       # for the phone: https://omi.gabrielprice.com
    OMI_APP_BASE=http://127.0.0.1:8387 OMI_ENV_FILE=<test env> zsh scripts/build_fork.sh   # simulator vs a test bridge

The script resets `~/omi-fork` (it refuses to touch a tree without its
`.omi-omy-build` marker), applies the patches, runs Omi's own `setup.sh` dev steps
(`prepare_mobile_build_env`, `setup_firebase`, `generate_ios_dev_info_plist.sh`,
`generate_ios_custom_config`), `pod install`, `pod_patches.py`, `build_runner`, and
`flutter build ios --simulator --debug --flavor dev`. The simulator needs a UDID; the
script resolves `OMI_SIM_DEVICE` (default "iPhone 17").

**Device / TestFlight build.** `OMI_FORK_DEVICE=1` (plus `OMI_FORK_UPLOAD=1` to send it
to TestFlight) runs `app-fork/device_overlay.py` after Omi's setup steps. It replaces Omi's
Apple team with `OMI_APPLE_TEAM` in every target, sets the dev flavor's bundle id and name
(`OMI_BUNDLE_ID`, default `com.gabrielprice.omiomy`; `OMI_APP_NAME`, default "Omi Omy"; the
watch app and widget follow as `.watchapp` and `.widget`), points Firebase's
`GoogleService-Info.plist` at that bundle id, renames the app group to `group.<bundle id>`
in the entitlements and the Swift sources that use it, and drops entitlements the build does
not use (push, Sign in with Apple, Omi's associated domains, HealthKit). Every edit must
match or the build stops. The script then runs `flutter build ios --release --no-codesign`,
`xcodebuild archive` with automatic signing (`-allowProvisioningUpdates` and the App Store
Connect key from `ASC_KEY_PATH`, `ASC_KEY_ID`, `ASC_ISSUER_ID`), and, when asked, uploads with
`app-fork/ExportOptions-TestFlight.plist`. Build numbers are `yyyymmddHHMM`. All defines,
the secret included, go through an owner-only file passed with `--dart-define-from-file`;
Flutter also writes them into `ios/Flutter/Generated.xcconfig`, which the script makes
owner-only.

Known limit (2026-11-11): with Xcode 27 the API key alone did not let `xcodebuild` create
the app's identifiers and profiles ("Authentication failed"), although the same key works
against the App Store Connect API. Archiving needed an Apple ID signed in to Xcode
(Settings > Accounts) on the Mac mini; uploading works with the key. A new app also needs
its App Store Connect record created by hand (Apps > + > New App) before the first upload;
Apple's API cannot create it.

## Checking a build

1. `.venv/bin/python -m pytest -q tests` (contract tests, no models needed).
2. Start a test bridge on scratch data and other ports (never the running service):
   an env file with `OMIBRIDGE_TOKEN`, `OMIBRIDGE_APP_SECRET`, `OMIBRIDGE_DATA=<scratch>`,
   `OMIBRIDGE_PORT=8387`; import a sample through `POST http://127.0.0.1:8388/api/import`.
3. Build with `OMI_FORK_AUTOPILOT=1 OMIOMY_EXPECT_TITLE=<word in the title>
   OMIOMY_EXPECT_DETAIL=<word in its summary>` and run it in the simulator.
   `app-fork/tests/omi_omy_autopilot.dart` boots the real app, taps the sign-in button,
   skips Omi's permissions step (it waits on iOS's notification prompt, which the
   simulator cannot answer; a person taps Allow there), opens the conversation, and logs
   `OMIOMY ...` lines (read them with `xcrun simctl spawn <udid> log stream
   --predicate 'eventMessage CONTAINS "OMIOMY"'`). Take screenshots with
   `xcrun simctl io <udid> screenshot`. The bridge log should show, in order:
   `custom-token`, `verifyCustomToken`, `getAccountInfo`, `token`, `v1/users/onboarding`,
   `v1/conversations` (the bridge logs one `app METHOD path -> status` line per request).
   The Firebase session lives in the simulator's keychain and survives reinstalling the
   app, so later runs start signed in; erase the simulator to test sign-in again.
4. The live socket needs a real WebSocket server (uvicorn uses the `websockets`
   package; the in-process test client does not). Stream a recording through
   `/app/<secret>/v4/listen` as Opus packets, press stop (`POST v1/conversations`), and
   expect `memory_processing_started` then `memory_created` with transcript lines.

## Moving to a new Omi release

1. Pick the new upstream commit. Read Omi's changes since the pinned one
   (`git log -p <old>..<new> -- app/lib backend/routers`), looking at: the listen
   socket (`app/lib/services/sockets/transcription_service.dart`, `message_event.dart`,
   `capture_controller.dart` event handling), the conversation model
   (`schema/gen/conversation_wire.g.dart`), sync uploads
   (`backend/http/api/conversations.dart` `uploadLocalFilesV2`, `services/wals/`),
   sign-in (`services/auth_service.dart` local-dev path, `lib/main.dart` emulator
   setup), the env guards (`lib/env/env.dart`, `environment_profile.dart`), and any
   new calls made at startup that would now get a 404 (check `core/app_shell.dart`
   and the home page's init).
2. Put the commit on line 1 of `app-fork/UPSTREAM`, and the Flutter version from
   `.github/workflows/mobile-app-checks.yml` in `FLUTTER=`; `fvm install` it.
3. Run the build. If a patch fails to apply, redo it by hand as described above and
   regenerate it with `git diff` in `~/omi-fork`. If `pod_patches.py` stops the build,
   update its expected lines for the new FirebaseAuth version.
4. If the app's API changed, change `omibridge/appapi.py` and the contract tests first.
5. Run the checks above; only then rebuild for the phone.

## Known risks

- The app's API is Omi's private one and changes often; pin, and update on purpose.
- A startup request that newly needs a real answer will show up as an empty or stuck
  screen, not an error. Watch the bridge log for 404s on new paths after an update.
- The FirebaseAuth emulator path is a debug feature of the SDK; Firebase could change
  or drop it. `pod_patches.py` catches changes to the lines it edits, not changes in
  behaviour.
- The app relies on the Mac mini and the tunnel. During an outage the app should keep
  the audio log on the phone and upload it through offline sync later; this follows
  from Omi's code (`capture_controller.dart`, WAL handling) but has not been tested
  end to end yet.
- Crashlytics is initialised against the dummy Firebase project; reports go nowhere.
- Anyone with the built app has the secret. Rotate `OMIBRIDGE_APP_SECRET` and rebuild
  if a phone is lost.
