#!/bin/zsh
# Builds Omi's own iPhone app in its "dev" flavor, pointed at this bridge instead
# of Omi's servers. Nothing in Omi's tree is committed: the build tree
# (~/Documents/projects/omi-omy/app-build by default, OMI_FORK_DIR to change) is reset to the pinned upstream commit in
# app-fork/UPSTREAM on every run, Omi's own setup.sh steps prepare it, and the
# patches in app-fork/patches/ are applied on top.
#
#   zsh scripts/build_fork.sh                                    # for the public address
#   OMI_APP_BASE=http://127.0.0.1:8387 zsh scripts/build_fork.sh  # simulator against a local test bridge
#
# The app reaches the bridge at <OMI_APP_BASE>/app/<OMIBRIDGE_APP_SECRET>/ (the secret
# comes from .env and ends up inside the app; it is never printed). Firebase's sign-in
# emulator calls go to the same host: over https on port 443 for an https base (the
# Firebase SDK is patched for that by app-fork/pod_patches.py), otherwise plain http
# on the base's own port (simulator tests on loopback).
#
# Env: OMI_APP_BASE (default https://omi.gabrielprice.com), OMI_SIM_DEVICE (simulator
# name or id, default "iPhone 17"), OMI_ENV_FILE (default .env), OMI_FORK_DIR, FLUTTER_BIN.
# OMI_FORK_AUTOPILOT=1 builds the test entry point in app-fork/tests/ instead, which taps through
# the app by itself (OMIOMY_EXPECT_TITLE names a conversation the bridge holds);
# OMI_FORK_SKIP_PREP=1 reuses the prepared tree.
#
# Device / TestFlight build (OMI_FORK_DEVICE=1): signs with your Apple team as
# OMI_BUNDLE_ID (default com.gabrielprice.omiomy, display name "Omi Omy"), archives, and
# with OMI_FORK_UPLOAD=1 uploads to TestFlight. Needs OMI_APPLE_TEAM, and the App Store
# Connect API key as ASC_KEY_PATH, ASC_KEY_ID, ASC_ISSUER_ID (not stored in this repo).
# Prints the built Runner.app path on the last line.
set -eu -o pipefail
HERE=${0:A:h:h}
PIN=$(head -1 "$HERE/app-fork/UPSTREAM")
FLUTTER_VERSION=$(sed -n 's/^FLUTTER=//p' "$HERE/app-fork/UPSTREAM")
FORK=${OMI_FORK_DIR:-$HOME/Documents/projects/omi-omy/app-build}
BASE=${OMI_APP_BASE:-https://omi.gabrielprice.com}
BASE=${BASE%/}
DEVICE=${OMI_SIM_DEVICE:-iPhone 17}
FLUTTER_BIN=${FLUTTER_BIN:-$HOME/fvm/versions/$FLUTTER_VERSION/bin}
export PATH="$FLUTTER_BIN:/opt/homebrew/bin:$PATH" LANG=en_US.UTF-8
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
[[ -x $FLUTTER_BIN/flutter ]] || { echo "Flutter $FLUTTER_VERSION not found; run: fvm install $FLUTTER_VERSION"; exit 1; }

# The secret, read with the bridge's own .env parser, kept out of argv and output.
ENV_FILE=${OMI_ENV_FILE:-$HERE/.env}
SECRET=$(cd "$HERE" && .venv/bin/python -I -c 'import sys; sys.path.insert(0, "."); from pathlib import Path
from omibridge.config import parse_env_file; print(parse_env_file(Path(sys.argv[1])).get("OMIBRIDGE_APP_SECRET", ""))' "$ENV_FILE")
[[ ${#SECRET} -ge 32 ]] || { echo "No OMIBRIDGE_APP_SECRET in $ENV_FILE; run scripts/install_mac.sh"; exit 1; }
SCHEME=${BASE%%://*}
HOSTPORT=${BASE#*://}; HOSTPORT=${HOSTPORT%%/*}
HOST=${HOSTPORT%%:*}
if [[ $HOSTPORT == *:* ]]; then PORT=${HOSTPORT##*:}; elif [[ $SCHEME == https ]]; then PORT=443; else PORT=80; fi
[[ $SCHEME == https ]] && PUBLIC_HOST=$HOST || PUBLIC_HOST=
export OMI_BUILD_API="$BASE/app/$SECRET/"

# Flutter hands the device straight to xcodebuild as an id, so resolve a simulator name to its UDID.
if [[ ! $DEVICE =~ '^[0-9A-F-]{36}$' ]]; then
  DEVICE=$(xcrun simctl list devices available -j | /usr/bin/python3 -c 'import json,sys
name=sys.argv[1]; devs=[d for r in json.load(sys.stdin)["devices"].values() for d in r if d["name"]==name]
devs.sort(key=lambda d: d["state"]!="Booted"); print(devs[0]["udid"] if devs else "")' "$DEVICE")
  [[ -n $DEVICE ]] || { echo "No simulator named ${OMI_SIM_DEVICE:-iPhone 17}"; exit 1; }
fi

if [[ -z ${OMI_FORK_SKIP_PREP:-} ]]; then
# 1. Upstream at the pinned commit. The tree belongs to this script: it is reset every run.
if [[ ! -d $FORK/.git ]]; then
  git clone --filter=blob:none https://github.com/BasedHardware/omi.git "$FORK"
  touch "$FORK/.omi-omy-build"
fi
[[ -f $FORK/.omi-omy-build ]] || { echo "$FORK was not made by this script; refusing to reset it."; exit 1; }
cd "$FORK"
git cat-file -e "$PIN^{commit}" 2>/dev/null || git fetch --filter=blob:none origin "$PIN"
git checkout -q --detach -f "$PIN"
git reset -q --hard "$PIN"
git clean -q -fdx app/lib app/ios/Flutter app/ios/Runner app/ios/Config app/android/app/src 2>/dev/null || true
for p in "$HERE"/app-fork/patches/*.patch(N); do
  echo "applying ${p:t}"
  git apply --3way --whitespace=nowarn "$p" 2>/dev/null || git apply --whitespace=nowarn "$p"
done
# New files (no conflicts possible): copied over the tree, e.g. lib/omiomy_silence_gate.dart.
cp -R "$HERE/app-fork/overlay/." app/

# 2. Omi's own setup steps for the dev flavor (setup.sh is bash; its main block only runs when executed).
cd app
OMI_DEV_HOST=$HOST bash -c 'set -euo pipefail
  source ./setup.sh >/dev/null
  prepare_mobile_build_env dev "$OMI_BUILD_API"
  setup_firebase
  bash scripts/generate_ios_dev_info_plist.sh
  generate_ios_custom_config Dev omi-dev'
# No analytics or support chat: their keys must be absent, not empty.
sed -i '' -E '/^[[:space:]]*(POSTHOG|INTERCOM)_/d' .dev.env
if [[ -n ${OMI_FORK_DEVICE:-} ]]; then
  : ${OMI_APPLE_TEAM:?set OMI_APPLE_TEAM to your Apple team id}
  /usr/bin/python3 "$HERE/app-fork/device_overlay.py" . "$OMI_APPLE_TEAM" "${OMI_BUNDLE_ID:-com.gabrielprice.omiomy}" "${OMI_APP_NAME:-Omi Omy}"
fi

# 3. Build.
flutter pub get >/dev/null
(cd ios && pod install >/dev/null 2>&1)
# Firebase's emulator mode over https when the port is 443 (fails here if the SDK changed).
/usr/bin/python3 "$HERE/app-fork/pod_patches.py" ios/Pods
dart run build_runner build --delete-conflicting-outputs >/dev/null
if [[ -n ${OMI_FORK_PREPARE_ONLY:-} ]]; then echo "prepared $FORK (no build)"; exit 0; fi
else
  cd "$FORK/app"
fi
# Build defines go through an owner-only file so the secret is in no process's argument list.
DEFINES_FILE=$(mktemp -t omiomy-defines); chmod 600 "$DEFINES_FILE"
trap 'rm -f "$DEFINES_FILE"' EXIT
OMI_PUBLIC_HOST=$PUBLIC_HOST OMI_EMU_HOST=$HOST OMI_EMU_PORT=$PORT /usr/bin/python3 -c 'import json, os, sys
d = {"OMIOMY": "true", "OMI_API_BASE_URL": os.environ["OMI_BUILD_API"],
     "OMI_FIREBASE_AUTH_EMULATOR_HOST": os.environ["OMI_EMU_HOST"], "OMI_FIREBASE_AUTH_EMULATOR_PORT": os.environ["OMI_EMU_PORT"],
     "OMIOMY_EXPECT_TITLE": os.environ.get("OMIOMY_EXPECT_TITLE", ""), "OMIOMY_EXPECT_DETAIL": os.environ.get("OMIOMY_EXPECT_DETAIL", "")}
if os.environ["OMI_PUBLIC_HOST"]: d["OMI_DEV_PUBLIC_API_HOST"] = os.environ["OMI_PUBLIC_HOST"]
json.dump(d, open(sys.argv[1], "w"))' "$DEFINES_FILE"
DEFINES=(--dart-define-from-file="$DEFINES_FILE")
TARGET=()
if [[ -n ${OMI_FORK_AUTOPILOT:-} ]]; then
  # Test build: the autopilot entry point taps through the app by itself (see app-fork/tests/).
  cp "$HERE"/app-fork/tests/omi_omy_autopilot.dart integration_test/
  TARGET=(-t integration_test/omi_omy_autopilot.dart)
fi
if [[ -n ${OMI_FORK_DEVICE:-} ]]; then
  : ${ASC_KEY_PATH:?} ${ASC_KEY_ID:?} ${ASC_ISSUER_ID:?}
  AUTH=(-allowProvisioningUpdates -authenticationKeyPath "$ASC_KEY_PATH" -authenticationKeyID "$ASC_KEY_ID"
        -authenticationKeyIssuerID "$ASC_ISSUER_ID")
  BUILD_NUMBER=${OMI_BUILD_NUMBER:-$(date +%Y%m%d%H%M)}
  flutter build ios --release --flavor dev --no-codesign --build-number="$BUILD_NUMBER" "${DEFINES[@]}" 2>&1 | grep -v -F "$SECRET" | tail -5
  ARCHIVE=build/omiomy.xcarchive
  rm -rf "$ARCHIVE" build/omiomy-export
  # Archive with the Apple ID signed in to Xcode (Settings > Accounts): with Xcode 27 the API key
  # alone could not create this app's identifiers and profiles. The key is used for the upload.
  xcodebuild -workspace ios/Runner.xcworkspace -scheme dev -configuration Release-dev -destination 'generic/platform=iOS' \
    -archivePath "$ARCHIVE" -allowProvisioningUpdates archive 2>&1 | grep -v -F "$SECRET" | grep -E "error:|warning: .*[Pp]rovision|ARCHIVE (SUCCEEDED|FAILED)|Signing" | tail -20
  [[ -d $ARCHIVE ]] || { echo "archive failed"; exit 1; }
  # Flutter writes the defines (secret included) into Generated.xcconfig; keep it owner-only.
  chmod 600 ios/Flutter/Generated.xcconfig ios/Flutter/flutter_export_environment.sh 2>/dev/null || true
  if [[ -n ${OMI_FORK_UPLOAD:-} ]]; then
    xcodebuild -exportArchive -archivePath "$ARCHIVE" -exportPath build/omiomy-export \
      -exportOptionsPlist "$HERE/app-fork/ExportOptions-TestFlight.plist" "${AUTH[@]}" 2>&1 \
      | grep -E "error|Upload|EXPORT (SUCCEEDED|FAILED)|uploaded|Progress" | tail -20
  fi
  echo "$FORK/app/$ARCHIVE (build $BUILD_NUMBER)"
  exit 0
fi
flutter build ios --simulator --debug --flavor dev -d "$DEVICE" "${DEFINES[@]}" "${TARGET[@]}" 2>&1 | grep -v -F "$SECRET"
echo "$FORK/app/build/ios/iphonesimulator/Runner.app"
