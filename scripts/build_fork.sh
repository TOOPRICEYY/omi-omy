#!/bin/zsh
# Builds Omi's own iPhone app in its "dev" flavor, pointed at this bridge instead
# of Omi's servers. Nothing in Omi's tree is committed: the build tree
# (~/omi-fork by default) is reset to the pinned upstream commit in
# app-fork/UPSTREAM on every run, Omi's own setup.sh steps prepare it, and the
# patches in app-fork/patches/ are applied on top.
#
#   zsh scripts/build_fork.sh                                    # for the public address
#   OMI_APP_BASE=http://127.0.0.1:8387 zsh scripts/build_fork.sh  # simulator against a local test bridge
#
# The app reaches the bridge at <OMI_APP_BASE>/app/<OMIBRIDGE_APP_SECRET>/ (the secret
# comes from .env and ends up inside the app; it is never printed). Firebase's sign-in
# emulator calls go to the same host over plain http: port 80 for an https base,
# otherwise the base's own port.
#
# Env: OMI_APP_BASE (default https://omi.gabrielprice.com), OMI_SIM_DEVICE (simulator
# name or id, default "iPhone 17"), OMI_ENV_FILE (default .env), OMI_FORK_DIR, FLUTTER_BIN.
# Prints the built Runner.app path on the last line.
set -eu -o pipefail
HERE=${0:A:h:h}
PIN=$(head -1 "$HERE/app-fork/UPSTREAM")
FLUTTER_VERSION=$(sed -n 's/^FLUTTER=//p' "$HERE/app-fork/UPSTREAM")
FORK=${OMI_FORK_DIR:-$HOME/omi-fork}
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
if [[ $HOSTPORT == *:* ]]; then PORT=${HOSTPORT##*:}; else PORT=80; fi
[[ $SCHEME == https ]] && PUBLIC_HOST=$HOST || PUBLIC_HOST=
export OMI_BUILD_API="$BASE/app/$SECRET/"

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

# 3. Build.
flutter pub get >/dev/null
(cd ios && pod install >/dev/null 2>&1)
dart run build_runner build --delete-conflicting-outputs >/dev/null
DEFINES=(--dart-define=OMI_API_BASE_URL="$OMI_BUILD_API"
         --dart-define=OMI_FIREBASE_AUTH_EMULATOR_HOST="$HOST"
         --dart-define=OMI_FIREBASE_AUTH_EMULATOR_PORT="$PORT")
[[ -n $PUBLIC_HOST ]] && DEFINES+=(--dart-define=OMI_DEV_PUBLIC_API_HOST="$PUBLIC_HOST")
flutter build ios --simulator --debug --flavor dev -d "$DEVICE" "${DEFINES[@]}" 2>&1 | grep -v -F "$SECRET"
echo "$FORK/app/build/ios/iphonesimulator/Runner.app"
