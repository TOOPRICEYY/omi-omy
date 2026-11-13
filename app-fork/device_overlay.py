"""Build-time identity changes for an installable (device / TestFlight) build of the
omi-omy app. Run by scripts/build_fork.sh in device mode, after Omi's setup steps and
before `pod install`. Never committed to Omi's tree.

    python3 app-fork/device_overlay.py <path to omi app/> <team id> <bundle id> <display name> <version> <build>

What it changes, and why:
- The Apple team: Omi's (9536L8KLMP) becomes the owner's, in every target.
- The dev flavor's bundle id, name and display name (ios/Flutter/dev*.xcconfig and the
  hard-coded display names in the project file): the app must not be confusable with
  Omi's own app. The watch app and widget derive their ids from it (.watchapp, .widget).
- Firebase's GoogleService-Info.plist BUNDLE_ID: the native SDK refuses to start when it
  does not match the running app.
- Version and build number in every target: the watch app and widget hard-code 1.0 / 1,
  and App Store Connect refuses a watch app whose version differs from its iPhone app.
- The app group: Omi's group id is registered to Omi's team, so the app, widget and
  Swift code that share data through it get group.<bundle id>.
- Entitlements the owner's build does not use and that would need extra provisioning:
  push (no Omi push service), Sign in with Apple (sign-in is the bridge's), Omi's
  associated domains (h.omi.me, try.omi.me), and HealthKit (the integration is hidden).
  Wi-Fi hotspot and Wi-Fi info stay: the pendant's Wi-Fi sync uses them.

Every edit must find what it expects, or the script stops the build.
"""
import plistlib
import re
import sys
from pathlib import Path

OMI_TEAM = "9536L8KLMP"
OMI_GROUP = "group.com.friend-app-with-wearable.ios12"
DROP_ENTITLEMENTS = ["aps-environment", "com.apple.developer.applesignin",
                     "com.apple.developer.associated-domains", "com.apple.developer.healthkit",
                     "com.apple.developer.healthkit.access"]


def fail(msg: str) -> None:
    sys.exit(f"device_overlay: {msg}. Omi's iOS project changed; update app-fork/device_overlay.py.")


def sub(path: Path, pattern: str, repl: str, min_count: int = 1) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n < min_count:
        fail(f"expected {pattern!r} in {path.name}, found {n}")
    path.write_text(new)


def main(app: Path, team: str, bundle: str, name: str, version: str, build: str) -> None:
    ios = app / "ios"
    group = "group." + bundle
    pbx = ios / "Runner.xcodeproj" / "project.pbxproj"

    sub(pbx, re.escape(f"DEVELOPMENT_TEAM = {OMI_TEAM};"), f"DEVELOPMENT_TEAM = {team};", min_count=5)
    sub(pbx, r'INFOPLIST_KEY_CFBundleDisplayName = (?:Omi|omi|"Omi Dev");', f'INFOPLIST_KEY_CFBundleDisplayName = "{name}";')
    sub(pbx, r"MARKETING_VERSION = [^;]+;", f"MARKETING_VERSION = {version};", min_count=3)
    sub(pbx, r"CURRENT_PROJECT_VERSION = [^;]+;", f"CURRENT_PROJECT_VERSION = {build};", min_count=3)

    for cfg in ("devDebug", "devProfile", "devRelease"):
        p = ios / "Flutter" / f"{cfg}.xcconfig"
        sub(p, r"^APP_BUNDLE_IDENTIFIER=.*$", f"APP_BUNDLE_IDENTIFIER={bundle}")
        sub(p, r"^BUNDLE_NAME=.*$", f"BUNDLE_NAME={name}")
        sub(p, r"^BUNDLE_DISPLAY_NAME=.*$", f"BUNDLE_DISPLAY_NAME={name}")

    for p in (ios / "Runner" / "GoogleService-Info.plist", ios / "Config" / "Dev" / "GoogleService-Info.plist"):
        d = plistlib.loads(p.read_bytes())
        if "BUNDLE_ID" not in d:
            fail(f"no BUNDLE_ID in {p}")
        d["BUNDLE_ID"] = bundle
        p.write_bytes(plistlib.dumps(d))

    ents = sorted((ios / "Runner").glob("Runner*.entitlements")) + [ios / "BatteryWidget" / "BatteryWidget.entitlements"]
    for p in ents:
        d = plistlib.loads(p.read_bytes())
        for k in DROP_ENTITLEMENTS:
            d.pop(k, None)
        if "com.apple.security.application-groups" in d:
            d["com.apple.security.application-groups"] = [group]
        p.write_bytes(plistlib.dumps(d))

    code = [p for p in list(ios.rglob("*.swift")) + list(ios.rglob("*.m")) + list((app / "lib").rglob("*.dart"))
            if "Pods" not in p.parts and OMI_GROUP in p.read_text(errors="ignore")]
    if not code:
        fail(f"no source uses {OMI_GROUP}")
    for p in code:
        p.write_text(p.read_text().replace(OMI_GROUP, group))
    print(f"device_overlay: team {team}, bundle {bundle}, name {name!r}, version {version} ({build}), group {group}, "
          f"{len(ents)} entitlement files, {len(code)} source files")


if __name__ == "__main__":
    main(Path(sys.argv[1]), *sys.argv[2:7])
