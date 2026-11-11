"""Build-time edits to CocoaPods sources, run by scripts/build_fork.sh after `pod install`.

FirebaseAuth only speaks plain http to an Auth emulator. omi-omy's sign-in endpoints
stand in for the emulator on the bridge's public HTTPS address, so when the emulator
port is 443 the SDK must use https. Each edit names the exact text it expects and
fails the build if it is not there exactly as often as expected, so a Firebase SDK
update shows up as a failed build, never as a phone that cannot sign in.

    python3 app-fork/pod_patches.py <path to app/ios/Pods>
"""
import sys
from pathlib import Path

HTTPS_IF_443 = 'emulatorHostAndPort.hasSuffix(":443") ? kHttpsProtocol : kHttpProtocol  // omi-omy'

EDITS = [
    ("FirebaseAuth/FirebaseAuth/Sources/Swift/Backend/IdentityToolkitRequest.swift", [
        ("      if let emulatorHostAndPort = emulatorHostAndPort {\n        apiProtocol = kHttpProtocol\n",
         "      if let emulatorHostAndPort = emulatorHostAndPort {\n        apiProtocol = " + HTTPS_IF_443 + "\n",
         2),
    ]),
    ("FirebaseAuth/FirebaseAuth/Sources/Swift/Backend/RPC/SecureTokenRequest.swift", [
        ('      urlString = "http://\\(emulatorHostAndPort)/\\(gAPIHost)/v1/token?key=\\(apiKey)"\n',
         '      urlString = "\\(emulatorHostAndPort.hasSuffix(":443") ? "https" : "http")'
         '://\\(emulatorHostAndPort)/\\(gAPIHost)/v1/token?key=\\(apiKey)"  // omi-omy\n',
         1),
    ]),
]


def main(pods: Path) -> None:
    for rel, edits in EDITS:
        path = pods / rel
        text = path.read_text()
        for old, new, count in edits:
            if new in text and old not in text:
                continue                      # already applied
            found = text.count(old)
            if found != count:
                sys.exit(f"pod_patches: expected {count} match(es) in {rel}, found {found}. "
                         "The Firebase SDK changed; update app-fork/pod_patches.py.")
            text = text.replace(old, new)
        path.chmod(0o644)
        path.write_text(text)
        print(f"pod_patches: patched {rel}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
