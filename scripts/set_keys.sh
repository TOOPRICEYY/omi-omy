#!/bin/zsh
# Saves OPENROUTER_API_KEY into .env without it showing on screen, in shell history or in `ps`.
#   zsh scripts/set_keys.sh
# Then restart the bridge: zsh bridgectl.sh restart
set -eu
cd "${0:A:h}/.."
[[ -f .env ]] || { echo "No .env yet; run scripts/install_mac.sh first."; exit 1; }
read -rs "KEY?OpenRouter API key (input hidden): "; echo
[[ -n "$KEY" ]] || { echo "Nothing entered; .env unchanged."; exit 1; }
# The key goes to Python on stdin, never as an argument.
print -rn -- "$KEY" | .venv/bin/python -I -c '
import os, sys, tempfile
key = sys.stdin.read().strip()
lines = open(".env").read().splitlines()
out, done = [], False
for l in lines:
    if l.split("=", 1)[0].strip() == "OPENROUTER_API_KEY":
        out.append("OPENROUTER_API_KEY=" + key); done = True
    else:
        out.append(l)
if not done:
    out.append("OPENROUTER_API_KEY=" + key)
fd, tmp = tempfile.mkstemp(dir=".")
with os.fdopen(fd, "w") as f:
    f.write("\n".join(out) + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, ".env")
'
unset KEY
echo "Saved OPENROUTER_API_KEY to .env (owner-only). Restart the bridge: zsh bridgectl.sh restart"
