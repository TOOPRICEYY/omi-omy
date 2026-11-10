#!/bin/zsh
# One-time setup: publishes ONLY the bridge's webhook port (8377) at https://omi.gabrielprice.com
# through its own Cloudflare tunnel, then starts the bridge and the tunnel as login agents.
# Run it yourself in Terminal:
#   zsh ~/omi-omy/setup-tunnel.sh
# Safe to re-run: it reuses the tunnel and DNS record if they already exist.
set -eu
HERE=${0:A:h}
NAME=omi-bridge
HOST=omi.gabrielprice.com
CF=/opt/homebrew/bin/cloudflared
PY=$HERE/.venv/bin/python
[[ -x $PY && -f $HERE/.env ]] || { echo "Run scripts/install_mac.sh first."; exit 1; }

if ! $CF tunnel info $NAME >/dev/null 2>&1; then
  echo "Creating Cloudflare tunnel '$NAME'..."
  $CF tunnel create $NAME
fi
ID=$($CF tunnel list --name $NAME --output json | $PY -I -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')
echo "Tunnel id: $ID"
echo "Pointing $HOST at the tunnel..."
$CF tunnel route dns $NAME $HOST 2>&1 | tail -1 || true

cat > "$HERE/cloudflared.yml" <<YML
# Cloudflare tunnel for the Omi bridge (written by setup-tunnel.sh).
# Publishes ONLY the webhook port 8377. Never add 8378 (review page and transcripts).
tunnel: $ID
credentials-file: $HOME/.cloudflared/$ID.json
no-autoupdate: true
ingress:
  - hostname: $HOST
    service: http://127.0.0.1:8377
    originRequest:
      connectTimeout: 10s
  - service: http_status:404
YML
$CF tunnel --config "$HERE/cloudflared.yml" ingress validate

# The URL to paste into the Omi app (contains the secret token): owner-only file, git-ignored.
( umask 077; $PY -I -c '
import sys
tok = [l.split("=", 1)[1].strip() for l in open(sys.argv[1]) if l.startswith("OMIBRIDGE_TOKEN=")][0]
print(f"https://{sys.argv[2]}/omi/audio/{tok}")' "$HERE/.env" $HOST > "$HERE/WEBHOOK_URL.txt" )
chmod 600 "$HERE/WEBHOOK_URL.txt"

mkdir -p "$HERE/logs"
for a in local.omi-bridge.server local.omi-bridge.tunnel; do
  sed "s|__ROOT__|$HERE|g" "$HERE/scripts/$a.plist" > ~/Library/LaunchAgents/$a.plist
done
zsh "$HERE/bridgectl.sh" restart
echo "Waiting for the models to load and the tunnel to connect..."
sleep 20
zsh "$HERE/bridgectl.sh" status
echo
echo "Webhook URL for Omi (Settings > Developer Mode > Realtime audio bytes) is in: $HERE/WEBHOOK_URL.txt"
