#!/bin/zsh
# Control the Omi bridge and its Cloudflare tunnel (both run as login agents).
#   zsh bridgectl.sh status | start | stop | restart | logs
set -u
HERE=${0:A:h}
AGENTS=(local.omi-bridge.server local.omi-bridge.tunnel)
D="gui/$(id -u)"
HOST=omi.gabrielprice.com

case "${1:-status}" in
  start)
    mkdir -p "$HERE/logs" ~/Library/LaunchAgents
    for a in $AGENTS; do
      [[ -f ~/Library/LaunchAgents/$a.plist ]] || sed "s|__ROOT__|$HERE|g" "$HERE/scripts/$a.plist" > ~/Library/LaunchAgents/$a.plist
      launchctl bootstrap $D ~/Library/LaunchAgents/$a.plist 2>/dev/null || launchctl kickstart $D/$a
    done
    echo "started" ;;
  stop)
    for a in $AGENTS; do launchctl bootout $D/$a 2>/dev/null; done
    pkill -f "$HERE/.venv/bin/python -m omibridge"; echo "stopped (stays stopped until 'start' or next login)" ;;
  restart)
    zsh "${0:A}" stop >/dev/null; sleep 2; zsh "${0:A}" start ;;
  logs)
    grep -v '"GET /api' "$HERE/logs/server.log" 2>/dev/null | tail -n 20; echo "--- tunnel"; tail -n 5 "$HERE/logs/tunnel.log" 2>/dev/null ;;
  status|*)
    for a in $AGENTS; do
      if launchctl print $D/$a >/dev/null 2>&1; then printf '%-26s loaded, %s\n' $a "$(launchctl print $D/$a | awk -F'= ' '/^\tstate/ {print $2; exit}')"; else printf '%-26s not loaded\n' $a; fi
    done
    printf 'local health:  '; curl -s -m 5 http://127.0.0.1:8378/health || echo "unreachable"; echo
    printf 'public, wrong token (expect 403): '; curl -s -m 10 -o /dev/null -w '%{http_code}\n' -X POST "https://$HOST/omi/audio/wrong-token?uid=t&sample_rate=16000" --data-binary @/dev/null
    printf 'public / (expect 404, or 200 when the app API is on): '; curl -s -m 10 -o /dev/null -w '%{http_code}\n' "https://$HOST/"
    printf 'public app API, wrong secret (expect 404): '; curl -s -m 10 -o /dev/null -w '%{http_code}\n' "https://$HOST/app/wrong-secret/v1/health"
    printf 'public /api/conversations (expect 404): '; curl -s -m 10 -o /dev/null -w '%{http_code}\n' "https://$HOST/api/conversations" ;;
esac
