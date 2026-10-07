#!/usr/bin/env bash
# Solve a captcha from the shell — no SDK, just curl.
#
#   ./examples/solve.sh captcha.png
#   YSOLVER_URL=http://my-box:8000 ./examples/solve.sh captcha.png
set -euo pipefail

URL="${YSOLVER_URL:-http://localhost:8000}"
KEY="${YSOLVER_KEY:-demo}"
IMAGE="${1:?usage: solve.sh <captcha.png>}"

BODY="$(base64 -w0 "$IMAGE" 2>/dev/null || base64 "$IMAGE" | tr -d '\n')"

CREATED="$(curl -s "$URL/in.php" -d "key=$KEY" -d "method=base64" \
  --data-urlencode "body=$BODY")"
[[ "$CREATED" == OK\|* ]] || { echo "submit failed: $CREATED" >&2; exit 1; }
ID="${CREATED#OK|}"

for _ in $(seq 1 120); do
  RESULT="$(curl -s "$URL/res.php?key=$KEY&action=get&id=$ID")"
  case "$RESULT" in
    OK\|*) echo "${RESULT#OK|}"; exit 0 ;;
    CAPCHA_NOT_READY) sleep 0.5 ;;
    *) echo "failed: $RESULT" >&2; exit 1 ;;
  esac
done
echo "timed out waiting for $ID" >&2
exit 1
