#!/usr/bin/env bash
# Copy of ink-app ci/write_deploy_key.sh for standalone deploy workflow.
set -euo pipefail
KEY="${1:-/tmp/deploy_key}"
if [[ -z "${SSH_PRIVATE_KEY:-}" && -n "${PRIVATE_SSH_KEY:-}" ]]; then
  export SSH_PRIVATE_KEY="${PRIVATE_SSH_KEY}"
fi
: "${SSH_PRIVATE_KEY:?SSH_PRIVATE_KEY or PRIVATE_SSH_KEY required}"
rm -f "$KEY"
python3 - "$KEY" <<'PY'
import os, sys
k = os.environ["SSH_PRIVATE_KEY"].strip()
path = sys.argv[1]
if not k:
    raise SystemExit("SSH_PRIVATE_KEY empty")
if "BEGIN" in k and "\n" not in k:
    begin = "-----BEGIN OPENSSH PRIVATE KEY-----"
    end = "-----END OPENSSH PRIVATE KEY-----"
    if begin in k and end in k:
        body = k.split(begin, 1)[1].split(end, 1)[0].strip()
        text = "\n".join([begin, *(body[i : i + 70] for i in range(0, len(body), 70)), end]) + "\n"
    else:
        mid = k.find("-----", 10) + 5
        stop = k.rfind("-----END")
        header, body, footer = k[:mid], k[mid:stop].strip(), k[stop:]
        text = header + "\n" + "\n".join(body[i : i + 70] for i in range(0, len(body), 70)) + "\n" + footer + "\n"
elif "BEGIN" in k:
    text = k if k.endswith("\n") else k + "\n"
else:
    import base64
    raw = base64.b64decode("".join(k.split()))
    open(path, "wb").write(raw)
    sys.exit(0)
open(path, "w").write(text)
PY
chmod 600 "$KEY"
