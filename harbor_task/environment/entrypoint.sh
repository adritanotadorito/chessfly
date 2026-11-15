#!/bin/bash
# Starts the root-owned referee daemon (as UID 1001 'referee'), then runs the
# container command unchanged (Harbor passes `sh -c "sleep infinity"`).
mkdir -p /run/counterplay /tmp/counterplay_ipc /logs/verifier /referee/logs
chmod 777 /tmp/counterplay_ipc /run/counterplay
chmod 755 /logs/verifier
chown -R referee:referee /referee
chmod 700 /referee /referee/logs /referee/bin/stockfish 2>/dev/null || true
if ! pgrep -f referee_daemon.py >/dev/null; then
  su referee -s /bin/bash -c "python3 /referee/referee_daemon.py" > /var/log/referee_daemon.log 2>&1 &
  for i in $(seq 1 50); do [ -S /run/counterplay/referee.sock ] && break; sleep 0.2; done
  chmod 666 /run/counterplay/referee.sock 2>/dev/null || true
fi
if [ $# -gt 0 ]; then exec "$@"; else exec sleep infinity; fi
