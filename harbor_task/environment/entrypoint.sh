#!/bin/bash
set -e

# Create necessary runtime directories
mkdir -p /run/counterplay /tmp/counterplay_ipc /logs/verifier
chmod 777 /tmp/counterplay_ipc /run/counterplay /logs/verifier

# Ensure referee owns /referee and its logs
mkdir -p /referee/logs
chown -R referee:referee /referee
chmod 700 /referee
chmod 700 /referee/logs
if [ -f /referee/bin/stockfish ]; then
    chmod 700 /referee/bin/stockfish
fi

# Start the authoritative referee daemon as the referee user in background
su - referee -c "python3 /referee/referee_daemon.py" > /tmp/referee_daemon.log 2>&1 &

# Wait for daemon IPC to be ready (up to 10 seconds)
for i in {1..50}; do
    if [ -S /run/counterplay/referee.sock ] || [ -d /tmp/counterplay_ipc/inbox ]; then
        break
    fi
    sleep 0.2
done

# Set permissions on socket if present
if [ -S /run/counterplay/referee.sock ]; then
    chmod 666 /run/counterplay/referee.sock
fi

# Execute command or keep container running
if [ $# -gt 0 ]; then
    # If root, switch to agent user for executing command unless command is root-required
    if [ "$(id -u)" = "0" ]; then
        exec su - agent -c "$*"
    else
        exec "$@"
    fi
else
    exec tail -f /dev/null
fi
