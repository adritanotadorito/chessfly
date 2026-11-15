FROM python:3.11-slim-bookworm

# Install required system tools and Stockfish engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    procps \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (python-chess)
RUN pip install --no-cache-dir chess

# Create users: referee (UID 1001) and agent (UID 1000)
RUN useradd -m -u 1001 -s /bin/bash referee && \
    useradd -m -u 1000 -s /bin/bash agent

# Set up isolated referee environment
# Move Stockfish into /referee/bin with restricted permissions (accessible only to referee)
RUN mkdir -p /referee/bin /referee/logs && \
    cp $(which stockfish || echo "/usr/games/stockfish") /referee/bin/stockfish && \
    rm -f /usr/games/stockfish /usr/bin/stockfish && \
    chmod 700 /referee/bin/stockfish

# Copy daemon, scenarios, and authoritative verifier into referee directory
COPY referee_daemon.py scenarios.py verifier.py /referee/
RUN chown -R referee:referee /referee && \
    chmod -R 700 /referee

# Install public counterplay CLI
COPY counterplay_cli.py /usr/local/bin/counterplay
RUN chmod 755 /usr/local/bin/counterplay

# Prepare runtime IPC directories
RUN mkdir -p /run/counterplay /tmp/counterplay_ipc /logs/verifier /workspace && \
    chmod 777 /run/counterplay /tmp/counterplay_ipc /logs/verifier && \
    chown -R agent:agent /workspace

# Install entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

WORKDIR /workspace
USER root

ENTRYPOINT ["/entrypoint.sh"]
