# ChessFly: Counterplay — Harbor comparison (observed trials only)

| Agent | Scored / trials | Errored (excluded) | Mean reward | Min | Max |
|---|---|---|---|---|---|
| reference (oracle) | 1/1 | 0 | 1.0 | 1.0 | 1.0 |
| opencode:gemini/gemini-2.5-pro | 0/1 | 1 | None | None | None |
| opencode:google/gemini-2.5-pro | 0/1 | 1 | None | None | None |
| baseline:stop-early-1 | 1/1 | 0 | 0.0625 | 0.0625 | 0.0625 |
| baseline:stop-early-2 | 1/1 | 0 | 0.25 | 0.25 | 0.25 |
| do-nothing (nop) | 1/1 | 0 | 0.0 | 0.0 | 0.0 |

## Per-scenario (score / final cp from agent's side)

| Agent | Trial | S1 | S2 | S3 | S4 | Reward |
|---|---|---|---|---|---|---|
| reference (oracle) | harbor_task__6HPsQeb | 0.25 / 664 | 0.25 / 30 | 0.25 / 384 | 0.25 / 155 | 1.0 |
| opencode:gemini/gemini-2.5-pro | harbor_task__KcBDiuV | 0.0 / None | 0.0 / None | 0.0 / None | 0.0 / None | 0.0 (ERROR: NonZeroAgentExitCodeError) |
| opencode:google/gemini-2.5-pro | harbor_task__nPCEFNZ | 0.0 / None | 0.0 / None | 0.0 / None | 0.0 / None | 0.0 (ERROR: NonZeroAgentExitCodeError) |
| baseline:stop-early-1 | stop-early-1__tZjryGE | 0.25 / 664 | 0.0 / None | 0.0 / None | 0.0 / None | 0.0625 |
| baseline:stop-early-2 | stop-early-2__WcfazsJ | 0.25 / 664 | 0.25 / 30 | 0.0 / None | 0.0 / None | 0.25 |
| do-nothing (nop) | harbor_task__73Jq9PT | 0.0 / None | 0.0 / None | 0.0 / None | 0.0 / None | 0.0 |
