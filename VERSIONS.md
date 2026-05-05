# CodeGames Version Matrix

This repository publishes three runnable cuts of the CodeGames contract intelligence app so they can be downloaded, run independently, and compared.

## Published Cuts

| Cut | Git tag | Git branch | What it shows |
| --- | --- | --- | --- |
| Version 1 - Agentic MVP | `codegames-v1-agentic-mvp` | `versions/v1-agentic-mvp` | Original reviewed-context learning loop: ingest, baseline extraction, agent critique, review packet, second run, benchmark. |
| Version 2 - Consolidated UI | `codegames-v2-consolidated-ui` | `versions/v2-consolidated-ui` | Single web workbench with upload, split, agent run, decisions, benchmark, and counterfactual panels. |
| Version 3 - Discovery App | `codegames-v3-discovery-current` | `versions/v3-discovery-current` | Current discovery-first app: find one contract type in a haystack, use active review, and grow a clause library. |

## Download

Download each cut from GitHub:

- Version 1: `https://github.com/cptntrps/contract-discovery-mvp/archive/refs/tags/codegames-v1-agentic-mvp.zip`
- Version 2: `https://github.com/cptntrps/contract-discovery-mvp/archive/refs/tags/codegames-v2-consolidated-ui.zip`
- Version 3: `https://github.com/cptntrps/contract-discovery-mvp/archive/refs/tags/codegames-v3-discovery-current.zip`

Or clone once and switch between cuts:

```bash
git clone https://github.com/cptntrps/contract-discovery-mvp.git
cd contract-discovery-mvp

git checkout codegames-v1-agentic-mvp
git checkout codegames-v2-consolidated-ui
git checkout codegames-v3-discovery-current
```

## Run A Cut

From the checked-out cut:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
contract-intel ui
```

Open the URL printed by the command, usually `http://127.0.0.1:8765`.

## Compare

Use separate folders so each cut has isolated generated `data/` outputs:

```bash
git clone https://github.com/cptntrps/contract-discovery-mvp.git codegames-v1
git clone https://github.com/cptntrps/contract-discovery-mvp.git codegames-v2
git clone https://github.com/cptntrps/contract-discovery-mvp.git codegames-v3

cd codegames-v1 && git checkout codegames-v1-agentic-mvp
cd ../codegames-v2 && git checkout codegames-v2-consolidated-ui
cd ../codegames-v3 && git checkout codegames-v3-discovery-current
```

Run each one in its own virtualenv and choose different UI ports if running them at the same time:

```bash
contract-intel ui --port 8765
contract-intel ui --port 8766
contract-intel ui --port 8767
```

## Notes

- These are source snapshots, not production releases.
- Some model-backed paths require Ollama or OpenAI configuration. The app keeps deterministic fallback paths where implemented.
- Generated files under `data/` are local run outputs and should not be compared as source unless you intentionally run the same corpus through each cut.
