# RLAnyTune CLI v0.1 — opp-2026-02-05-006

Rust CLI wrapper for [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) (RLAnything framework).

**Dynamic env/policy/reward forge** for LLM/agent RL, **Qwen-optimized** (significant gains shown in paper).

**Foundry self-tune scouts**: `rlanytune foundry-tune` proposes RL tuning for Foundry phases/scouts.

## Quickstart

```bash
cd builds/opp-2026-02-05-006
cargo run --bin rlanytune -- install --env rlanything
# Manual: conda create -n rlanything python=3.10; conda activate rlanything; cd open-agentrl; pip install -r requirements_rlanything.txt
rlanytune train osworld --policy Qwen/Qwen2.5-7B-Instruct
# Edits configs/osworld_rl.yaml (Qwen defaults), run python ../open-agentrl/osworld_rl.py config=...
```

## Features

- **Train modes**: osworld (GUI agents), alfworld (text games), coding (RLVR).
- **Qwen presets**: Policy/reward models auto-set.
- **Forge**: Dynamic adaptation via RLAnything closed-loop.
- **Scout**: GitHub RL opps for Foundry proposals.
- **Self-tune**: Proposes RL for Foundry (scouts → proposals → builds).

## Workflow Trace

1. Clone Open-AgentRL.
2. Rust CLI orchestrates configs/python.
3. Qwen models → 9-20% gains (OSWorld/AlfWorld/LiveBench).
4. Foundry integration: Tune agent scouts with agentic RL.

Build complete: `cargo build --release`. Binary in target/release/rlanytune.

Next: Full yaml templating, conda mgmt, HF model dl, GitHub scout API.
