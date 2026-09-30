# rlanytune

A small command-line helper for training with **RLAnything**, the reinforcement-learning framework in [Gen-Verse/Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) ([paper](https://arxiv.org/abs/2602.02488)). RLAnything trains a policy model, a reward model and an environment model together, in three settings: OSWorld (computer control), AlfWorld (text games) and coding.

Open-AgentRL's training scripts are driven by YAML configs that you fill in by hand with absolute paths, node counts and (for OSWorld) cloud credentials. `rlanytune` does the setup, fills the configs in, tells you what is still missing, and launches the run.

> **Hardware:** this is large-scale training. Upstream's configs assume one or more machines with 8 GPUs each (their experiments used 4–12 nodes). `rlanytune` doesn't change those requirements.

## Install

```bash
cargo install --git https://github.com/ShapeZeroSZ/rlanytune-rs
```

Or from a clone: `cargo build --release` (binary at `target/release/rlanytune`). Needs Rust 1.88+.

Running training also needs `git`, [conda](https://docs.conda.io/en/latest/miniconda.html) and NVIDIA GPUs. `rlanytune doctor` checks for them.

## Quick start (coding)

```bash
# 1. Clone Open-AgentRL into ./Open-AgentRL and create the `rlanything` conda env
rlanytune setup

# 2. Get models as local directories, e.g.
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/policy

# 3. Fill in configs/coding_rl.yaml
rlanytune configure coding \
  --policy ~/models/policy --reward ~/models/reward --env-model ~/models/env \
  --nodes 1

# 4. Download the coding datasets (see Open-AgentRL/data), then
rlanytune train coding
```

Point `--dir` (or `OPEN_AGENTRL_DIR`) at an existing checkout to use it instead of `./Open-AgentRL`.

## Commands

| Command | What it does |
|---------|--------------|
| `setup [--env NAME] [--no-env]` | Clones Open-AgentRL if it isn't there yet, creates the conda env (Python 3.10) and installs `requirements_rlanything.txt` into it. |
| `configure <mode> [options]` | Writes values into `configs/<mode>_rl.yaml` in the Open-AgentRL checkout, keeping upstream's comments, then runs `check`. |
| `check <mode>` | Lists every required field that is unset, model paths that don't exist, and a node count that doesn't match the DeepSpeed config. Exits 1 if anything is wrong. |
| `train <mode> [--dry-run] [--no-conda]` | Runs `check`, then `python <mode>_rl.py config=configs/<mode>_rl.yaml` inside the conda env named in the config. `--no-conda` uses `python` from PATH instead. |
| `doctor` | Shows which tools are installed, the Open-AgentRL commit, and which modes are ready. |

Modes: `coding`, `alfworld`, `osworld`.

### `configure` options

| Option | Config key |
|--------|-----------|
| `--policy DIR`, `--reward DIR`, `--env-model DIR` | `model.policy_model`, `model.reward_model`, `model.environment_model` (must be existing local directories) |
| `--nodes N` | `experiment.num_node` and the matching `experiment.deepspeed_file` (1–10, 12, 15 or 16 nodes, as shipped by Open-AgentRL) |
| `--hf-home DIR` | `system.HF_HOME` (defaults to `$HF_HOME` if set) |
| `--envs-dir DIR` | `system.envs_dir` (defaults to conda's envs directory if conda is installed) |
| `--conda-env NAME` | `system.env_name` |
| `--alfworld-data DIR` | `dataset.environment_data_dir` (AlfWorld only) |

`system.rl_base_dir` (and, for AlfWorld, `dataset.environment_file_dir`) are always set from the checkout's location.

The config is edited in place because Open-AgentRL's scripts reload it from `configs/<mode>_rl.yaml` when they launch their worker processes. Use `git -C Open-AgentRL diff configs` to see what changed and `git -C Open-AgentRL checkout configs` to reset.

## Mode notes

- **coding:** download the training and evaluation data first; see `Open-AgentRL/data`.
- **alfworld:** run `alfworld-download` inside the conda env, then pass the directory it creates (it contains `detectors`, `json_2.1.1` and `logic`) as `--alfworld-data`.
- **osworld:** runs on a pool of Volcengine cloud VMs. Follow [OSWorld's Volcengine guide](https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/providers/volcengine/VOLCENGINE_GUIDELINE_CN.md), then export `VOLCENGINE_ACCESS_KEY_ID`, `VOLCENGINE_SECRET_ACCESS_KEY`, `VOLCENGINE_REGION`, `VOLCENGINE_IMAGE_ID`, `VOLCENGINE_SUBNET_ID`, `VOLCENGINE_SECURITY_GROUP_ID`, `VOLCENGINE_ZONE_ID` and `VOLCENGINE_DEFAULT_PASSWORD` before `rlanytune configure osworld`. Secrets are written to the config file but never printed.

For multi-node runs, start `rlanytune train <mode>` on the head node only; Open-AgentRL's README shows how the other nodes should wait.

> **Note:** Open-AgentRL's training scripts append `export` lines (HF_HOME, proxy settings, AlfWorld data path) to `~/.bashrc` on the machines they run on. That is upstream behaviour, not something `rlanytune` does.

## Development

```bash
cargo test
```

The tests run against copies of Open-AgentRL's real configs in `tests/fixtures`.
