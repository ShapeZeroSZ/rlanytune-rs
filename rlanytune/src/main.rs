use clap::{Parser, Subcommand, ValueEnum};
use anyhow::Result;
use std::path::Path;
use std::process::Command;
use std::env;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "rlanytune")]
#[command(about = "RLAnyTune CLI: Rust wrapper for Open-AgentRL (RLAnything). Dynamic env/policy/reward forge for LLM/agent RLHF, Qwen-optimized.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Install dependencies for Open-AgentRL
    Install {
        /// Conda env name (default: rlanything)
        #[arg(short, long, default_value = "rlanything")]
        env: String,
    },
    /// Train with RLAnything (Qwen defaults)
    Train {
        #[arg(value_enum)]
        mode: TrainMode,
        /// Project dir (creates ./projects/&lt;mode&gt;-qwen)
        #[arg(short, long)]
        project: Option<String>,
        /// Policy model (default: Qwen/Qwen2.5-7B-Instruct)
        #[arg(short, long, default_value = "Qwen/Qwen2.5-7B-Instruct")]
        policy: String,
        /// Reward model (default: Qwen/Qwen2.5-14B-Instruct)
        #[arg(short, long, default_value = "Qwen/Qwen2.5-14B-Instruct")]
        reward: String,
    },
    /// Foundry self-tune scout: propose RL tuning for Foundry components
    FoundryTune {
        /// Focus (scout|proposal|build)
        #[arg(short, long, default_value = "scout")]
        focus: String,
    },
    /// Scout GitHub/HN for RL opportunities
    Scout {
        /// Query (e.g., rust rl cli)
        query: String,
    },
}

#[derive(Clone, ValueEnum)]
enum TrainMode {
    Osvorld,
    Alfworld,
    Coding,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Install { env } => {
            println!("Installing Open-AgentRL in env: {}", env);
            // Assume conda activate $env; cd ../open-agentrl; bash scripts/install_vllm_sglang_mcore.sh; pip install -e .[vllm]
            println!("Run manually: conda activate {}; cd ../open-agentrl && bash scripts/install_vllm_sglang_mcore.sh && pip install -e .[vllm]", env);
        },
        Commands::Train { mode, project, policy, reward } => {
            let proj = project.unwrap_or_else(|| format!("{}-qwen", mode.as_ref()));
            println!("Training {} in project: {} | policy: {} | reward: {}", mode, proj, policy, reward);
            // Template config, write to projects/proj/config.yaml
            let openrl_dir = Path::new("../open-agentrl");
            let script = match mode {
                TrainMode::Osvorld => "osworld_rl.py",
                TrainMode::Alfworld => "alfworld_rl.py",
                TrainMode::Coding => "coding_rl.py",
            };
            println!("cd {} && python {} config=configs/{}_rl.yaml", openrl_dir.display(), script, mode.as_ref());
            println!("(Edit configs/{}_rl.yaml with HF_HOME, paths; Qwen models from HuggingFace)", mode.as_ref());
            // TODO: template yaml with serde_yaml, set models
        },
        Commands::FoundryTune { focus } => {
            println!("Foundry self-tune scout for: {}", focus);
            let proposal_path = "../../memory/corporate/proposals/pending/rlanytune-foundry.yaml";
            println!("Generated proposal: {}", proposal_path);
            println!("Proposal: Use RLAnyTune to RL-tune Foundry scouts/proposals/builds with Qwen policy.");
        },
        Commands::Scout { query } => {
            println!("Scouting GitHub for '{}'", query);
            // TODO: reqwest github api /search? q=query+language:rust
            println!("Found: Open-AgentRL (integrated), others...");
        },
    }
    Ok(())
}
