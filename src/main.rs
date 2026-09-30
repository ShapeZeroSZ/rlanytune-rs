mod config;
mod yaml;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use config::{Mode, Settings};

const OPEN_AGENTRL_URL: &str = "https://github.com/Gen-Verse/Open-AgentRL";

#[derive(Parser)]
#[command(name = "rlanytune", version)]
#[command(about = "Set up, configure and launch Open-AgentRL (RLAnything) training runs")]
struct Cli {
    /// Open-AgentRL checkout to use
    #[arg(
        long,
        global = true,
        env = "OPEN_AGENTRL_DIR",
        default_value = "Open-AgentRL"
    )]
    dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Clone Open-AgentRL and create its conda environment
    Setup {
        /// Conda env to create and install into
        #[arg(long, default_value = "rlanything")]
        env: String,
        /// Only clone; skip creating the conda env and installing requirements
        #[arg(long)]
        no_env: bool,
    },
    /// Fill in a mode's training config (configs/<mode>_rl.yaml) in place
    Configure {
        #[arg(value_enum)]
        mode: Mode,
        /// Local directory of the policy model
        #[arg(long)]
        policy: Option<PathBuf>,
        /// Local directory of the reward model
        #[arg(long)]
        reward: Option<PathBuf>,
        /// Local directory of the environment model
        #[arg(long)]
        env_model: Option<PathBuf>,
        /// Number of 8-GPU machines to train on
        #[arg(long)]
        nodes: Option<u32>,
        /// Hugging Face cache dir [default: $HF_HOME, if set]
        #[arg(long, env = "HF_HOME")]
        hf_home: Option<PathBuf>,
        /// Conda envs directory [default: detected from conda]
        #[arg(long)]
        envs_dir: Option<PathBuf>,
        /// Conda env the training runs in
        #[arg(long)]
        conda_env: Option<String>,
        /// AlfWorld data directory (created by `alfworld-download`)
        #[arg(long)]
        alfworld_data: Option<PathBuf>,
    },
    /// Report anything in a mode's config that would stop training
    Check {
        #[arg(value_enum)]
        mode: Mode,
    },
    /// Check the config, then start training
    Train {
        #[arg(value_enum)]
        mode: Mode,
        /// Print the command instead of running it
        #[arg(long)]
        dry_run: bool,
        /// Use `python` from PATH instead of `conda run -n <env>`
        #[arg(long)]
        no_conda: bool,
    },
    /// Show which prerequisites are installed and which configs are ready
    Doctor,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let dir = &cli.dir;
    match cli.command {
        Commands::Setup { env, no_env } => setup(dir, &env, no_env)?,
        Commands::Configure {
            mode,
            policy,
            reward,
            env_model,
            nodes,
            hf_home,
            envs_dir,
            conda_env,
            alfworld_data,
        } => {
            let settings = Settings {
                policy,
                reward,
                env_model,
                nodes,
                hf_home,
                envs_dir: envs_dir.or_else(conda_envs_dir),
                conda_env,
                alfworld_data,
            };
            let changes = config::configure(dir, mode, &settings, |k| std::env::var(k).ok())?;
            println!("Updated {}:", config::config_path(dir, mode).display());
            for (key, value) in changes {
                println!("  {key} = {value}");
            }
            return report(dir, mode);
        }
        Commands::Check { mode } => return report(dir, mode),
        Commands::Train {
            mode,
            dry_run,
            no_conda,
        } => return train(dir, mode, dry_run, no_conda),
        Commands::Doctor => doctor(dir),
    }
    Ok(ExitCode::SUCCESS)
}

fn report(dir: &Path, mode: Mode) -> Result<ExitCode> {
    let problems = config::problems(dir, mode)?;
    if problems.is_empty() {
        println!(
            "{} config is ready: rlanytune train {}",
            mode.name(),
            mode.name()
        );
        return Ok(ExitCode::SUCCESS);
    }
    println!("{} config is not ready yet:", mode.name());
    for p in problems {
        println!("  - {p}");
    }
    Ok(ExitCode::FAILURE)
}

fn setup(dir: &Path, env: &str, no_env: bool) -> Result<()> {
    if dir.join(".git").is_dir() {
        println!("Open-AgentRL already at {}", dir.display());
    } else {
        run_cmd(
            Command::new("git")
                .args(["clone", "--depth", "1", OPEN_AGENTRL_URL])
                .arg(dir),
        )?;
    }
    if no_env {
        return Ok(());
    }
    if !has("conda") {
        bail!(
            "conda not found. Install Miniconda, then rerun `rlanytune setup`, or do it by hand:\n  \
             conda create -n {env} python=3.10\n  \
             conda run -n {env} pip install -r {}/requirements_rlanything.txt",
            dir.display()
        );
    }
    if conda_env_exists(env) {
        println!("conda env {env} already exists");
    } else {
        run_cmd(Command::new("conda").args(["create", "-y", "-n", env, "python=3.10"]))?;
    }
    run_cmd(
        Command::new("conda")
            .args([
                "run",
                "-n",
                env,
                "--no-capture-output",
                "pip",
                "install",
                "-r",
                "requirements_rlanything.txt",
            ])
            .current_dir(dir),
    )?;
    println!(
        "\nSetup done. Next: rlanytune configure <osworld|alfworld|coding> --policy ... --reward ... --env-model ..."
    );
    Ok(())
}

fn train(dir: &Path, mode: Mode, dry_run: bool, no_conda: bool) -> Result<ExitCode> {
    let problems = config::problems(dir, mode)?;
    if !problems.is_empty() {
        let list: Vec<String> = problems.iter().map(|p| format!("  - {p}")).collect();
        bail!("{} config is not ready:\n{}", mode.name(), list.join("\n"));
    }
    let script = mode.script();
    let cfg_arg = format!("config={}", mode.config_rel().display());
    let mut cmd = if no_conda {
        Command::new("python")
    } else {
        let mut c = Command::new("conda");
        c.args([
            "run",
            "-n",
            &config::env_name(dir, mode)?,
            "--no-capture-output",
            "python",
        ]);
        c
    };
    cmd.arg(&script).arg(&cfg_arg).current_dir(dir);

    if dry_run {
        println!("cd {} && {}", dir.display(), describe(&cmd));
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!("$ {}", describe(&cmd));
    let status = cmd
        .status()
        .with_context(|| format!("starting {}", cmd.get_program().to_string_lossy()))?;
    Ok(ExitCode::from(
        status.code().unwrap_or(1).clamp(0, 255) as u8
    ))
}

fn doctor(dir: &Path) {
    println!("Tools:");
    for (tool, arg) in [
        ("git", "--version"),
        ("conda", "--version"),
        ("python", "--version"),
        ("nvidia-smi", "-L"),
    ] {
        match Command::new(tool).arg(arg).output() {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout);
                let summary = if tool == "nvidia-smi" {
                    format!(
                        "{} GPU(s)",
                        text.lines().filter(|l| l.starts_with("GPU ")).count()
                    )
                } else {
                    text.lines().next().unwrap_or("").trim().to_string()
                };
                println!("  ok       {tool}: {summary}");
            }
            _ => println!("  missing  {tool}"),
        }
    }

    println!("\nOpen-AgentRL ({}):", dir.display());
    if !dir.join(".git").is_dir() {
        println!("  not found; run `rlanytune setup` or pass --dir");
        return;
    }
    if let Ok(out) = Command::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["log", "-1", "--format=%h %cs"])
        .output()
    {
        println!("  commit {}", String::from_utf8_lossy(&out.stdout).trim());
    }

    println!("\nConfigs:");
    for mode in [Mode::Coding, Mode::Alfworld, Mode::Osworld] {
        match config::problems(dir, mode) {
            Ok(p) if p.is_empty() => println!("  ready    {}", mode.name()),
            Ok(p) => println!(
                "  {} issue(s) {}  (rlanytune check {})",
                p.len(),
                mode.name(),
                mode.name()
            ),
            Err(e) => println!("  error    {}: {e:#}", mode.name()),
        }
    }
}

fn run_cmd(cmd: &mut Command) -> Result<()> {
    eprintln!("$ {}", describe(cmd));
    let status = cmd
        .status()
        .with_context(|| format!("starting {}", cmd.get_program().to_string_lossy()))?;
    if !status.success() {
        bail!("`{}` failed ({status})", describe(cmd));
    }
    Ok(())
}

fn describe(cmd: &Command) -> String {
    std::iter::once(cmd.get_program())
        .chain(cmd.get_args())
        .map(|a| a.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

fn has(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn conda_info() -> Option<serde_json::Value> {
    let out = Command::new("conda")
        .args(["info", "--json"])
        .output()
        .ok()?;
    serde_json::from_slice(&out.stdout).ok()
}

fn conda_envs_dir() -> Option<PathBuf> {
    conda_info()?["envs_dirs"][0].as_str().map(PathBuf::from)
}

fn conda_env_exists(name: &str) -> bool {
    conda_info().is_some_and(|info| {
        info["envs"].as_array().is_some_and(|envs| {
            envs.iter()
                .filter_map(|e| e.as_str())
                .any(|p| Path::new(p).file_name().is_some_and(|f| f == name))
        })
    })
}
