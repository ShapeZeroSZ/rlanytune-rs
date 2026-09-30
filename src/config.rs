//! Filling in and validating Open-AgentRL's RLAnything training configs.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;

use crate::yaml;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Computer control (GUI agent) on OSWorld; needs Volcengine cloud VMs
    Osworld,
    /// Text-based games on AlfWorld
    Alfworld,
    /// RLVR coding
    Coding,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Osworld => "osworld",
            Mode::Alfworld => "alfworld",
            Mode::Coding => "coding",
        }
    }

    /// The training script at the Open-AgentRL root.
    pub fn script(self) -> String {
        format!("{}_rl.py", self.name())
    }

    /// Config path relative to the Open-AgentRL root. The scripts relaunch
    /// themselves with `config=../configs/<project>.yaml`, so the config has
    /// to live at exactly this path.
    pub fn config_rel(self) -> PathBuf {
        PathBuf::from(format!("configs/{}_rl.yaml", self.name()))
    }

    /// Keys that must be set before training, with a hint for each.
    fn required(self) -> Vec<(&'static str, &'static str)> {
        let mut req = vec![
            (
                "system.rl_base_dir",
                "set automatically by `rlanytune configure`",
            ),
            ("model.policy_model", "--policy <local model dir>"),
            ("model.reward_model", "--reward <local model dir>"),
            ("model.environment_model", "--env-model <local model dir>"),
        ];
        match self {
            Mode::Alfworld => {
                req.push((
                    "dataset.environment_data_dir",
                    "--alfworld-data <dir from `alfworld-download`>",
                ));
                req.push((
                    "dataset.environment_file_dir",
                    "set automatically by `rlanytune configure`",
                ));
            }
            Mode::Osworld => {
                req.push(("system.region", "env VOLCENGINE_REGION, or edit the config"));
                for key in OSWORLD_VOLCENGINE_KEYS {
                    req.push((key, "env var of the same name, or edit the config"));
                }
            }
            Mode::Coding => {}
        }
        req
    }
}

const OSWORLD_VOLCENGINE_KEYS: [&str; 8] = [
    "system.VOLCENGINE_ACCESS_KEY_ID",
    "system.VOLCENGINE_SECRET_ACCESS_KEY",
    "system.VOLCENGINE_REGION",
    "system.VOLCENGINE_IMAGE_ID",
    "system.VOLCENGINE_SUBNET_ID",
    "system.VOLCENGINE_SECURITY_GROUP_ID",
    "system.VOLCENGINE_ZONE_ID",
    "system.VOLCENGINE_DEFAULT_PASSWORD",
];

/// Values to write into a config. `None` leaves the current value alone.
#[derive(Default, Debug)]
pub struct Settings {
    pub policy: Option<PathBuf>,
    pub reward: Option<PathBuf>,
    pub env_model: Option<PathBuf>,
    pub nodes: Option<u32>,
    pub hf_home: Option<PathBuf>,
    pub envs_dir: Option<PathBuf>,
    pub conda_env: Option<String>,
    pub alfworld_data: Option<PathBuf>,
}

pub fn config_path(base: &Path, mode: Mode) -> PathBuf {
    base.join(mode.config_rel())
}

fn read(base: &Path, mode: Mode) -> Result<String> {
    let path = config_path(base, mode);
    std::fs::read_to_string(&path).with_context(|| {
        format!(
            "reading {} (is --dir pointing at an Open-AgentRL checkout? run `rlanytune setup`)",
            path.display()
        )
    })
}

fn existing_dir(p: &Path, what: &str) -> Result<PathBuf> {
    let abs = std::path::absolute(p)?;
    if !abs.is_dir() {
        bail!("{what} {} is not a directory", abs.display());
    }
    Ok(abs)
}

fn deepspeed_file(nodes: u32) -> String {
    format!("{nodes}_node_8_gpus_deepspeed_zero3")
}

/// Applies edits to config text and records what changed.
struct Editor {
    text: String,
    changes: Vec<(String, String)>,
}

impl Editor {
    fn raw(&mut self, key: &str, raw: &str, shown: &str) -> Result<()> {
        self.text = yaml::set(&self.text, key, raw).map_err(anyhow::Error::msg)?;
        self.changes.push((key.to_string(), shown.to_string()));
        Ok(())
    }

    fn string(&mut self, key: &str, value: &str) -> Result<()> {
        self.raw(key, &yaml::quote(value), value)
    }

    fn path(&mut self, key: &str, value: &Path) -> Result<()> {
        self.string(key, &value.display().to_string())
    }
}

/// Writes `settings` (plus the derived paths) into the mode's config and
/// returns the `(key, value)` pairs that were set.
pub fn configure(
    base: &Path,
    mode: Mode,
    settings: &Settings,
    env: impl Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>> {
    let base = existing_dir(base, "Open-AgentRL directory")?;
    let mut ed = Editor {
        text: read(&base, mode)?,
        changes: Vec::new(),
    };

    ed.path("system.rl_base_dir", &base)?;

    for (key, value, what) in [
        ("model.policy_model", &settings.policy, "policy model"),
        ("model.reward_model", &settings.reward, "reward model"),
        (
            "model.environment_model",
            &settings.env_model,
            "environment model",
        ),
    ] {
        if let Some(p) = value {
            let dir = existing_dir(p, what).context(
                "model paths must be local directories; download with e.g. `hf download <repo-id> --local-dir <dir>`",
            )?;
            ed.path(key, &dir)?;
        }
    }

    if let Some(n) = settings.nodes {
        let ds = deepspeed_file(n);
        if !base
            .join("accelerate_configs")
            .join(format!("{ds}.yaml"))
            .is_file()
        {
            bail!(
                "Open-AgentRL has no accelerate config for {n} node(s) (accelerate_configs/{ds}.yaml)"
            );
        }
        ed.raw("experiment.num_node", &n.to_string(), &n.to_string())?;
        ed.string("experiment.deepspeed_file", &ds)?;
    }

    if let Some(p) = &settings.hf_home {
        ed.path("system.HF_HOME", &std::path::absolute(p)?)?;
    }
    if let Some(p) = &settings.envs_dir {
        ed.path("system.envs_dir", &std::path::absolute(p)?)?;
    }
    if let Some(name) = &settings.conda_env {
        ed.string("system.env_name", name)?;
    }

    match mode {
        Mode::Alfworld => {
            ed.path(
                "dataset.environment_file_dir",
                &base.join("alfworld_master"),
            )?;
            if let Some(p) = &settings.alfworld_data {
                ed.path(
                    "dataset.environment_data_dir",
                    &existing_dir(p, "AlfWorld data directory")?,
                )?;
            }
        }
        Mode::Osworld => {
            for key in OSWORLD_VOLCENGINE_KEYS {
                let var = key.trim_start_matches("system.");
                let Some(v) = env(var).filter(|v| !v.is_empty()) else {
                    continue;
                };
                if var.contains("SECRET") || var.contains("PASSWORD") {
                    ed.raw(key, &yaml::quote(&v), "(from env, hidden)")?;
                } else {
                    ed.string(key, &v)?;
                }
                if var == "VOLCENGINE_REGION" {
                    ed.string("system.region", &v)?;
                }
            }
        }
        Mode::Coding => {}
    }

    std::fs::write(config_path(&base, mode), &ed.text)?;
    Ok(ed.changes)
}

/// Lists everything that would stop training from starting.
pub fn problems(base: &Path, mode: Mode) -> Result<Vec<String>> {
    let text = read(base, mode)?;
    let mut out = Vec::new();

    for (key, hint) in mode.required() {
        if yaml::get(&text, key).is_none() {
            out.push(format!("{key} is not set ({hint})"));
        }
    }

    for key in [
        "model.policy_model",
        "model.reward_model",
        "model.environment_model",
    ] {
        if let Some(p) = yaml::get(&text, key)
            && !Path::new(&p).is_dir()
        {
            out.push(format!("{key}: {p} does not exist on this machine"));
        }
    }

    match yaml::get(&text, "experiment.deepspeed_file") {
        Some(ds)
            if !base
                .join("accelerate_configs")
                .join(format!("{ds}.yaml"))
                .is_file() =>
        {
            out.push(format!(
                "experiment.deepspeed_file: accelerate_configs/{ds}.yaml not found"
            ));
        }
        _ => {}
    }
    if let Some(n) = yaml::get(&text, "experiment.num_node").and_then(|n| n.parse::<u32>().ok())
        && yaml::get(&text, "experiment.deepspeed_file").as_deref()
            != Some(deepspeed_file(n).as_str())
    {
        out.push(format!(
            "experiment.deepspeed_file should be \"{}\" to match num_node: {n} (use --nodes {n})",
            deepspeed_file(n)
        ));
    }

    Ok(out)
}

/// The conda env named in the config (`system.env_name`).
pub fn env_name(base: &Path, mode: Mode) -> Result<String> {
    Ok(yaml::get(&read(base, mode)?, "system.env_name").unwrap_or_else(|| "rlanything".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake Open-AgentRL checkout holding the real upstream configs.
    fn checkout() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/configs");
        std::fs::create_dir(dir.path().join("configs")).unwrap();
        for m in ["osworld", "alfworld", "coding"] {
            let f = format!("{m}_rl.yaml");
            std::fs::copy(fixtures.join(&f), dir.path().join("configs").join(&f)).unwrap();
        }
        std::fs::create_dir(dir.path().join("accelerate_configs")).unwrap();
        for n in [1, 4, 8, 12] {
            std::fs::write(
                dir.path()
                    .join(format!("accelerate_configs/{}.yaml", deepspeed_file(n))),
                "",
            )
            .unwrap();
        }
        for m in ["policy", "reward", "envm", "alfdata"] {
            std::fs::create_dir(dir.path().join(m)).unwrap();
        }
        dir
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn models(dir: &Path) -> Settings {
        Settings {
            policy: Some(dir.join("policy")),
            reward: Some(dir.join("reward")),
            env_model: Some(dir.join("envm")),
            ..Default::default()
        }
    }

    #[test]
    fn fresh_upstream_config_reports_missing_fields() {
        let dir = checkout();
        let p = problems(dir.path(), Mode::Coding).unwrap();
        assert!(
            p.iter()
                .any(|s| s.starts_with("model.policy_model is not set")),
            "{p:?}"
        );
        assert!(
            p.iter()
                .any(|s| s.starts_with("system.rl_base_dir is not set")),
            "{p:?}"
        );
    }

    #[test]
    fn configure_coding_makes_it_ready() {
        let dir = checkout();
        let mut s = models(dir.path());
        s.nodes = Some(1);
        s.hf_home = Some("/data/hf".into());
        configure(dir.path(), Mode::Coding, &s, no_env).unwrap();

        assert_eq!(
            problems(dir.path(), Mode::Coding).unwrap(),
            Vec::<String>::new()
        );
        let text = read(dir.path(), Mode::Coding).unwrap();
        assert_eq!(
            yaml::get(&text, "experiment.num_node").as_deref(),
            Some("1")
        );
        assert_eq!(
            yaml::get(&text, "experiment.deepspeed_file").as_deref(),
            Some("1_node_8_gpus_deepspeed_zero3")
        );
        assert_eq!(
            yaml::get(&text, "system.HF_HOME").as_deref(),
            Some("/data/hf")
        );
        // Upstream's instructions survive.
        assert!(text.contains("# the number of machines you have"));
    }

    #[test]
    fn missing_model_dir_is_rejected() {
        let dir = checkout();
        let s = Settings {
            policy: Some(dir.path().join("nope")),
            ..Default::default()
        };
        let err = configure(dir.path(), Mode::Coding, &s, no_env).unwrap_err();
        assert!(format!("{err:#}").contains("is not a directory"));
    }

    #[test]
    fn unsupported_node_count_is_rejected() {
        let dir = checkout();
        let s = Settings {
            nodes: Some(3),
            ..Default::default()
        };
        assert!(configure(dir.path(), Mode::Coding, &s, no_env).is_err());
    }

    #[test]
    fn mismatched_deepspeed_file_is_reported() {
        let dir = checkout();
        let path = config_path(dir.path(), Mode::Coding);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, yaml::set(&text, "experiment.num_node", "1").unwrap()).unwrap();
        let p = problems(dir.path(), Mode::Coding).unwrap();
        assert!(
            p.iter().any(|s| s.contains("to match num_node: 1")),
            "{p:?}"
        );
    }

    #[test]
    fn alfworld_needs_and_sets_data_dirs() {
        let dir = checkout();
        configure(dir.path(), Mode::Alfworld, &models(dir.path()), no_env).unwrap();
        let p = problems(dir.path(), Mode::Alfworld).unwrap();
        assert!(
            p.iter()
                .any(|s| s.starts_with("dataset.environment_data_dir")),
            "{p:?}"
        );

        let mut s = models(dir.path());
        s.alfworld_data = Some(dir.path().join("alfdata"));
        s.nodes = Some(8);
        configure(dir.path(), Mode::Alfworld, &s, no_env).unwrap();
        assert_eq!(
            problems(dir.path(), Mode::Alfworld).unwrap(),
            Vec::<String>::new()
        );
        let text = read(dir.path(), Mode::Alfworld).unwrap();
        let file_dir = yaml::get(&text, "dataset.environment_file_dir").unwrap();
        assert!(file_dir.ends_with("/alfworld_master"), "{file_dir}");
    }

    #[test]
    fn osworld_reads_volcengine_settings_from_env() {
        let dir = checkout();
        let env = |k: &str| Some(format!("val-{k}"));
        let mut s = models(dir.path());
        s.nodes = Some(12);
        let changes = configure(dir.path(), Mode::Osworld, &s, env).unwrap();
        assert_eq!(
            problems(dir.path(), Mode::Osworld).unwrap(),
            Vec::<String>::new()
        );
        let text = read(dir.path(), Mode::Osworld).unwrap();
        assert_eq!(
            yaml::get(&text, "system.region").as_deref(),
            Some("val-VOLCENGINE_REGION")
        );
        // Secrets are written but never echoed back.
        let shown: Vec<_> = changes.iter().map(|(_, v)| v.as_str()).collect();
        assert!(!shown.contains(&"val-VOLCENGINE_SECRET_ACCESS_KEY"));
        // The instance-type list under system is untouched.
        assert!(text.contains("        - ecs.e-c1m2.large"));
    }
}
