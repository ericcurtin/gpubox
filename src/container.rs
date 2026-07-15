//! Persistent named containers (`gpubox enter --name <name>` /
//! `gpubox rm <name>`).
//!
//! Without `--name`, every `gpubox enter`/`gpubox run` launches a fresh
//! `--rm` container (see `backend::linux`): anything installed via `apt`
//! inside it vanishes the moment the shell exits, and the Vulkan/CPU
//! fallback's `mesa-vulkan-drivers` et al. get reinstalled - a network
//! hit plus a wait - on every single launch. `--name` opts into a
//! long-lived container instead, the same model distrobox/toolbox use:
//! the first `enter`/`run` creates it (detached, initialized once via the
//! usual passwd/group/package wrapper), every subsequent invocation just
//! `exec`s into the already-running container, and it persists across
//! host reboots until `gpubox rm` deletes it.
//!
//! Linux (Docker/Podman) only - Seatbelt runs natively on the host with
//! nothing to persist, and Windows Sandbox always boots a clean VM by
//! design.

use crate::backend::linux::ContainerEngine;
use crate::backend::{Invocation, LaunchSpec};
use crate::mounts;
use anyhow::{Context, Result};
use std::process::Command;

/// Turn a user-supplied `--name` into the actual docker/podman container
/// name gpubox creates, namespaced so it doesn't collide with unrelated
/// containers on the host.
pub fn container_name(name: &str) -> String {
    format!("gpubox-{name}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerState {
    Missing,
    Stopped,
    Running,
}

/// Query whether the named container exists and, if so, whether it's
/// running. Best-effort: any failure (engine not installed, container
/// genuinely missing) is treated as [`ContainerState::Missing`], since
/// the caller's next step (create it) is the right recovery either way.
pub fn inspect(engine_program: &str, name: &str) -> ContainerState {
    let output = Command::new(engine_program)
        .args(["inspect", "-f", "{{.State.Running}}", name])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            if String::from_utf8_lossy(&out.stdout).trim() == "true" {
                ContainerState::Running
            } else {
                ContainerState::Stopped
            }
        }
        _ => ContainerState::Missing,
    }
}

/// Build the `<engine> run -d --name <name> ... sleep infinity` invocation
/// that creates the persistent container the first time `--name` is used.
/// Deliberately omits `--rm` (the whole point is that it survives) and
/// runs detached; the passwd/group fixup and any one-time package install
/// still happen here via the normal wrapper script (see
/// `backend::linux::command_argv_with_tail`), so they only ever run once
/// per container rather than on every `enter`.
pub fn create_invocation(engine: ContainerEngine, spec: &LaunchSpec, name: &str) -> Invocation {
    let mut args = vec!["run".to_string(), "-d".to_string(), "--name".to_string()];
    args.push(name.to_string());
    args.extend(crate::backend::linux::run_args_common(spec));
    args.push(spec.image.clone());

    let keep_alive = ["sleep".to_string(), "infinity".to_string()];
    args.extend(crate::backend::linux::command_argv_with_tail(
        spec,
        &keep_alive,
    ));

    Invocation {
        program: engine.program().to_string(),
        args,
        generated_files: Vec::new(),
    }
}

/// Build the `<engine> exec ... <name> <command>` invocation that runs
/// `spec`'s command (or an interactive shell) inside an already-running
/// named container. No wrapper script is needed here: the passwd/group
/// fixup already happened once, during [`create_invocation`], so this
/// just execs directly as the real host uid/gid via `--user`.
pub fn exec_invocation(engine: ContainerEngine, spec: &LaunchSpec, name: &str) -> Invocation {
    let identity = mounts::current_identity();
    let mut args = vec!["exec".to_string()];
    if spec.interactive {
        args.push("-it".to_string());
    } else {
        args.push("-i".to_string());
    }
    args.push("--user".to_string());
    args.push(format!("{}:{}", identity.uid, identity.gid));
    for (key, value) in &spec.env {
        args.push("-e".to_string());
        args.push(format!("{key}={value}"));
    }
    if let Some(workdir) = &spec.workdir {
        args.push("-w".to_string());
        args.push(workdir.display().to_string());
    }
    args.push(name.to_string());
    if spec.command.is_empty() {
        args.push("/bin/bash".to_string());
    } else {
        args.extend(spec.command.iter().cloned());
    }

    Invocation {
        program: engine.program().to_string(),
        args,
        generated_files: Vec::new(),
    }
}

/// Build the `<engine> start <name>` invocation used to wake a stopped
/// (but not removed) named container back up before exec'ing into it.
pub fn start_invocation(engine: ContainerEngine, name: &str) -> Invocation {
    Invocation {
        program: engine.program().to_string(),
        args: vec!["start".to_string(), name.to_string()],
        generated_files: Vec::new(),
    }
}

/// Remove a named container outright (`gpubox rm <name>`), so the next
/// `enter --name <name>` starts completely fresh.
pub fn remove(engine_program_name: &str, name: &str) -> Result<()> {
    let status = Command::new(engine_program_name)
        .args(["rm", "-f", name])
        .status()
        .with_context(|| format!("failed to run `{engine_program_name} rm -f {name}`"))?;
    if !status.success() {
        anyhow::bail!("`{engine_program_name} rm -f {name}` failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mounts::BindMount;
    use std::path::PathBuf;

    fn sample_spec() -> LaunchSpec {
        LaunchSpec {
            image: "rocm/rocm-terminal:6.1".to_string(),
            stack: "rocm".to_string(),
            env: vec![("HSA_OVERRIDE_GFX_VERSION".to_string(), "9.0.0".to_string())],
            mounts: vec![BindMount {
                host: PathBuf::from("/home/alice"),
                container: PathBuf::from("/home/alice"),
                read_only: false,
            }],
            device_args: vec!["--device".to_string(), "/dev/dri".to_string()],
            extra_args: vec![],
            packages: vec![],
            command: vec![],
            interactive: true,
            workdir: None,
        }
    }

    #[test]
    fn container_name_is_namespaced() {
        assert_eq!(container_name("ml"), "gpubox-ml");
    }

    #[test]
    fn create_invocation_is_detached_and_survives() {
        let inv = create_invocation(ContainerEngine::Docker, &sample_spec(), "gpubox-ml");
        assert_eq!(inv.program, "docker");
        assert!(inv.args.contains(&"-d".to_string()));
        assert!(!inv.args.contains(&"--rm".to_string()));
        assert!(inv.args.contains(&"--name".to_string()));
        assert!(inv.args.contains(&"gpubox-ml".to_string()));
        // Keeps itself alive rather than running the user's real command.
        assert_eq!(inv.args.last(), Some(&"infinity".to_string()));
        assert!(inv.args.iter().any(|a| a == "sleep"));
    }

    #[test]
    fn create_invocation_still_runs_setup_wrapper() {
        let mut spec = sample_spec();
        spec.packages = vec!["mesa-vulkan-drivers".to_string()];
        let inv = create_invocation(ContainerEngine::Docker, &spec, "gpubox-ml");
        let script = inv
            .args
            .iter()
            .find(|a| a.contains("setpriv"))
            .expect("wrapper script must still run once at creation");
        assert!(script.contains("apt-get"));
    }

    #[test]
    fn exec_invocation_uses_user_flag_not_wrapper() {
        let inv = exec_invocation(ContainerEngine::Docker, &sample_spec(), "gpubox-ml");
        assert_eq!(inv.program, "docker");
        assert!(inv.args.contains(&"exec".to_string()));
        assert!(inv.args.contains(&"--user".to_string()));
        assert!(!inv.args.iter().any(|a| a.contains("setpriv")));
        assert_eq!(inv.args.last(), Some(&"/bin/bash".to_string()));
    }

    #[test]
    fn exec_invocation_non_interactive_uses_dash_i_and_explicit_command() {
        let mut spec = sample_spec();
        spec.interactive = false;
        spec.command = vec!["python".to_string(), "train.py".to_string()];
        let inv = exec_invocation(ContainerEngine::Podman, &spec, "gpubox-ml");
        assert_eq!(inv.program, "podman");
        assert!(inv.args.contains(&"-i".to_string()));
        assert!(!inv.args.contains(&"-it".to_string()));
        assert_eq!(
            &inv.args[inv.args.len() - 2..],
            &["python".to_string(), "train.py".to_string()]
        );
    }

    #[test]
    fn start_invocation_targets_the_right_engine_and_name() {
        let inv = start_invocation(ContainerEngine::Podman, "gpubox-ml");
        assert_eq!(inv.program, "podman");
        assert_eq!(inv.args, vec!["start", "gpubox-ml"]);
    }
}
