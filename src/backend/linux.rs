//! Docker/Podman argv construction.

use super::{Invocation, LaunchSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerEngine {
    Docker,
    Podman,
}

impl ContainerEngine {
    fn program(self) -> &'static str {
        match self {
            ContainerEngine::Docker => "docker",
            ContainerEngine::Podman => "podman",
        }
    }
}

pub fn build_invocation(engine: ContainerEngine, spec: &LaunchSpec) -> Invocation {
    let mut args = vec!["run".to_string(), "--rm".to_string()];

    if spec.interactive {
        args.push("-it".to_string());
    }

    for mount in &spec.mounts {
        let ro = if mount.read_only { ":ro" } else { "" };
        args.push("-v".to_string());
        args.push(format!(
            "{}:{}{}",
            mount.host.display(),
            mount.container.display(),
            ro
        ));
    }

    for (key, value) in &spec.env {
        args.push("-e".to_string());
        args.push(format!("{key}={value}"));
    }

    args.extend(spec.device_args.iter().cloned());
    args.extend(spec.extra_args.iter().cloned());

    // Engine-specific integration flags matching distrobox's behavior.
    match engine {
        ContainerEngine::Podman => {
            // Rootless podman: allow supplementary groups (video/render)
            // to carry through so /dev/dri permissions actually apply.
            args.push("--group-add".to_string());
            args.push("keep-groups".to_string());
        }
        ContainerEngine::Docker => {}
    }

    if let Some(workdir) = &spec.workdir {
        args.push("-w".to_string());
        args.push(workdir.display().to_string());
    }

    args.push(spec.image.clone());

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mounts::BindMount;
    use std::path::PathBuf;

    fn sample_spec() -> LaunchSpec {
        LaunchSpec {
            image: "gpubox/rocm:6.1".to_string(),
            stack: "rocm".to_string(),
            env: vec![("HSA_OVERRIDE_GFX_VERSION".to_string(), "9.0.0".to_string())],
            mounts: vec![BindMount {
                host: PathBuf::from("/home/alice"),
                container: PathBuf::from("/home/alice"),
                read_only: false,
            }],
            device_args: vec!["--device".to_string(), "/dev/dri".to_string()],
            extra_args: vec![],
            command: vec![],
            interactive: true,
            workdir: None,
        }
    }

    #[test]
    fn docker_invocation_has_expected_shape() {
        let inv = build_invocation(ContainerEngine::Docker, &sample_spec());
        assert_eq!(inv.program, "docker");
        assert!(inv.args.contains(&"run".to_string()));
        assert!(inv.args.contains(&"-v".to_string()));
        assert!(inv.args.iter().any(|a| a == "/home/alice:/home/alice"));
        assert!(inv.args.contains(&"gpubox/rocm:6.1".to_string()));
        assert_eq!(inv.args.last(), Some(&"/bin/bash".to_string()));
    }

    #[test]
    fn podman_gets_keep_groups() {
        let inv = build_invocation(ContainerEngine::Podman, &sample_spec());
        assert!(inv.args.contains(&"keep-groups".to_string()));
        assert_eq!(inv.program, "podman");
    }

    #[test]
    fn non_interactive_run_uses_explicit_command() {
        let mut spec = sample_spec();
        spec.interactive = false;
        spec.command = vec!["python".to_string(), "train.py".to_string()];
        let inv = build_invocation(ContainerEngine::Docker, &spec);
        assert!(!inv.args.contains(&"-it".to_string()));
        assert_eq!(
            &inv.args[inv.args.len() - 2..],
            &["python".to_string(), "train.py".to_string()]
        );
    }
}
