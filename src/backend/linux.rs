//! Docker/Podman argv construction.

use super::{Invocation, LaunchSpec};
use crate::mounts::{self, HostIdentity};
use std::fmt::Write as _;

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

/// Builds the in-container wrapper script that runs as the image's
/// default user (root, for every image gpubox defaults to) before the
/// actual workload:
///
/// 1. Forces an `/etc/passwd`/`/etc/group` entry for the *real* host
///    uid/gid, overwriting any placeholder account the base image
///    already happens to define at that uid (e.g. Ubuntu's OCI images
///    ship their own `ubuntu` uid-1000 account) - without this, a host
///    user whose uid collides with that placeholder effectively gets
///    "logged in" as it instead: wrong `$HOME`, wrong `whoami`, wrong
///    prompt, even though file ownership on the bind mounts still
///    happens to line up.
/// 2. Installs any stack-specific apt packages (the Vulkan/CPU
///    fallback's `mesa-vulkan-drivers` et al; see `data/quirks.toml`).
/// 3. Drops from root to the real host uid/gid via `setpriv` (part of
///    util-linux, present on every image gpubox defaults to) and execs
///    the actual command.
///
/// This assumes the image starts as root (true for every image gpubox
/// defaults to) and, when `packages` is non-empty, that it's
/// Debian/Ubuntu-based (also true of every default image). A custom
/// `--image` override that doesn't start as root, or isn't apt-based
/// while relying on `packages`, won't work with this wrapper.
fn wrapper_script(identity: &HostIdentity, packages: &[String]) -> String {
    let mut script = String::from("set -e\n");

    let _ = writeln!(
        script,
        "sed -i '/^[^:]*:[^:]*:{uid}:/d' /etc/passwd 2>/dev/null || true",
        uid = identity.uid
    );
    let _ = writeln!(
        script,
        "echo '{username}:x:{uid}:{gid}:{username}:{home}:{shell}' >> /etc/passwd",
        username = identity.username,
        uid = identity.uid,
        gid = identity.gid,
        home = identity.home.display(),
        shell = identity.shell,
    );
    let _ = writeln!(
        script,
        "sed -i '/^[^:]*:[^:]*:{gid}:/d' /etc/group 2>/dev/null || true",
        gid = identity.gid
    );
    let _ = writeln!(
        script,
        "echo '{groupname}:x:{gid}:' >> /etc/group",
        groupname = identity.groupname,
        gid = identity.gid,
    );

    if !packages.is_empty() {
        let _ = writeln!(
            script,
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
             --no-install-recommends {}",
            packages.join(" ")
        );
    }

    let _ = writeln!(
        script,
        "exec setpriv --reuid {uid} --regid {gid} --clear-groups \"$@\"",
        uid = identity.uid,
        gid = identity.gid,
    );

    script
}

/// The argv tail that runs `spec`'s command (or an interactive shell) as
/// the real host user inside the container. Shared between the actual
/// `docker run`/`podman run` invocation and `gpubox generate --format
/// quadlet`, so generated artifacts stay faithful to what `enter`/`run`
/// actually does.
pub fn command_argv(spec: &LaunchSpec) -> Vec<String> {
    let identity = mounts::current_identity();
    let script = wrapper_script(&identity, &spec.packages);

    let mut argv = vec![
        "sh".to_string(),
        "-c".to_string(),
        script,
        "gpubox".to_string(),
    ];
    if spec.command.is_empty() {
        argv.push("/bin/bash".to_string());
    } else {
        argv.extend(spec.command.iter().cloned());
    }
    argv
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

    if let Some(workdir) = &spec.workdir {
        args.push("-w".to_string());
        args.push(workdir.display().to_string());
    }

    args.push(spec.image.clone());
    args.extend(command_argv(spec));

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
    fn docker_invocation_has_expected_shape() {
        let inv = build_invocation(ContainerEngine::Docker, &sample_spec());
        assert_eq!(inv.program, "docker");
        assert!(inv.args.contains(&"run".to_string()));
        assert!(inv.args.contains(&"-v".to_string()));
        assert!(inv.args.iter().any(|a| a == "/home/alice:/home/alice"));
        assert!(inv.args.contains(&"rocm/rocm-terminal:6.1".to_string()));
        // Default interactive command is still the tail of the argv.
        assert_eq!(inv.args.last(), Some(&"/bin/bash".to_string()));
    }

    #[test]
    fn wrapper_rewrites_passwd_and_drops_privileges() {
        let inv = build_invocation(ContainerEngine::Docker, &sample_spec());
        // The wrapper script is the 3rd-from-last-before-command arg
        // (`sh`, `-c`, `<script>`, `gpubox`, ...command); just look for it
        // by content rather than position.
        let script = inv
            .args
            .iter()
            .find(|a| a.contains("setpriv"))
            .expect("wrapper script must be present in argv");
        assert!(script.contains("/etc/passwd"));
        assert!(script.contains("/etc/group"));
        assert!(script.contains("exec setpriv --reuid"));
        assert!(script.contains("--clear-groups"));
        // No packages requested for this stack, so no apt-get.
        assert!(!script.contains("apt-get"));
    }

    #[test]
    fn wrapper_installs_packages_when_present() {
        let mut spec = sample_spec();
        spec.packages = vec![
            "mesa-vulkan-drivers".to_string(),
            "vulkan-tools".to_string(),
        ];
        let inv = build_invocation(ContainerEngine::Docker, &spec);
        let script = inv
            .args
            .iter()
            .find(|a| a.contains("setpriv"))
            .expect("wrapper script must be present in argv");
        assert!(script.contains("apt-get update"));
        assert!(script.contains("mesa-vulkan-drivers"));
        assert!(script.contains("vulkan-tools"));
    }

    #[test]
    fn extra_args_are_passed_through_verbatim() {
        // Engine-specific policy (e.g. podman's --userns=keep-id /
        // --group-add keep-groups) is decided in mounts::plan, not here;
        // this backend just passes extra_args through mechanically.
        let mut spec = sample_spec();
        spec.extra_args = vec!["--userns=keep-id".to_string()];
        let inv = build_invocation(ContainerEngine::Podman, &spec);
        assert!(inv.args.contains(&"--userns=keep-id".to_string()));
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
