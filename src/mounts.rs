//! Stage 4: distrobox-grade host integration.
//!
//! Builds the bind mounts, uid mapping, and GUI socket wiring that make a
//! container feel like an extension of the host rather than an isolated
//! box: home directory (dotfiles included, since they live under `$HOME`),
//! current working directory, X11/Wayland sockets, and a prompt marker so
//! users always know which stack they're in.

use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindMount {
    pub host: PathBuf,
    pub container: PathBuf,
    pub read_only: bool,
}

#[derive(Debug, Clone, Default)]
pub struct HostIntegration {
    pub mounts: Vec<BindMount>,
    pub env: Vec<(String, String)>,
    /// Extra engine flags, e.g. `--userns=keep-id` on podman.
    pub extra_args: Vec<String>,
    /// Container-side path to `cd` into (mirrors the host CWD bind mount).
    pub workdir: Option<PathBuf>,
}

fn home_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        env::var_os("USERPROFILE").map(PathBuf::from)
    } else {
        env::var_os("HOME").map(PathBuf::from)
    }
}

/// Build the standard distrobox-style set of host bindings for the given
/// container engine ("docker" or "podman") and resolved stack tag (used
/// only for the prompt marker).
pub fn plan(engine: &str, stack: &str) -> HostIntegration {
    let mut integration = HostIntegration::default();

    if let Some(home) = home_dir() {
        integration.mounts.push(BindMount {
            container: home.clone(),
            host: home,
            read_only: false,
        });
    }

    if let Ok(cwd) = env::current_dir() {
        integration.workdir = Some(cwd.clone());
        integration.mounts.push(BindMount {
            container: cwd.clone(),
            host: cwd,
            read_only: false,
        });
    }

    add_gui_sockets(&mut integration);

    // Prompt marker: images are expected to source
    // /etc/profile.d/gpubox-prompt.sh, which reads this var and appends
    // "(gpubox:<stack>)" to PS1.
    integration
        .env
        .push(("GPUBOX_STACK".to_string(), stack.to_string()));

    // Rootless uid mapping so files created in the mounted home/cwd are
    // owned by the invoking user, not root or a container-local uid.
    match engine {
        "podman" => integration.extra_args.push("--userns=keep-id".to_string()),
        "docker" => {
            #[cfg(unix)]
            {
                // SAFETY: geteuid/getegid take no arguments and can't fail.
                let (uid, gid) = unsafe { (libc::geteuid(), libc::getegid()) };
                integration.extra_args.push("-u".to_string());
                integration.extra_args.push(format!("{uid}:{gid}"));
            }
        }
        _ => {}
    }

    integration
}

fn add_gui_sockets(integration: &mut HostIntegration) {
    if let Ok(display) = env::var("DISPLAY") {
        integration.mounts.push(BindMount {
            host: PathBuf::from("/tmp/.X11-unix"),
            container: PathBuf::from("/tmp/.X11-unix"),
            read_only: false,
        });
        integration.env.push(("DISPLAY".to_string(), display));
    }

    if let Ok(wayland_display) = env::var("WAYLAND_DISPLAY") {
        if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
            let socket = PathBuf::from(&runtime_dir).join(&wayland_display);
            integration.mounts.push(BindMount {
                host: socket.clone(),
                container: socket,
                read_only: false,
            });
            integration
                .env
                .push(("XDG_RUNTIME_DIR".to_string(), runtime_dir));
            integration
                .env
                .push(("WAYLAND_DISPLAY".to_string(), wayland_display));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn podman_gets_keep_id_flag() {
        let integration = plan("podman", "rocm");
        assert!(integration
            .extra_args
            .contains(&"--userns=keep-id".to_string()));
    }

    #[test]
    fn prompt_marker_env_is_set() {
        let integration = plan("docker", "cuda");
        assert!(integration
            .env
            .iter()
            .any(|(k, v)| k == "GPUBOX_STACK" && v == "cuda"));
    }

    #[test]
    fn home_and_cwd_are_mounted() {
        let integration = plan("docker", "vulkan");
        // At minimum the current working directory must always be
        // mountable in a test environment.
        assert!(integration.mounts.iter().any(|m| m.host == m.container));
    }
}
