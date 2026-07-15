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

/// The invoking host user's identity, used to make the container actually
/// *be* that user instead of whatever account (if any) happens to already
/// own the mapped uid inside the image - which, for the common case of a
/// host uid 1000 against a stock `ubuntu:24.04` image (which itself ships
/// a baked-in `ubuntu` uid-1000 account), silently "impersonates" the
/// image's placeholder account: wrong `$HOME`, wrong `whoami`, wrong
/// prompt, even though file ownership on the bind mounts still happens to
/// line up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostIdentity {
    pub uid: u32,
    pub gid: u32,
    pub username: String,
    pub groupname: String,
    pub home: PathBuf,
    pub shell: String,
}

#[cfg(unix)]
pub fn current_identity() -> HostIdentity {
    use std::ffi::CStr;

    let uid = unsafe { libc::geteuid() };
    let gid = unsafe { libc::getegid() };

    let mut buf: Vec<libc::c_char> = vec![0; 16384];
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result: *mut libc::passwd = std::ptr::null_mut();
    // SAFETY: buf is sized well above typical NSS backend requirements;
    // getpwuid_r writes into `pwd` and points `result` at it on success,
    // or leaves `result` null (checked below) on failure/not-found.
    let looked_up = unsafe {
        let rc = libc::getpwuid_r(uid, &mut pwd, buf.as_mut_ptr(), buf.len(), &mut result);
        (rc == 0 && !result.is_null()).then(|| {
            (
                CStr::from_ptr(pwd.pw_name).to_string_lossy().into_owned(),
                CStr::from_ptr(pwd.pw_dir).to_string_lossy().into_owned(),
                CStr::from_ptr(pwd.pw_shell).to_string_lossy().into_owned(),
            )
        })
    };

    let (username, home, shell) = looked_up.unwrap_or_else(|| {
        (
            "gpubox".to_string(),
            env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()),
            "/bin/sh".to_string(),
        )
    });

    let mut gbuf: Vec<libc::c_char> = vec![0; 16384];
    let mut grp: libc::group = unsafe { std::mem::zeroed() };
    let mut gresult: *mut libc::group = std::ptr::null_mut();
    // SAFETY: same reasoning as getpwuid_r above.
    let groupname = unsafe {
        let rc = libc::getgrgid_r(gid, &mut grp, gbuf.as_mut_ptr(), gbuf.len(), &mut gresult);
        if rc == 0 && !gresult.is_null() {
            CStr::from_ptr(grp.gr_name).to_string_lossy().into_owned()
        } else {
            username.clone()
        }
    };

    HostIdentity {
        uid,
        gid,
        username,
        groupname,
        home: PathBuf::from(home),
        shell: if shell.is_empty() {
            "/bin/sh".to_string()
        } else {
            shell
        },
    }
}

#[cfg(not(unix))]
pub fn current_identity() -> HostIdentity {
    // Only meaningfully used by the Linux (Docker/Podman) backend; a
    // plausible fallback here purely so the code compiles everywhere.
    HostIdentity {
        uid: 0,
        gid: 0,
        username: "gpubox".to_string(),
        groupname: "gpubox".to_string(),
        home: home_dir().unwrap_or_else(|| PathBuf::from("/tmp")),
        shell: "/bin/sh".to_string(),
    }
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

/// Build the standard set of host bindings for the given container engine
/// ("docker" or "podman") and resolved stack tag (used only for the prompt
/// marker).
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

    match engine {
        "podman" | "docker" => {
            // `--userns=keep-id` (Podman) makes the container's numeric
            // uid/gid actually *mean* the host user on the user-namespace
            // level - required for a rootless-Podman privilege drop (see
            // backend::linux) to land on files with the right owner.
            // Docker's default (non-rootless) uid namespace already maps
            // 1:1 to the host, so it needs no equivalent flag.
            if engine == "podman" {
                integration.extra_args.push("--userns=keep-id".to_string());
                integration.extra_args.push("--group-add".to_string());
                integration.extra_args.push("keep-groups".to_string());
            }
            // Force $HOME/$USER/$LOGNAME to the *real* host user
            // regardless of what /etc/passwd says inside the container -
            // belt-and-braces alongside backend::linux's passwd-rewriting
            // wrapper, since env vars are what most tools actually read.
            let identity = current_identity();
            integration
                .env
                .push(("HOME".to_string(), identity.home.display().to_string()));
            integration
                .env
                .push(("USER".to_string(), identity.username.clone()));
            integration
                .env
                .push(("LOGNAME".to_string(), identity.username));
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

    #[test]
    fn docker_gets_identity_env_not_a_raw_uid_flag() {
        let integration = plan("docker", "cuda");
        // $HOME/$USER/$LOGNAME are forced to the real host identity...
        assert!(integration.env.iter().any(|(k, _)| k == "HOME"));
        assert!(integration.env.iter().any(|(k, _)| k == "USER"));
        assert!(integration.env.iter().any(|(k, _)| k == "LOGNAME"));
        // ...instead of the old raw `-u uid:gid` flag: uid mapping is now
        // handled inside the container by backend::linux's wrapper
        // script, since a raw `-u` would prevent it from running the
        // passwd fixup / package install as root first.
        assert!(!integration.extra_args.contains(&"-u".to_string()));
    }

    #[test]
    fn podman_gets_identity_env_and_keep_groups() {
        let integration = plan("podman", "rocm");
        assert!(integration.env.iter().any(|(k, _)| k == "HOME"));
        assert!(integration.extra_args.contains(&"--group-add".to_string()));
        assert!(integration.extra_args.contains(&"keep-groups".to_string()));
    }

    #[test]
    fn current_identity_reports_a_real_looking_uid() {
        let identity = current_identity();
        // On unix this must be the actual process uid; on the
        // non-unix fallback it's a placeholder, but should still at least
        // produce a non-empty username to build a passwd line from.
        assert!(!identity.username.is_empty());
        assert!(!identity.groupname.is_empty());
    }
}
