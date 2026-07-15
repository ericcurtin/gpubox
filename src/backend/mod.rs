//! Backend abstraction: turns a fully-resolved [`LaunchSpec`] into an
//! executable command for whichever sandboxing technology fits the host
//! platform.
//!
//! * Linux: Docker or Podman containers (Docker is the default).
//! * macOS: Seatbelt (`sandbox-exec`), Apple's native process sandboxing.
//!   There's no Linux kernel to pass a device node through, so instead of
//!   containerizing we sandbox the process directly on the host and let it
//!   use Metal natively.
//! * Windows: Windows Sandbox, a lightweight Hyper-V-backed VM with opt-in
//!   GPU (vGPU) passthrough. Deliberately not WSL: WSL2 is a Linux VM, and
//!   the point of the Windows backend is a native, non-Linux sandbox.

pub mod linux;
pub mod macos;
pub mod windows;

use crate::mounts::BindMount;
use anyhow::{bail, Result};
use std::env;
use std::path::PathBuf;

/// Everything a backend needs to know to launch the sandbox.
#[derive(Debug, Clone)]
pub struct LaunchSpec {
    pub image: String,
    pub stack: String,
    pub env: Vec<(String, String)>,
    pub mounts: Vec<BindMount>,
    /// Raw engine device flags, e.g. `--gpus all` or `--device /dev/dri`
    /// (Linux only; ignored by the macOS/Windows backends).
    pub device_args: Vec<String>,
    /// Additional raw engine flags (e.g. `--userns=keep-id`, `-u uid:gid`).
    pub extra_args: Vec<String>,
    /// Apt packages the Linux backend must install inside the container
    /// before running `command` (see [`crate::stack::ResolvedStack::packages`]).
    /// Ignored by the macOS/Windows backends.
    pub packages: Vec<String>,
    /// Command to run inside the sandbox. Empty means "interactive shell".
    pub command: Vec<String>,
    pub interactive: bool,
    /// Working directory to `cd` into inside the sandbox (the container
    /// path corresponding to the host CWD bind mount).
    pub workdir: Option<PathBuf>,
}

/// A concrete argv ready to be executed (or printed, for `--dry-run`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    pub program: String,
    pub args: Vec<String>,
    /// Any files that had to be written to disk to support this
    /// invocation (e.g. a generated Seatbelt profile or `.wsb` config).
    pub generated_files: Vec<(PathBuf, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Docker,
    Podman,
    Seatbelt,
    WindowsSandbox,
}

impl Engine {
    pub fn name(self) -> &'static str {
        match self {
            Engine::Docker => "docker",
            Engine::Podman => "podman",
            Engine::Seatbelt => "seatbelt",
            Engine::WindowsSandbox => "windows-sandbox",
        }
    }

    pub fn parse(s: &str) -> Option<Engine> {
        match s.to_ascii_lowercase().as_str() {
            "docker" => Some(Engine::Docker),
            "podman" => Some(Engine::Podman),
            "seatbelt" | "sandbox-exec" => Some(Engine::Seatbelt),
            "windows-sandbox" | "wsb" => Some(Engine::WindowsSandbox),
            _ => None,
        }
    }

    /// The default engine for the current host platform. Linux defaults to
    /// Docker (falling back to Podman if Docker isn't installed).
    pub fn default_for_platform() -> Engine {
        if cfg!(target_os = "macos") {
            Engine::Seatbelt
        } else if cfg!(target_os = "windows") {
            Engine::WindowsSandbox
        } else if !Engine::Docker.is_available() && Engine::Podman.is_available() {
            Engine::Podman
        } else {
            Engine::Docker
        }
    }

    /// Whether the underlying tool for this engine is installed on PATH
    /// (or, for Windows Sandbox, present in its usual System32 location).
    pub fn is_available(self) -> bool {
        match self {
            Engine::Docker => which("docker"),
            Engine::Podman => which("podman"),
            Engine::Seatbelt => which("sandbox-exec"),
            Engine::WindowsSandbox => {
                which("WindowsSandbox.exe")
                    || PathBuf::from(r"C:\Windows\System32\WindowsSandbox.exe").exists()
            }
        }
    }

    pub fn all() -> [Engine; 4] {
        [
            Engine::Docker,
            Engine::Podman,
            Engine::Seatbelt,
            Engine::WindowsSandbox,
        ]
    }
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Build the executable invocation for `spec` under `engine`.
pub fn build_invocation(engine: Engine, spec: &LaunchSpec) -> Result<Invocation> {
    match engine {
        Engine::Docker => Ok(linux::build_invocation(
            linux::ContainerEngine::Docker,
            spec,
        )),
        Engine::Podman => Ok(linux::build_invocation(
            linux::ContainerEngine::Podman,
            spec,
        )),
        Engine::Seatbelt => macos::build_invocation(spec),
        Engine::WindowsSandbox => windows::build_invocation(spec),
    }
}

/// True if `bin` (with a platform-appropriate extension) can be found on
/// `PATH`. Deliberately simple - existence, not "is it actually
/// executable" - since the latter varies awkwardly across platforms and
/// engines already fail loudly and clearly if invoked incorrectly.
fn which(bin: &str) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    let candidates: Vec<String> = if cfg!(windows) && !bin.ends_with(".exe") {
        vec![bin.to_string(), format!("{bin}.exe")]
    } else {
        vec![bin.to_string()]
    };
    env::split_paths(&path).any(|dir| candidates.iter().any(|c| dir.join(c).is_file()))
}

/// Ensure the chosen engine is actually usable, with a clear error message
/// pointing at how to fix it (install it, or pick another with
/// `--backend`).
pub fn ensure_available(engine: Engine) -> Result<()> {
    if engine.is_available() {
        return Ok(());
    }
    bail!(
        "backend `{engine}` was selected but isn't available on this host.\n\
         Install it, or choose another with --backend (available choices for this \
         platform: {}).",
        platform_engines_hint()
    );
}

fn platform_engines_hint() -> &'static str {
    if cfg!(target_os = "linux") {
        "docker, podman"
    } else if cfg!(target_os = "macos") {
        "seatbelt"
    } else if cfg!(target_os = "windows") {
        "windows-sandbox"
    } else {
        "none supported on this platform"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_engine_names() {
        assert_eq!(Engine::parse("docker"), Some(Engine::Docker));
        assert_eq!(Engine::parse("Podman"), Some(Engine::Podman));
        assert_eq!(Engine::parse("sandbox-exec"), Some(Engine::Seatbelt));
        assert_eq!(Engine::parse("wsb"), Some(Engine::WindowsSandbox));
        assert_eq!(Engine::parse("bogus"), None);
    }

    #[test]
    fn default_engine_matches_current_platform_family() {
        let engine = Engine::default_for_platform();
        #[cfg(target_os = "macos")]
        assert_eq!(engine, Engine::Seatbelt);
        #[cfg(target_os = "windows")]
        assert_eq!(engine, Engine::WindowsSandbox);
        #[cfg(target_os = "linux")]
        assert!(matches!(engine, Engine::Docker | Engine::Podman));
    }
}
