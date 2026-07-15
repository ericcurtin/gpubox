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
pub mod windows_container;

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
    /// Process-isolated Windows container with GPU compute passthrough -
    /// the CUDA-capable alternative to Windows Sandbox's `<VGpu>`, which
    /// only provides WDDM/DirectX paravirtualization. See
    /// `backend::windows_container` for the full rationale.
    WindowsContainer,
}

impl Engine {
    pub fn name(self) -> &'static str {
        match self {
            Engine::Docker => "docker",
            Engine::Podman => "podman",
            Engine::Seatbelt => "seatbelt",
            Engine::WindowsSandbox => "windows-sandbox",
            Engine::WindowsContainer => "windows-container",
        }
    }

    pub fn parse(s: &str) -> Option<Engine> {
        match s.to_ascii_lowercase().as_str() {
            "docker" => Some(Engine::Docker),
            "podman" => Some(Engine::Podman),
            "seatbelt" | "sandbox-exec" => Some(Engine::Seatbelt),
            "windows-sandbox" | "wsb" => Some(Engine::WindowsSandbox),
            "windows-container" | "wincontainer" | "win-container" => {
                Some(Engine::WindowsContainer)
            }
            _ => None,
        }
    }

    /// The default engine for the current host platform. Linux defaults to
    /// Docker (falling back to Podman if Docker isn't installed). Windows
    /// defaults to Windows Sandbox, since it works out of the box with no
    /// setup and needs no GPU for most uses; callers that know they need
    /// real GPU compute (CUDA/ROCm/oneAPI, not just CPU or Vulkan/DirectX
    /// rendering) should prefer [`Engine::default_for_platform_and_class`]
    /// instead, which switches to [`Engine::WindowsContainer`] in that
    /// case when it's available.
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

    /// Like [`Engine::default_for_platform`], but on Windows additionally
    /// prefers [`Engine::WindowsContainer`] over Windows Sandbox when the
    /// detected stack actually needs vendor GPU compute (`cuda`/`rocm`/
    /// `oneapi`) - Windows Sandbox's `<VGpu>` cannot run those workloads
    /// at all, so silently picking it there would be a launch that looks
    /// like it worked but where `nvidia-smi`/CUDA calls simply fail.
    pub fn default_for_platform_and_stack(stack: &str) -> Engine {
        if cfg!(target_os = "windows") {
            let needs_gpu_compute = matches!(stack, "cuda" | "rocm" | "oneapi");
            if needs_gpu_compute && Engine::WindowsContainer.is_available() {
                return Engine::WindowsContainer;
            }
        }
        Engine::default_for_platform()
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
            // Process-isolated Windows containers are driven by the same
            // `docker` CLI as Linux; whether the local Docker daemon is
            // actually configured for Windows containers (vs. Linux
            // containers via WSL2) can't be told from PATH alone, so
            // `ensure_available` doubles as the point where a misconfigured
            // daemon fails loudly instead of silently.
            Engine::WindowsContainer => which("docker"),
        }
    }

    pub fn all() -> [Engine; 5] {
        [
            Engine::Docker,
            Engine::Podman,
            Engine::Seatbelt,
            Engine::WindowsSandbox,
            Engine::WindowsContainer,
        ]
    }

    /// If this engine is one of the Linux docker/podman engines, the
    /// corresponding [`linux::ContainerEngine`] - used by
    /// [`crate::container`] and [`crate::cache`], both of which only make
    /// sense for a real, nameable, image-buildable container engine
    /// (Seatbelt has no containers to persist or build images for;
    /// Windows Sandbox always boots clean; Windows Containers use their
    /// own separate, Windows-image-only invocation builder).
    pub fn as_container_engine(self) -> Option<linux::ContainerEngine> {
        match self {
            Engine::Docker => Some(linux::ContainerEngine::Docker),
            Engine::Podman => Some(linux::ContainerEngine::Podman),
            _ => None,
        }
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
        Engine::WindowsContainer => windows_container::build_invocation(spec),
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
        "windows-sandbox, windows-container"
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
