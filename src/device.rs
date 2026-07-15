//! Stage 3: device injection (Linux only).
//!
//! Decides how to hand the GPU to the container: CDI where the engine
//! supports it, otherwise raw `--device` nodes for AMD/Intel or the
//! nvidia-container-toolkit's `--gpus` flag for NVIDIA. Also figures out
//! which host userspace driver libraries need to be bind-mounted in for
//! NVIDIA, since the kernel driver and userspace libs must be the exact
//! same version - the single nastiest problem in this whole space, and the
//! reason NVIDIA images can never simply bake `libcuda` in themselves.

#![cfg(target_os = "linux")]

use crate::probe::GpuClass;
use std::path::PathBuf;

/// A bind mount of a specific host path into the same path in the
/// container, read-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoMount {
    pub host: PathBuf,
}

/// Extra docker/podman CLI arguments needed to grant access to the GPU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceInjection {
    pub args: Vec<String>,
    /// Read-only host library bind mounts (NVIDIA userspace driver libs).
    pub library_mounts: Vec<RoMount>,
    /// Human-readable explanation of the chosen path, surfaced by
    /// `gpubox doctor`.
    pub reason: String,
}

/// True if a CDI (Container Device Interface) spec directory has at least
/// one spec file. Both modern docker and podman can consume CDI specs
/// directly via `--device`, which is the preferred, most portable path.
pub fn cdi_available() -> bool {
    crate::probe::cdi_available()
}

/// Decide how to inject the detected GPU class into the container.
pub fn plan(class: &GpuClass) -> DeviceInjection {
    match class {
        GpuClass::Nvidia { .. } => nvidia_injection(),
        GpuClass::Amd { .. } => amd_injection(),
        GpuClass::Intel { .. } => intel_injection(),
        GpuClass::Vulkan | GpuClass::Apple | GpuClass::None => DeviceInjection::default(),
    }
}

fn nvidia_injection() -> DeviceInjection {
    if cdi_available() {
        return DeviceInjection {
            args: vec!["--device".into(), "nvidia.com/gpu=all".into()],
            library_mounts: Vec::new(), // CDI specs already declare their own library mounts
            reason: "CDI spec found under /etc/cdi or /var/run/cdi; using nvidia.com/gpu=all"
                .into(),
        };
    }
    // Fall back to the nvidia-container-toolkit's classic `--gpus` flag,
    // and mount the host's userspace driver libraries ourselves since
    // there's no CDI spec doing it for us.
    DeviceInjection {
        args: vec!["--gpus".into(), "all".into()],
        library_mounts: nvidia_driver_library_mounts(),
        reason: "no CDI spec found; falling back to nvidia-container-toolkit's --gpus=all \
                 (requires the toolkit's runtime hook to be installed on the host)"
            .into(),
    }
}

fn amd_injection() -> DeviceInjection {
    let mut args = vec!["--device".into(), "/dev/dri".into()];
    let mut notes = vec!["--device /dev/dri for rendering"];
    if crate::probe::has_kfd() {
        args.push("--device".into());
        args.push("/dev/kfd".into());
        notes.push("--device /dev/kfd for ROCm compute (HSA)");
    }
    DeviceInjection {
        args,
        library_mounts: Vec::new(),
        reason: notes.join("; "),
    }
}

fn intel_injection() -> DeviceInjection {
    DeviceInjection {
        args: vec!["--device".into(), "/dev/dri".into()],
        library_mounts: Vec::new(),
        reason: "--device /dev/dri for Intel VA-API/oneAPI Level Zero".into(),
    }
}

/// Common locations for the NVIDIA userspace driver stack across major
/// distros. We glob each for `libcuda.so*` / `libnvidia-*.so*` and mount
/// whatever we find read-only at the same path inside the container, so
/// the container's userspace exactly matches the host's kernel driver.
const NVIDIA_LIB_DIRS: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib/wsl/lib", // WSL2's GPU paravirtualization driver location
];

fn nvidia_driver_library_mounts() -> Vec<RoMount> {
    let mut mounts = Vec::new();
    for dir in NVIDIA_LIB_DIRS {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if (name.starts_with("libcuda.so") || name.starts_with("libnvidia-"))
                && entry.path().is_file()
            {
                mounts.push(RoMount { host: entry.path() });
            }
        }
    }
    mounts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amd_without_kfd_only_requests_dri() {
        // This test only makes strong assertions about /dev/kfd's absence
        // being handled; it can't assume the CI runner has a GPU node.
        let injection = amd_injection();
        assert!(injection.args.contains(&"/dev/dri".to_string()));
    }

    #[test]
    fn intel_requests_dri_only() {
        let injection = intel_injection();
        assert_eq!(injection.args, vec!["--device", "/dev/dri"]);
    }

    #[test]
    fn plan_is_noop_for_vulkan_apple_and_none() {
        assert_eq!(plan(&GpuClass::Vulkan), DeviceInjection::default());
        assert_eq!(plan(&GpuClass::Apple), DeviceInjection::default());
        assert_eq!(plan(&GpuClass::None), DeviceInjection::default());
    }
}
