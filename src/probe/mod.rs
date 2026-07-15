//! Stage 1: hardware probe.
//!
//! Walks host-native interfaces only (sysfs on Linux, WMI on Windows,
//! architecture detection on macOS) - deliberately never shells out to a
//! vendor tool like `nvidia-smi` or `rocm-smi`, since the entire premise of
//! gpubox is that the host has nothing GPU-vendor-specific installed.

// Every platform module is compiled on every target (so its unit tests run
// everywhere in CI), but only the module matching the current target is
// ever called from `probe_platform` below; suppress the resulting
// dead-code warnings on the other two platforms.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod linux;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod macos;
mod pciids;
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
mod windows;

pub use pciids::classify;

#[cfg(target_os = "linux")]
pub use linux::{cdi_available, has_kfd, has_vulkan_icd};

use std::fmt;

/// Normalized classification of a piece of GPU hardware, independent of the
/// raw vendor/device id pair used to derive it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuClass {
    /// NVIDIA GPU, `arch` is a CUDA compute-capability tag, e.g. `"sm_86"`,
    /// or `"unknown"` if the device id wasn't recognized.
    Nvidia { arch: String },
    /// AMD GPU, `arch` is a gfx architecture tag, e.g. `"gfx1100"`, or
    /// `"unknown"` if the device id wasn't recognized.
    Amd { arch: String },
    /// Intel GPU, `class` is one of `"arc"`, `"xe"`, `"igpu"`.
    Intel { class: String },
    /// Apple Silicon integrated GPU.
    Apple,
    /// A GPU was found but couldn't be attributed to a known vendor;
    /// generic Vulkan is the only thing we can reasonably promise.
    Vulkan,
    /// No usable GPU found - CPU fallback.
    None,
}

impl fmt::Display for GpuClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuClass::Nvidia { arch } => write!(f, "NVIDIA ({arch})"),
            GpuClass::Amd { arch } => write!(f, "AMD ({arch})"),
            GpuClass::Intel { class } => write!(f, "Intel ({class})"),
            GpuClass::Apple => write!(f, "Apple Silicon"),
            GpuClass::Vulkan => write!(f, "unrecognized GPU (Vulkan capable)"),
            GpuClass::None => write!(f, "none"),
        }
    }
}

/// A single detected GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// PCI vendor id, e.g. `0x10de` for NVIDIA. `None` on platforms (or
    /// device classes) where PCI enumeration doesn't apply, e.g. Apple
    /// Silicon.
    pub vendor_id: Option<u16>,
    /// PCI device id.
    pub device_id: Option<u16>,
    /// Host-side identifier useful for diagnostics: a DRM render node path
    /// on Linux (`/dev/dri/renderD128`), a PCI bus address, or a PNP device
    /// id fragment on Windows.
    pub node: Option<String>,
    pub class: GpuClass,
}

impl GpuDevice {
    fn none() -> Self {
        GpuDevice {
            vendor_id: None,
            device_id: None,
            node: None,
            class: GpuClass::None,
        }
    }
}

/// Probe the host for GPUs. Never fails - hardware probing is best-effort
/// and any error (permissions, missing sysfs, etc.) degrades to an empty
/// device list, which stack resolution then treats as the CPU fallback.
pub fn probe_host() -> Vec<GpuDevice> {
    let devices = probe_platform();
    if devices.is_empty() {
        vec![GpuDevice::none()]
    } else {
        devices
    }
}

#[cfg(target_os = "linux")]
fn probe_platform() -> Vec<GpuDevice> {
    linux::probe()
}

#[cfg(target_os = "macos")]
fn probe_platform() -> Vec<GpuDevice> {
    macos::probe()
}

#[cfg(target_os = "windows")]
fn probe_platform() -> Vec<GpuDevice> {
    windows::probe()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn probe_platform() -> Vec<GpuDevice> {
    Vec::new()
}

/// Known AMD integrated-GPU gfx architecture tags, used only to rank
/// discrete cards above iGPUs when multiple AMD devices are present.
const AMD_IGPU_ARCHES: &[&str] = &["gfx902", "gfx90c"];

/// Pick the "best" device to drive stack resolution when multiple GPUs are
/// present. Priority: discrete NVIDIA > discrete AMD > Intel Arc/Xe > any
/// remaining AMD/Intel iGPU > Apple > generic Vulkan > none.
pub fn pick_primary(devices: &[GpuDevice]) -> &GpuDevice {
    fn rank(d: &GpuDevice) -> u8 {
        match &d.class {
            GpuClass::Nvidia { .. } => 0,
            GpuClass::Amd { arch } if !AMD_IGPU_ARCHES.contains(&arch.as_str()) => 1,
            GpuClass::Intel { class } if class == "arc" || class == "xe" => 2,
            GpuClass::Amd { .. } => 3,
            GpuClass::Intel { .. } => 4,
            GpuClass::Apple => 5,
            GpuClass::Vulkan => 6,
            GpuClass::None => 7,
        }
    }
    devices
        .iter()
        .min_by_key(|d| rank(d))
        .unwrap_or(&devices[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_host_never_empty() {
        // Even on an unsupported platform / with no permissions, we must
        // always return at least the "none" sentinel so downstream stack
        // resolution has something to match against.
        let devices = probe_host();
        assert!(!devices.is_empty());
    }

    #[test]
    fn pick_primary_prefers_nvidia_over_amd() {
        let devices = vec![
            GpuDevice {
                vendor_id: Some(0x1002),
                device_id: Some(0x744c),
                node: None,
                class: GpuClass::Amd {
                    arch: "gfx1100".into(),
                },
            },
            GpuDevice {
                vendor_id: Some(0x10de),
                device_id: Some(0x2684),
                node: None,
                class: GpuClass::Nvidia {
                    arch: "sm_89".into(),
                },
            },
        ];
        let primary = pick_primary(&devices);
        assert!(matches!(primary.class, GpuClass::Nvidia { .. }));
    }
}
