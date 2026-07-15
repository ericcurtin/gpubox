//! Linux hardware probe: walks `/sys/class/drm`, reading PCI vendor/device
//! ids straight out of sysfs. Requires no root and no vendor tooling.
//!
//! Not every GPU is a PCI device: SoC-integrated GPUs (Apple Silicon under
//! Asahi Linux, NVIDIA's Tegra/Grace-family SoCs used in Jetson and
//! DGX Spark-class systems, and ARM SBCs generally) show up under
//! `/sys/class/drm` as *platform* devices, which have no `vendor`/`device`
//! attribute files - those are PCI-specific. Treating "no PCI id" as "no
//! GPU" silently drops all of these to the CPU fallback, which is wrong:
//! there's a real render node right there. So a card lacking PCI ids is
//! still reported - as [`GpuClass::Vulkan`] by default, or a best-effort
//! vendor guess from the device tree `compatible` string when we can get
//! one - unless it's a known software-only test driver (`vgem`/`vkms`).

use super::{classify, GpuClass, GpuDevice};
use std::fs;
use std::path::{Path, PathBuf};

/// Parses a sysfs id file like `0x10de\n` into a `u16`.
fn read_hex_id(path: &Path) -> Option<u16> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    let hex = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u16::from_str_radix(hex, 16).ok()
}

/// Kernel DRM drivers that back a virtual/software-only device rather
/// than real GPU hardware (used in CI and for testing KMS). These also
/// show up with no PCI vendor/device ids, so without this exclusion
/// they'd otherwise be misreported as a real GPU.
const VIRTUAL_DRM_DRIVERS: &[&str] = &["vgem", "vkms"];

/// The kernel driver bound to a `/sys/class/drm/cardN/device` node, read
/// from the `driver` symlink's target (e.g. `.../drivers/nouveau` ->
/// `"nouveau"`).
fn driver_name(device_dir: &Path) -> Option<String> {
    let target = fs::read_link(device_dir.join("driver")).ok()?;
    target
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
}

/// Best-effort vendor guess for a platform-device GPU (no PCI ids) from
/// its device-tree `compatible` string, e.g. `"nvidia,gb10-gpu"`. Falls
/// back to the generic [`GpuClass::Vulkan`] when nothing more specific
/// can be said - which still gets a real `/dev/dri` render node handed to
/// the container (see `src/device.rs`), just without a vendor-specific
/// runtime stack.
fn classify_platform_device(device_dir: &Path) -> GpuClass {
    let compatible = fs::read(device_dir.join("of_node/compatible"))
        .map(|bytes| String::from_utf8_lossy(&bytes).to_lowercase())
        .unwrap_or_default();
    if compatible.contains("nvidia") {
        // Confident about the vendor, not the exact chip - resolves via
        // the `nvidia.unknown` quirks.toml rule, and still gets proper
        // NVIDIA device injection (CDI / --gpus all / driver lib mounts)
        // rather than being treated as a vendor-neutral Vulkan device.
        GpuClass::Nvidia {
            arch: "unknown".to_string(),
        }
    } else if compatible.contains("amd") {
        GpuClass::Amd {
            arch: "unknown".to_string(),
        }
    } else {
        GpuClass::Vulkan
    }
}

pub fn probe() -> Vec<GpuDevice> {
    probe_at(Path::new("/sys/class/drm"))
}

/// Testable core: probes an arbitrary root instead of always
/// `/sys/class/drm`, so unit tests can point it at a fixture directory.
fn probe_at(drm_root: &Path) -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    let Ok(entries) = fs::read_dir(drm_root) else {
        return devices;
    };

    // Only look at top-level "cardN" nodes (not connectors like
    // "cardN-DP-1") to avoid enumerating the same GPU multiple times.
    let mut card_dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("card") && !n.contains('-'))
                .unwrap_or(false)
        })
        .collect();
    card_dirs.sort();

    for card_dir in card_dirs {
        let device_dir = card_dir.join("device");
        if !device_dir.is_dir() {
            continue;
        }

        let vendor_id = read_hex_id(&device_dir.join("vendor"));
        let device_id = read_hex_id(&device_dir.join("device"));

        let class = match (vendor_id, device_id) {
            (Some(vendor_id), Some(device_id)) => classify(vendor_id, device_id),
            _ => {
                // No PCI ids: either a platform-device GPU (see module
                // docs) or a software-only test driver we should ignore.
                if driver_name(&device_dir)
                    .is_some_and(|d| VIRTUAL_DRM_DRIVERS.contains(&d.as_str()))
                {
                    continue;
                }
                classify_platform_device(&device_dir)
            }
        };

        let node = card_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| format!("/dev/dri/{n}"));

        devices.push(GpuDevice {
            vendor_id,
            device_id,
            node,
            class,
        });
    }

    devices
}

/// True if a CDI (Container Device Interface) spec directory with at least
/// one spec file exists. Used by the device-injection stage to prefer CDI
/// over raw `--device` flags when the container engine supports it.
pub fn cdi_available() -> bool {
    for dir in ["/etc/cdi", "/var/run/cdi"] {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                    || path.extension().and_then(|e| e.to_str()) == Some("json")
                {
                    return true;
                }
            }
        }
    }
    false
}

/// True if the kernel exposes an AMD KFD (compute) node, meaning ROCm's
/// HSA runtime can attach to the GPU.
pub fn has_kfd() -> bool {
    Path::new("/dev/kfd").exists()
}

/// True if the system has at least one Vulkan ICD manifest installed,
/// which is a reasonable proxy for "some Vulkan driver already works
/// here" and is used as a last-resort signal for unrecognized GPUs.
pub fn has_vulkan_icd() -> bool {
    for dir in ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"] {
        if let Ok(entries) = fs::read_dir(dir) {
            if entries.flatten().next().is_some() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_id(path: &Path, value: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, value).unwrap();
    }

    #[test]
    fn probes_single_nvidia_card() {
        let tmp = tempfile::tempdir().unwrap();
        let card = tmp.path().join("card0/device");
        write_id(&card.join("vendor"), "0x10de\n");
        write_id(&card.join("device"), "0x2684\n");

        let devices = probe_at(tmp.path());
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vendor_id, Some(0x10de));
        assert_eq!(devices[0].node.as_deref(), Some("/dev/dri/card0"));
    }

    #[test]
    fn ignores_connector_nodes() {
        let tmp = tempfile::tempdir().unwrap();
        let card = tmp.path().join("card0/device");
        write_id(&card.join("vendor"), "0x1002\n");
        write_id(&card.join("device"), "0x744c\n");
        fs::create_dir_all(tmp.path().join("card0-DP-1")).unwrap();

        let devices = probe_at(tmp.path());
        assert_eq!(devices.len(), 1);
    }

    #[test]
    fn missing_root_yields_no_devices() {
        let devices = probe_at(Path::new("/nonexistent/path/for/gpubox/tests"));
        assert!(devices.is_empty());
    }

    /// A GPU with no PCI vendor/device attribute files at all (no driver
    /// symlink either) - e.g. Apple Silicon's GPU under Asahi Linux - must
    /// still be reported, as the generic Vulkan fallback, rather than
    /// silently dropped to "no GPU".
    #[test]
    fn platform_device_without_pci_ids_is_reported_as_vulkan() {
        let tmp = tempfile::tempdir().unwrap();
        let device_dir = tmp.path().join("card0/device");
        fs::create_dir_all(&device_dir).unwrap();

        let devices = probe_at(tmp.path());
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vendor_id, None);
        assert_eq!(devices[0].class, GpuClass::Vulkan);
        assert_eq!(devices[0].node.as_deref(), Some("/dev/dri/card0"));
    }

    #[cfg(unix)]
    #[test]
    fn platform_device_compatible_string_identifies_nvidia() {
        let tmp = tempfile::tempdir().unwrap();
        let device_dir = tmp.path().join("card0/device");
        let of_node = device_dir.join("of_node");
        fs::create_dir_all(&of_node).unwrap();
        fs::write(of_node.join("compatible"), b"nvidia,gb10-gpu\0nvidia,gpu\0").unwrap();

        let devices = probe_at(tmp.path());
        assert_eq!(devices.len(), 1);
        assert_eq!(
            devices[0].class,
            GpuClass::Nvidia {
                arch: "unknown".to_string()
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn virtual_test_drivers_are_ignored() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let device_dir = tmp.path().join("card0/device");
        fs::create_dir_all(&device_dir).unwrap();
        let driver_target = tmp.path().join("bus/platform/drivers/vgem");
        fs::create_dir_all(&driver_target).unwrap();
        symlink(&driver_target, device_dir.join("driver")).unwrap();

        let devices = probe_at(tmp.path());
        assert!(devices.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn non_virtual_platform_driver_is_still_reported() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let device_dir = tmp.path().join("card0/device");
        fs::create_dir_all(&device_dir).unwrap();
        let driver_target = tmp.path().join("bus/platform/drivers/asahi");
        fs::create_dir_all(&driver_target).unwrap();
        symlink(&driver_target, device_dir.join("driver")).unwrap();

        let devices = probe_at(tmp.path());
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].class, GpuClass::Vulkan);
    }
}
