//! Linux hardware probe: walks `/sys/class/drm`, reading PCI vendor/device
//! ids straight out of sysfs. Requires no root and no vendor tooling.

use super::{classify, GpuDevice};
use std::fs;
use std::path::{Path, PathBuf};

/// Parses a sysfs id file like `0x10de\n` into a `u16`.
fn read_hex_id(path: &Path) -> Option<u16> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    let hex = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u16::from_str_radix(hex, 16).ok()
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
        let Some(vendor_id) = read_hex_id(&device_dir.join("vendor")) else {
            continue;
        };
        let Some(device_id) = read_hex_id(&device_dir.join("device")) else {
            continue;
        };

        let class = classify(vendor_id, device_id);
        let node = card_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| format!("/dev/dri/{n}"));

        devices.push(GpuDevice {
            vendor_id: Some(vendor_id),
            device_id: Some(device_id),
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
}
