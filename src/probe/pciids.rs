//! Loads the embedded `data/pci_ids.toml` hardware classification database
//! and turns a raw (vendor_id, device_id) pair into a [`GpuClass`].

use super::GpuClass;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

const RAW: &str = include_str!("../../data/pci_ids.toml");

const VENDOR_NVIDIA: u16 = 0x10de;
const VENDOR_AMD: u16 = 0x1002;
const VENDOR_INTEL: u16 = 0x8086;

#[derive(Debug, Deserialize)]
struct Db {
    #[serde(default)]
    nvidia: HashMap<String, String>,
    #[serde(default)]
    amd: HashMap<String, String>,
    #[serde(default)]
    intel: HashMap<String, String>,
}

static DB: OnceLock<Db> = OnceLock::new();

fn db() -> &'static Db {
    DB.get_or_init(|| {
        toml::from_str(RAW).expect("data/pci_ids.toml is embedded and must always parse")
    })
}

fn key(device_id: u16) -> String {
    format!("0x{device_id:04x}")
}

/// Classify a (vendor_id, device_id) pair read from PCI configuration
/// space (e.g. `/sys/class/drm/card0/device/{vendor,device}` on Linux).
pub fn classify(vendor_id: u16, device_id: u16) -> GpuClass {
    let k = key(device_id);
    match vendor_id {
        VENDOR_NVIDIA => GpuClass::Nvidia {
            arch: db()
                .nvidia
                .get(&k)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
        },
        VENDOR_AMD => GpuClass::Amd {
            arch: db()
                .amd
                .get(&k)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
        },
        VENDOR_INTEL => {
            let class = db()
                .intel
                .get(&k)
                .cloned()
                .unwrap_or_else(|| "igpu".to_string());
            GpuClass::Intel { class }
        }
        _ => GpuClass::Vulkan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_nvidia_ada_device() {
        // AD102 - RTX 4090
        assert_eq!(
            classify(0x10de, 0x2684),
            GpuClass::Nvidia {
                arch: "sm_89".into()
            }
        );
    }

    #[test]
    fn known_amd_rdna3_device() {
        // Navi 31 - RX 7900 XTX
        assert_eq!(
            classify(0x1002, 0x744c),
            GpuClass::Amd {
                arch: "gfx1100".into()
            }
        );
    }

    #[test]
    fn known_amd_igpu_quirk_device() {
        // Renoir iGPU, needs HSA_OVERRIDE_GFX_VERSION quirk (see quirks.toml)
        assert_eq!(
            classify(0x1002, 0x1636),
            GpuClass::Amd {
                arch: "gfx90c".into()
            }
        );
    }

    #[test]
    fn known_intel_arc_device() {
        assert_eq!(
            classify(0x8086, 0x56a0),
            GpuClass::Intel {
                class: "arc".into()
            }
        );
    }

    #[test]
    fn unknown_nvidia_device_falls_back_to_unknown_arch() {
        assert_eq!(
            classify(0x10de, 0xffff),
            GpuClass::Nvidia {
                arch: "unknown".into()
            }
        );
    }

    #[test]
    fn unrecognized_vendor_is_generic_vulkan() {
        assert_eq!(classify(0x1234, 0x0001), GpuClass::Vulkan);
    }

    #[test]
    fn unknown_intel_device_defaults_to_igpu_class() {
        assert_eq!(
            classify(0x8086, 0x0000),
            GpuClass::Intel {
                class: "igpu".into()
            }
        );
    }
}
