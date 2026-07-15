//! macOS hardware probe.
//!
//! Apple Silicon Macs have an integrated GPU that's inseparable from the
//! SoC, so there's no PCI vendor/device pair to read - detecting the CPU
//! architecture is sufficient. Intel Macs are out of scope for GPU
//! passthrough (their dGPUs, when present, aren't exposed to Seatbelt-
//! sandboxed processes in any useful way); they fall through to CPU.

use super::{GpuClass, GpuDevice};

pub fn probe() -> Vec<GpuDevice> {
    if cfg!(target_arch = "aarch64") {
        vec![GpuDevice {
            vendor_id: None,
            device_id: None,
            node: Some("Apple GPU (integrated)".to_string()),
            class: GpuClass::Apple,
        }]
    } else {
        Vec::new()
    }
}
