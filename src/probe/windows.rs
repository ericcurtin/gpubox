//! Windows hardware probe.
//!
//! There's no sysfs equivalent, so we query WMI's `Win32_VideoController`
//! class via PowerShell for each adapter's `PNPDeviceID`, which embeds the
//! same PCI `VEN_xxxx&DEV_xxxx` pair Linux exposes through sysfs. This is a
//! stock OS management interface, not a GPU vendor tool, so it fits the
//! "nothing vendor-specific installed" requirement.

use super::{classify, GpuDevice};
use std::process::Command;

const PS_QUERY: &str =
    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID";

pub fn probe() -> Vec<GpuDevice> {
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", PS_QUERY])
        .output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().filter_map(parse_pnp_device_id).collect()
}

/// Parses a PnP device id like
/// `PCI\VEN_10DE&DEV_2684&SUBSYS_...&REV_A1\4&1a2b3c4d&0&0008` into a
/// [`GpuDevice`].
fn parse_pnp_device_id(line: &str) -> Option<GpuDevice> {
    let line = line.trim();
    let vendor_id = extract_hex_field(line, "VEN_")?;
    let device_id = extract_hex_field(line, "DEV_")?;
    Some(GpuDevice {
        vendor_id: Some(vendor_id),
        device_id: Some(device_id),
        node: Some(line.to_string()),
        class: classify(vendor_id, device_id),
    })
}

fn extract_hex_field(haystack: &str, marker: &str) -> Option<u16> {
    let start = haystack.find(marker)? + marker.len();
    let rest = &haystack[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(rest.len());
    u16::from_str_radix(&rest[..end], 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_pnp_device_id() {
        let line = r"PCI\VEN_10DE&DEV_2684&SUBSYS_87131458&REV_A1\4&1a2b3c4d&0&0008";
        let device = parse_pnp_device_id(line).unwrap();
        assert_eq!(device.vendor_id, Some(0x10de));
        assert_eq!(device.device_id, Some(0x2684));
    }

    #[test]
    fn parses_amd_pnp_device_id() {
        let line = r"PCI\VEN_1002&DEV_744C&SUBSYS_0B361002&REV_C8\4&1a2b3c4d&0&0008";
        let device = parse_pnp_device_id(line).unwrap();
        assert_eq!(device.vendor_id, Some(0x1002));
        assert_eq!(device.device_id, Some(0x744c));
    }

    #[test]
    fn rejects_malformed_line() {
        assert!(parse_pnp_device_id("not a pnp id").is_none());
    }

    /// WMI's `Win32_VideoController` lists one row per adapter, so a
    /// hybrid laptop (Intel iGPU + NVIDIA dGPU) surfaces as multiple
    /// lines from a single `Get-CimInstance` call; both must parse.
    #[test]
    fn multiple_lines_yield_multiple_devices() {
        let output = concat!(
            r"PCI\VEN_8086&DEV_46A6&SUBSYS_00000000&REV_0C\3&11583659&0&10",
            "\n",
            r"PCI\VEN_10DE&DEV_2684&SUBSYS_87131458&REV_A1\4&1a2b3c4d&0&0008",
        );
        let devices: Vec<GpuDevice> = output.lines().filter_map(parse_pnp_device_id).collect();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].vendor_id, Some(0x8086));
        assert_eq!(devices[1].vendor_id, Some(0x10de));
    }
}
