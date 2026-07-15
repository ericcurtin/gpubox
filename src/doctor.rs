//! Stage 5: `gpubox doctor` - explain what was detected, what was chosen,
//! and how to override it. Good diagnostics are what make people trust
//! auto-detection.
//!
//! Three output modes:
//! - [`report`] - the default human-readable text.
//! - [`report_json`] - the same information, structured, for scripting
//!   (`gpubox doctor --json`).
//! - [`probe_snapshot_json`] - a hardware-probe-only snapshot with no
//!   paths, usernames, or other host-identifying detail, suitable for
//!   pasting into a bug report (`gpubox doctor --report`) and, since its
//!   shape mirrors the fixtures already used in `src/probe/*.rs`'s unit
//!   tests, for growing that fixture corpus with real-world hardware.

use crate::backend::Engine;
use crate::launch::{self, Overrides};
use crate::probe::GpuDevice;
use anyhow::Result;
use serde::Serialize;
use std::fmt::Write as _;

pub fn report(overrides: &Overrides) -> Result<String> {
    let plan = launch::plan(overrides, Vec::new(), true)?;
    let mut out = String::new();

    let _ = writeln!(out, "gpubox doctor");
    let _ = writeln!(out, "=============");
    let _ = writeln!(out);

    if plan.devices.len() > 1 {
        let _ = writeln!(out, "Detected GPUs     :");
        for (index, device) in plan.devices.iter().enumerate() {
            let marker = if device.class == plan.class { "*" } else { " " };
            let _ = writeln!(out, "  {marker} [{index}] {}", device.class);
        }
        let _ = writeln!(
            out,
            "                    (* = selected; override with --gpu <index|vendor>, e.g. \
             `--gpu 1` or `--gpu nvidia`)"
        );
    }
    let _ = writeln!(out, "Detected hardware : {}", plan.class);
    let _ = writeln!(out, "Resolved stack    : {}", plan.resolved.stack);
    let _ = writeln!(out, "Container image   : {}", plan.spec.image);
    let _ = writeln!(out, "Matched rule      : {}", plan.resolved.rule_key);

    if let Some(notes) = &plan.resolved.notes {
        let _ = writeln!(out, "Notes             :");
        for line in notes.trim().lines() {
            let _ = writeln!(out, "  {line}");
        }
    }

    if !plan.spec.env.is_empty() {
        let _ = writeln!(out, "Quirk env vars    :");
        for (k, v) in &plan.spec.env {
            let _ = writeln!(out, "  {k}={v}");
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "Backend           : {}", plan.engine);
    let available = plan.engine.is_available();
    let _ = writeln!(
        out,
        "  available       : {}",
        if available {
            "yes"
        } else {
            "NO - not found on PATH"
        }
    );

    if let Some(reason) = &plan.device_reason {
        let _ = writeln!(out, "Device injection  : {reason}");
    }

    if plan.engine.as_container_engine().is_some() {
        let _ = writeln!(
            out,
            "Container         : {} (persistent by default; override with --name, or use \
             --rm for a throwaway one)",
            crate::container::DEFAULT_CONTAINER_NAME
        );
    }

    if plan.engine == Engine::Podman {
        let _ = writeln!(
            out,
            "Podman mode       : {}",
            if is_rootless_podman() {
                "rootless (uid-mapped via --userns=keep-id; first-class, no root required on \
                 the host)"
            } else {
                "rootful (running as root, or a setuid/root-daemon Podman install)"
            }
        );
    }

    if plan.engine == Engine::WindowsContainer {
        let _ = writeln!(
            out,
            "Note              : process-isolated Windows container with GPU compute \
             passthrough - CUDA-capable, unlike Windows Sandbox's <VGpu> (WDDM/DirectX-only \
             paravirtualization). The host's own GPU driver is used automatically; don't bake \
             one into the image."
        );
        if overrides.image.is_none() && plan.resolved.windows_image.is_none() {
            let _ = writeln!(
                out,
                "WARNING           : no `windows_image` quirk and no --image override - the \
                 image in use (`{}`) is a Linux image and will NOT run under a Windows \
                 container. Pass --image with a Windows Server Core/Nano Server-based image.",
                plan.spec.image
            );
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "Other backends available on this platform:");
    for engine in Engine::all() {
        if engine == plan.engine {
            continue;
        }
        if platform_supports(engine) {
            let mark = if engine.is_available() { "yes" } else { "no" };
            let _ = writeln!(out, "  {engine:<15} {mark}");
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "Overrides:");
    let _ = writeln!(out, "  --backend <name>       force a specific backend (docker, podman, seatbelt, windows-sandbox, windows-container)");
    let _ = writeln!(
        out,
        "  --image <ref>          use a custom image instead of {}",
        plan.resolved.image
    );
    let _ = writeln!(out, "  --gfx-override <arch>  force a hardware classification (e.g. sm_86, gfx1100, arc, apple, vulkan, cpu)");
    let _ = writeln!(
        out,
        "  --gpu <index|vendor>   pick a specific GPU on a multi-GPU/hybrid host"
    );
    let _ = writeln!(
        out,
        "  --name <name>          [run] use this container name instead of the default (`{}`)",
        crate::container::DEFAULT_CONTAINER_NAME
    );
    let _ = writeln!(
        out,
        "  --rm                   [run] use a throwaway container for this run instead of the \
         persistent default"
    );
    let _ = writeln!(
        out,
        "  --no-home              don't mount $HOME into the sandbox at all"
    );
    let _ = writeln!(
        out,
        "  --read-only-home       mount $HOME read-only instead of read-write"
    );

    Ok(out)
}

fn platform_supports(engine: Engine) -> bool {
    match engine {
        Engine::Docker | Engine::Podman => cfg!(target_os = "linux"),
        Engine::Seatbelt => cfg!(target_os = "macos"),
        Engine::WindowsSandbox | Engine::WindowsContainer => cfg!(target_os = "windows"),
    }
}

/// Best-effort detection of rootless Podman: true if we're not running as
/// uid 0 (root). This is exactly the condition under which
/// `mounts::plan`'s `--userns=keep-id`/`--group-add keep-groups` handling
/// matters - rootless is Podman's normal, first-class mode of operation,
/// not a fallback, so this is purely informational (surfaced by `gpubox
/// doctor`) rather than a warning.
#[cfg(unix)]
fn is_rootless_podman() -> bool {
    unsafe { libc::geteuid() != 0 }
}

#[cfg(not(unix))]
fn is_rootless_podman() -> bool {
    true
}

/// Structured, scriptable equivalent of [`report`] (`gpubox doctor
/// --json`). Unlike [`probe_snapshot_json`], this includes the fully
/// resolved plan (image, env, backend) for the *current* host - not meant
/// for sharing, just for feeding into other tools.
pub fn report_json(overrides: &Overrides) -> Result<String> {
    let plan = launch::plan(overrides, Vec::new(), true)?;

    #[derive(Serialize)]
    struct DoctorJson<'a> {
        devices: &'a [GpuDevice],
        selected: &'a crate::probe::GpuClass,
        resolved: &'a crate::stack::ResolvedStack,
        backend: String,
        backend_available: bool,
        device_reason: &'a Option<String>,
    }

    let json = DoctorJson {
        devices: &plan.devices,
        selected: &plan.class,
        resolved: &plan.resolved,
        backend: plan.engine.name().to_string(),
        backend_available: plan.engine.is_available(),
        device_reason: &plan.device_reason,
    };

    Ok(serde_json::to_string_pretty(&json)?)
}

/// An anonymized hardware-probe snapshot (`gpubox doctor --report`): just
/// the vendor/device ids and resulting classification for every detected
/// GPU, plus host OS/arch - no paths, no usernames, nothing host-
/// identifying. Meant to be pasted directly into a bug report when gpubox
/// misidentifies (or fails to identify) a card, and its shape - a plain
/// list of `(vendor_id, device_id) -> class` - is exactly what a
/// `data/pci_ids.toml` PR or a new `probe::linux`/`probe::windows`
/// fixture test needs, so real-world reports can grow that corpus
/// directly.
pub fn probe_snapshot_json() -> Result<String> {
    #[derive(Serialize)]
    struct Snapshot<'a> {
        os: &'a str,
        arch: &'a str,
        devices: &'a [GpuDevice],
    }

    let devices = crate::probe::probe_host();
    let snapshot = Snapshot {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        devices: &devices,
    };

    Ok(serde_json::to_string_pretty(&snapshot)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_mentions_detected_and_resolved_stack() {
        let overrides = Overrides {
            backend: None,
            image: None,
            gfx_override: Some("gfx1100".to_string()),
            ..Default::default()
        };
        let text = report(&overrides).unwrap();
        assert!(text.contains("AMD (gfx1100)"));
        assert!(text.contains("Resolved stack    : rocm"));
        assert!(text.contains("--gfx-override"));
    }

    #[test]
    fn windows_container_without_windows_image_warns() {
        // amd.gfx1100 has no `windows_image` quirk and no `--image`
        // override, so the (Linux) image in `resolved.image` would
        // silently be wrong under a Windows container - `doctor` must
        // call that out instead of staying quiet.
        let overrides = Overrides {
            backend: Some("windows-container".to_string()),
            gfx_override: Some("gfx1100".to_string()),
            ..Default::default()
        };
        let text = report(&overrides).unwrap();
        assert!(text.contains("WARNING"));
    }

    #[test]
    fn windows_container_with_windows_image_quirk_does_not_warn() {
        let overrides = Overrides {
            backend: Some("windows-container".to_string()),
            gfx_override: Some("sm_86".to_string()),
            ..Default::default()
        };
        let text = report(&overrides).unwrap();
        assert!(!text.contains("WARNING"));
    }
}
