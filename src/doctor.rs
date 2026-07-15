//! Stage 5: `gpubox doctor` - explain what was detected, what was chosen,
//! and how to override it. Good diagnostics are what make people trust
//! auto-detection.

use crate::backend::Engine;
use crate::launch::{self, Overrides};
use anyhow::Result;
use std::fmt::Write as _;

pub fn report(overrides: &Overrides) -> Result<String> {
    let plan = launch::plan(overrides, Vec::new(), true)?;
    let mut out = String::new();

    let _ = writeln!(out, "gpubox doctor");
    let _ = writeln!(out, "=============");
    let _ = writeln!(out);
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
    let _ = writeln!(out, "  --backend <name>       force a specific backend (docker, podman, seatbelt, windows-sandbox)");
    let _ = writeln!(
        out,
        "  --image <ref>          use a custom image instead of {}",
        plan.resolved.image
    );
    let _ = writeln!(out, "  --gfx-override <arch>  force a hardware classification (e.g. sm_86, gfx1100, arc, apple, vulkan, cpu)");

    Ok(out)
}

fn platform_supports(engine: Engine) -> bool {
    match engine {
        Engine::Docker | Engine::Podman => cfg!(target_os = "linux"),
        Engine::Seatbelt => cfg!(target_os = "macos"),
        Engine::WindowsSandbox => cfg!(target_os = "windows"),
    }
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
        };
        let text = report(&overrides).unwrap();
        assert!(text.contains("AMD (gfx1100)"));
        assert!(text.contains("Resolved stack    : rocm"));
        assert!(text.contains("--gfx-override"));
    }
}
