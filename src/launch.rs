//! Wires stages 1-4 (probe, stack resolution, device injection, host
//! integration) together into a single [`LaunchSpec`] ready to hand to a
//! [`crate::backend`].

use crate::backend::{Engine, LaunchSpec};
use crate::mounts;
use crate::probe::{self, GpuClass};
use crate::stack::{self, ResolvedStack};
use anyhow::{bail, Result};

/// User-facing overrides, one per `gpubox` global CLI flag.
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub backend: Option<String>,
    pub image: Option<String>,
    /// Force a specific hardware classification instead of probing the
    /// host, e.g. `sm_86`, `gfx1100`, `arc`, `apple`, `vulkan`, `cpu`.
    pub gfx_override: Option<String>,
    /// Select a specific detected GPU on a multi-GPU/hybrid host, by
    /// 0-based index (`--gpu 1`) or coarse vendor name (`--gpu nvidia`).
    /// `None` keeps the default [`probe::pick_primary`] behavior.
    pub gpu: Option<String>,
    /// Give the container a name (`--name ml`) instead of the default
    /// (`container::DEFAULT_CONTAINER_NAME`, `"gpubox"`, shared across
    /// every stack/hardware config): either way, the container is created
    /// once and reattached on subsequent `gpubox run` invocations instead
    /// of being torn down. See [`crate::container`].
    pub name: Option<String>,
    /// Use a throwaway `--rm` container for this invocation instead of a
    /// persistent one (the pre-persistence default). Mutually exclusive
    /// with `name`.
    pub ephemeral: bool,
    /// Don't mount `$HOME` into the container at all.
    pub no_home: bool,
    /// Mount `$HOME` read-only instead of read-write.
    pub read_only_home: bool,
}

/// Parse a `--gfx-override` value into a [`GpuClass`]. Prefixes match the
/// same arch-tag conventions used throughout `data/pci_ids.toml`.
pub fn parse_gfx_override(value: &str) -> Result<GpuClass> {
    let lower = value.to_ascii_lowercase();
    let class = if lower.starts_with("sm_") {
        GpuClass::Nvidia { arch: lower }
    } else if lower.starts_with("gfx") {
        GpuClass::Amd { arch: lower }
    } else if matches!(lower.as_str(), "arc" | "xe" | "igpu") {
        GpuClass::Intel { class: lower }
    } else if lower == "apple" {
        GpuClass::Apple
    } else if lower == "vulkan" {
        GpuClass::Vulkan
    } else if matches!(lower.as_str(), "cpu" | "none") {
        GpuClass::None
    } else {
        bail!(
            "unrecognized --gfx-override value `{value}`; expected an sm_* / gfx* arch tag, \
             one of arc/xe/igpu, or apple/vulkan/cpu"
        );
    };
    Ok(class)
}

pub struct Plan {
    pub class: GpuClass,
    /// Every GPU the probe detected (before `--gpu`/primary selection),
    /// for `gpubox doctor` to surface on multi-GPU/hybrid hosts.
    pub devices: Vec<probe::GpuDevice>,
    pub resolved: ResolvedStack,
    pub engine: Engine,
    pub spec: LaunchSpec,
    /// Explanation of the device-injection path chosen (Linux only).
    pub device_reason: Option<String>,
}

/// Build the full launch plan: probe (or apply `--gfx-override`/`--gpu`),
/// resolve the stack, plan device injection, plan host integration, and
/// merge it all into a [`LaunchSpec`] plus the [`Engine`] that will run
/// it.
pub fn plan(overrides: &Overrides, command: Vec<String>, interactive: bool) -> Result<Plan> {
    if overrides.no_home && overrides.read_only_home {
        bail!("--no-home and --read-only-home are mutually exclusive");
    }

    let devices = probe::probe_host();
    let class = match &overrides.gfx_override {
        Some(value) => parse_gfx_override(value)?,
        None => match &overrides.gpu {
            // Multi-GPU/hybrid hosts (e.g. an Intel iGPU alongside an
            // NVIDIA dGPU) shouldn't have their hardware silently chosen
            // for them - `--gpu <index|vendor>` lets the user pick
            // explicitly instead of trusting `pick_primary`'s ranking.
            Some(selector) => probe::select(&devices, selector)?.class,
            None => probe::pick_primary(&devices).class.clone(),
        },
    };

    let resolved = stack::resolve(&class)?;

    let engine = match &overrides.backend {
        Some(name) => {
            Engine::parse(name).ok_or_else(|| anyhow::anyhow!("unrecognized --backend `{name}`"))?
        }
        None => Engine::default_for_platform_and_stack(&resolved.stack),
    };

    // `resolved.image` is always a Linux image reference (the vendor's
    // published CUDA/ROCm/oneAPI container, or the Ubuntu-based Vulkan/CPU
    // fallback) - it cannot run under `Engine::WindowsContainer`, which
    // needs a Windows Server Core/Nano Server-based image instead. Prefer
    // `resolved.windows_image` in that case; `gpubox doctor` warns when
    // neither it nor `--image` is set (see `doctor::report`).
    let image = overrides.image.clone().unwrap_or_else(|| {
        if engine == Engine::WindowsContainer {
            resolved
                .windows_image
                .clone()
                .unwrap_or_else(|| resolved.image.clone())
        } else {
            resolved.image.clone()
        }
    });

    let (device_args, device_reason, library_mounts) = device_injection(&class);

    let home_mode = match (overrides.no_home, overrides.read_only_home) {
        (true, _) => mounts::HomeMode::None,
        (false, true) => mounts::HomeMode::ReadOnly,
        (false, false) => mounts::HomeMode::ReadWrite,
    };
    let integration = mounts::plan_with_home_mode(engine.name(), &resolved.stack, home_mode);

    let mut env: Vec<(String, String)> = resolved.env.clone().into_iter().collect();
    env.extend(integration.env.clone());

    let mut spec_mounts = integration.mounts.clone();
    spec_mounts.extend(library_mounts);

    let spec = LaunchSpec {
        image,
        stack: resolved.stack.clone(),
        env,
        mounts: spec_mounts,
        device_args,
        extra_args: integration.extra_args.clone(),
        packages: resolved.packages.clone(),
        command,
        interactive,
        workdir: integration.workdir.clone(),
    };

    Ok(Plan {
        class,
        devices,
        resolved,
        engine,
        spec,
        device_reason,
    })
}

#[cfg(target_os = "linux")]
fn device_injection(
    class: &GpuClass,
) -> (Vec<String>, Option<String>, Vec<crate::mounts::BindMount>) {
    use crate::device;
    let injection = device::plan(class);
    let library_mounts = injection
        .library_mounts
        .into_iter()
        .map(|m| crate::mounts::BindMount {
            container: m.host.clone(),
            host: m.host,
            read_only: true,
        })
        .collect();
    (injection.args, Some(injection.reason), library_mounts)
}

#[cfg(not(target_os = "linux"))]
fn device_injection(
    _class: &GpuClass,
) -> (Vec<String>, Option<String>, Vec<crate::mounts::BindMount>) {
    (Vec::new(), None, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx_override_parses_nvidia_arch() {
        let class = parse_gfx_override("sm_86").unwrap();
        assert_eq!(
            class,
            GpuClass::Nvidia {
                arch: "sm_86".into()
            }
        );
    }

    #[test]
    fn gfx_override_parses_amd_arch() {
        let class = parse_gfx_override("gfx1100").unwrap();
        assert_eq!(
            class,
            GpuClass::Amd {
                arch: "gfx1100".into()
            }
        );
    }

    #[test]
    fn gfx_override_parses_intel_class() {
        let class = parse_gfx_override("arc").unwrap();
        assert_eq!(
            class,
            GpuClass::Intel {
                class: "arc".into()
            }
        );
    }

    #[test]
    fn gfx_override_rejects_garbage() {
        assert!(parse_gfx_override("banana").is_err());
    }

    #[test]
    fn plan_with_override_skips_hardware_probe() {
        let overrides = Overrides {
            backend: None,
            image: None,
            gfx_override: Some("gfx1100".to_string()),
            ..Default::default()
        };
        let plan = plan(&overrides, vec![], true).unwrap();
        assert_eq!(plan.resolved.stack, "rocm");
        assert_eq!(plan.spec.image, "rocm/rocm-terminal:6.1");
    }

    #[test]
    fn plan_honors_image_override() {
        let overrides = Overrides {
            backend: None,
            image: Some("mycorp/custom:latest".to_string()),
            gfx_override: Some("cpu".to_string()),
            ..Default::default()
        };
        let plan = plan(&overrides, vec![], true).unwrap();
        assert_eq!(plan.spec.image, "mycorp/custom:latest");
    }

    #[test]
    fn windows_container_backend_prefers_windows_image_over_linux_image() {
        let overrides = Overrides {
            backend: Some("windows-container".to_string()),
            gfx_override: Some("sm_86".to_string()),
            ..Default::default()
        };
        let plan = plan(&overrides, vec![], true).unwrap();
        assert_eq!(plan.resolved.image, "nvidia/cuda:12.9.2-devel-ubuntu24.04");
        assert_eq!(
            plan.spec.image,
            "mcr.microsoft.com/windows/servercore:ltsc2022"
        );
    }

    #[test]
    fn windows_container_backend_honors_explicit_image_override() {
        let overrides = Overrides {
            backend: Some("windows-container".to_string()),
            image: Some("myregistry/my-cuda-windows:latest".to_string()),
            gfx_override: Some("sm_86".to_string()),
            ..Default::default()
        };
        let plan = plan(&overrides, vec![], true).unwrap();
        assert_eq!(plan.spec.image, "myregistry/my-cuda-windows:latest");
    }

    #[test]
    fn plan_rejects_unknown_backend() {
        let overrides = Overrides {
            backend: Some("bogus-engine".to_string()),
            image: None,
            gfx_override: Some("cpu".to_string()),
            ..Default::default()
        };
        assert!(plan(&overrides, vec![], true).is_err());
    }
}
