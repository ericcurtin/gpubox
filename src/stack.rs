//! Stage 2: stack resolution.
//!
//! Maps a [`GpuClass`] to a runtime stack name, a default container image,
//! and any quirk environment variables, by consulting the community-
//! maintained `data/quirks.toml` matrix (see that file for the rationale).

use crate::probe::GpuClass;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

const RAW: &str = include_str!("../data/quirks.toml");

#[derive(Debug, Clone, Deserialize)]
pub struct StackRule {
    pub stack: String,
    pub image: String,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub notes: Option<String>,
    /// Apt packages to layer on top of `image` (used by the Vulkan/CPU
    /// fallbacks, which start from a plain distro base rather than a
    /// vendor-published image). Consumed by `gpubox generate --format
    /// dockerfile`; ignored otherwise.
    #[serde(default)]
    pub packages: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QuirksDb {
    #[serde(default)]
    nvidia: HashMap<String, StackRule>,
    #[serde(default)]
    amd: HashMap<String, StackRule>,
    #[serde(default)]
    intel: HashMap<String, StackRule>,
    #[serde(default)]
    apple: HashMap<String, StackRule>,
    #[serde(default)]
    vulkan: HashMap<String, StackRule>,
    #[serde(default)]
    none: HashMap<String, StackRule>,
}

static DB: OnceLock<QuirksDb> = OnceLock::new();

fn db() -> &'static QuirksDb {
    DB.get_or_init(|| {
        toml::from_str(RAW).expect("data/quirks.toml is embedded and must always parse")
    })
}

/// The final, fully-resolved answer to "what should I run, and how".
#[derive(Debug, Clone)]
pub struct ResolvedStack {
    /// Which detected hardware this was resolved from (human-readable).
    pub detected: String,
    /// Runtime stack name: "cuda", "rocm", "oneapi", "metal", "vulkan", or
    /// "cpu".
    pub stack: String,
    /// Default container image tag for this stack.
    pub image: String,
    /// Extra environment variables that must be set inside the container
    /// for this specific piece of hardware to work (quirks).
    pub env: HashMap<String, String>,
    /// Human-readable explanation, surfaced by `gpubox doctor`.
    pub notes: Option<String>,
    /// Which quirks.toml table.key produced this rule, for diagnostics.
    pub rule_key: String,
    /// Apt packages to layer on top of `image` (see [`StackRule::packages`]).
    pub packages: Vec<String>,
}

fn lookup(
    table: &HashMap<String, StackRule>,
    key: &str,
    vendor: &str,
) -> Result<(StackRule, String)> {
    if let Some(rule) = table.get(key) {
        return Ok((rule.clone(), format!("{vendor}.{key}")));
    }
    for fallback in ["unknown", "default"] {
        if let Some(rule) = table.get(fallback) {
            return Ok((rule.clone(), format!("{vendor}.{fallback}")));
        }
    }
    anyhow::bail!(
        "data/quirks.toml has no rule for `{vendor}.{key}` and no `{vendor}.unknown`/`{vendor}.default` fallback"
    )
}

/// Resolve a detected [`GpuClass`] into a concrete stack/image/env.
pub fn resolve(class: &GpuClass) -> Result<ResolvedStack> {
    let db = db();
    let (rule, rule_key) = match class {
        GpuClass::Nvidia { arch } => lookup(&db.nvidia, arch, "nvidia"),
        GpuClass::Amd { arch } => lookup(&db.amd, arch, "amd"),
        GpuClass::Intel { class } => lookup(&db.intel, class, "intel"),
        GpuClass::Apple => lookup(&db.apple, "default", "apple"),
        GpuClass::Vulkan => lookup(&db.vulkan, "default", "vulkan"),
        GpuClass::None => lookup(&db.none, "default", "none"),
    }
    .with_context(|| format!("resolving stack for detected hardware {class}"))?;

    Ok(ResolvedStack {
        detected: class.to_string(),
        stack: rule.stack,
        image: rule.image,
        env: rule.env,
        notes: rule.notes,
        rule_key,
        packages: rule.packages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_nvidia_arch_to_cuda() {
        let resolved = resolve(&GpuClass::Nvidia {
            arch: "sm_86".into(),
        })
        .unwrap();
        assert_eq!(resolved.stack, "cuda");
        assert!(resolved.image.contains("cuda"));
        assert_eq!(resolved.rule_key, "nvidia.sm_86");
    }

    #[test]
    fn resolves_gfx90c_with_hsa_override_quirk() {
        let resolved = resolve(&GpuClass::Amd {
            arch: "gfx90c".into(),
        })
        .unwrap();
        assert_eq!(resolved.stack, "rocm");
        assert_eq!(
            resolved
                .env
                .get("HSA_OVERRIDE_GFX_VERSION")
                .map(String::as_str),
            Some("9.0.0")
        );
        assert!(resolved.notes.is_some());
    }

    #[test]
    fn unknown_nvidia_arch_falls_back_to_vulkan() {
        let resolved = resolve(&GpuClass::Nvidia {
            arch: "unknown".into(),
        })
        .unwrap();
        assert_eq!(resolved.stack, "vulkan");
        assert_eq!(resolved.rule_key, "nvidia.unknown");
    }

    #[test]
    fn intel_arc_resolves_to_oneapi() {
        let resolved = resolve(&GpuClass::Intel {
            class: "arc".into(),
        })
        .unwrap();
        assert_eq!(resolved.stack, "oneapi");
    }

    #[test]
    fn apple_resolves_to_metal() {
        let resolved = resolve(&GpuClass::Apple).unwrap();
        assert_eq!(resolved.stack, "metal");
    }

    #[test]
    fn none_resolves_to_cpu() {
        let resolved = resolve(&GpuClass::None).unwrap();
        assert_eq!(resolved.stack, "cpu");
    }

    #[test]
    fn vulkan_class_resolves_to_vulkan() {
        let resolved = resolve(&GpuClass::Vulkan).unwrap();
        assert_eq!(resolved.stack, "vulkan");
        assert_eq!(resolved.image, "ubuntu:24.04");
        assert!(resolved.packages.iter().any(|p| p == "mesa-vulkan-drivers"));
    }

    #[test]
    fn cuda_and_rocm_images_carry_no_apt_packages() {
        // Vendor images (CUDA/ROCm/oneAPI) already ship their full stack;
        // only the plain-distro Vulkan/CPU fallbacks need `packages`.
        let cuda = resolve(&GpuClass::Nvidia {
            arch: "sm_86".into(),
        })
        .unwrap();
        assert!(cuda.packages.is_empty());
    }
}
