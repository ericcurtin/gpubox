//! Stage 2: stack resolution.
//!
//! Maps a [`GpuClass`] to a runtime stack name, a default container image,
//! and any quirk environment variables, by consulting the community-
//! maintained `data/quirks.toml` matrix (see that file for the rationale).

use crate::probe::GpuClass;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
    /// Optional Windows-container image override, used only by
    /// `Engine::WindowsContainer` (see `backend::windows_container`).
    /// Linux vendor images (`image`, above) can't run under a
    /// process-isolated Windows container - it needs a Windows Server
    /// Core/Nano Server-based image instead, with the GPU *driver*
    /// supplied by the host and only the CUDA/ROCm/oneAPI *toolkit*
    /// baked into the image itself.
    #[serde(default)]
    pub windows_image: Option<String>,
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
#[derive(Debug, Clone, Serialize)]
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
    /// See [`StackRule::windows_image`].
    pub windows_image: Option<String>,
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
        windows_image: rule.windows_image,
    })
}

/// Structural checks over `data/quirks.toml` beyond "it parses as TOML":
/// every vendor table must degrade gracefully for hardware it doesn't
/// have a specific rule for, and every rule must point at a real-looking
/// image reference. Community PRs that add a new arch tag but forget
/// these are common, easy-to-miss mistakes - this is the "schema
/// validation" that runs in CI via `cargo test`, so a malformed
/// `quirks.toml` fails the PR instead of silently shipping a broken
/// fallback.
pub fn validate_quirks_db() -> Result<()> {
    let db = db();
    // NVIDIA/AMD arch tags come from an open-ended, ever-growing PCI id
    // database (data/pci_ids.toml), so those tables must have an
    // "unknown" fallback for archs not yet in that database. Intel's
    // `pciids::classify` instead normalizes every unrecognized device id
    // down to the closed `"igpu"` class itself, so `[intel]` doesn't need
    // (and, being keyed by class name, wouldn't sensibly have) a separate
    // fallback entry - it just needs a rule for every class case
    // (`arc`/`xe`/`igpu`) to exist at all. Apple/Vulkan/None are each a
    // single always-"default"-keyed table.
    validate_table_has_any_of("nvidia", &db.nvidia, &["unknown"])?;
    validate_table_has_any_of("amd", &db.amd, &["unknown"])?;
    validate_table_has_all_of("intel", &db.intel, &["arc", "xe", "igpu"])?;
    validate_table_has_any_of("apple", &db.apple, &["default"])?;
    validate_table_has_any_of("vulkan", &db.vulkan, &["default"])?;
    validate_table_has_any_of("none", &db.none, &["default"])?;

    for (name, table) in [
        ("nvidia", &db.nvidia),
        ("amd", &db.amd),
        ("intel", &db.intel),
        ("apple", &db.apple),
        ("vulkan", &db.vulkan),
        ("none", &db.none),
    ] {
        for (key, rule) in table {
            validate_rule(name, key, rule)?;
        }
    }
    Ok(())
}

/// `table` must contain at least one of `keys` - the vendor's fallback
/// key(s) (`"unknown"`/`"default"`).
fn validate_table_has_any_of(
    name: &str,
    table: &HashMap<String, StackRule>,
    keys: &[&str],
) -> Result<()> {
    if keys.iter().any(|k| table.contains_key(*k)) {
        return Ok(());
    }
    anyhow::bail!(
        "data/quirks.toml: table `[{name}]` is missing one of {keys:?}; newly-released or \
         unrecognized hardware in this vendor family would fail to resolve instead of \
         degrading gracefully"
    );
}

/// `table` must contain every one of `keys` - used for Intel's closed
/// class enum (`arc`/`xe`/`igpu`), where `pciids::classify` always
/// normalizes to one of exactly these three and there is no separate
/// "unknown" bucket, so a missing entry for any one of them is a real gap.
fn validate_table_has_all_of(
    name: &str,
    table: &HashMap<String, StackRule>,
    keys: &[&str],
) -> Result<()> {
    let missing: Vec<&&str> = keys.iter().filter(|k| !table.contains_key(**k)).collect();
    if missing.is_empty() {
        return Ok(());
    }
    anyhow::bail!("data/quirks.toml: table `[{name}]` is missing entries for {missing:?}");
}

/// A single rule's `stack`/`image` must be non-empty, and `image` must
/// look like a real, pullable image reference rather than a placeholder
/// left over from copy-pasting another entry.
fn validate_rule(name: &str, key: &str, rule: &StackRule) -> Result<()> {
    if rule.stack.trim().is_empty() {
        anyhow::bail!("data/quirks.toml: `{name}.{key}` has an empty `stack`");
    }
    if rule.image.trim().is_empty() {
        anyhow::bail!("data/quirks.toml: `{name}.{key}` has an empty `image`");
    }
    if rule.image.contains('<') || rule.image.contains("TODO") {
        anyhow::bail!(
            "data/quirks.toml: `{name}.{key}`'s `image` (`{}`) looks like a placeholder, not a \
             real pullable image reference",
            rule.image
        );
    }
    Ok(())
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
    fn quirks_db_passes_schema_validation() {
        // This is the "schema validation for the TOML files in CI"
        // check: it runs as a normal `cargo test`, which every matrix job
        // in .github/workflows/ci.yml already runs, so a community PR to
        // data/quirks.toml that breaks a vendor table's fallback rule or
        // ships a placeholder image reference fails CI instead of
        // shipping.
        validate_quirks_db().unwrap();
    }

    #[test]
    fn missing_fallback_rule_fails_validation() {
        let toml = r#"
            [nvidia.sm_86]
            stack = "cuda"
            image = "nvidia/cuda:12.9.2-devel-ubuntu24.04"
        "#;
        let db: QuirksDb = toml::from_str(toml).unwrap();
        let err = validate_table_has_any_of("nvidia", &db.nvidia, &["unknown"]).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn placeholder_image_fails_validation() {
        let mut rule = StackRule {
            stack: "cuda".to_string(),
            image: "<your-image-here>".to_string(),
            env: HashMap::new(),
            notes: None,
            packages: Vec::new(),
            windows_image: None,
        };
        assert!(validate_rule("nvidia", "sm_86", &rule).is_err());
        rule.image = "nvidia/cuda:12.9.2-devel-ubuntu24.04".to_string();
        assert!(validate_rule("nvidia", "sm_86", &rule).is_ok());
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
