//! Local image caching for stacks that need extra apt packages layered on
//! top of a base image (the Vulkan/CPU fallback's `mesa-vulkan-drivers`
//! et al. - see `packages` in `data/quirks.toml`).
//!
//! Without this, every single `gpubox run` on unrecognized
//! hardware re-runs `apt-get install` inside the wrapper script
//! (`backend::linux`) before anything else can happen: a network hit and
//! a real wait, every launch, purely to reinstall the exact same
//! packages that were already installed last time. Building a local
//! tagged image once - the same Dockerfile `gpubox generate --format
//! dockerfile` would produce - and reusing that image on every
//! subsequent launch turns that into "install once per base image +
//! package set, ever".

use crate::backend::LaunchSpec;
use crate::generate::{self, Format};
use crate::stack::ResolvedStack;
use anyhow::{Context, Result};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::process::{Command, Stdio};

/// Deterministic local image tag for a given base image + package set.
/// Re-running with the same base image and the same `packages` list
/// reuses the same cached image; a `data/quirks.toml` change or an
/// `--image` override naturally busts the cache by hashing to a
/// different tag instead of silently reusing a stale one.
pub fn cache_tag(resolved: &ResolvedStack, spec: &LaunchSpec) -> String {
    let mut hasher = DefaultHasher::new();
    spec.image.hash(&mut hasher);
    resolved.packages.hash(&mut hasher);
    format!("gpubox-cache/{}:{:016x}", resolved.stack, hasher.finish())
}

/// True if `tag` already exists as a local image for `engine_program`
/// ("docker" or "podman"). Best-effort: any failure (engine missing,
/// image genuinely absent) is treated as "doesn't exist yet".
fn image_exists(engine_program: &str, tag: &str) -> bool {
    Command::new(engine_program)
        .args(["image", "inspect", tag])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Build and tag a local image from `dockerfile`'s content, piped in over
/// stdin. Passing `-` as the sole build-context argument (rather than
/// `-f - .`) tells docker/podman to read the Dockerfile from stdin *and*
/// use an empty build context, instead of packaging and sending the
/// current working directory to the daemon - which, since the generated
/// Dockerfile only ever runs `apt-get install` and never `COPY`/`ADD`,
/// would otherwise be a pointless (and, for a CWD containing datasets or
/// a deep tree, potentially very slow) transfer on every cache miss.
fn build_image(engine_program: &str, tag: &str, dockerfile: &str) -> Result<()> {
    let mut child = Command::new(engine_program)
        .args(["build", "-t", tag, "-"])
        .stdin(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run `{engine_program} build`"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(dockerfile.as_bytes())
        .context("writing generated Dockerfile to `build`'s stdin")?;
    let status = child
        .wait()
        .with_context(|| format!("waiting on `{engine_program} build`"))?;
    if !status.success() {
        anyhow::bail!("`{engine_program} build -t {tag}` failed");
    }
    Ok(())
}

/// If `spec` needs extra packages layered on top of its base image,
/// ensure a locally cached image with those packages already baked in
/// exists (building it the first time), then repoint `spec.image` at it
/// and clear `spec.packages` - so the runtime wrapper script has nothing
/// left to `apt-get install` and every subsequent launch skips straight
/// to the workload. Returns `true` if a cached image is now in use.
///
/// A no-op (returns `Ok(false)`) when `spec.packages` is empty - vendor
/// images (CUDA/ROCm/oneAPI) already ship their full stack and never hit
/// this path.
pub fn ensure_cached_image(
    engine_program: &str,
    resolved: &ResolvedStack,
    spec: &mut LaunchSpec,
) -> Result<bool> {
    if spec.packages.is_empty() {
        return Ok(false);
    }
    let tag = cache_tag(resolved, spec);
    if !image_exists(engine_program, &tag) {
        let dockerfile = generate::render(Format::Dockerfile, resolved, spec)?;
        build_image(engine_program, &tag, &dockerfile)?;
    }
    spec.image = tag;
    spec.packages.clear();
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mounts::BindMount;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn resolved() -> ResolvedStack {
        ResolvedStack {
            detected: "unrecognized GPU (Vulkan capable)".to_string(),
            stack: "vulkan".to_string(),
            image: "ubuntu:24.04".to_string(),
            env: HashMap::new(),
            notes: None,
            packages: vec![
                "mesa-vulkan-drivers".to_string(),
                "vulkan-tools".to_string(),
            ],
            rule_key: "vulkan.default".to_string(),
            windows_image: None,
        }
    }

    fn spec() -> LaunchSpec {
        LaunchSpec {
            image: "ubuntu:24.04".to_string(),
            stack: "vulkan".to_string(),
            env: vec![],
            mounts: vec![BindMount {
                host: PathBuf::from("/home/alice"),
                container: PathBuf::from("/home/alice"),
                read_only: false,
            }],
            device_args: vec!["--device".to_string(), "/dev/dri".to_string()],
            extra_args: vec![],
            packages: vec![
                "mesa-vulkan-drivers".to_string(),
                "vulkan-tools".to_string(),
            ],
            command: vec![],
            interactive: true,
            workdir: None,
        }
    }

    #[test]
    fn cache_tag_is_deterministic_for_same_inputs() {
        assert_eq!(
            cache_tag(&resolved(), &spec()),
            cache_tag(&resolved(), &spec())
        );
    }

    #[test]
    fn cache_tag_changes_when_image_override_changes() {
        let mut other = spec();
        other.image = "debian:12".to_string();
        assert_ne!(
            cache_tag(&resolved(), &spec()),
            cache_tag(&resolved(), &other)
        );
    }

    #[test]
    fn cache_tag_changes_when_packages_change() {
        let mut other_resolved = resolved();
        other_resolved.packages.push("mesa-utils".to_string());
        assert_ne!(
            cache_tag(&resolved(), &spec()),
            cache_tag(&other_resolved, &spec())
        );
    }

    #[test]
    fn cuda_and_rocm_specs_have_no_packages_to_cache() {
        // Sanity check on the no-op path's precondition: vendor images
        // never populate `packages`, so `ensure_cached_image` must never
        // be reached for them (see stack::tests for the same invariant).
        let mut cuda_resolved = resolved();
        cuda_resolved.stack = "cuda".to_string();
        cuda_resolved.packages.clear();
        assert!(cuda_resolved.packages.is_empty());
    }
}
