//! Process-isolated Windows container backend - the CUDA-capable
//! alternative to Windows Sandbox.
//!
//! Windows Sandbox's `<VGpu>Enable</VGpu>` only gives WDDM/DirectX
//! paravirtualization (RemoteFX-successor GPU-P), which is enough for
//! DirectX/OpenGL rendering but is *not* a CUDA-capable path - `nvidia-smi`
//! and the CUDA runtime simply don't work inside it, because there's no
//! NVIDIA kernel-mode driver plumbed through the Hyper-V VM boundary the
//! way there is for a real GPU passthrough mechanism.
//!
//! Microsoft's actual native, non-Linux, CUDA-capable technique is
//! *process-isolated Windows containers* with GPU compute support
//! (`docker run --isolation process --device class/<GPU class GUID>`),
//! shipping since Windows 10 21H2 / Windows Server 2022. Process
//! isolation means the container shares the host's kernel directly (no
//! Hyper-V VM boundary at all, unlike Hyper-V-isolated Windows containers
//! or Windows Sandbox), so the host's already-installed NVIDIA driver -
//! kernel-mode component and userspace DLLs alike - is used as-is by
//! whatever CUDA app runs inside the container. This is deliberately not
//! WSL2 or any other Linux VM: the container image, and the process
//! running inside it, are both native Windows.
//!
//! Trade-off versus Windows Sandbox: no disposable-VM isolation (process
//! isolation is a weaker boundary than a full VM), and the container
//! image must be Windows-based (a Linux `nvidia/cuda:*-ubuntu*` image
//! cannot run here) - so `--image` should point at a Windows Server Core-
//! or Nano Server-based image with the CUDA *toolkit* installed; the
//! *driver* comes from the host automatically and must not be baked into
//! the image (in fact doing so can conflict with the host's driver
//! version - the same "don't ship libcuda in the image" rule that applies
//! to Linux, just enforced differently).

use super::{Invocation, LaunchSpec};
use anyhow::Result;

/// The stable device class GUID for the "Display" (GPU) device class,
/// used by Windows containers' GPU compute/rendering support to select
/// which host devices to project into the container. Documented by
/// Microsoft under "GPU acceleration in Windows containers".
pub const GPU_DEVICE_CLASS_GUID: &str = "5B45201D-F2F2-4F3B-85BB-30FF1F953599";

/// Build the `docker run --isolation process --device class/<GUID> ...`
/// invocation. Structurally similar to the Linux docker/podman backend
/// (same `docker run` verb, `--mount`/`-e` flags), but deliberately
/// doesn't reuse `backend::linux`'s wrapper script: Windows containers
/// have no `/bin/sh`, no `setpriv`, and no POSIX uid/gid model to
/// rewrite, so commands run directly as the container's configured user
/// (`ContainerUser` by default on process-isolated Windows containers).
pub fn build_invocation(spec: &LaunchSpec) -> Result<Invocation> {
    let mut args = vec![
        "run".to_string(),
        "--rm".to_string(),
        "--isolation".to_string(),
        "process".to_string(),
        "--device".to_string(),
        format!("class/{GPU_DEVICE_CLASS_GUID}"),
    ];

    if spec.interactive {
        args.push("-it".to_string());
    }

    for mount in &spec.mounts {
        // `--mount type=bind,source=...,target=...` instead of `-v
        // host:container[:ro]`: Windows paths themselves contain a colon
        // (`C:\Users\alice`), so appending `:ro` on top of that turns `-v`
        // into an ambiguous, colon-delimited mess that both docker and
        // podman parse fragilely (or outright reject) on Windows.
        // `--mount`'s comma-separated `key=value` syntax has no such
        // ambiguity.
        let ro = if mount.read_only { ",readonly" } else { "" };
        args.push("--mount".to_string());
        args.push(format!(
            "type=bind,source={},target={}{}",
            mount.host.display(),
            mount.container.display(),
            ro
        ));
    }

    for (key, value) in &spec.env {
        args.push("-e".to_string());
        args.push(format!("{key}={value}"));
    }

    args.extend(spec.extra_args.iter().cloned());

    if let Some(workdir) = &spec.workdir {
        args.push("-w".to_string());
        args.push(workdir.display().to_string());
    }

    args.push(spec.image.clone());

    if spec.command.is_empty() {
        args.push("cmd.exe".to_string());
    } else {
        args.extend(spec.command.iter().cloned());
    }

    Ok(Invocation {
        program: "docker".to_string(),
        args,
        generated_files: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mounts::BindMount;
    use std::path::PathBuf;

    fn sample_spec() -> LaunchSpec {
        LaunchSpec {
            image: "mcr.microsoft.com/windows/servercore:ltsc2022".to_string(),
            stack: "cuda".to_string(),
            env: vec![],
            mounts: vec![BindMount {
                host: PathBuf::from(r"C:\Users\alice"),
                container: PathBuf::from(r"C:\Users\alice"),
                read_only: false,
            }],
            device_args: vec![],
            extra_args: vec![],
            packages: vec![],
            command: vec![],
            interactive: true,
            workdir: None,
        }
    }

    #[test]
    fn invocation_uses_process_isolation_and_gpu_class_device() {
        let inv = build_invocation(&sample_spec()).unwrap();
        assert_eq!(inv.program, "docker");
        assert!(inv.args.contains(&"--isolation".to_string()));
        assert!(inv.args.contains(&"process".to_string()));
        assert!(inv
            .args
            .iter()
            .any(|a| a == &format!("class/{GPU_DEVICE_CLASS_GUID}")));
    }

    #[test]
    fn no_posix_wrapper_command_tail_is_used_directly() {
        let mut spec = sample_spec();
        spec.command = vec!["nvidia-smi.exe".to_string()];
        spec.interactive = false;
        let inv = build_invocation(&spec).unwrap();
        assert_eq!(inv.args.last(), Some(&"nvidia-smi.exe".to_string()));
        assert!(!inv.args.iter().any(|a| a.contains("setpriv")));
    }

    #[test]
    fn interactive_default_command_is_cmd_exe() {
        let inv = build_invocation(&sample_spec()).unwrap();
        assert_eq!(inv.args.last(), Some(&"cmd.exe".to_string()));
    }

    #[test]
    fn mounts_use_mount_flag_not_dash_v() {
        // A Windows path already contains a colon (`C:\Users\alice`), so
        // `-v host:container:ro`'s colon-delimited syntax is ambiguous
        // there; `--mount type=bind,source=...,target=...` has no such
        // problem.
        let inv = build_invocation(&sample_spec()).unwrap();
        assert!(!inv.args.contains(&"-v".to_string()));
        assert!(inv.args.contains(&"--mount".to_string()));
        assert!(inv
            .args
            .iter()
            .any(|a| a == r"type=bind,source=C:\Users\alice,target=C:\Users\alice"));
    }

    #[test]
    fn read_only_mount_gets_readonly_suffix() {
        let mut spec = sample_spec();
        spec.mounts[0].read_only = true;
        let inv = build_invocation(&spec).unwrap();
        assert!(inv
            .args
            .iter()
            .any(|a| a == r"type=bind,source=C:\Users\alice,target=C:\Users\alice,readonly"));
    }
}
