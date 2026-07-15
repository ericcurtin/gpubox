//! gpubox: auto-detecting GPU container launcher.
//!
//! The core flow, from the top:
//!
//! 1. [`probe`] - hardware probe (host side, no root, no vendor tools).
//! 2. [`stack`] - map detected hardware to a runtime stack + image via the
//!    community-maintained quirks database.
//! 3. [`device`] - device injection (Linux: CDI / raw device nodes /
//!    nvidia-container-toolkit, plus userspace driver library mounts).
//! 4. [`mounts`] - distrobox-grade host integration (home, CWD, GUI
//!    sockets, uid mapping, prompt marker).
//! 5. [`doctor`] - explain what was detected/chosen and how to override it.
//!
//! [`launch`] wires 1-4 into a [`backend::LaunchSpec`], and [`backend`]
//! turns that into an executable invocation for Docker/Podman (Linux),
//! Seatbelt (macOS), or Windows Sandbox (Windows). [`generate`] renders
//! the same information as an inspectable Containerfile/Compose/Quadlet/
//! Seatbelt-profile/`.wsb` file instead of running anything.

pub mod backend;
pub mod cli;
#[cfg(target_os = "linux")]
pub mod device;
pub mod doctor;
pub mod generate;
pub mod launch;
pub mod mounts;
pub mod probe;
pub mod stack;
