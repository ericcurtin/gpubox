# gpubox

Auto-detecting GPU container launcher. Type one command - `gpubox enter` -
and land in a shell where `nvidia-smi`, `rocm-smi`, or `vulkaninfo` just
works, with your home directory, dotfiles, and current working directory
mounted through, distrobox-style. The host needs nothing GPU-vendor
specific installed; gpubox figures out what's there and picks the right
container image for you.

```
$ gpubox enter
(gpubox:rocm) you@host:~/project$ rocm-smi
...
```

## How it works

Five stages, each its own module:

1. **Hardware probe** (`src/probe/`) - host-side, no root, no vendor
   tools. Walks `/sys/class/drm` and PCI vendor/device ids on Linux,
   queries `Win32_VideoController` via WMI on Windows, and checks CPU
   architecture on macOS (Apple Silicon = Apple GPU). Classifies into
   NVIDIA (+ CUDA compute capability), AMD (+ gfx architecture), Intel
   (Arc / Xe / older iGPU), Apple Silicon, generic Vulkan, or CPU
   fallback. Raw ID -> arch tag mappings live in `data/pci_ids.toml`.

2. **Stack resolution** (`src/stack.rs`) - maps a classification to a
   runtime stack, a default container image, and any environment
   variable quirks, via the community-maintained matrix in
   `data/quirks.toml`. This file is the real product: if your card needs
   `HSA_OVERRIDE_GFX_VERSION=9.0.0` or similar folk knowledge to work,
   that lives here, not in code.

3. **Device injection** (`src/device.rs`, Linux only) - prefers CDI
   (Container Device Interface) when a spec is present under `/etc/cdi`
   or `/var/run/cdi`, falls back to raw `--device /dev/dri`/`/dev/kfd`
   for AMD/Intel, and to the nvidia-container-toolkit's `--gpus all` for
   NVIDIA, additionally bind-mounting the host's userspace NVIDIA driver
   libraries read-only so they match the host kernel driver's version.

4. **Host integration** (`src/mounts.rs`) - mounts `$HOME` (dotfiles
   included) and the current working directory, maps the host uid/gid in
   (`--userns=keep-id` on Podman, `-u uid:gid` on Docker), forwards
   X11/Wayland sockets for GUI apps, and sets `GPUBOX_STACK` so shells can
   show a `(gpubox:rocm)`-style prompt marker.

5. **Doctor** (`src/doctor.rs`) - `gpubox doctor` prints what was
   detected, which stack was chosen and why, and how to override it.

`src/launch.rs` wires stages 1-4 into a `backend::LaunchSpec`;
`src/backend/` turns that into something executable per platform:

| Platform | Backend                          | Notes |
|----------|-----------------------------------|-------|
| Linux    | Docker (default) or Podman        | CDI / `--device` / `--gpus all` |
| macOS    | Seatbelt (`sandbox-exec`)          | No Linux kernel to pass a device through; the sandboxed process runs natively against the host's Metal stack |
| Windows  | Windows Sandbox                   | Hyper-V-backed, non-Linux (unlike WSL2), with opt-in GPU passthrough via `<VGpu>Enable</VGpu>` |

`src/generate.rs` renders the same plan as an inspectable file instead of
running anything: a Containerfile, a Compose file, a Podman Quadlet unit,
a Seatbelt `.sb` profile, or a Windows Sandbox `.wsb` config - so the
"magic" is reproducible in CI or checked into a repo.

## Usage

```
gpubox enter                     # interactive shell in the auto-detected sandbox
gpubox run -- python train.py    # non-interactive
gpubox generate --format compose -o compose.yaml
gpubox doctor                    # explain what was detected and why
```

Global overrides (work with every subcommand):

```
--backend <docker|podman|seatbelt|windows-sandbox>
--image <ref>
--gfx-override <sm_86|gfx1100|arc|apple|vulkan|cpu>
--dry-run
```

## Contributing hardware support

Two data files, both plain TOML, no code changes required:

- `data/pci_ids.toml` - add a missing vendor/device id -> arch tag
  mapping.
- `data/quirks.toml` - add or fix a stack/image/env mapping for an arch
  tag, e.g. "this gfx target needs `HSA_OVERRIDE_GFX_VERSION` set to
  work with ROCm".

## Development

```
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
```

CI (`.github/workflows/ci.yml`) builds and tests natively on Linux
x86_64/aarch64, macOS aarch64, and Windows x86_64/aarch64.
