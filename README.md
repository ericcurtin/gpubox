# gpubox

_See also: [Introducing gpubox](https://ericcurtin.github.io/gpubox/blog/2026-07-15-introducing-gpubox.html), the announcement post._

Auto-detecting GPU container launcher. Type one command - `gpubox run` -
and land in a shell where `nvidia-smi`, `rocm-smi`, or `vulkaninfo` just
works, with your home directory, dotfiles, and current working directory
mounted through. The host needs nothing GPU-vendor
specific installed; gpubox figures out what's there and picks the right
container image for you.

```
$ gpubox run
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
   fallback. Raw ID -> arch tag mappings live in `data/pci_ids.toml`. Not
   every GPU is a PCI device: SoC-integrated GPUs (Apple Silicon under
   Asahi Linux, NVIDIA's Tegra/Grace-family SoCs) show up as *platform*
   devices with no PCI vendor/device attribute files - these are still
   detected (not silently dropped to the CPU fallback), classified via a
   best-effort guess from the device tree `compatible` string when
   possible, or generic Vulkan otherwise.

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
   included) and the current working directory, forwards X11/Wayland
   sockets for GUI apps, and sets `GPUBOX_STACK` so shells can show a
   `(gpubox:rocm)`-style prompt marker. On Linux, the container is always
   made to *be* the real host user rather than trusting whatever account
   happens to already own the mapped uid in the base image - `ubuntu:24.04`
   itself ships a baked-in `ubuntu:x:1000:1000:...:/home/ubuntu` account,
   and 1000 is the default first-user uid on most distros, so without
   this a very common uid collision would silently "log you in" as that
   placeholder instead of yourself. `$HOME`/`$USER`/`$LOGNAME` are forced
   via `-e`, and the container runs a small wrapper (`src/backend/linux.rs`)
   that, as root, rewrites `/etc/passwd`/`/etc/group` with your real
   identity, installs any stack-specific apt packages (e.g. the Vulkan
   fallback's `mesa-vulkan-drivers`), and only then drops to your uid/gid
   via `setpriv` before exec'ing your shell or command.

5. **Doctor** (`src/doctor.rs`) - `gpubox doctor` prints what was
   detected, which stack was chosen and why, and how to override it.

`src/launch.rs` wires stages 1-4 into a `backend::LaunchSpec`;
`src/backend/` turns that into something executable per platform:

| Platform | Backend                          | Notes |
|----------|-----------------------------------|-------|
| Linux    | Docker (default) or Podman        | CDI / `--device` / `--gpus all`. Podman's rootless mode is first-class here - no root required on the host (see `gpubox doctor`'s "Podman mode" line). |
| macOS    | Seatbelt (`sandbox-exec`)          | No Linux kernel to pass a device through; the sandboxed process runs natively against the host's Metal stack |
| Windows  | Windows Sandbox (default) or Windows Container | Windows Sandbox's `<VGpu>Enable</VGpu>` is WDDM/DirectX paravirtualization only - **not CUDA-capable**. For `cuda`/`rocm`/`oneapi` stacks, gpubox instead defaults to a **process-isolated Windows container** (`docker run --isolation process --device class/<GPU class GUID>`), which shares the host kernel directly so the host's real GPU driver (kernel-mode + userspace) is used as-is - CUDA-capable, still not a Linux VM (unlike WSL2). Needs a Windows-based `--image`; see `src/backend/windows_container.rs`. |

`src/generate.rs` renders the same plan as an inspectable file instead of
running anything: a Dockerfile, a Compose file, a Podman Quadlet unit,
a Seatbelt `.sb` profile, a Windows Sandbox `.wsb` config, or a VS Code
`devcontainer.json` - so the "magic" is reproducible in CI, checked into a
repo, or opened straight in Dev Containers/Codespaces.

## One command: `gpubox run`

`enter` and `run` are the same command. With no trailing command, `gpubox
run` is an interactive shell; with one (after `--`), it runs
non-interactively and exits:

```
gpubox run                       # interactive shell
gpubox run -- python train.py    # non-interactive
```

## Persistent containers by default

Containers are persistent, the same model distrobox/toolbox use: `gpubox
run` creates a single container, named `gpubox`, the first time, then
reattaches to it on every later invocation instead of tearing it down -
regardless of which stack/hardware config resolved that particular
invocation - anything `apt install`ed inside it is still there next time,
and the Vulkan/CPU fallback's `mesa-vulkan-drivers` et al. only ever get
installed once instead of on every single launch:

```
gpubox run                     # first run: creates `gpubox`; every run after: reattaches
gpubox run -- python train.py
gpubox rm                      # delete the default container and start fresh next time
```

Override the container name with `--name`, or opt back into the old
one-off `--rm` behavior for a single invocation:

```
gpubox run --name ml                       # a separate, independently-named persistent container
gpubox run --name ml -- python train.py
gpubox rm ml

gpubox run --rm                            # throwaway container, torn down on exit (e.g. for CI)
```

(Docker/Podman only - Seatbelt runs natively on the host with nothing to
persist, and Windows Sandbox always boots a clean VM by design, so both
silently keep their original one-shot behavior.)

Separately, and regardless of persistence: any image that needs extra
apt packages layered on (the Vulkan/CPU fallback) is built and tagged
**once**, locally, and reused - so even a `--rm` run skips the
`apt-get install` network hit and wait after the first launch. See
`src/cache.rs`.

## Multi-GPU and hybrid systems

On a host with more than one GPU (a laptop with an Intel iGPU and an
NVIDIA dGPU, or a workstation with mixed vendors), `gpubox doctor` lists
every detected device, and `pick_primary`'s ranking (discrete NVIDIA >
discrete AMD > Intel Arc/Xe > other iGPUs > Apple > Vulkan > none) is
just the *default* - override it explicitly:

```
gpubox --gpu nvidia run    # coarse vendor name
gpubox --gpu 1 doctor      # 0-based index into the detected list
gpubox --gpu amd run -- rocm-smi
```

## Other flags

```
--no-home            don't mount $HOME into the sandbox at all
--read-only-home      mount $HOME read-only instead of read-write
```

Mounting `$HOME` read-write into a container that runs a root wrapper
before dropping privileges (see `src/backend/linux.rs`) means that, for
the brief window before the privilege drop, the process inside the
container *is* root and can read or write anything under `$HOME`
regardless of host file permissions. `--no-home`/`--read-only-home` exist
for anyone who wants gpubox's CWD mount / GUI sockets / identity
forwarding without handing over full write access to the home directory
on every sandbox.

## Diagnostics for scripts and bug reports

```
gpubox doctor --json      # the same report, structured, for scripting
gpubox doctor --report    # an anonymized hardware-probe snapshot (no
                           # paths, no usernames) to paste into a bug
                           # report - and to grow the fixture corpus in
                           # src/probe/*.rs's tests with real hardware
```

## Shell completions and man page

```
gpubox completions bash > /etc/bash_completion.d/gpubox
gpubox completions zsh   > "${fpath[1]}/_gpubox"
gpubox man > /usr/local/share/man/man1/gpubox.1
```

### Default images (Linux)

Real, pullable upstream images - never a placeholder namespace. Override
any of them per-invocation with `--image`, or permanently by editing
`data/quirks.toml`.

| Stack    | Default image                                    | Notes |
|----------|---------------------------------------------------|-------|
| `cuda`   | `nvidia/cuda:12.9.2-devel-ubuntu24.04`             | NVIDIA's official CUDA devel image |
| `rocm`   | `rocm/rocm-terminal:6.1`                           | AMD's official interactive ROCm image |
| `oneapi` | `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04`| Intel's official oneAPI base toolkit |
| `vulkan` | `ubuntu:24.04` + `mesa-vulkan-drivers`, `vulkan-tools`, `libvulkan1` | No single canonical vendor-neutral Vulkan image exists, so gpubox layers Mesa's Vulkan drivers onto plain Ubuntu; see `packages` in `data/quirks.toml` |
| `cpu`    | `ubuntu:24.04`                                     | Plain fallback, no GPU packages |
| `metal`  | *(none)*                                           | macOS only; Seatbelt runs the command natively on the host, there's no container image |

## Usage

```
gpubox run                       # interactive shell in the persistent `gpubox` container
gpubox run --name ml             # a separate, independently-named persistent container
gpubox run -- python train.py    # non-interactive, same persistence
gpubox run --rm                  # throwaway container for this run only
gpubox rm                        # delete the default container
gpubox rm ml                     # delete a specifically-named one
gpubox generate --format compose -o compose.yaml
gpubox doctor                    # explain what was detected and why
gpubox doctor --json             # same, structured
gpubox doctor --report           # anonymized probe snapshot for bug reports
gpubox completions bash          # shell completion script
gpubox man                       # man page (troff)
```

Global overrides (work with every subcommand):

```
--backend <docker|podman|seatbelt|windows-sandbox|windows-container>
--image <ref>
--gfx-override <sm_86|gfx1100|arc|apple|vulkan|cpu>
--gpu <index|vendor>       pick a specific GPU on a multi-GPU/hybrid host
--no-home                 don't mount $HOME into the sandbox at all
--read-only-home           mount $HOME read-only instead of read-write
--dry-run
```

`run`-specific:

```
--name <name>   use this container name instead of the default (`gpubox`)
--rm            use a throwaway container for this run instead of the persistent default
```

## Contributing hardware support

Two data files, both plain TOML, no code changes required:

- `data/pci_ids.toml` - add a missing vendor/device id -> arch tag
  mapping.
- `data/quirks.toml` - add or fix a stack/image/env mapping for an arch
  tag, e.g. "this gfx target needs `HSA_OVERRIDE_GFX_VERSION` set to
  work with ROCm".

Both files are structurally validated by `cargo test` (see
`tests/schema_validation.rs`, `stack::validate_quirks_db`, and
`probe::validate_pci_ids_db`) - every vendor table's fallback rule,
every device id's hex format, and every image reference is checked, so a
PR that breaks one of these fails CI instead of shipping a broken
fallback.

## Development

```
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
```

CI (`.github/workflows/ci.yml`) builds and tests natively on Linux
x86_64/aarch64, macOS aarch64, and Windows x86_64/aarch64.
