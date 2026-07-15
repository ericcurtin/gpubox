---
layout: default
title: "Introducing gpubox: a GPU-aware, container launcher"
date: 2026-07-15
---

# Introducing gpubox: a GPU-aware, container launcher

*Published 2026-07-15*

If you've ever tried to get CUDA, ROCm, or oneAPI working inside a
container, you already know the shape of the pain: the host needs a
kernel driver, the container needs a userspace library that matches that
driver *exactly*, and the incantation to wire the two together is
different on every distro, every container engine, and every GPU vendor.
Multiply that by "and also I want this to work on my Linux box, my Mac,
and my Windows machine" and most people just give up and install
everything directly on the host.

**gpubox** is a new, small Rust CLI that tries to make this a
non-problem. The whole interaction is meant to be one command:

```
$ gpubox enter
(gpubox:rocm) you@host:~/project$ rocm-smi
...
```

No driver installation on the host beyond what the GPU vendor already
requires, no manually figuring out which `--device` flags your card
needs, no picking a base image by hand. gpubox looks at what hardware
you have, decides what container image and runtime flags make sense, and
drops you into a shell with your home directory, dotfiles, and current
project mounted in.

It's a brand new project as of this post - the design is settled and the
core pipeline works end-to-end, but the community data files (see below)
are intentionally thin so far. That's by design: the interesting part of
this problem isn't the code, it's the accumulated hardware-specific
knowledge, and that's meant to grow from real users' real GPUs.

## The problem, restated

GPU passthrough into containers is really three separate problems people
tend to conflate:

1. **Which stack do I even need?** NVIDIA wants CUDA, AMD wants ROCm,
   Intel Arc wants oneAPI, and everything else is stuck hoping Vulkan
   works. Figuring this out today means running vendor tools that
   probably aren't installed on a clean machine.
2. **How do I get the device into the container?** Device nodes,
   `--gpus`, CDI specs, `nvidia-container-toolkit` hooks - the right
   answer depends on your container engine version and your distro.

gpubox is a straight-line pipeline through all three, plus a fourth
problem that turns out to matter just as much: **explaining itself**. GPU
detection heuristics that fail silently are worse than not having them,
so a big part of the design is `gpubox doctor` - a command whose entire
job is to answer "why did it pick that?"

## The five-stage pipeline

### 1. Hardware probe - no root, no vendor tools

The whole premise falls apart if gpubox itself requires you to install
`nvidia-smi` or `rocm-smi` just to be detected. So the probe stage reads
straight from OS-native interfaces instead of shelling out to anything
vendor-specific:

- **Linux**: walks `/sys/class/drm/card*/device/{vendor,device}` and
  reads the raw PCI ids as plain sysfs files.
- **Windows**: queries the `Win32_VideoController` WMI class via
  PowerShell for each adapter's `PNPDeviceID`, which embeds the same
  `VEN_xxxx&DEV_xxxx` pair Linux exposes through sysfs.
- **macOS**: Apple Silicon's GPU is inseparable from the SoC, so checking
  the CPU architecture is enough - if you're on `aarch64`, you have an
  Apple GPU.

The raw vendor/device id pair is classified into a normalized tag using
`data/pci_ids.toml` - NVIDIA gets a CUDA compute-capability tag
(`sm_86`), AMD gets a gfx architecture tag (`gfx1100`), Intel gets a
coarse class (`arc`, `xe`, or older `igpu`). Anything unrecognized
degrades gracefully instead of erroring out.

### 2. Stack resolution - the actual product

This is the part of gpubox that matters more than the code around it.
`data/quirks.toml` maps each classification to a runtime stack, a
default container image, and any environment variables needed to
actually make that hardware work:

```toml
[amd.gfx90c]
stack = "rocm"
image = "rocm/rocm-terminal:6.1"
env = { HSA_OVERRIDE_GFX_VERSION = "9.0.0" }
notes = """
Renoir (gfx90c) integrated graphics is not officially supported by ROCm \
but works when HSA_OVERRIDE_GFX_VERSION overrides it to report as \
gfx9.0.0.
"""
```

That's a real entry in the repo today. Renoir APU owners have known for
years that ROCm works if you lie to it about which GPU you have -
knowledge that's usually scattered across GitHub issues and forum posts.
`quirks.toml` is where that kind of folk knowledge is meant to live: a
plain TOML file, PR-able by anyone, with no code changes required to add
support for a new card or fix a broken one. Every vendor table also has
an `unknown`/`default` fallback entry, so hardware nobody's added yet
still resolves to a sane Vulkan or CPU stack instead of failing outright.

The default images are real, pullable, upstream images - not a
placeholder namespace:

| Stack    | Default image |
|----------|---------------|
| `cuda`   | `nvidia/cuda:12.9.2-devel-ubuntu24.04` |
| `rocm`   | `rocm/rocm-terminal:6.1` |
| `oneapi` | `intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04` |
| `vulkan` | `ubuntu:24.04` + Mesa's Vulkan drivers layered on top |
| `cpu`    | `ubuntu:24.04` |

There's no single canonical vendor-neutral Vulkan image on any registry,
so the Vulkan fallback is a plain Ubuntu base with
`mesa-vulkan-drivers`/`vulkan-tools`/`libvulkan1` installed via
`gpubox generate --format dockerfile` rather than a fictional image tag.

### 3. Device injection - hiding the nastiest problem in the space

On Linux, gpubox prefers [CDI](https://github.com/cncf-tags/container-device-interface)
(Container Device Interface) when a spec is present under `/etc/cdi` or
`/var/run/cdi` - both modern Docker and Podman can consume it directly.
Where CDI isn't available it falls back to raw `--device /dev/dri` /
`/dev/kfd` for AMD and Intel, or the `nvidia-container-toolkit`'s
`--gpus all` for NVIDIA.

NVIDIA gets one more thing: the kernel driver and userspace driver
libraries (`libcuda.so`, `libnvidia-*.so`) have to be the *exact* same
version, which is why NVIDIA images can never simply bake `libcuda` in
themselves. When gpubox falls back to the toolkit path, it scans the
common driver library locations on the host (including WSL2's
paravirtualized driver path) and bind-mounts whatever it finds read-only
into the container, so the container's userspace always matches
whatever the host kernel driver actually is.

### 4. Host integration - making it feel native

gpubox mounts `$HOME` (dotfiles included, since they live under `$HOME`
anyway) and the current working directory, maps your UID in
(`--userns=keep-id` on Podman, `-u uid:gid` on Docker), forwards
X11/Wayland sockets so GUI and OpenGL apps work, and sets a
`GPUBOX_STACK` environment variable so shells can render a prompt marker
like `(gpubox:rocm)` - so you always know which stack you're in without
having to think about it.

### 5. Doctor - because trust requires transparency

```
$ gpubox doctor --gfx-override gfx90c
gpubox doctor
=============

Detected hardware : AMD (gfx90c)
Resolved stack    : rocm
Container image   : rocm/rocm-terminal:6.1
Matched rule      : amd.gfx90c
Notes             :
  Renoir (gfx90c) integrated graphics is not officially supported by ROCm but works when HSA_OVERRIDE_GFX_VERSION overrides it to report as gfx9.0.0.
Quirk env vars    :
  HSA_OVERRIDE_GFX_VERSION=9.0.0
  GPUBOX_STACK=rocm

Backend           : seatbelt
  available       : yes

Overrides:
  --backend <name>       force a specific backend (docker, podman, seatbelt, windows-sandbox)
  --image <ref>          use a custom image instead of rocm/rocm-terminal:6.1
  --gfx-override <arch>  force a hardware classification (e.g. sm_86, gfx1100, arc, apple, vulkan, cpu)
```

Auto-detection that fails silently is worse than no auto-detection at
all, so `doctor` always shows exactly what was found, exactly which
`quirks.toml` rule matched, and exactly which flags to pass if you want
to override any part of the decision.

## Three platforms, three sandboxing technologies

The interesting design decision in gpubox isn't the Linux path - Docker
and Podman with CDI/device nodes is well-trodden ground. It's that
"enter a sandboxed shell with GPU access" means something structurally
different on each OS, and gpubox picks the right primitive for each
rather than forcing one model everywhere:

- **Linux** → Docker (default) or Podman. Straightforward OCI containers.
- **macOS** → **Seatbelt**, via `sandbox-exec`. There's no Linux kernel
  to pass a device node through on macOS, so instead of containerizing,
  gpubox sandboxes the target process directly on the host using Apple's
  own sandbox profile language - the same mechanism behind App Sandbox
  and much of macOS's system hardening. The process runs as the invoking
  user against the host's native Metal stack; there's no driver-matching
  problem at all here, because the GPU stack is part of the OS. The
  profile itself starts from `(allow default)` and layers a `file-write*`
  deny over the whole filesystem, re-allowing writes only under the
  mounted home directory and CWD - a strict default-deny profile turned
  out to be impractical, since GPU/Metal/WindowServer access alone needs
  dozens of mach-lookup service names that vary across macOS versions.
- **Windows** → **Windows Sandbox**, a lightweight Hyper-V-backed
  desktop VM that's been able to opt into virtualized GPU acceleration
  (`<VGpu>Enable</VGpu>`) since 2020. This is deliberately *not* WSL2 -
  WSL2 is itself a Linux VM, and the point of the Windows backend is a
  native, non-Linux sandbox with its own GPU story.

## Inspectable by design: `gpubox generate`

Auto-detection is only trustworthy if you can see exactly what it would
have done. `gpubox generate` renders the same resolved plan as a file
instead of running anything - a Dockerfile, a Compose file, a Podman
Quadlet unit, a Seatbelt `.sb` profile, or a Windows Sandbox `.wsb`
config:

```
$ gpubox generate --gfx-override gfx90c --format quadlet
# Generated by `gpubox generate` for stack: rocm
[Unit]
Description=gpubox rocm sandbox

[Container]
Image=rocm/rocm-terminal:6.1
Volume=/home/alice:/home/alice
Volume=/home/alice/project:/home/alice/project
Environment=HSA_OVERRIDE_GFX_VERSION=9.0.0
Environment=GPUBOX_STACK=rocm
WorkingDir=/home/alice/project

[Install]
WantedBy=default.target
```

Check that into a repo, run it straight in CI, hand it to a teammate who
doesn't have gpubox installed - the "magic" of hardware detection and
stack resolution becomes a diffable artifact instead of something that
only happens at runtime on your machine.

## What's next

This is a first release, not a finished product. The CLI and the
five-stage pipeline are solid and covered by unit tests across Linux
(x86_64/aarch64), macOS (aarch64), and Windows (x86_64/aarch64) in CI -
but `data/pci_ids.toml` and `data/quirks.toml` only have the device ids
and quirks I could source myself. The actual value of this project scales
with how many GPUs are represented in those two files, and neither
requires touching a line of Rust to extend:

- Missing hardware? Add a vendor/device id → arch tag mapping to
  `data/pci_ids.toml`.
- Know a quirk (an env var override, a stack that works better than the
  obvious one) for a specific chip? Add or fix an entry in
  `data/quirks.toml`.

The repository is at `github.com/ericcurtin/gpubox`. If you've fought
with `HSA_OVERRIDE_GFX_VERSION` or a mismatched `libcuda` before, your
two-line PR to a TOML file is worth more to the next person than most
code contributions would be.
