//! Command-line interface: `run`, `rm`, `generate`, `doctor`,
//! `completions`, `man`.

use crate::backend::{self, Invocation};
use crate::cache;
use crate::container;
use crate::generate::{self, Format};
use crate::launch::{self, Overrides, Plan};
use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "gpubox",
    version,
    about = "Auto-detecting GPU container launcher."
)]
pub struct Cli {
    /// Force a specific backend instead of the platform default (docker,
    /// podman, seatbelt, windows-sandbox, windows-container).
    #[arg(long, global = true)]
    backend: Option<String>,

    /// Use a custom image instead of the one resolved from detected
    /// hardware.
    #[arg(long, global = true)]
    image: Option<String>,

    /// Force a hardware classification instead of probing the host, e.g.
    /// `sm_86`, `gfx1100`, `arc`, `apple`, `vulkan`, `cpu`.
    #[arg(long = "gfx-override", global = true)]
    gfx_override: Option<String>,

    /// Pick a specific GPU on a multi-GPU/hybrid host, either by 0-based
    /// index (`--gpu 1`) or coarse vendor name (`--gpu nvidia`), e.g.
    /// `gpubox run --gpu nvidia`.
    #[arg(long, global = true)]
    gpu: Option<String>,

    /// Don't mount $HOME into the sandbox at all. See the threat-model
    /// note on `mounts::HomeMode`.
    #[arg(long, global = true, conflicts_with = "read_only_home")]
    no_home: bool,

    /// Mount $HOME read-only instead of read-write.
    #[arg(long, global = true)]
    read_only_home: bool,

    /// Print the command(s) that would be run instead of running them.
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Enter an interactive shell, or run a command, inside the
    /// auto-detected GPU sandbox. With no trailing command this is an
    /// interactive shell (what used to be the separate `enter`
    /// subcommand); with one (after `--`) it runs non-interactively and
    /// exits.
    Run {
        /// Use a container name other than the default
        /// (`gpubox`, shared across every stack/hardware config on this
        /// host). Either way the container is created once and
        /// reattached on every later `gpubox run`. Use `gpubox rm [name]`
        /// to remove it and start fresh. Docker/Podman only.
        #[arg(long, conflicts_with = "ephemeral")]
        name: Option<String>,
        /// Use a throwaway container for this run instead of the
        /// persistent default: torn down on exit, nothing installed
        /// inside it survives. Docker/Podman only.
        #[arg(long = "rm")]
        ephemeral: bool,
        /// Command to run non-interactively, e.g. `-- python train.py`.
        /// Omit entirely for an interactive shell.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    /// Remove a persistent container (see `run`/`run --name`), so the
    /// next invocation with the same name starts completely fresh.
    Rm {
        /// The container name (defaults to the same default `gpubox`
        /// container `run` uses, if omitted).
        name: Option<String>,
    },
    /// Emit the equivalent Dockerfile / Compose / Quadlet / Seatbelt /
    /// `.wsb` / devcontainer.json config instead of launching anything.
    Generate {
        /// dockerfile, compose, quadlet, seatbelt, windows-sandbox, or
        /// devcontainer. Defaults to whatever's idiomatic for the host
        /// platform.
        #[arg(long)]
        format: Option<String>,
        /// Write to this path instead of stdout.
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },
    /// Print detected hardware, chosen stack, and how to override it.
    Doctor {
        /// Structured output for scripting, instead of the human-readable
        /// report.
        #[arg(long, conflicts_with = "report")]
        json: bool,
        /// Print an anonymized hardware-probe snapshot (no paths, no
        /// usernames) suitable for pasting into a bug report.
        #[arg(long)]
        report: bool,
    },
    /// Print a shell completion script for the given shell.
    Completions { shell: clap_complete::Shell },
    /// Print a man page (troff) for gpubox.
    Man,
}

/// Build an [`Overrides`] from the global flags, taking ownership of each
/// (rather than cloning out of a `&Cli`) - safe since `run()` below
/// destructures `Cli` once, so nothing else needs these values afterward.
fn overrides_from(
    backend: Option<String>,
    image: Option<String>,
    gfx_override: Option<String>,
    gpu: Option<String>,
    no_home: bool,
    read_only_home: bool,
) -> Overrides {
    Overrides {
        backend,
        image,
        gfx_override,
        gpu,
        name: None,
        ephemeral: false,
        no_home,
        read_only_home,
    }
}

pub fn run() -> Result<i32> {
    // Destructure the whole struct in one go (rather than matching on
    // `cli.command` and separately reading other fields off `&cli`) so
    // every field can be moved into whichever single arm below actually
    // needs it, with no clones and no partial-move conflicts.
    let Cli {
        backend,
        image,
        gfx_override,
        gpu,
        no_home,
        read_only_home,
        dry_run,
        command,
    } = Cli::parse();

    match command {
        Command::Run {
            name,
            ephemeral,
            command,
        } => {
            let interactive = command.is_empty();
            let mut overrides =
                overrides_from(backend, image, gfx_override, gpu, no_home, read_only_home);
            overrides.name = name;
            overrides.ephemeral = ephemeral;
            execute(&overrides, command, interactive, dry_run)
        }
        Command::Rm { name } => rm_cmd(backend, dry_run, name.as_deref()),
        Command::Generate { format, output } => {
            let overrides =
                overrides_from(backend, image, gfx_override, gpu, no_home, read_only_home);
            generate_cmd(&overrides, format.as_deref(), output.as_deref())?;
            Ok(0)
        }
        Command::Doctor { json, report } => {
            let overrides =
                overrides_from(backend, image, gfx_override, gpu, no_home, read_only_home);
            if report {
                println!("{}", crate::doctor::probe_snapshot_json()?);
            } else if json {
                println!("{}", crate::doctor::report_json(&overrides)?);
            } else {
                print!("{}", crate::doctor::report(&overrides)?);
            }
            Ok(0)
        }
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
            Ok(0)
        }
        Command::Man => {
            let cmd = Cli::command();
            let man = clap_mangen::Man::new(cmd);
            man.render(&mut std::io::stdout())?;
            Ok(0)
        }
    }
}

fn execute(
    overrides: &Overrides,
    command: Vec<String>,
    interactive: bool,
    dry_run: bool,
) -> Result<i32> {
    let mut plan: Plan = launch::plan(overrides, command, interactive)?;

    // Persistence is the default (distrobox/toolbox's whole appeal is
    // that the box feels like a second home instead of vanishing on
    // exit): every `gpubox run` reattaches to a container - `gpubox`,
    // shared across every stack/hardware config, or `gpubox-<name>` with
    // `--name <name>` - rather than a one-off `--rm` container. `--rm`
    // opts back into the old throwaway behavior. Backends with no notion
    // of a persistent container (Seatbelt runs natively on the host;
    // Windows Sandbox always boots a clean VM by design) silently fall
    // through to the ephemeral/native path below - unless the user
    // *explicitly* asked for `--name` on one of them, which is an error
    // rather than being quietly ignored.
    if !overrides.ephemeral {
        let full_name = match &overrides.name {
            Some(name) => container::container_name(name),
            None => container::DEFAULT_CONTAINER_NAME.to_string(),
        };
        match plan.engine.as_container_engine() {
            Some(engine) => return execute_named(engine, &mut plan, &full_name, dry_run),
            None if overrides.name.is_some() => {
                anyhow::bail!(
                    "--name is only supported with the docker/podman backends (resolved \
                     backend: `{}`)",
                    plan.engine
                );
            }
            None => {} // fall through: this backend has no persistent-container story.
        }
    }

    if !dry_run {
        if let Some(engine) = plan.engine.as_container_engine() {
            cache::ensure_cached_image(engine.program(), &plan.resolved, &mut plan.spec)?;
        }
    }

    let invocation = backend::build_invocation(plan.engine, &plan.spec)?;

    if dry_run {
        print_dry_run(&plan, &invocation);
        return Ok(0);
    }

    backend::ensure_available(plan.engine)?;
    write_generated_files(&invocation)?;
    run_invocation(&invocation)
}

/// The persistent-container path (the default, or explicit `--name`):
/// create-or-reattach a persistent container, then exec into it. `name`
/// is already the fully-qualified container name (`gpubox` or
/// `gpubox-<name>` - see [`container::DEFAULT_CONTAINER_NAME`]/
/// [`container::container_name`]). See `container` module docs for the
/// full rationale.
fn execute_named(
    engine: backend::linux::ContainerEngine,
    plan: &mut Plan,
    full_name: &str,
    dry_run: bool,
) -> Result<i32> {
    let engine_program = engine.program();

    if !dry_run {
        backend::ensure_available(plan.engine)?;
        cache::ensure_cached_image(engine_program, &plan.resolved, &mut plan.spec)?;
    }

    let state = container::inspect(engine_program, full_name);

    let mut setup: Vec<Invocation> = Vec::new();
    match state {
        container::ContainerState::Missing => {
            setup.push(container::create_invocation(engine, &plan.spec, full_name));
        }
        container::ContainerState::Stopped => {
            setup.push(container::start_invocation(engine, full_name));
        }
        container::ContainerState::Running => {}
    }
    let exec_invocation = container::exec_invocation(engine, &plan.spec, full_name);

    if dry_run {
        println!(
            "# stack: {}  image: {}  backend: {}  container: {full_name} (state: {state:?})",
            plan.resolved.stack, plan.spec.image, plan.engine
        );
        for inv in &setup {
            print_invocation_line(inv);
        }
        print_invocation_line(&exec_invocation);
        return Ok(0);
    }

    for inv in &setup {
        let status = std::process::Command::new(&inv.program)
            .args(&inv.args)
            .status()
            .with_context(|| format!("failed to execute `{}`", inv.program))?;
        if !status.success() {
            anyhow::bail!(
                "`{} {}` failed while preparing container `{full_name}`",
                inv.program,
                inv.args.join(" ")
            );
        }
    }

    run_invocation(&exec_invocation)
}

fn rm_cmd(backend: Option<String>, dry_run: bool, name: Option<&str>) -> Result<i32> {
    // Unlike `run`, `rm` never needs a hardware probe: the default
    // container name is the fixed `container::DEFAULT_CONTAINER_NAME`
    // ("gpubox"), not derived from the resolved stack, so picking the
    // engine only ever needs `--backend`/the platform default.
    let engine = match &backend {
        Some(name) => backend::Engine::parse(name)
            .ok_or_else(|| anyhow::anyhow!("unrecognized --backend `{name}`"))?,
        None => backend::Engine::default_for_platform(),
    };
    let engine = engine.as_container_engine().ok_or_else(|| {
        anyhow::anyhow!("`gpubox rm` is only supported with the docker/podman backends")
    })?;
    let full_name = match name {
        Some(name) => container::container_name(name),
        None => container::DEFAULT_CONTAINER_NAME.to_string(),
    };

    if dry_run {
        println!("{} rm -f {full_name}", engine.program());
        return Ok(0);
    }

    container::remove(engine.program(), &full_name)?;
    println!("removed {full_name}");
    Ok(0)
}

fn run_invocation(invocation: &Invocation) -> Result<i32> {
    let status = std::process::Command::new(&invocation.program)
        .args(&invocation.args)
        .status()
        .with_context(|| format!("failed to execute `{}`", invocation.program))?;
    Ok(status.code().unwrap_or(1))
}

fn print_dry_run(plan: &Plan, invocation: &Invocation) {
    println!(
        "# stack: {}  image: {}  backend: {}",
        plan.resolved.stack, plan.spec.image, plan.engine
    );
    print_invocation_line(invocation);
    for (path, _content) in &invocation.generated_files {
        println!("# (would write {} )", path.display());
    }
}

fn print_invocation_line(invocation: &Invocation) {
    print!("{}", invocation.program);
    for arg in &invocation.args {
        print!(" {}", shell_quote(arg));
    }
    println!();
}

fn shell_quote(arg: &str) -> String {
    if arg
        .chars()
        .all(|c| c.is_alphanumeric() || "-_./:=@".contains(c))
    {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

fn write_generated_files(invocation: &Invocation) -> Result<()> {
    for (path, content) in &invocation.generated_files {
        std::fs::write(path, content)
            .with_context(|| format!("writing generated file {}", path.display()))?;
    }
    Ok(())
}

fn generate_cmd(
    overrides: &Overrides,
    format: Option<&str>,
    output: Option<&std::path::Path>,
) -> Result<()> {
    let plan = launch::plan(overrides, Vec::new(), true)?;
    let format = match format {
        Some(f) => Format::parse(f).with_context(|| format!("unrecognized --format `{f}`"))?,
        None => Format::default_for_platform(),
    };
    generate::validate_format_for_stack(format, &plan.resolved)?;
    let content = generate::render(format, &plan.resolved, &plan.spec)?;

    match output {
        Some(path) => {
            std::fs::write(path, &content)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => {
            std::io::stdout().write_all(content.as_bytes())?;
        }
    }
    Ok(())
}
