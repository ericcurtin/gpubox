//! Command-line interface: `enter`, `run`, `rm`, `generate`, `doctor`,
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
    /// index (`--gpu 1`) or coarse vendor name (`--gpu nvidia`). Also
    /// available as a positional argument on `enter`/`run`, e.g.
    /// `gpubox enter nvidia`.
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
    /// Enter an interactive shell inside the auto-detected GPU sandbox.
    Enter {
        /// Select a GPU by vendor name (nvidia, amd, intel, apple,
        /// vulkan, cpu) or index, same as `--gpu`. A positional
        /// convenience for the common case, e.g. `gpubox enter nvidia`.
        gpu: Option<String>,
        /// Use a container name other than the default (the resolved
        /// stack, e.g. `cuda`/`rocm`/`vulkan`). Either way the container
        /// is created once and reattached on every later `enter`/`run`.
        /// Use `gpubox rm [name]` to remove it and start fresh.
        /// Docker/Podman only.
        #[arg(long, conflicts_with = "ephemeral")]
        name: Option<String>,
        /// Use a throwaway container for this run instead of the
        /// persistent default: torn down on exit, nothing installed
        /// inside it survives. Docker/Podman only.
        #[arg(long = "rm")]
        ephemeral: bool,
    },
    /// Run a single command inside the sandbox, non-interactively.
    Run {
        /// See `enter`'s `--name`.
        #[arg(long, conflicts_with = "ephemeral")]
        name: Option<String>,
        /// See `enter`'s `--rm`.
        #[arg(long = "rm")]
        ephemeral: bool,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    /// Remove a persistent container (see `enter`/`run`), so the next
    /// invocation with the same name starts completely fresh.
    Rm {
        /// The container name (defaults to `enter`'s: the resolved
        /// stack, e.g. `cuda`/`rocm`/`vulkan`, if omitted).
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

impl Cli {
    fn overrides(&self, positional_gpu: Option<&str>) -> Overrides {
        Overrides {
            backend: self.backend.clone(),
            image: self.image.clone(),
            gfx_override: self.gfx_override.clone(),
            gpu: positional_gpu
                .map(str::to_string)
                .or_else(|| self.gpu.clone()),
            name: None,
            ephemeral: false,
            no_home: self.no_home,
            read_only_home: self.read_only_home,
        }
    }
}

pub fn run() -> Result<i32> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Enter {
            gpu,
            name,
            ephemeral,
        } => {
            let mut overrides = cli.overrides(gpu.as_deref());
            overrides.name = name.clone();
            overrides.ephemeral = *ephemeral;
            execute(&overrides, Vec::new(), true, cli.dry_run)
        }
        Command::Run {
            name,
            ephemeral,
            command,
        } => {
            if command.is_empty() {
                anyhow::bail!(
                    "`gpubox run` requires a command, e.g. `gpubox run -- python train.py`"
                );
            }
            let mut overrides = cli.overrides(None);
            overrides.name = name.clone();
            overrides.ephemeral = *ephemeral;
            execute(&overrides, command.clone(), false, cli.dry_run)
        }
        Command::Rm { name } => rm_cmd(&cli, name.as_deref()),
        Command::Generate { format, output } => {
            let overrides = cli.overrides(None);
            generate_cmd(&overrides, format.as_deref(), output.as_deref())?;
            Ok(0)
        }
        Command::Doctor { json, report } => {
            let overrides = cli.overrides(None);
            if *report {
                println!("{}", crate::doctor::probe_snapshot_json()?);
            } else if *json {
                println!("{}", crate::doctor::report_json(&overrides)?);
            } else {
                print!("{}", crate::doctor::report(&overrides)?);
            }
            Ok(0)
        }
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(*shell, &mut cmd, name, &mut std::io::stdout());
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
    // exit): every `enter`/`run` reattaches to a container named after
    // either `--name` or, absent that, the resolved stack
    // (`cuda`/`rocm`/`vulkan`/...), rather than a one-off `--rm`
    // container. `--rm` opts back into the old throwaway behavior.
    // Backends with no notion of a persistent container (Seatbelt runs
    // natively on the host; Windows Sandbox always boots a clean VM by
    // design) silently fall through to the ephemeral/native path below -
    // unless the user *explicitly* asked for `--name` on one of them,
    // which is an error rather than being quietly ignored.
    if !overrides.ephemeral {
        let name = overrides
            .name
            .clone()
            .unwrap_or_else(|| plan.resolved.stack.clone());
        match plan.engine.as_container_engine() {
            Some(engine) => return execute_named(engine, &mut plan, &name, dry_run),
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
/// create-or-reattach a persistent container, then exec into it. See
/// `container` module docs for the full rationale.
fn execute_named(
    engine: backend::linux::ContainerEngine,
    plan: &mut Plan,
    name: &str,
    dry_run: bool,
) -> Result<i32> {
    let full_name = container::container_name(name);
    let engine_program = engine.program();

    if !dry_run {
        backend::ensure_available(plan.engine)?;
        cache::ensure_cached_image(engine_program, &plan.resolved, &mut plan.spec)?;
    }

    let state = container::inspect(engine_program, &full_name);

    let mut setup: Vec<Invocation> = Vec::new();
    match state {
        container::ContainerState::Missing => {
            setup.push(container::create_invocation(engine, &plan.spec, &full_name));
        }
        container::ContainerState::Stopped => {
            setup.push(container::start_invocation(engine, &full_name));
        }
        container::ContainerState::Running => {}
    }
    let exec_invocation = container::exec_invocation(engine, &plan.spec, &full_name);

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

fn rm_cmd(cli: &Cli, name: Option<&str>) -> Result<i32> {
    // An explicit name needs nothing beyond the engine choice - skip the
    // hardware probe entirely (it walks /sys/class/drm on Linux or shells
    // out to WMI on Windows, so it's neither instant nor guaranteed to
    // succeed everywhere, and would be pure overhead here). Only fall
    // back to a full `launch::plan` when no name was given: `gpubox rm`
    // with no arguments has to know the resolved stack, since that's the
    // same name `enter`/`run` would use without `--name`, so it "just
    // works" as the mirror image of plain `gpubox enter`.
    let (engine, name) = match name {
        Some(name) => {
            let engine = match &cli.backend {
                Some(name) => backend::Engine::parse(name)
                    .ok_or_else(|| anyhow::anyhow!("unrecognized --backend `{name}`"))?,
                None => backend::Engine::default_for_platform(),
            };
            (engine, name.to_string())
        }
        None => {
            let overrides = cli.overrides(None);
            let plan = launch::plan(&overrides, Vec::new(), true)?;
            (plan.engine, plan.resolved.stack)
        }
    };

    let engine = engine.as_container_engine().ok_or_else(|| {
        anyhow::anyhow!("`gpubox rm` is only supported with the docker/podman backends")
    })?;
    let full_name = container::container_name(&name);

    if cli.dry_run {
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
