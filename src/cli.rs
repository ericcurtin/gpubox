//! Command-line interface: `enter`, `run`, `generate`, `doctor`.

use crate::backend::{self, Invocation};
use crate::generate::{self, Format};
use crate::launch::{self, Overrides, Plan};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "gpubox",
    version,
    about = "Auto-detecting GPU container launcher - distrobox-style host integration, \
             no host GPU driver install required."
)]
pub struct Cli {
    /// Force a specific backend instead of the platform default (docker,
    /// podman, seatbelt, windows-sandbox).
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

    /// Print the command(s) that would be run instead of running them.
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Enter an interactive shell inside the auto-detected GPU sandbox.
    Enter,
    /// Run a single command inside the sandbox, non-interactively.
    Run {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    /// Emit the equivalent Containerfile / Compose / Quadlet / Seatbelt /
    /// `.wsb` config instead of launching anything.
    Generate {
        /// containerfile, compose, quadlet, seatbelt, or windows-sandbox.
        /// Defaults to whatever's idiomatic for the host platform.
        #[arg(long)]
        format: Option<String>,
        /// Write to this path instead of stdout.
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },
    /// Print detected hardware, chosen stack, and how to override it.
    Doctor,
}

impl Cli {
    fn overrides(&self) -> Overrides {
        Overrides {
            backend: self.backend.clone(),
            image: self.image.clone(),
            gfx_override: self.gfx_override.clone(),
        }
    }
}

pub fn run() -> Result<i32> {
    let cli = Cli::parse();
    let overrides = cli.overrides();

    match &cli.command {
        Command::Enter => execute(&overrides, Vec::new(), true, cli.dry_run),
        Command::Run { command } => {
            if command.is_empty() {
                anyhow::bail!(
                    "`gpubox run` requires a command, e.g. `gpubox run -- python train.py`"
                );
            }
            execute(&overrides, command.clone(), false, cli.dry_run)
        }
        Command::Generate { format, output } => {
            generate_cmd(&overrides, format.as_deref(), output.as_deref())?;
            Ok(0)
        }
        Command::Doctor => {
            print!("{}", crate::doctor::report(&overrides)?);
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
    let plan: Plan = launch::plan(overrides, command, interactive)?;
    let invocation = backend::build_invocation(plan.engine, &plan.spec)?;

    if dry_run {
        print_dry_run(&plan, &invocation);
        return Ok(0);
    }

    backend::ensure_available(plan.engine)?;
    write_generated_files(&invocation)?;

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
    print!("{}", invocation.program);
    for arg in &invocation.args {
        print!(" {}", shell_quote(arg));
    }
    println!();
    for (path, _content) in &invocation.generated_files {
        println!("# (would write {} )", path.display());
    }
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
