//! End-to-end CLI tests for the "persistent by default" behavior: every
//! `gpubox run` should reattach to a single default container (`gpubox`)
//! unless `--rm` (throwaway) or `--name <other>` says otherwise. Runs the
//! actual compiled binary with `--dry-run` (which never touches
//! docker/podman) so these exercise the real argument parsing and
//! dispatch in `src/cli.rs`, not just the library functions it calls.

use std::process::Command;

fn gpubox() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gpubox"))
}

fn stdout_of(cmd: &mut Command) -> String {
    let output = cmd.output().expect("failed to run gpubox binary");
    assert!(
        output.status.success(),
        "gpubox exited with {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn interactive_run_defaults_to_the_single_persistent_gpubox_container() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "run",
        "--dry-run",
    ]));
    assert!(out.contains("container: gpubox "));
    assert!(out.contains("docker run -d --name gpubox "));
    assert!(out.contains("docker exec"));
    // No `--rm` container flag anywhere in the persistent path.
    assert!(!out.contains("run --rm"));
}

#[test]
fn run_rm_falls_back_to_the_old_throwaway_container() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "run",
        "--rm",
        "--dry-run",
    ]));
    assert!(out.contains("docker run --rm -it"));
    assert!(!out.contains("--name"));
}

#[test]
fn run_name_uses_that_name_instead_of_the_default() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "run",
        "--name",
        "ml",
        "--dry-run",
    ]));
    assert!(out.contains("container: gpubox-ml"));
}

#[test]
fn name_and_rm_together_is_rejected() {
    let output = gpubox()
        .args([
            "--gfx-override",
            "cpu",
            "--backend",
            "docker",
            "run",
            "--name",
            "ml",
            "--rm",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot be used with"));
}

#[test]
fn rm_with_no_name_defaults_to_the_gpubox_container() {
    let out = stdout_of(gpubox().args(["--backend", "docker", "--dry-run", "rm"]));
    assert!(out.contains("docker rm -f gpubox"));
    // Exactly `gpubox`, not e.g. `gpubox-cpu` - the default no longer
    // depends on the resolved stack, so this must never probe hardware.
    assert!(out.trim().ends_with("gpubox"));
}

#[test]
fn rm_with_explicit_name_uses_it() {
    let out = stdout_of(gpubox().args(["--backend", "docker", "--dry-run", "rm", "ml"]));
    assert!(out.contains("docker rm -f gpubox-ml"));
}

#[test]
fn run_with_a_trailing_command_is_non_interactive_and_persistent_too() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "run",
        "--dry-run",
        "--",
        "python",
        "train.py",
    ]));
    assert!(out.contains("container: gpubox "));
    assert!(out.contains("python train.py"));
}

#[test]
fn backends_with_no_persistence_story_silently_use_the_native_ephemeral_path() {
    // Seatbelt has no container to persist - `run` (no explicit --name)
    // must fall through to the old native path without erroring.
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "seatbelt",
        "run",
        "--dry-run",
    ]));
    assert!(out.contains("sandbox-exec"));
}

#[test]
fn explicit_name_on_an_unsupported_backend_is_an_error() {
    let output = gpubox()
        .args([
            "--gfx-override",
            "cpu",
            "--backend",
            "seatbelt",
            "run",
            "--name",
            "ml",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--name is only supported with the docker/podman backends"));
}
