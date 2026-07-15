//! End-to-end CLI tests for the "persistent by default" behavior: every
//! `enter`/`run` should reattach to a container named after the resolved
//! stack unless `--rm` (throwaway) or `--name <other>` says otherwise.
//! Runs the actual compiled binary with `--dry-run` (which never touches
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
fn enter_defaults_to_a_persistent_container_named_after_the_stack() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "enter",
        "--dry-run",
    ]));
    assert!(out.contains("container: gpubox-cpu"));
    assert!(out.contains("docker run -d --name gpubox-cpu"));
    assert!(out.contains("docker exec"));
    // No `--rm` container flag anywhere in the persistent path.
    assert!(!out.contains("run --rm"));
}

#[test]
fn enter_rm_falls_back_to_the_old_throwaway_container() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "enter",
        "--rm",
        "--dry-run",
    ]));
    assert!(out.contains("docker run --rm -it"));
    assert!(!out.contains("--name"));
}

#[test]
fn enter_name_uses_that_name_instead_of_the_stack() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "enter",
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
            "enter",
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
fn rm_with_no_name_defaults_to_the_stack_container() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "--dry-run",
        "rm",
    ]));
    assert!(out.contains("docker rm -f gpubox-cpu"));
}

#[test]
fn rm_with_explicit_name_uses_it() {
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "docker",
        "--dry-run",
        "rm",
        "ml",
    ]));
    assert!(out.contains("docker rm -f gpubox-ml"));
}

#[test]
fn run_is_persistent_by_default_too() {
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
    assert!(out.contains("container: gpubox-cpu"));
    assert!(out.contains("python train.py"));
}

#[test]
fn backends_with_no_persistence_story_silently_use_the_native_ephemeral_path() {
    // Seatbelt has no container to persist - `enter` (no explicit --name)
    // must fall through to the old native path without erroring.
    let out = stdout_of(gpubox().args([
        "--gfx-override",
        "cpu",
        "--backend",
        "seatbelt",
        "enter",
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
            "enter",
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
