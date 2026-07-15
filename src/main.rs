use std::process::ExitCode;

fn main() -> ExitCode {
    match gpubox::cli::run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            eprintln!("gpubox: error: {err:?}");
            ExitCode::FAILURE
        }
    }
}
