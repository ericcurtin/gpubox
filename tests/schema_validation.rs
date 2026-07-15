//! Integration-level schema validation for the two community-maintained
//! TOML databases (`data/pci_ids.toml`, `data/quirks.toml`). Runs as a
//! normal `cargo test`, which every job in `.github/workflows/ci.yml`
//! already executes on every push and PR - so a community PR that breaks
//! either file's structural invariants (a missing vendor fallback rule, a
//! malformed device id, a placeholder image reference, etc.) fails CI
//! instead of silently shipping.

#[test]
fn pci_ids_toml_is_structurally_valid() {
    gpubox::probe::validate_pci_ids_db().expect("data/pci_ids.toml failed schema validation");
}

#[test]
fn quirks_toml_is_structurally_valid() {
    gpubox::stack::validate_quirks_db().expect("data/quirks.toml failed schema validation");
}
