use thiserror::Error;

#[derive(Error, Debug)]
pub enum GoldyError {
    #[error("Failed to handle file.")]
    FileError(#[from] std::io::Error),
    #[error("invalid value (expected {expected:?}, found {found:?})")]
    ValueError { expected: String, found: String },
    #[error("expected value to be greater than 0, but was {found:?}")]
    NegativValue { found: String },
}
