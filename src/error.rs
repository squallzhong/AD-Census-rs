use std::{ffi::NulError, fmt};

#[derive(Debug)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl Error {
    pub fn new(code: i32, message: &str) -> Self {
        Self {
            code: code,
            message: message.to_string(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (code: {})", self.message, self.code)
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::new(1000, ERROR_1000.into())
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = ::std::result::Result<T, E>;

pub static ERROR_1000: &'static str = "Passed Rust string contains nul byte";
pub static ERROR_1001: &'static str = "Image width or height can not be less than 1";
pub static ERROR_1002: &'static str = "Image disparity range can not be less than 1";

pub static ERROR_1003: &'static str = "The argument: n or m must be odd";
