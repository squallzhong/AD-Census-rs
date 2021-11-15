use getset::{CopyGetters, Getters, MutGetters, Setters};

#[derive(Getters, Setters, MutGetters, CopyGetters, Copy, Clone, Debug)]
pub struct Point {
    #[getset(get = "pub", set = "pub")]
    x: usize,
    #[getset(get = "pub", set = "pub")]
    y: usize,
}
impl Point {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}
impl Default for Point {
    fn default() -> Self {
        Self {
            x: 0usize,
            y: 0usize,
        }
    }
}
