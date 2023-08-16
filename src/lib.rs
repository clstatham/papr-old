#[macro_use]
pub mod dsp;
#[macro_use]
pub mod graph;
pub mod app;
pub mod io;
pub mod parser2;
pub mod parser3;

cfg_if::cfg_if! {
    if #[cfg(feature = "double")] {
        pub type Scalar = f64;
        pub const PI: f64 = std::f64::consts::PI;
        pub const TAU: f64 = std::f64::consts::TAU;
    } else {
        pub type Scalar = f32;
        pub const PI: f32 = std::f32::consts::PI;
        pub const TAU: f32 = std::f32::consts::TAU;
    }
}
