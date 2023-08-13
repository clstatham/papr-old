use std::path::PathBuf;

use app::PaprApp;
use clap::Parser;
use eframe::{egui::Visuals, NativeOptions};
use papr_proc_macro::node_constructor;

pub mod app;
#[macro_use]
pub mod dsp;
#[macro_use]
pub mod graph;
pub mod io;
pub mod parser2;

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

#[derive(clap::Parser)]
struct Args {
    script_path: Option<PathBuf>,

    #[arg(short, long, default_value_t = 1000)]
    control_rate: u64,
    #[arg(short, long, default_value_t = 48000)]
    sample_rate: u64,
    #[arg(short, long, default_value_t = 1024)]
    buffer_len: u64,
}

fn main() {
    env_logger::init();
    log::trace!("Logger initialized.");
    let args = Args::parse();
    eframe::run_native(
        "PAPR",
        NativeOptions::default(),
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(Visuals::dark());

            let mut app = PaprApp::new(
                args.script_path.clone(),
                args.sample_rate as Scalar,
                args.control_rate as Scalar,
                args.buffer_len as usize,
            );
            if args.script_path.is_some() {
                app.init();
                app.spawn();
            }

            Box::new(app)
        }),
    )
    .unwrap();
}
