use app::PaprApp;
use clap::Parser;
use eframe::{egui::Visuals, NativeOptions};

pub mod app;
#[macro_use]
pub mod dsp;
#[macro_use]
pub mod graph;
pub mod parser;

cfg_if::cfg_if! {
    if #[cfg(feature = "double")] {
        pub type Scalar = f64;
        pub const PI: f64 = std::f64::consts::PI;
    } else {
        pub type Scalar = f32;
        pub const PI: f32 = std::f32::consts::PI;
    }
}

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long, default_value_t = 400)]
    control_rate: u64,
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

            let mut app = PaprApp::new(args.control_rate as Scalar);
            app.init();
            app.spawn();

            Box::new(app)
        }),
    )
    .unwrap();
}
