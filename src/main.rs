// #![warn(clippy::unwrap_used)]
use std::{error::Error, path::PathBuf, time::Duration};

use app::PaprApp;
use clap::Parser;
use eframe::{egui::Visuals, NativeOptions};

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
    /// Path to the `.papr` script to load
    script_path: Option<PathBuf>,

    /// The desired control rate to use (in samples per second)
    #[arg(short, long, default_value_t = 1000, value_name = "HERTZ")]
    control_rate: u64,

    /// The desired audio sample rate to request (in samples per second)
    #[arg(short, long, default_value_t = 48000, value_name = "HERTZ")]
    sample_rate: u64,

    /// The desired audio buffer length to request (in samples)
    #[arg(short, long, default_value_t = 1024, value_name = "SAMPLES")]
    buffer_len: u64,

    /// The MIDI input port id to use
    #[arg(short, long, default_value_t = 0, value_name = "PORT")]
    midi_port: usize,

    /// On Linux, Force ALSA as the audio backend, instead of JACK
    #[arg(long, default_value_t = false)]
    force_alsa: bool,

    /// Don't start the GUI and instead run the loaded script with default input values
    #[arg(long, default_value_t = false)]
    headless: bool,

    /// In headless mode, how long to run in milliseconds (0 to run until ctrl+c unless --out-path specified)
    #[arg(long, default_value_t = 0, value_name = "MILLISECONDS")]
    run_for: u64,

    /// In headless mode, file to write audio data to (offline) instead of using the realtime audio backend
    #[arg(short, long)]
    out_path: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::trace!("Logger initialized.");
    let args = Args::parse();
    if args.headless {
        let script_path = args
            .script_path
            .expect("A script path is required when running with --headless");
        let mut app = PaprApp::new(
            Some(script_path),
            args.sample_rate as Scalar,
            args.control_rate as Scalar,
            args.buffer_len as usize,
            args.midi_port,
            #[cfg(target_os = "linux")]
            args.force_alsa,
            args.out_path.clone(),
            Some(args.run_for),
        );
        app.init();
        app.load_script_file();
        app.spawn()?;

        if args.out_path.is_none() {
            if args.run_for > 0 {
                std::thread::sleep(Duration::from_millis(args.run_for));
            } else {
                loop {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }

        Ok(())
    } else {
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
                    args.midi_port,
                    #[cfg(target_os = "linux")]
                    args.force_alsa,
                    None,
                    None,
                );
                if args.script_path.is_some() {
                    app.init();
                    app.load_script_file();
                }

                Box::new(app)
            }),
        )?;
        Ok(())
    }
}
