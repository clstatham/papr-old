#![warn(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::result_large_err)]
use std::{path::PathBuf, time::Duration};

use app::PaprApp;
use clap::Parser;
use eframe::{egui::Visuals, NativeOptions};
use miette::{Diagnostic, Result};
use thiserror::Error;

use crate::app::PaprRuntime;

pub mod app;
#[macro_use]
pub mod dsp;
#[macro_use]
pub mod graph;
pub mod io;
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

#[derive(Debug, Error, Diagnostic)]
pub enum PaprError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("DSP error: {0}")]
    Dsp(#[from] dsp::DspError),
    #[error("Parse error: {0}")]
    Parse(#[from] parser3::ParseError),
    #[error("MIDI error: {0}")]
    Midi(#[from] io::midi::MidiError),
    #[error("Graph error: {0}")]
    Graph(#[from] graph::GraphError),
    #[error("Runtime error: {0}")]
    Runtime(#[from] app::RuntimeError),
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

fn main() -> Result<()> {
    env_logger::init();
    log::trace!("Logger initialized.");
    let args = Args::parse();
    if args.headless {
        let script_path = args
            .script_path
            .ok_or(PaprError::Runtime(app::RuntimeError::ScriptPathRequired))?;
        let rt = PaprRuntime::new(
            args.sample_rate as Scalar,
            args.control_rate as Scalar,
            args.out_path.clone(),
            Some(args.run_for),
        );
        let mut app = PaprApp::new(rt);
        app.init_midi()?;
        app.init_audio(
            #[cfg(target_os = "linux")]
            args.force_alsa,
        )?;
        app.rt.init();
        app.load_script_file(&script_path).map_err(|e| {
            log::error!("Error loading script: {}", e);
            e
        })?;
        app.rt.spawn(&script_path, app.audio_cx.take())?;

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
        // the app constructor can't return a Result
        #[allow(clippy::unwrap_used)]
        eframe::run_native(
            "PAPR",
            NativeOptions::default(),
            Box::new(move |cc| {
                cc.egui_ctx.set_visuals(Visuals::dark());

                let rt = PaprRuntime::new(
                    args.sample_rate as Scalar,
                    args.control_rate as Scalar,
                    None,
                    None,
                );
                let mut app = PaprApp::new(rt);
                if let Some(script_path) = args.script_path.as_ref() {
                    app.init_midi().unwrap();
                    app.init_audio(
                        #[cfg(target_os = "linux")]
                        args.force_alsa,
                    )
                    .unwrap();
                    app.rt.init();
                    app.load_script_file(script_path).unwrap();
                }

                Box::new(app)
            }),
        );
        Ok(())
    }
}
