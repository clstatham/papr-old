use std::time::Duration;

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};

use tokio::runtime::Runtime;

use crate::{
    dsp::{
        generators::{SineOsc, SineOscA},
        AudioProcessor, AudioSignal, SignalImpl,
    },
    graph::{AudioConnection, AudioGraph, ControlGraph, CreateNode},
    Scalar,
};

pub struct AudioContext {
    host: cpal::Host,
    out_device: cpal::Device,
}

impl AudioContext {
    fn write_data<T>(
        sample_rate: Scalar,
        graph: &mut AudioGraph,
        output: &mut [T],
        channels: usize,
        t: Scalar,
    ) where
        T: SizedSample + FromSample<Scalar>,
    {
        for (frame_idx, frame) in output.chunks_mut(channels).enumerate() {
            let mut out = vec![AudioSignal(0 as Scalar); channels];
            graph.process_audio(
                t as Scalar + frame_idx as Scalar / sample_rate,
                &[],
                &mut out,
            );
            for (c, sample) in frame.iter_mut().enumerate() {
                *sample = T::from_sample(out[c].value());
            }
        }
    }

    pub fn run<T>(self, mut graph: AudioGraph)
    where
        T: SizedSample + FromSample<Scalar>,
    {
        std::thread::spawn(move || {
            let config = self.out_device.default_output_config().unwrap();
            let sample_rate = config.sample_rate().0 as Scalar;
            let channels = config.channels() as usize;

            let mut sample_clock = 0 as Scalar;
            let err_fn = |err| eprintln!("Error occurred on stream: {err}");

            let stream = self
                .out_device
                .build_output_stream(
                    &config.into(),
                    move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                        sample_clock += (data.len() as Scalar / channels as Scalar) / sample_rate;
                        Self::write_data(
                            sample_rate as Scalar,
                            &mut graph,
                            data,
                            channels,
                            sample_clock,
                        );
                    },
                    err_fn,
                    None,
                )
                .unwrap();
            stream.play().unwrap();
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        });
    }
}

pub struct PaprApp {
    audio_cx: Option<AudioContext>,
    audio_graph: Option<AudioGraph>,
    control_graph: Option<ControlGraph>,
    rt: Option<Runtime>,
    control_rate: Scalar,
}

impl PaprApp {
    pub fn new(control_rate: Scalar) -> Self {
        Self {
            audio_cx: None,
            audio_graph: None,
            control_graph: None,
            rt: None,
            control_rate,
        }
    }

    pub fn init(&mut self) {
        if self.audio_cx.is_none() {
            let host = cpal::default_host();
            let out_device = host
                .default_output_device()
                .expect("PaprApp::init(): failed to find output device");
            let inner = AudioContext { host, out_device };
            self.audio_cx = Some(inner);
        } else {
            eprintln!("PaprApp::init(): audio context already initialized");
        }

        // if self.audio_graph.is_none() {
        let mut audio_graph = AudioGraph::new();
        let mut control_graph = ControlGraph::new();

        // test stuff follows

        let dac0 = audio_graph.add_dac();
        let dac1 = audio_graph.add_dac();

        let (an, cn) = SineOsc::create_nodes();

        let sine_an = audio_graph.add_node(an);
        audio_graph.add_edge(
            sine_an,
            dac0,
            AudioConnection {
                source_output_index: 0,
                sink_input_index: 0,
            },
        );
        audio_graph.add_edge(
            sine_an,
            dac1,
            AudioConnection {
                source_output_index: 0,
                sink_input_index: 0,
            },
        );
        let sine_cn = control_graph.add_node(cn);

        // end test stuff

        // }

        // if self.control_graph.is_none() {

        self.audio_graph = Some(audio_graph);
        self.control_graph = Some(control_graph);
        // } else {
        // eprintln!("PaprApp::init(): graph already initialized");
        // }

        if self.rt.is_none() {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .enable_time()
                .build()
                .expect("PaprApp::init(): error creating new tokio runtime");
            self.rt = Some(rt);
        } else {
            eprintln!("PaprApp::init(): tokio runtime already initialized");
        }
    }

    pub fn spawn(&mut self) {
        let rt = self
            .rt
            .take()
            .expect("PaprApp::spawn(): runtime not initialized");
        let audio_cx = self
            .audio_cx
            .take()
            .expect("PaprApp::spawn(): audio context not initialized");
        let audio_graph = self
            .audio_graph
            .take()
            .expect("PaprApp::spawn(): audio graph not initialized");
        let mut control_graph = self
            .control_graph
            .take()
            .expect("PaprApp::spawn(): control graph not initialized");

        let config = audio_cx.out_device.default_output_config().unwrap();

        println!("Output device: {}", audio_cx.out_device.name().unwrap());
        println!("Output config: {:?}", config);
        println!("Control rate: {} Hz", self.control_rate);

        match config.sample_format() {
            cpal::SampleFormat::I8 => audio_cx.run::<i8>(audio_graph),
            cpal::SampleFormat::I16 => audio_cx.run::<i16>(audio_graph),
            cpal::SampleFormat::I32 => audio_cx.run::<i32>(audio_graph),
            cpal::SampleFormat::I64 => audio_cx.run::<i64>(audio_graph),
            cpal::SampleFormat::U8 => audio_cx.run::<u8>(audio_graph),
            cpal::SampleFormat::U16 => audio_cx.run::<u16>(audio_graph),
            cpal::SampleFormat::U32 => audio_cx.run::<u32>(audio_graph),
            cpal::SampleFormat::U64 => audio_cx.run::<u64>(audio_graph),
            cpal::SampleFormat::F32 => audio_cx.run::<f32>(audio_graph),
            cpal::SampleFormat::F64 => audio_cx.run::<f64>(audio_graph),
            f => panic!("PaprApp::spawn(): unsupported sample format {f:?}"),
        }
        let control_rate = self.control_rate;
        std::thread::Builder::new()
            .name("PAPR I/O".into())
            .spawn(move || {
                rt.block_on(async move {
                    let mut clk = tokio::time::interval(Duration::from_secs_f64(
                        (control_rate as f64).recip(),
                    ));
                    let mut t = 0 as Scalar;
                    loop {
                        control_graph.process_control(t);
                        clk.tick().await;
                        t += control_rate.recip();
                    }
                });
            })
            .expect("PaprApp::spawn(): error spawning I/O worker thread");
    }
}

impl eframe::App for PaprApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
    }
}
