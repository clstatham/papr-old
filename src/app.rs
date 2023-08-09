use std::{sync::Arc, time::Duration};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};

use eframe::egui::CentralPanel;
use rustc_hash::FxHashMap;
use tokio::runtime::Runtime;

use crate::{
    dsp::{
        basic::{DebugNode, UiInput},
        AudioRate, ControlRate, Processor, Signal,
    },
    graph::{Connection, Graph, InputName, Node, NodeName, OutputName},
    parser::{parse_script, DualGraphs},
    Scalar,
};

pub struct AudioContext {
    out_device: cpal::Device,
}

impl AudioContext {
    fn write_data<T>(
        sample_rate: Scalar,
        graph: &Arc<Graph<AudioRate>>,
        output: &mut [T],
        channels: usize,
        t: Scalar,
    ) where
        T: SizedSample + FromSample<Scalar>,
    {
        let mut out = FxHashMap::default();
        for c in 0..channels {
            out.insert(OutputName(format!("dac{c}")), Signal::new_audio(0.0));
        }
        for (frame_idx, frame) in output.chunks_mut(channels).enumerate() {
            graph.process_graph(
                t as Scalar + frame_idx as Scalar / sample_rate,
                sample_rate,
                &FxHashMap::default(),
                &mut out,
            );
            for (c, sample) in frame.iter_mut().enumerate() {
                *sample = T::from_sample(out[&OutputName(format!("dac{c}"))].value());
            }
        }
    }

    pub fn run<T>(self, graph: Arc<Graph<AudioRate>>)
    where
        T: SizedSample + FromSample<Scalar>,
    {
        std::thread::Builder::new()
            .name("PAPR Audio".into())
            .spawn(move || {
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
                            sample_clock +=
                                (data.len() as Scalar / channels as Scalar) / sample_rate;
                            Self::write_data(
                                sample_rate as Scalar,
                                &graph,
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
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
            .expect("AudioContext::run(): error spawning audio rate thread");
    }
}

pub struct PaprApp {
    audio_cx: Option<AudioContext>,
    audio_graph: Option<Arc<Graph<AudioRate>>>,
    control_graph: Option<Arc<Graph<ControlRate>>>,
    ui_control_inputs: Vec<Arc<Node<ControlRate>>>,
    rt: Option<Runtime>,
    control_rate: Scalar,
}

impl PaprApp {
    pub fn new(control_rate: Scalar) -> Self {
        Self {
            audio_cx: None,
            audio_graph: None,
            control_graph: None,
            ui_control_inputs: Vec::new(),
            rt: None,
            control_rate,
        }
    }

    pub fn create_graphs(&mut self, sample_rate: Scalar, control_rate: Scalar) {
        let main_graphs = parse_script(
            include_str!("../test-scripts/test3.papr"),
            sample_rate,
            control_rate,
        )
        .remove(&NodeName("main".to_owned()))
        .unwrap();
        let DualGraphs {
            mut audio,
            name: _,
            mut control,
        } = main_graphs;

        for c_in_idx in control.graph_inputs.clone() {
            let c_in = &control.digraph[c_in_idx];
            let ui_name = c_in.name.0.to_owned();
            let (an, cn) = UiInput::create_nodes(c_in.inputs[&InputName::default()].clone());
            self.ui_control_inputs.push(cn.clone());
            audio.add_node(an);
            let c_in_ui_idx = control.add_node(cn);
            control.add_edge(
                c_in_ui_idx,
                c_in_idx,
                Connection {
                    source_output: OutputName(ui_name),
                    sink_input: InputName::default(),
                },
            );
        }

        for c_out_idx in control.graph_outputs.clone() {
            let c_out = &control.digraph[c_out_idx];
            let dbg_name = format!("{}_dbg", &c_out.name.0);
            let (_an, cn) = DebugNode::create_nodes(dbg_name.to_owned());
            let debug0_cn = control.add_node(cn);
            control.add_edge(
                c_out_idx,
                debug0_cn,
                Connection {
                    source_output: OutputName::default(),
                    sink_input: InputName::default(),
                },
            );
        }

        self.audio_graph = Some(Arc::new(audio));
        self.control_graph = Some(Arc::new(control));
    }

    pub fn init(&mut self) {
        if self.audio_cx.is_none() {
            let host = cpal::host_from_id(cpal::HostId::Jack)
                .expect("PaprApp::init(): no JACK host available");
            let out_device = host
                .default_output_device()
                .expect("PaprApp::init(): failed to find output device");
            let inner = AudioContext { out_device };
            self.audio_cx = Some(inner);
        } else {
            eprintln!("PaprApp::init(): audio context already initialized");
        }

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

        self.create_graphs(
            self.audio_cx
                .as_ref()
                .unwrap()
                .out_device
                .default_output_config()
                .unwrap()
                .sample_rate()
                .0 as Scalar,
            self.control_rate as Scalar,
        );
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
        let control_graph = self
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
            .name("PAPR Control".into())
            .spawn(move || {
                rt.block_on(async {
                    #[allow(clippy::unnecessary_cast)]
                    let mut clk = tokio::time::interval(Duration::from_secs_f64(
                        (control_rate as f64).recip(),
                    ));
                    let mut t = 0 as Scalar;
                    loop {
                        control_graph.process_graph(
                            t,
                            control_rate,
                            &FxHashMap::default(),
                            &mut FxHashMap::default(),
                        );
                        clk.tick().await;
                        t += control_rate.recip();
                    }
                });
            })
            .expect("PaprApp::spawn(): error spawning control rate thread");
    }
}

impl eframe::App for PaprApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // ctx.request_repaint();

        CentralPanel::default().show(ctx, |ui| {
            for inp in self.ui_control_inputs.iter() {
                inp.processor.ui_update(ui);
            }
        });
    }
}
